import { initGpu, readBuffer } from './gpu';
import { Solver3D, PRESETS3, type Precision } from './solver3d';
import { deriveTau, maxReynolds } from './units';
import { mean } from './analysis';
import { ForceChart } from './chart';
import { initMenu, isShown, showToast } from './menu';
import { icon } from './icons';
import { OrbitCamera, invert } from './camera3d';
import { Renderer3D } from './render3d';
import { presetsThatFit, fallbackPreset, nextSmaller, buildBody, coefficientScale, tuneStepsPerFrame, hasDiverged, fieldDiverged, type PresetName } from './tunnel3d';
import type { Obstacle3 } from './geometry3d';
import type { Vec3 } from './camera3d';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const input = (id: string) => $<HTMLInputElement>(id);
const select = (id: string) => $<HTMLSelectElement>(id);

const U_TARGET = 0.1;
const RAMP_STEPS = 3000;
const FORCE_EVERY = 4;

const canvas = $<HTMLCanvasElement>('view');

let gpu: Awaited<ReturnType<typeof initGpu>>;
try {
  gpu = await initGpu({ f16: true });
} catch (e) {
  showError(`${(e as Error).message} This wind tunnel needs WebGPU: current Chrome, Edge or Safari 26, or Firefox on Windows or Apple Silicon.`);
  throw e;
}
const { device } = gpu;
const context = canvas.getContext('webgpu')!;
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
const renderer = new Renderer3D(device, context, format);
await renderer.init();
const chart = new ForceChart($<HTMLCanvasElement>('chart'), $('chartTip'));

function showError(text: string) {
  const box = $('nogpu');
  box.hidden = false;
  box.textContent = text;
  canvas.hidden = true;
}

const state = {
  /** Blow-ups recovered from since load, for tests. */
  recoveries: 0,
  /** A recovery happened and nothing has changed since, so another blow-up means the setup itself is unstable. */
  recentRecovery: false,
  /** Lattice steps run since the last recovery, independent of resets. */
  stepsSinceRecovery: 0,
  /** Bumped on every flow reset, so field readings taken before a reset are ignored. */
  flowGen: 0,
  solver: null as Solver3D | null,
  camera: null as OrbitCamera | null,
  paused: false,
  spf: 5,
  rampFrom: 0,
  lRef: 1,
  area: 0,
  bounds: null as [Vec3, Vec3] | null,
  gpuMs: 0,
  frames: 0,
  rate: { t: 0, step: 0, value: 0 },
};

function settings() {
  return {
    grid: select('grid').value as PresetName,
    precision: select('precision').value as Precision,
    obstacle: select('obstacle').value as Obstacle3,
    size: Number(input('size').value),
    angle: Number(input('angle').value),
    re: Math.max(1, Number(input('re').value) || 1),
    smag: input('smag').checked,
    cs: Number(input('cs').value),
  };
}

function note(id: string, text: string) {
  $(id).hidden = !text;
  $(id).textContent = text;
}

/** Enables exactly the grid presets that fit the device at this precision. */
function refreshPresets(precision: Precision) {
  const fit = presetsThatFit(precision, device.limits);
  const bytesPer = precision === 'fp16' ? 2 : 4;
  for (const opt of Array.from(select('grid').options)) {
    const [W, H, D] = PRESETS3[opt.value as PresetName];
    const ok = fit.includes(opt.value as PresetName);
    opt.disabled = !ok;
    opt.title = ok ? '' : `Needs ${(19 * bytesPer * W * H * D).toLocaleString()} bytes per buffer; this device allows ${device.limits.maxStorageBufferBindingSize.toLocaleString()} per binding and ${device.limits.maxBufferSize.toLocaleString()} per buffer`;
  }
}

function applyPhysics() {
  state.recentRecovery = false;
  const solver = state.solver;
  if (!solver) return;
  const s = settings();
  const { tau, clamped, effectiveRe, mach } = deriveTau(s.re, U_TARGET, state.lRef);
  solver.setTau(tau);
  solver.setSmagorinsky(s.smag ? s.cs : 0);
  $('csOut').textContent = s.cs.toFixed(3);
  $('tauNote').textContent = `τ = ${tau.toFixed(4)}, ν = ${((tau - 0.5) / 3).toExponential(2)}, Mach ${mach.toFixed(2)}, L = ${state.lRef.toFixed(1)} cells`;
  const notes: string[] = [];
  if (clamped && state.area > 0)
    notes.push(`Re ${s.re} needs τ below 0.51, which is unstable. Running at Re ${effectiveRe.toFixed(0)}; enlarge the body or the grid for more (max ${maxReynolds(U_TARGET, state.lRef).toFixed(0)} at this size).`);
  if (s.smag) notes.push('Smagorinsky here stabilizes coarse grids; the grid is far too coarse to resolve turbulence.');
  note('reNote', notes.join(' '));
}

function resetForces() {
  state.solver?.resetForces();
  chart.clear();
}

/** Swaps in the current obstacle; the flow keeps running, as the 2D resetBody does. */
function resetBody() {
  state.recentRecovery = false;
  const solver = state.solver;
  if (!solver) return;
  const s = settings();
  const body = buildBody({ obstacle: s.obstacle, sizeFraction: s.size, angleDeg: s.angle }, solver.W, solver.H, solver.D);
  solver.setSdf(body.sdf);
  state.lRef = body.lRef;
  state.area = body.area;
  state.bounds = body.bounds;
  $('sizeOut').textContent = (s.size * solver.H).toFixed(1);
  $('angleOut').textContent = String(s.angle);
  input('angle').disabled = s.obstacle !== 'cube' && s.obstacle !== 'wing';
  for (const el of Array.from(document.querySelectorAll<HTMLElement>('.coef'))) el.hidden = s.obstacle === 'none';
  applyPhysics();
  resetForces();
}

const BLOWUP_MESSAGE =
  'The flow became unstable: the local speed near the obstacle exceeded what the lattice can represent. Try a smaller obstacle or a lower Reynolds number.';
const BLOWUP_REPEAT = "The flow keeps becoming unstable with this setup, so it's paused. Change the obstacle or lower the Reynolds number, then press play.";

/** Repeat blow-ups within this many steps of a recovery pause the flow instead of looping. */
const REPEAT_STEPS = 10_000;

function recoverFromBlowup(message: string, repeatMessage: string) {
  const repeat = state.recentRecovery;
  resetFlow();
  state.recoveries++;
  state.recentRecovery = true;
  state.stepsSinceRecovery = 0;
  if (repeat) {
    if (!state.paused) $('pause').click();
    showToast(repeatMessage, true);
  } else showToast(message);
}

function resetFlow() {
  state.flowGen++;
  // Start from rest: initField writes the equilibrium at the current inlet speed, and the ramp restarts from 0.
  state.solver?.setInlet(0);
  state.solver?.initField();
  state.rampFrom = 0;
  resetForces();
}

let rebuildToken = 0;

/** Rebuilds the solver; the grid note shows the caller's note plus any allocation fallbacks, or nothing. */
async function rebuild(callerNote = '') {
  state.recentRecovery = false;
  const token = ++rebuildToken;
  const s = settings();
  state.solver?.destroy();
  state.solver = null;
  let grid: PresetName | null = s.grid;
  let solver: Solver3D | null = null;
  const failures: string[] = [];
  while (grid && !solver) {
    const [W, H, D] = PRESETS3[grid];
    try {
      solver = await Solver3D.create(device, { width: W, height: H, depth: D, precision: s.precision, tau: 0.6, uIn: 0, spongeFraction: 0.15, spongeTau: 1, absorb: 0.02, forceEvery: FORCE_EVERY });
    } catch (e) {
      const msg = (e as Error).message;
      if (!msg.startsWith('Solver3D allocation failed')) throw e;
      const smaller = nextSmaller(grid);
      failures.push(`Grid ${grid} didn't fit (${msg})${smaller ? `; using ${smaller}` : ''}.`);
      grid = smaller;
    }
  }
  // A later rebuild started while this one was allocating; its settings win.
  if (token !== rebuildToken) {
    solver?.destroy();
    return;
  }
  if (!solver) {
    showError(failures.join(' '));
    return;
  }
  select('grid').value = grid!;
  note('gridNote', [callerNote, ...failures].filter(Boolean).join(' '));
  state.solver = solver;
  state.camera = new OrbitCamera([(solver.W - 1) / 2, (solver.H - 1) / 2, (solver.D - 1) / 2], 1.6 * solver.W);
  renderer.attach(solver);
  resetBody();
  resetFlow();
  state.rate = { t: performance.now(), step: 0, value: 0 };
}

/** Sizes the drawing buffer to the canvas's exact device pixels; the menu overlay never changes this size. */
function watchCanvasSize(onResize: (w: number, h: number) => void) {
  new ResizeObserver(([e]) => {
    const dp = e.devicePixelContentBoxSize?.[0];
    const w = dp ? dp.inlineSize : Math.round(e.contentBoxSize[0].inlineSize * devicePixelRatio);
    const h = dp ? dp.blockSize : Math.round(e.contentBoxSize[0].blockSize * devicePixelRatio);
    if (w > 0 && h > 0 && (canvas.width !== w || canvas.height !== h)) {
      canvas.width = w;
      canvas.height = h;
      onResize(w, h);
    }
  }).observe(canvas);
}

// Orbit on drag, zoom on wheel.
let drag: [number, number] | null = null;
canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  drag = [e.clientX, e.clientY];
});
canvas.addEventListener('pointermove', (e) => {
  if (!drag || !state.camera) return;
  state.camera.rotate(e.clientX - drag[0], e.clientY - drag[1]);
  drag = [e.clientX, e.clientY];
});
canvas.addEventListener('pointerup', () => (drag = null));
canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    state.camera?.zoom(Math.exp(e.deltaY * 0.001));
  },
  { passive: false },
);

select('grid').addEventListener('change', () => void rebuild(f16Note));
select('precision').addEventListener('change', () => {
  const s = settings();
  refreshPresets(s.precision);
  const grid = fallbackPreset(s.grid, s.precision, device.limits);
  let fallback = '';
  if (grid && grid !== s.grid) {
    select('grid').value = grid;
    fallback = `Grid ${s.grid} doesn't fit at ${s.precision.toUpperCase()}; using ${grid}.`;
  }
  void rebuild([f16Note, fallback].filter(Boolean).join(' '));
});
select('obstacle').addEventListener('change', resetBody);
select('sliceAxis').addEventListener('change', () => (input('slicePos').disabled = select('sliceAxis').value === 'off'));
for (const id of ['size', 'angle']) input(id).addEventListener('input', resetBody);
for (const id of ['re', 'smag', 'cs']) input(id).addEventListener('input', applyPhysics);
$('pause').addEventListener('click', () => {
  state.paused = !state.paused;
  const label = state.paused ? 'Run' : 'Pause';
  $('pause').innerHTML = icon(state.paused ? 'play' : 'pause');
  $('pause').setAttribute('aria-label', label);
  $('pause').title = label;
});
$('resetFlow').addEventListener('click', resetFlow);
addEventListener('keydown', (e) => {
  const t = e.target as HTMLElement;
  if (e.code === 'Space' && !t.closest('input, select, textarea, button, a, summary')) {
    e.preventDefault();
    $('pause').click();
  }
});

// GPU timing: the compute pass is bracketed by timestamps, read back without stalling the frame loop.
const querySet = gpu.timestamps ? device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
const resolveBuf = querySet ? device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
const stagingPool: GPUBuffer[] = querySet
  ? [0, 1, 2].map(() => device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }))
  : [];
let lastFrameTime = performance.now();
let forcesInFlight = false;
let fieldInFlight = false;

function tuneSteps(frameMs: number) {
  // A fixed budget, whatever the display rate: GPU time when timestamps have measured it, frame time otherwise.
  state.spf = tuneStepsPerFrame(state.spf, state.gpuMs > 0 ? state.gpuMs : frameMs - 3);
}

function updateStats() {
  const { cd, cl } = chart.data;
  if (cd.length < 10) return;
  const tail = Math.floor(cd.length / 2);
  $('cd').textContent = mean(cd.slice(tail)).toFixed(4);
  $('cl').textContent = mean(cl.slice(tail)).toFixed(4);
}

function frame(now: number) {
  requestAnimationFrame(frame);
  const solver = state.solver;
  const camera = state.camera;
  if (!solver || !camera) return;
  const frameMs = now - lastFrameTime;
  lastFrameTime = now;
  state.frames++;
  // Paused frames measure nothing; tuning on a stale cost would drift toward MAX_SPF.
  if (!state.paused) tuneSteps(frameMs);
  const n = state.paused ? 0 : state.spf;
  state.stepsSinceRecovery += n;
  if (state.recentRecovery && state.stepsSinceRecovery > REPEAT_STEPS) state.recentRecovery = false;

  const t = Math.min(1, (solver.step - state.rampFrom) / RAMP_STEPS);
  solver.setInlet(U_TARGET * t * t * (3 - 2 * t));

  const enc = device.createCommandEncoder();
  const staging = stagingPool.find((b) => b.mapState === 'unmapped' && !(b as GPUBuffer & { busy?: boolean }).busy);
  const timed = querySet && staging && n > 0;
  const pass = enc.beginComputePass(timed ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
  solver.encodeSteps(pass, n);
  solver.encodeMacro(pass);
  pass.end();
  if (timed) {
    enc.resolveQuerySet(querySet, 0, 2, resolveBuf!, 0);
    enc.copyBufferToBuffer(resolveBuf!, 0, staging, 0, 16);
  }
  const viewProj = camera.viewProj(canvas.width / canvas.height);
  const box = state.bounds;
  renderer.encode(enc, {
    viewProj,
    invViewProj: invert(viewProj),
    eye: camera.eye(),
    sliceAxis: (select('sliceAxis').value === 'off' ? 0 : Number(select('sliceAxis').value)) as 0 | 1 | 2,
    slice: select('sliceAxis').value !== 'off',
    slicePos: Number(input('slicePos').value),
    mode: Number(select('viewMode').value) as 0 | 1,
    uRef: U_TARGET,
    boxMin: box ? box[0] : [0, 0, 0],
    boxMax: box ? box[1] : [0, 0, 0],
    hasBody: box !== null,
    tracers: input('tracers').checked,
    steps: n,
    count: 0,
    frame: state.frames,
  });
  device.queue.submit([enc.finish()]);
  renderer.afterSubmit();
  hook.ready = true;

  if (timed) {
    const b = staging as GPUBuffer & { busy?: boolean };
    b.busy = true;
    b.mapAsync(GPUMapMode.READ).then(() => {
      const ts = new BigUint64Array(b.getMappedRange());
      const ms = Number(ts[1] - ts[0]) / 1e6;
      b.unmap();
      b.busy = false;
      if (ms > 0 && ms < 1000) {
        state.gpuMs = 0.8 * state.gpuMs + 0.2 * ms;
        $('mlups').textContent = `${((solver.N * n) / (ms * 1e3)).toFixed(0)} MLUPS · ${n} steps/frame`;
      }
    });
  } else if (!querySet && state.frames % 30 === 0) {
    $('mlups').textContent = `${n} steps/frame (no GPU timer)`;
  }

  // Read even without a body: readForces is what trims the solver's pending sample list.
  if (!forcesInFlight && state.frames % 6 === 0) {
    forcesInFlight = true;
    const scale = coefficientScale(U_TARGET, state.area);
    solver.readForces().then((samples) => {
      if (solver === state.solver && hasDiverged(samples)) {
        // Once NaN appears every later step stays NaN: restart the flow, keeping the obstacle and settings.
        recoverFromBlowup(BLOWUP_MESSAGE, BLOWUP_REPEAT);
      } else if (solver === state.solver && scale > 0) for (const f of samples) chart.push(f.step, f.fx * scale, f.fy * scale);
      forcesInFlight = false;
      updateStats();
    });
  }

  // Forces are zero without an obstacle, so also watch a row of the field; a reading from before the latest reset is ignored.
  if (!fieldInFlight && state.frames % 15 === 0) {
    fieldInFlight = true;
    const gen = state.flowGen;
    readBuffer(device, solver.macro, (Math.floor(solver.H / 2) + solver.H * Math.floor(solver.D / 2)) * solver.W * 16, solver.W * 16).then((buf) => {
      fieldInFlight = false;
      if (solver === state.solver && gen === state.flowGen && fieldDiverged(new Float32Array(buf))) recoverFromBlowup(BLOWUP_MESSAGE, BLOWUP_REPEAT);
    });
  }
  if (state.frames % 10 === 0) {
    $('step').textContent = solver.step.toLocaleString();
    const r = state.rate;
    if (now - r.t >= 1000) {
      r.value = ((solver.step - r.step) * 1000) / (now - r.t);
      r.t = now;
      r.step = solver.step;
      $('sps').textContent = `${r.value.toFixed(0)} steps/s`;
    }
  }
  if (isShown(chartCanvas)) chart.draw();
}

refreshPresets(select('precision').value as Precision);
const f16Note = gpu.f16 ? '' : "FP16 isn't available on this device.";
if (!gpu.f16) {
  const fp16 = select('precision').options[0];
  fp16.disabled = true;
  select('precision').value = 'fp32';
  refreshPresets('fp32');
}
{
  const s = settings();
  const grid = fallbackPreset(s.grid, s.precision, device.limits) ?? 'low';
  select('grid').value = grid;
}
/** Test hook for tests/tunnel3d.spec.ts; ready once the first frame is submitted. */
const hook = {
  ready: false,
  step: () => state.solver?.step ?? 0,
  cd: () => {
    const { cd } = chart.data;
    return cd.length ? mean(cd.slice(Math.floor(cd.length / 2))) : NaN;
  },
  stepsPerSecond: () => state.rate.value,
  stepsPerFrame: () => state.spf,
  pixelCount: () => renderer.requestPixelCount(),
};
(window as unknown as { tunnel3d: typeof hook }).tunnel3d = hook;

const chartCanvas = $('chart');
(window as unknown as { flowResets: () => number }).flowResets = () => state.recoveries;
(window as unknown as { solverStep: () => number }).solverStep = () => state.solver?.step ?? 0;
/** Test hook: the camera angles, so tests can tell whether a canvas press orbited. */
(window as unknown as { interactionState: () => string }).interactionState = () => JSON.stringify([state.camera?.yaw, state.camera?.pitch]);
initMenu(() => chart.invalidate());
await new Promise<void>((resolve) => watchCanvasSize((w, h) => (renderer.resize(w, h), resolve())));
await rebuild(f16Note);
requestAnimationFrame(frame);
