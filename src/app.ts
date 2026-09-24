import { initGpu, readBuffer } from './gpu';
import { Solver } from './solver';
import { XMode, YMode } from './lattice';
import { emptySdf, addCircle, addBox, addPolygon, nacaPolygon } from './geometry';
import { deriveTau, maxReynolds } from './units';
import { periodFromCrossings, mean } from './analysis';
import { Renderer, ViewMode, TracerMode } from './render';
import { ForceChart } from './chart';

type Obstacle = 'cylinder' | 'naca' | 'square' | 'plate' | 'none';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const input = (id: string) => $<HTMLInputElement>(id);
const select = (id: string) => $<HTMLSelectElement>(id);

const RAMP_STEPS = 3000;
const FORCE_EVERY = 4;
/** Fraction of the display frame interval given to simulation compute. */
const FRAME_BUDGET = 0.85;
const MAX_SPF = 400;
/** Default reference length as a fraction of the grid height. */
const DEFAULT_SIZE: Record<Obstacle, number> = { cylinder: 0.1, naca: 0.25, square: 0.1, plate: 0.12, none: 0.1 };

const canvas = $<HTMLCanvasElement>('view');

let gpu: Awaited<ReturnType<typeof initGpu>>;
try {
  gpu = await initGpu();
} catch (e) {
  const box = $('nogpu');
  box.hidden = false;
  box.textContent = `${(e as Error).message} This wind tunnel needs WebGPU: current Chrome, Edge or Safari 26, or Firefox on Windows or Apple Silicon.`;
  canvas.hidden = true;
  throw e;
}
const { device } = gpu;
const context = canvas.getContext('webgpu')!;
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format, alphaMode: 'opaque' });
const renderer = new Renderer(device, context, format);
await renderer.init();
const chart = new ForceChart($<HTMLCanvasElement>('chart'), $('chartTip'));

const state = {
  solver: null as Solver | null,
  W: 0,
  H: 0,
  paused: false,
  spf: 20,
  rampFrom: 0,
  lRef: 1,
  drawnBox: null as [number, number, number, number] | null,
  probe: null as [number, number] | null,
  probeValue: '',
  gpuMs: 0,
  frames: 0,
  interval: 16.7,
};

function settings() {
  const [W, H] = select('grid').value.split('x').map(Number);
  return {
    W,
    H,
    channel: select('tunnel').value === 'channel',
    obstacle: select('obstacle').value as Obstacle,
    size: Number(input('size').value),
    angle: (Number(input('angle').value) * Math.PI) / 180,
    re: Math.max(1, Number(input('re').value) || 1),
    u: Number(input('u').value),
    smag: input('smag').checked,
    cs: Number(input('cs').value),
  };
}

/** Inlet speed used for Re and the force coefficients: the mean speed in the channel, the free stream otherwise. */
function uRef() {
  const s = settings();
  return s.channel ? (2 / 3) * s.u : s.u;
}

function physics() {
  const s = settings();
  const d = deriveTau(s.re, uRef(), state.lRef);
  return { ...d, s };
}

function applyPhysics() {
  const solver = state.solver;
  if (!solver) return;
  const { tau, clamped, effectiveRe, mach, s } = physics();
  solver.update({ tau, smagorinsky: s.smag ? s.cs : 0, spongeTau: Math.max(1, tau) });
  $('uOut').textContent = s.u.toFixed(3);
  $('csOut').textContent = s.cs.toFixed(3);
  $('tauNote').textContent = `τ = ${tau.toFixed(4)}, ν = ${((tau - 0.5) / 3).toExponential(2)}, Mach ${mach.toFixed(2)}, L = ${state.lRef.toFixed(0)} cells`;
  const notes: string[] = [];
  if (clamped)
    notes.push(`Re ${s.re} needs τ below 0.51, which is unstable. Running at Re ${effectiveRe.toFixed(0)}; enlarge the body or the grid for more (max ${maxReynolds(uRef(), state.lRef).toFixed(0)} at this size).`);
  if (Math.round(effectiveRe) > 200)
    notes.push('Above Re ≈ 200 real wakes turn three-dimensional, and 2D turbulence cascades energy the wrong way. Treat this run as qualitative.');
  if (s.smag) notes.push('Smagorinsky here stabilizes coarse grids; it does not model real turbulence in 2D.');
  $('reNote').hidden = notes.length === 0;
  $('reNote').textContent = notes.join(' ');
}

function buildSdf(W: number, H: number) {
  const s = settings();
  const sdf = emptySdf(W, H);
  const L = s.size;
  const cx = s.channel ? W / 5 : W / 4;
  // Slightly off the centre line so vortex shedding starts on its own.
  const cy = s.channel ? H * (0.2 / 0.41) - 0.5 : H / 2 + 0.3;
  switch (s.obstacle) {
    case 'cylinder': addCircle(sdf, W, H, cx, cy, L / 2); break;
    case 'square': addBox(sdf, W, H, cx, cy, L / 2, L / 2, -s.angle); break;
    case 'plate': addBox(sdf, W, H, cx, cy, Math.max(1.5, L / 20), L / 2, -s.angle); break;
    case 'naca': addPolygon(sdf, W, H, nacaPolygon(cx - L / 3, cy, L, 0.12, s.angle)); break;
    case 'none': break;
  }
  state.lRef = s.obstacle === 'none' ? 1 : L;
  state.drawnBox = null;
  return sdf;
}

function resetForces() {
  state.solver?.resetForces();
  chart.clear();
}

function resetFlow() {
  const solver = state.solver!;
  solver.initField();
  state.rampFrom = 0;
  renderer.clearTrails();
  resetForces();
}

async function rebuild() {
  const s = settings();
  state.solver?.destroy();
  state.solver = null;
  const channel = s.channel;
  const solver = await Solver.create(device, {
    width: s.W,
    height: s.H,
    left: XMode.Open,
    right: XMode.Open,
    bottom: channel ? YMode.NoSlip : YMode.Slip,
    top: channel ? YMode.NoSlip : YMode.Slip,
    tau: 0.6,
    parabolic: channel,
    uIn: 0,
    spongeFraction: 0.15,
    spongeTau: 1,
    absorb: 0.02,
    forceEvery: FORCE_EVERY,
  });
  state.W = s.W;
  state.H = s.H;
  solver.setSdf(buildSdf(s.W, s.H));
  state.solver = solver;
  renderer.attach(solver);
  resetFlow();
  applyPhysics();
  layout();
}

function layout() {
  const stage = $('stage');
  const availW = stage.clientWidth - 24;
  const availH = stage.clientHeight - 24;
  const scale = Math.min(availW / state.W, availH / state.H);
  const cssW = Math.floor(state.W * scale);
  const cssH = Math.floor(state.H * scale);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const pw = Math.round(cssW * devicePixelRatio);
  const ph = Math.round(cssH * devicePixelRatio);
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
    renderer.resize(pw, ph);
  }
}

function toGrid(e: PointerEvent): [number, number] {
  const r = canvas.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * state.W - 0.5, (1 - (e.clientY - r.top) / r.height) * state.H - 0.5];
}

function tool() {
  return (document.querySelector('input[name="tool"]:checked') as HTMLInputElement).value;
}

let last: [number, number] | null = null;
function paint(e: PointerEvent) {
  const solver = state.solver;
  if (!solver) return;
  const p = toGrid(e);
  const r = Number(input('brush').value);
  const pts: [number, number][] = [];
  const from = last ?? p;
  const n = Math.max(1, Math.ceil(Math.hypot(p[0] - from[0], p[1] - from[1]) / Math.max(0.5, r / 2)));
  for (let k = 1; k <= n; k++) pts.push([from[0] + ((p[0] - from[0]) * k) / n, from[1] + ((p[1] - from[1]) * k) / n]);
  const erase = tool() === 'erase';
  solver.brushStroke(pts, r, erase);
  if (!erase) {
    const b = state.drawnBox ?? [p[0], p[1], p[0], p[1]];
    for (const [x, y] of pts) {
      b[0] = Math.min(b[0], x - r); b[1] = Math.min(b[1], y - r);
      b[2] = Math.max(b[2], x + r); b[3] = Math.max(b[3], y + r);
    }
    state.drawnBox = b;
    if (settings().obstacle === 'none') {
      state.lRef = Math.max(1, b[3] - b[1]);
      applyPhysics();
    }
  }
  last = p;
}

canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId);
  if (tool() === 'probe') {
    state.probe = toGrid(e).map(Math.round) as [number, number];
    return;
  }
  last = null;
  paint(e);
  resetForces();
});
canvas.addEventListener('pointermove', (e) => {
  if (e.buttons & 1 && tool() !== 'probe') paint(e);
});
canvas.addEventListener('pointerup', () => (last = null));

const onPhysics = () => applyPhysics();
for (const id of ['re', 'u', 'smag', 'cs']) input(id).addEventListener('input', onPhysics);
input('u').addEventListener('change', resetForces);
for (const id of ['tunnel', 'grid']) select(id).addEventListener('change', () => void rebuild());
select('obstacle').addEventListener('change', () => {
  const s = settings();
  input('size').value = String(Math.round(DEFAULT_SIZE[s.obstacle] * s.H));
  resetBody();
});
for (const id of ['size', 'angle']) input(id).addEventListener('input', () => resetBody());

function resetBody() {
  const solver = state.solver;
  if (!solver) return;
  $('sizeOut').textContent = input('size').value;
  $('angleOut').textContent = input('angle').value;
  solver.setSdf(buildSdf(state.W, state.H));
  applyPhysics();
  resetForces();
}

$('pause').addEventListener('click', () => {
  state.paused = !state.paused;
  $('pause').textContent = state.paused ? 'Run' : 'Pause';
});
$('resetFlow').addEventListener('click', resetFlow);
$('resetBody').addEventListener('click', resetBody);
input('brush').addEventListener('input', () => ($('brushOut').textContent = input('brush').value));
addEventListener('resize', layout);
addEventListener('keydown', (e) => {
  if (e.code === 'Space' && e.target === document.body) {
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
let probeInFlight = false;

function tuneSteps(frameMs: number) {
  const mode = select('spfMode').value;
  if (mode !== 'auto') {
    state.spf = Number(mode);
    return;
  }
  // Track the display's frame interval (60 Hz, 120 Hz...) as the shortest recent frame.
  state.interval = Math.min(Math.max(state.interval * 1.002, 4), Math.max(frameMs, 4));
  if (frameMs > 1.5 * state.interval) {
    state.spf = Math.max(1, Math.floor(state.spf * 0.85));
    return;
  }
  // Fill most of the frame: a partly idle GPU gets down-clocked, which makes each step slower.
  const budget = FRAME_BUDGET * state.interval;
  const cost = state.gpuMs > 0 ? state.gpuMs : frameMs - 3;
  const ratio = budget / Math.max(0.5, cost);
  state.spf = Math.round(Math.min(MAX_SPF, Math.max(1, state.spf * Math.min(1.1, Math.max(0.8, ratio)))));
}

function updateStats() {
  const { steps, cd, cl } = chart.data;
  const n = steps.length;
  if (n < 10) return;
  const u = uRef();
  const tail = Math.floor(n / 2);
  const p = periodFromCrossings(steps.slice(tail), cl.slice(tail));
  let window = cd.slice(tail);
  let windowL = cl.slice(tail);
  if (p && p.crossings >= 3) {
    const t1 = steps[n - 1];
    const k0 = steps.findIndex((s) => s >= t1 - Math.floor((t1 - steps[tail]) / p.period) * p.period);
    window = cd.slice(k0);
    windowL = cl.slice(k0);
    $('st').textContent = (state.lRef / (p.period * u)).toFixed(4);
  } else {
    $('st').textContent = '–';
  }
  $('cd').textContent = mean(window).toFixed(4);
  $('cl').textContent = mean(windowL).toFixed(4);
}

function frame(now: number) {
  requestAnimationFrame(frame);
  const solver = state.solver;
  if (!solver) return;
  const frameMs = now - lastFrameTime;
  lastFrameTime = now;
  state.frames++;
  tuneSteps(frameMs);
  const n = state.paused ? 0 : state.spf;

  const t = Math.min(1, (solver.step - state.rampFrom) / RAMP_STEPS);
  const uIn = settings().u * t * t * (3 - 2 * t);
  if (uIn !== solver.cfg.uIn) solver.update({ uIn });

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
  renderer.encode(
    enc,
    {
      mode: Number(select('viewMode').value) as ViewMode,
      contrast: Number(input('contrast').value),
      uRef: Math.max(0.01, settings().u),
      tracers: Number(select('tracers').value) as TracerMode,
      trailFade: select('tracers').value === '2' ? 0.985 : 0.94,
    },
    n,
  );
  device.queue.submit([enc.finish()]);

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

  if (!forcesInFlight && state.frames % 6 === 0) {
    forcesInFlight = true;
    const u = uRef();
    const scale = 2 / (u * u * state.lRef);
    solver.readForces().then((samples) => {
      if (solver === state.solver) for (const f of samples) chart.push(f.step, f.fx * scale, f.fy * scale);
      forcesInFlight = false;
      updateStats();
    });
  }

  if (state.probe && !probeInFlight && state.frames % 4 === 0) {
    const [x, y] = state.probe;
    if (x >= 0 && y >= 0 && x < state.W && y < state.H) {
      probeInFlight = true;
      readBuffer(device, solver.macro, (y * state.W + x) * 16, 16).then((buf) => {
        const [rho, ux, uy, solid] = new Float32Array(buf);
        const u = settings().u;
        state.probeValue = solid ? `(${x}, ${y}) solid` : `(${x}, ${y}) |u|/U ${(Math.hypot(ux, uy) / u).toFixed(3)}, Cp ${((2 * (rho - 1)) / 3 / (u * u)).toFixed(3)}`;
        probeInFlight = false;
      });
    }
  }
  if (state.frames % 10 === 0) {
    $('step').textContent = solver.step.toLocaleString();
    if (state.probe) $('probe').textContent = state.probeValue;
  }
  $('spfOut').textContent = String(n);
  chart.draw();
}

input('size').value = String(Math.round(DEFAULT_SIZE.cylinder * 512));
$('sizeOut').textContent = input('size').value;
$('angleOut').textContent = input('angle').value;
$('brushOut').textContent = input('brush').value;
await rebuild();
requestAnimationFrame(frame);
