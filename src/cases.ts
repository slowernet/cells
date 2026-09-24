import { Solver, ForceSample, SolverConfig } from './solver';
import { XMode, YMode } from './lattice';
import { emptySdf, addCircle, addPolygon, nacaPolygon } from './geometry';
import { periodFromCrossings, mean, streamFunction, primaryVortex } from './analysis';

export interface Metric {
  name: string;
  value: number;
  /** Published reference range. */
  ref: [number, number];
  /** Acceptance range for this solver, fixed before the case was first run. */
  accept: [number, number];
  pass: boolean;
}

export interface CaseResult {
  name: string;
  metrics: Metric[];
  pass: boolean;
  steps: number;
  cells: number;
  seconds: number;
  notes: string[];
}

type Case = (device: GPUDevice, log: (s: string) => void) => Promise<Omit<CaseResult, 'pass' | 'seconds'>>;

function metric(name: string, value: number, ref: [number, number], accept: [number, number]): Metric {
  return { name, value, ref, accept, pass: value >= accept[0] && value <= accept[1] };
}

const around = (center: number, rel: number): [number, number] => [center * (1 - rel), center * (1 + rel)];

const HEARTBEAT_MS = 10_000;
let lastBeat = 0;

/** Advances the solver, ramping the inlet smoothly to uTarget, collecting force samples along the way. */
async function runRamped(s: Solver, steps: number, uTarget: number, rampSteps: number, forces: ForceSample[] = [], chunk = 400, log?: (s: string) => void) {
  const end = s.step + steps;
  let beatStep = s.step;
  let beatAt = performance.now();
  while (s.step < end) {
    const now = performance.now();
    if (log && now - beatAt > HEARTBEAT_MS && now - lastBeat > HEARTBEAT_MS) {
      log(`step ${s.step}, ${((s.N * (s.step - beatStep)) / ((now - beatAt) * 1e3)).toFixed(0)} MLUPS`);
      beatStep = s.step;
      beatAt = lastBeat = now;
    }
    const r = Math.min(1, s.step / rampSteps);
    const uIn = uTarget * r * r * (3 - 2 * r);
    if (uIn !== s.cfg.uIn) s.update({ uIn });
    await s.run(Math.min(chunk, end - s.step), chunk);
    forces.push(...(await s.readForces()));
  }
  return forces;
}

const poiseuille: Case = async (device) => {
  const H = 32;
  const uMax = 0.05;
  const metrics: Metric[] = [];
  const notes: string[] = [];
  let steps = 0;
  for (const tau of [0.6, 1.0, 1.5]) {
    const nu = (tau - 0.5) / 3;
    const fx = (8 * nu * uMax) / (H * H);
    const s = await Solver.create(device, {
      width: 4, height: H, left: XMode.Periodic, right: XMode.Periodic,
      bottom: YMode.NoSlip, top: YMode.NoSlip, tau, bodyForce: [fx, 0], forceEvery: 0,
    });
    s.initField();
    const n = Math.ceil((10 * H * H) / nu);
    await s.run(n, 2000);
    steps += n;
    const m = await s.readMacro();
    let num = 0, den = 0;
    for (let y = 0; y < H; y++) {
      const e = (fx / (2 * nu)) * (y + 0.5) * (H - y - 0.5);
      num += (m[(y * 4 + 1) * 4 + 1] - e) ** 2;
      den += e * e;
    }
    const err = Math.sqrt(num / den);
    metrics.push(metric(`L2 error, tau=${tau}`, err, [0, 0], [0, 5e-3]));
    s.destroy();
  }
  notes.push('TRT with Lambda = 3/16 puts the halfway bounce-back wall exactly at y = -1/2 for every tau, so the parabola is reproduced to round-off.');
  return { name: 'Plane Poiseuille', metrics, steps, cells: 4 * H, notes };
};

const taylorGreen: Case = async (device) => {
  const tau = 0.8;
  const nu = (tau - 0.5) / 3;
  const errors: number[] = [];
  const nuMeasured: number[] = [];
  const sizes = [32, 64, 128];
  let steps = 0;
  for (const n of sizes) {
    const u0 = (0.04 * 32) / n;
    const k = (2 * Math.PI) / n;
    const s = await Solver.create(device, {
      width: n, height: n, left: XMode.Periodic, right: XMode.Periodic,
      bottom: YMode.Periodic, top: YMode.Periodic, tau, forceEvery: 0,
    });
    const field = (t: number) => {
      const a = u0 * Math.exp(-2 * nu * k * k * t);
      const m = new Float32Array(n * n * 4);
      for (let y = 0; y < n; y++)
        for (let x = 0; x < n; x++) {
          const i = (y * n + x) * 4;
          m[i] = 1 - 0.75 * a * a * (Math.cos(2 * k * x) + Math.cos(2 * k * y));
          m[i + 1] = -a * Math.cos(k * x) * Math.sin(k * y);
          m[i + 2] = a * Math.sin(k * x) * Math.cos(k * y);
        }
      return m;
    };
    s.initField(field(0));
    const t = Math.round(Math.log(2) / (2 * nu * k * k));
    const energy = (m: Float32Array) => {
      let e = 0;
      for (let i = 0; i < n * n; i++) e += m[i * 4 + 1] ** 2 + m[i * 4 + 2] ** 2;
      return e;
    };
    await s.run(Math.round(t / 2), 1000);
    const mid = await s.readMacro();
    await s.run(t - Math.round(t / 2), 1000);
    const end = await s.readMacro();
    steps += t;
    const exact = field(t);
    let num = 0, den = 0;
    for (let i = 0; i < n * n; i++)
      for (const c of [1, 2]) {
        num += (end[i * 4 + c] - exact[i * 4 + c]) ** 2;
        den += exact[i * 4 + c] ** 2;
      }
    errors.push(Math.sqrt(num / den));
    const dt = t - Math.round(t / 2);
    nuMeasured.push(Math.log(energy(mid) / energy(end)) / (4 * k * k * dt));
    s.destroy();
  }
  const order = Math.log2(errors[0] / errors[2]) / 2;
  return {
    name: 'Taylor-Green vortex',
    metrics: [
      metric('nu measured / nu (N=128)', nuMeasured[2] / nu, [1, 1], [0.99, 1.01]),
      metric('velocity L2 error (N=128)', errors[2], [0, 0], [0, 0.01]),
      metric('convergence order 32 to 128', order, [2, 2], [1.8, Infinity]),
    ],
    steps,
    cells: 128 * 128,
    notes: [`L2 errors for N=${sizes.join('/')}: ${errors.map((e) => e.toExponential(2)).join(', ')} (diffusive scaling, tau fixed)`],
  };
};

/** Schäfer-Turek 2D channel with a cylinder of D cells; returns a configured solver at rest. */
async function schaferTurek(device: GPUDevice, D: number, re: number, forceEvery: number, uMax: number) {
  const W = 22 * D;
  const H = Math.round(4.1 * D);
  const uMean = (2 / 3) * uMax;
  const nu = (uMean * D) / re;
  const cx = 2 * D;
  const cy = 2 * D - 0.5;
  const cfg: SolverConfig = {
    width: W, height: H, left: XMode.Open, right: XMode.Open, bottom: YMode.NoSlip, top: YMode.NoSlip,
    tau: 3 * nu + 0.5, parabolic: true, uIn: 0, forceEvery, spongeFraction: 0.1, spongeTau: 1, absorb: 0.1,
  };
  const s = await Solver.create(device, cfg);
  const sdf = emptySdf(W, H);
  addCircle(sdf, W, H, cx, cy, D / 2);
  s.setSdf(sdf);
  s.initField();
  const coef = (f: number) => (2 * f) / (uMean * uMean * D);
  return { s, W, H, cx, cy, uMax, uMean, coef };
}

const schaferTurek1: Case = async (device, log) => {
  const D = 40;
  const { s, W, H, cx, cy, uMax, uMean, coef } = await schaferTurek(device, D, 20, 50, 0.05);
  const forces: ForceSample[] = [];
  let prev = Infinity;
  const block = 10000;
  for (let k = 0; k < 20; k++) {
    await runRamped(s, block, uMax, 5000, forces, 400, log);
    const cd = coef(mean(forces.slice(-20).map((f) => f.fx)));
    log(`check ${k + 1}/20 (stops once converged), step ${s.step}: c_D ${cd.toFixed(5)}`);
    if (k >= 4 && Math.abs(cd - prev) < 1e-4) break;
    prev = cd;
  }
  const last = forces.slice(-20);
  const cd = coef(mean(last.map((f) => f.fx)));
  const cl = coef(mean(last.map((f) => f.fy)));
  const m = await s.readMacro();
  const rhoAt = (x: number) => {
    // Linear interpolation between the two rows straddling the centre line.
    const y0 = Math.floor(cy);
    const w = cy - y0;
    return (1 - w) * m[(y0 * W + x) * 4] + w * m[((y0 + 1) * W + x) * 4];
  };
  const uxAt = (x: number) => {
    const y0 = Math.floor(cy);
    const w = cy - y0;
    return (1 - w) * m[(y0 * W + x) * 4 + 1] + w * m[((y0 + 1) * W + x) * 4 + 1];
  };
  // Pressure at the stagnation points, extrapolated linearly from the two nearest fluid nodes.
  const xf = cx - D / 2;
  const xb = cx + D / 2;
  const a = Math.floor(xf), b = Math.ceil(xb);
  const pFront = rhoAt(a) + (rhoAt(a) - rhoAt(a - 1)) * (xf - a);
  const pBack = rhoAt(b) + (rhoAt(b) - rhoAt(b + 1)) * (b - xb);
  const dP = (pFront - pBack) / 3 / (uMean * uMean);
  let la = NaN;
  for (let x = b; x < W - 1; x++) {
    const u0 = uxAt(x), u1 = uxAt(x + 1);
    if (u0 < 0 && u1 >= 0) {
      la = (x + -u0 / (u1 - u0) - xb) / D;
      break;
    }
  }
  s.destroy();
  return {
    name: 'Schäfer-Turek 2D-1 (Re 20)',
    metrics: [
      metric('c_D', cd, [5.57, 5.59], around(5.58, 0.02)),
      metric('c_L', cl, [0.0104, 0.011], around(0.0107, 0.1)),
      metric('L_a / D', la, [0.842, 0.852], around(0.847, 0.03)),
      metric('dP / (rho U_mean^2)', dP, [0.1172 / 0.04, 0.1176 / 0.04], around(0.1174 / 0.04, 0.02)),
    ],
    steps: s.step,
    cells: W * H,
    notes: [`D = ${D} cells, grid ${W}x${H}, u_max = ${uMax}, tau = ${s.cfg.tau.toFixed(4)}, Bouzidi links from the circle SDF`],
  };
};

const schaferTurek2: Case = async (device, log) => {
  const D = 40;
  const { s, W, H, uMax, uMean, coef } = await schaferTurek(device, D, 100, 5, 0.05);
  const forces: ForceSample[] = [];
  const period = D / (0.3 * uMean);
  let window: ForceSample[] = [];
  let prevAmp = 0;
  for (let k = 0; k < 40; k++) {
    await runRamped(s, Math.round(5 * period), uMax, 5000, forces, 400, log);
    window = forces.filter((f) => f.step > s.step - 5 * period);
    const cls = window.map((f) => coef(f.fy));
    const amp = Math.max(...cls) - Math.min(...cls);
    log(`check ${k + 1}/40 (stops once periodic), step ${s.step}: c_L amplitude ${amp.toFixed(4)}`);
    if (k >= 6 && amp > 0.5 && Math.abs(amp - prevAmp) < 1e-3 * amp) break;
    prevAmp = amp;
    forces.splice(0, Math.max(0, forces.length - 20000));
  }
  const t = window.map((f) => f.step);
  const cl = window.map((f) => coef(f.fy));
  const cd = window.map((f) => coef(f.fx));
  const p = periodFromCrossings(t, cl);
  const st = p ? D / (p.period * uMean) : NaN;
  s.destroy();
  return {
    name: 'Schäfer-Turek 2D-2 (Re 100)',
    metrics: [
      metric('c_D,max', Math.max(...cd), [3.22, 3.24], around(3.23, 0.02)),
      metric('c_L,max', Math.max(...cl), [0.99, 1.01], around(1.0, 0.05)),
      metric('St', st, [0.295, 0.305], around(0.3, 0.02)),
    ],
    steps: s.step,
    cells: W * H,
    notes: [`D = ${D} cells, grid ${W}x${H}, tau = ${s.cfg.tau.toFixed(4)}, forces sampled every 5 steps`],
  };
};

const unconfinedCylinder: Case = async (device, log) => {
  // The inlet fixes the inflow speed, so the body needs room: 8D of upstream run confined it enough to raise St by 3% and C_D by 5%.
  const D = 32, lx = 40, ly = 48, xc = 16, u = 0.05;
  const W = lx * D, H = ly * D;
  const nu = (u * D) / 100;
  const s = await Solver.create(device, {
    width: W, height: H, left: XMode.Open, right: XMode.Open, bottom: YMode.Slip, top: YMode.Slip,
    tau: 3 * nu + 0.5, forceEvery: 5, spongeFraction: 0.15, spongeTau: 1, absorb: 0.02,
  });
  const sdf = emptySdf(W, H);
  // Slightly off-centre so shedding starts without an explicit perturbation.
  addCircle(sdf, W, H, xc * D, H / 2 + 0.3, D / 2);
  s.setSdf(sdf);
  s.initField();
  const coef = (f: number) => (2 * f) / (u * u * D);
  const period = D / (0.165 * u);
  const forces: ForceSample[] = [];
  let window: ForceSample[] = [];
  let prevAmp = 0;
  for (let k = 0; k < 60; k++) {
    await runRamped(s, Math.round(5 * period), u, 5000, forces, 400, log);
    window = forces.filter((f) => f.step > s.step - 5 * period);
    const cls = window.map((f) => coef(f.fy));
    const amp = Math.max(...cls) - Math.min(...cls);
    log(`check ${k + 1}/60 (stops once periodic), step ${s.step}: c_L amplitude ${amp.toFixed(4)}`);
    if (k >= 6 && amp > 0.3 && Math.abs(amp - prevAmp) < 2e-3 * amp) break;
    prevAmp = amp;
    forces.splice(0, Math.max(0, forces.length - 20000));
  }
  const t = window.map((f) => f.step);
  const p = periodFromCrossings(t, window.map((f) => f.fy));
  const st = p ? D / (p.period * u) : NaN;
  // Average C_D over whole shedding periods only.
  const n = p ? Math.floor((t[t.length - 1] - t[0]) / p.period) : 0;
  const span = window.filter((f) => f.step > t[t.length - 1] - n * (p?.period ?? 0));
  const cd = coef(mean(span.map((f) => f.fx)));
  s.destroy();
  return {
    name: 'Unconfined cylinder (Re 100)',
    metrics: [
      metric('St', st, [0.1643, 0.1670], around(0.165, 0.04)),
      metric('mean C_D', cd, [1.33, 1.36], around(1.345, 0.05)),
    ],
    steps: s.step,
    cells: W * H,
    notes: [
      `D = ${D} cells, domain ${lx}D x ${ly}D with slip side walls (blockage ${(100 / ly).toFixed(1)}%), cylinder ${xc}D from the inlet, u = ${u}, tau = ${s.cfg.tau.toFixed(4)}`,
      'References: Williamson fit St 0.1643; arXiv:2007.07347 Table II St 0.1646-0.1670, mean C_D 1.33-1.36 (the research doc quoted 1.39-1.40, which is the Re 60 row).',
    ],
  };
};

const cavity: Case = async (device, log) => {
  const N = 128;
  const u = 0.1;
  const nu = (u * N) / 1000;
  const s = await Solver.create(device, {
    width: N, height: N, left: XMode.Wall, right: XMode.Wall, bottom: YMode.NoSlip, top: YMode.MovingWall,
    tau: 3 * nu + 0.5, uLid: u, forceEvery: 0,
  });
  s.initField();
  let v = { x: 0, y: 0, psi: 0 };
  let prev = { x: -1, y: -1 };
  for (let k = 0; k < 40; k++) {
    await s.run(20000, 2000);
    const m = await s.readMacro();
    const ux = new Float32Array(N * N), uy = new Float32Array(N * N);
    for (let i = 0; i < N * N; i++) { ux[i] = m[i * 4 + 1]; uy[i] = m[i * 4 + 2]; }
    v = primaryVortex(streamFunction(ux, uy, N, N), N, N);
    log(`check ${k + 1}/40 (stops once converged), step ${s.step}: vortex (${((v.x + 0.5) / N).toFixed(4)}, ${((v.y + 0.5) / N).toFixed(4)})`);
    if (Math.hypot(v.x - prev.x, v.y - prev.y) < 0.01 && k >= 5) break;
    prev = v;
  }
  s.destroy();
  return {
    name: 'Ghia lid-driven cavity (Re 1000)',
    metrics: [
      metric('vortex x / L', (v.x + 0.5) / N, [0.5313, 0.5313], [0.5213, 0.5413]),
      metric('vortex y / L', (v.y + 0.5) / N, [0.5625, 0.5625], [0.5525, 0.5725]),
    ],
    steps: s.step,
    cells: N * N,
    notes: [`N = ${N}, lid u = ${u}, tau = ${s.cfg.tau.toFixed(4)}; Ghia's grid spacing is 1/128, so the reference itself is uncertain to about 0.008.`],
  };
};

const naca: Case = async (device, log) => {
  const c = 100;
  const W = 16 * c, H = 8 * c;
  const u = 0.1;
  const nu = (u * c) / 500;
  const s = await Solver.create(device, {
    width: W, height: H, left: XMode.Open, right: XMode.Open, bottom: YMode.Slip, top: YMode.Slip,
    tau: 3 * nu + 0.5, forceEvery: 50, spongeFraction: 0.15, spongeTau: 1, absorb: 0.1, inletLayerFraction: 0.08,
  });
  const sdf = emptySdf(W, H);
  addPolygon(sdf, W, H, nacaPolygon(4 * c, H / 2 - 0.5, c, 0.12, 0));
  s.setSdf(sdf);
  s.initField();
  const coef = (f: number) => (2 * f) / (u * u * c);
  const forces: ForceSample[] = [];
  let prev = Infinity;
  let cd = NaN;
  for (let k = 0; k < 30; k++) {
    await runRamped(s, 10000, u, 5000, forces, 400, log);
    cd = coef(mean(forces.slice(-20).map((f) => f.fx)));
    log(`check ${k + 1}/30 (stops once converged), step ${s.step}: C_D ${cd.toFixed(5)}`);
    if (k >= 5 && Math.abs(cd - prev) < 2e-4) break;
    prev = cd;
  }
  s.destroy();
  return {
    name: 'NACA0012 (Re 500, alpha 0)',
    metrics: [metric('C_D', cd, [0.1762, 0.178], around(0.1762, 0.05))],
    steps: s.step,
    cells: W * H,
    notes: [
      `chord ${c} cells, domain 16c x 8c with slip side walls, closed trailing edge, tau = ${s.cfg.tau.toFixed(4)}`,
      'References: Lockard et al. 0.1762; arXiv:1901.08766 reports 0.178 (the research doc quoted 0.176 for the latter).',
    ],
  };
};

export const CASES: Record<string, Case> = {
  poiseuille,
  taylorGreen,
  schaferTurek1,
  schaferTurek2,
  unconfinedCylinder,
  cavity,
  naca,
};

export async function runCase(device: GPUDevice, key: string, log: (s: string) => void = () => {}): Promise<CaseResult> {
  const t0 = performance.now();
  const stamped = (msg: string) => log(`[${key} +${((performance.now() - t0) / 1000).toFixed(0)}s] ${msg.trim()}`);
  stamped('started');
  const r = await CASES[key](device, stamped);
  return { ...r, pass: r.metrics.every((m) => m.pass), seconds: (performance.now() - t0) / 1000 };
}
