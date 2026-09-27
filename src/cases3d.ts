import { metric, type Case } from './cases';
import { Solver3D, PRESETS3, type Precision } from './solver3d';
import { emptySdf3, addSphere, referenceArea } from './geometry3d';
import { deriveTau } from './units';
import { mean } from './analysis';

const HEARTBEAT_MS = 10_000;

const smoothstep = (t: number) => t * t * (3 - 2 * t);

const sphereFp16: Case = async (device, log) => {
  const [W, H, D] = PRESETS3.medium;
  const N = W * H * D;
  const name = 'Sphere C_D, FP16 vs FP32 (Re 100)';
  if (!device.features.has('shader-f16')) return { name, metrics: [], skipped: 'shader-f16 not available on this device', steps: 0, cells: N, notes: [] };
  const d = 16;
  const u = 0.1;
  const tau = deriveTau(100, u, d).tau;
  const sdf = emptySdf3(W, H, D);
  addSphere(sdf, W, H, D, W / 4, H / 2, D / 2, d / 2);
  const scale = 2 / (u * u * referenceArea('sphere', d, D));
  const notes: string[] = [];
  const cd: Record<Precision, number> = { fp32: NaN, fp16: NaN };
  let steps = 0;

  for (const precision of ['fp32', 'fp16'] as const) {
    let s: Solver3D;
    try {
      s = await Solver3D.create(device, { width: W, height: H, depth: D, precision, tau, spongeFraction: 0.15, spongeTau: 1, absorb: 0.02, forceEvery: 4 });
    } catch (e) {
      const msg = (e as Error).message;
      if (msg.startsWith('Solver3D allocation failed')) return { name, metrics: [], skipped: msg, steps, cells: N, notes };
      throw e;
    }
    try {
      s.setSdf(sdf);
      s.initField();
      let prev = NaN;
      let converged = false;
      let beatStep = 0;
      let beatAt = performance.now();
      for (let k = 1; k <= 15; k++) {
        const block: number[] = [];
        const end = k * 2000;
        while (s.step < end) {
          s.setInlet(u * smoothstep(Math.min(1, s.step / 3000)));
          await s.run(Math.min(400, end - s.step), 400);
          block.push(...(await s.readForces()).map((f) => f.fx));
          const now = performance.now();
          if (now - beatAt > HEARTBEAT_MS) {
            log(`${precision} step ${s.step}, ${((N * (s.step - beatStep)) / ((now - beatAt) * 1e3)).toFixed(0)} MLUPS`);
            beatStep = s.step;
            beatAt = now;
          }
        }
        const c = mean(block) * scale;
        log(`${precision} check ${k}/15 C_D=${c.toFixed(5)}`);
        cd[precision] = c;
        if (k >= 5 && Math.abs(c - prev) / Math.abs(prev) < 1e-4) {
          converged = true;
          break;
        }
        prev = c;
      }
      if (!converged) notes.push(`${precision} reached 15 blocks without converging`);
      steps += s.step;
    } finally {
      s.destroy();
    }
  }

  notes.push(`C_D FP32 ${cd.fp32.toFixed(5)}, FP16 ${cd.fp16.toFixed(5)}`);
  notes.push('published sphere C_D at Re 100 ≈ 1.09 (Johnson & Patel 1999, unverified), not gated');
  return {
    name,
    metrics: [metric('C_D relative difference FP16 vs FP32', Math.abs(cd.fp16 - cd.fp32) / cd.fp32, [0, 0], [0, 0.01])],
    steps,
    cells: N,
    notes,
  };
};

export const CASES3D: Record<string, Case> = { sphereFp16 };
