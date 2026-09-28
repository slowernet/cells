/** Pure logic for the 3D page: grid presets, obstacle geometry and force scaling. */
import { fitsLimits, PRESETS3, type Precision } from './shaders/common3d';
import { emptySdf3, addSphere, addBox3, addCylinderZ, addWing, referenceArea, type Obstacle3 } from './geometry3d';
import type { Vec3 } from './camera3d';

export type PresetName = keyof typeof PRESETS3;
type Limits = { maxStorageBufferBindingSize: number; maxBufferSize: number };

const ORDER: PresetName[] = ['low', 'medium', 'high'];

export function presetsThatFit(precision: Precision, limits: Limits): PresetName[] {
  return ORDER.filter((p) => {
    const [W, H, D] = PRESETS3[p];
    return fitsLimits(W, H, D, precision, limits);
  });
}

/** The largest preset that fits, no larger than preferred; null when even the smallest doesn't. */
export function fallbackPreset(preferred: PresetName, precision: Precision, limits: Limits): PresetName | null {
  const fit = presetsThatFit(precision, limits);
  for (let i = ORDER.indexOf(preferred); i >= 0; i--) if (fit.includes(ORDER[i])) return ORDER[i];
  return null;
}

export function nextSmaller(p: PresetName): PresetName | null {
  const i = ORDER.indexOf(p);
  return i > 0 ? ORDER[i - 1] : null;
}

export interface Body3 {
  obstacle: Obstacle3;
  /** Diameter, edge or chord as a fraction of the grid height. */
  sizeFraction: number;
  angleDeg: number;
}

/** Builds the obstacle SDF with its reference length and area (cells) and the march bounds grown by 2 cells. */
export function buildBody(body: Body3, W: number, H: number, D: number) {
  const sdf = emptySdf3(W, H, D);
  const L = body.sizeFraction * H;
  const [cx, cy, cz] = [W / 4, H / 2, D / 2];
  const a = (body.angleDeg * Math.PI) / 180;
  let half: Vec3;
  switch (body.obstacle) {
    case 'sphere':
      addSphere(sdf, W, H, D, cx, cy, cz, L / 2);
      half = [L / 2, L / 2, L / 2];
      break;
    case 'cube': {
      addBox3(sdf, W, H, D, cx, cy, cz, L / 2, L / 2, L / 2, -a);
      const r = (L / 2) * (Math.abs(Math.cos(a)) + Math.abs(Math.sin(a)));
      half = [r, r, L / 2];
      break;
    }
    case 'cylinder':
      addCylinderZ(sdf, W, H, D, cx, cy, L / 2);
      half = [L / 2, L / 2, D / 2];
      break;
    case 'wing':
      addWing(sdf, W, H, D, cx, cy, cz, L, 0.6 * D, a);
      half = [L / 2, L / 2, 0.3 * D];
      break;
    case 'none':
      return { sdf, lRef: 1, area: 0, bounds: null };
  }
  const c: Vec3 = [cx, cy, cz];
  const dims: Vec3 = [W, H, D];
  const lo = c.map((v, k) => Math.max(-0.5, v - half[k] - 2)) as Vec3;
  const hi = c.map((v, k) => Math.min(dims[k] - 0.5, v + half[k] + 2)) as Vec3;
  return { sdf, lRef: L, area: referenceArea(body.obstacle, L, D), bounds: [lo, hi] as [Vec3, Vec3] };
}

/** Converts a lattice force into a coefficient: 2 / (u² A); 0 without a body. */
export function coefficientScale(uTarget: number, area: number): number {
  return area > 0 ? 2 / (uTarget * uTarget * area) : 0;
}

export const MAX_SPF = 400;

/** Scales steps per frame toward the frame budget by at most ×0.8..×1.1, always gaining a step when there is room. */
export function nextStepsPerFrame(spf: number, budgetRatio: number): number {
  let next = Math.round(spf * Math.min(1.1, Math.max(0.8, budgetRatio)));
  if (budgetRatio > 1 && next <= spf) next = spf + 1;
  return Math.min(MAX_SPF, Math.max(1, next));
}

/** Simulation time per frame the tuner aims for: 0.85 of a 60 Hz frame, whatever the display's refresh rate. */
export const FRAME_BUDGET_MS = 0.85 * 16.7;

/** Steps per frame for the fixed budget; costMs is GPU compute time when timestamps measure it, otherwise frame time. */
export function tuneStepsPerFrame(spf: number, costMs: number): number {
  return nextStepsPerFrame(spf, FRAME_BUDGET_MS / Math.max(0.5, costMs));
}
