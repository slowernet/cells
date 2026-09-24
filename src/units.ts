export const TAU_MIN = 0.51;

export interface Derived {
  nu: number;
  tau: number;
  /** True when the requested Re needed tau below TAU_MIN and was clamped. */
  clamped: boolean;
  /** Reynolds number the clamped tau actually delivers. */
  effectiveRe: number;
  mach: number;
}

/** Lattice relaxation time for Re = u L / nu, with c_s^2 = 1/3 so nu = (tau - 1/2) / 3. */
export function deriveTau(re: number, uLat: number, length: number): Derived {
  const nu = (uLat * length) / re;
  const raw = 3 * nu + 0.5;
  const tau = Math.max(raw, TAU_MIN);
  const nuEff = (tau - 0.5) / 3;
  return { nu: nuEff, tau, clamped: raw < TAU_MIN, effectiveRe: (uLat * length) / nuEff, mach: uLat * Math.sqrt(3) };
}

export function maxReynolds(uLat: number, length: number): number {
  return (uLat * length * 3) / (TAU_MIN - 0.5);
}
