import { MAGIC_LAMBDA } from './lattice';

// D3Q19 velocity set. Odd i and i + 1 are opposite and form TRT pairs.
export const CX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0] as const;
export const CY = [0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1] as const;
export const CZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1] as const;
export const Q3 = 19;
export const W: readonly number[] = Array.from({ length: Q3 }, (_, i) => (i === 0 ? 1 / 3 : i <= 6 ? 1 / 18 : 1 / 36));
export const OPP: readonly number[] = Array.from({ length: Q3 }, (_, i) => (i === 0 ? 0 : i % 2 ? i + 1 : i - 1));

function find(cx: number, cy: number, cz: number): number {
  for (let i = 0; i < Q3; i++) if (CX[i] === cx && CY[i] === cy && CZ[i] === cz) return i;
  throw new Error(`no direction (${cx}, ${cy}, ${cz})`);
}

/** Direction with its y component negated, for slip on the y faces. */
export const MIRROR_Y: readonly number[] = Array.from({ length: Q3 }, (_, i) => find(CX[i], -CY[i], CZ[i]));
/** Direction with its z component negated, for slip on the z faces. */
export const MIRROR_Z: readonly number[] = Array.from({ length: Q3 }, (_, i) => find(CX[i], CY[i], -CZ[i]));

/** Each unordered pair (i, OPP[i]) once, with i the lower index. */
export const PAIRS: ReadonlyArray<readonly [number, number]> = Array.from({ length: 9 }, (_, k) => [2 * k + 1, 2 * k + 2] as const);

export const FLAG_SOLID = 1 << 19;
export const FLAG_INLET = 1 << 20;
export const FLAG_OUTLET = 1 << 21;

/** TRT Λ: the magic 3/16 where stable, shrinking toward BGK below τ ≈ 0.561, where the inlet drives odd modes unstable. */
export function trtLambda3(tau: number): number {
  return Math.min(MAGIC_LAMBDA, 50 * (tau - 0.5) ** 2);
}

export function equilibrium3(rho: number, ux: number, uy: number, uz: number, out: Float32Array | number[] = new Array(Q3)) {
  const usq = 1.5 * (ux * ux + uy * uy + uz * uz);
  for (let i = 0; i < Q3; i++) {
    const cu = 3 * (CX[i] * ux + CY[i] * uy + CZ[i] * uz);
    out[i] = W[i] * rho * (1 + cu + 0.5 * cu * cu - usq);
  }
  return out;
}
