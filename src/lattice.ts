// D2Q9 velocity set. Direction i and OPP[i] form TRT pairs.
export const CX = [0, 1, 0, -1, 0, 1, -1, -1, 1] as const;
export const CY = [0, 0, 1, 0, -1, 1, 1, -1, -1] as const;
export const W = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36] as const;
export const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6] as const;
export const MIRROR_Y = [0, 1, 4, 3, 2, 8, 7, 6, 5] as const;
export const Q = 9;

/** Each unordered pair (i, OPP[i]) once, with i the lower index. */
export const PAIRS: ReadonlyArray<readonly [number, number]> = [
  [1, 3],
  [2, 4],
  [5, 7],
  [6, 8],
];

export const enum XMode {
  Periodic = 0,
  Wall = 1,
  Open = 2, // inlet on the left, outlet on the right
}

export const enum YMode {
  Periodic = 0,
  NoSlip = 1,
  Slip = 2,
  MovingWall = 3,
}

export const FLAG_SOLID = 1 << 9;
export const FLAG_INLET = 1 << 10;
export const FLAG_OUTLET = 1 << 11;
export const BIDX_SHIFT = 12;

/** TRT magic parameter that places the halfway bounce-back wall independently of viscosity. */
export const MAGIC_LAMBDA = 3 / 16;

export function equilibrium(rho: number, ux: number, uy: number, out: Float32Array | number[] = new Array(Q)) {
  const usq = 1.5 * (ux * ux + uy * uy);
  for (let i = 0; i < Q; i++) {
    const cu = 3 * (CX[i] * ux + CY[i] * uy);
    out[i] = W[i] * rho * (1 + cu + 0.5 * cu * cu - usq);
  }
  return out;
}
