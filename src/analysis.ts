export interface Periodic {
  /** Period in the same units as t, from the mean spacing of upward zero crossings. */
  period: number;
  crossings: number;
}

/** Period of an oscillating signal from linearly interpolated upward crossings of its mean. */
export function periodFromCrossings(t: number[], v: number[]): Periodic | null {
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const times: number[] = [];
  for (let k = 1; k < v.length; k++) {
    const a = v[k - 1] - mean;
    const b = v[k] - mean;
    if (a < 0 && b >= 0) times.push(t[k - 1] + ((t[k] - t[k - 1]) * -a) / (b - a));
  }
  if (times.length < 3) return null;
  return { period: (times[times.length - 1] - times[0]) / (times.length - 1), crossings: times.length };
}

export function mean(v: number[]): number {
  return v.reduce((a, b) => a + b, 0) / v.length;
}

/**
 * Stream function by integrating u_y along x on each row after integrating u_x up the first column,
 * returned as psi[y * W + x]. Solid cells should carry zero velocity.
 */
export function streamFunction(ux: Float32Array | number[], uy: Float32Array | number[], W: number, H: number): Float64Array {
  const psi = new Float64Array(W * H);
  for (let y = 1; y < H; y++) psi[y * W] = psi[(y - 1) * W] + 0.5 * (ux[y * W] + ux[(y - 1) * W]);
  for (let y = 0; y < H; y++)
    for (let x = 1; x < W; x++) {
      const i = y * W + x;
      psi[i] = psi[i - 1] - 0.5 * (uy[i] + uy[i - 1]);
    }
  return psi;
}

/** Location of the extremum of psi with the largest magnitude, refined by a quadratic fit in each axis. */
export function primaryVortex(psi: Float64Array, W: number, H: number): { x: number; y: number; psi: number } {
  let best = 1;
  for (let y = 1; y < H - 1; y++)
    for (let x = 1; x < W - 1; x++) if (Math.abs(psi[y * W + x]) > Math.abs(psi[best])) best = y * W + x;
  const bx = best % W;
  const by = Math.floor(best / W);
  const refine = (m: number, c: number, p: number) => {
    const den = m - 2 * c + p;
    return den === 0 ? 0 : (0.5 * (m - p)) / den;
  };
  return {
    x: bx + refine(psi[best - 1], psi[best], psi[best + 1]),
    y: by + refine(psi[best - W], psi[best], psi[best + W]),
    psi: psi[best],
  };
}
