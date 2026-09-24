/** Signed distance fields on a W x H node grid (node (x, y) sits at integer coordinates), negative inside solids. */

export const FAR = 1e6;

export function emptySdf(W: number, H: number): Float32Array {
  return new Float32Array(W * H).fill(FAR);
}

function stamp(sdf: Float32Array, W: number, H: number, box: [number, number, number, number], d: (x: number, y: number) => number) {
  const x0 = Math.max(0, Math.floor(box[0]));
  const y0 = Math.max(0, Math.floor(box[1]));
  const x1 = Math.min(W - 1, Math.ceil(box[2]));
  const y1 = Math.min(H - 1, Math.ceil(box[3]));
  for (let y = y0; y <= y1; y++)
    for (let x = x0; x <= x1; x++) {
      const i = y * W + x;
      sdf[i] = Math.min(sdf[i], d(x, y));
    }
}

const MARGIN = 3;

export function addCircle(sdf: Float32Array, W: number, H: number, cx: number, cy: number, r: number) {
  stamp(sdf, W, H, [cx - r - MARGIN, cy - r - MARGIN, cx + r + MARGIN, cy + r + MARGIN], (x, y) => Math.hypot(x - cx, y - cy) - r);
}

/** Rectangle of half-extents (hx, hy) rotated by angle (radians, counter-clockwise). */
export function addBox(sdf: Float32Array, W: number, H: number, cx: number, cy: number, hx: number, hy: number, angle = 0) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const R = Math.hypot(hx, hy) + MARGIN;
  stamp(sdf, W, H, [cx - R, cy - R, cx + R, cy + R], (x, y) => {
    const px = (x - cx) * c + (y - cy) * s;
    const py = -(x - cx) * s + (y - cy) * c;
    const qx = Math.abs(px) - hx;
    const qy = Math.abs(py) - hy;
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0)) + Math.min(Math.max(qx, qy), 0);
  });
}

/** NACA 4-digit symmetric section with a closed trailing edge, leading edge at (x0, y0), pitched nose-up by alpha. */
export function nacaPolygon(x0: number, y0: number, chord: number, thickness: number, alpha: number, n = 400): [number, number][] {
  const half = (t: number) =>
    5 * thickness * (0.2969 * Math.sqrt(t) - 0.126 * t - 0.3516 * t * t + 0.2843 * t ** 3 - 0.1036 * t ** 4);
  const pts: [number, number][] = [];
  for (let k = 0; k <= n; k++) {
    const t = (1 - Math.cos((Math.PI * k) / n)) / 2;
    pts.push([t, half(t)]);
  }
  for (let k = n - 1; k > 0; k--) {
    const t = (1 - Math.cos((Math.PI * k) / n)) / 2;
    pts.push([t, -half(t)]);
  }
  const c = Math.cos(-alpha);
  const s = Math.sin(-alpha);
  return pts.map(([u, v]) => [x0 + chord * (u * c - v * s), y0 + chord * (u * s + v * c)]);
}

export function addPolygon(sdf: Float32Array, W: number, H: number, poly: [number, number][]) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of poly) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  stamp(sdf, W, H, [minX - MARGIN, minY - MARGIN, maxX + MARGIN, maxY + MARGIN], (x, y) => {
    let d2 = Infinity;
    let inside = false;
    for (let a = 0, b = poly.length - 1; a < poly.length; b = a++) {
      const [ax, ay] = poly[a];
      const [bx, by] = poly[b];
      const ex = bx - ax, ey = by - ay;
      const t = Math.max(0, Math.min(1, ((x - ax) * ex + (y - ay) * ey) / (ex * ex + ey * ey)));
      const dx = x - ax - t * ex, dy = y - ay - t * ey;
      d2 = Math.min(d2, dx * dx + dy * dy);
      if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
    }
    const d = Math.sqrt(d2);
    return inside ? -d : d;
  });
}
