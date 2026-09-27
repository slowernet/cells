/** Signed distance fields on a W x H x D node grid (node (x, y, z) at integer coordinates, index x + W(y + Hz)), negative inside solids. */
import { FAR, nacaPolygon } from './geometry';

export type Obstacle3 = 'sphere' | 'cube' | 'cylinder' | 'wing' | 'none';

const MARGIN = 3;

export function emptySdf3(W: number, H: number, D: number): Float32Array {
  return new Float32Array(W * H * D).fill(FAR);
}

type Box3 = [number, number, number, number, number, number];

function stamp3(sdf: Float32Array, W: number, H: number, D: number, box: Box3, d: (x: number, y: number, z: number) => number) {
  const x0 = Math.max(0, Math.floor(box[0] - MARGIN));
  const y0 = Math.max(0, Math.floor(box[1] - MARGIN));
  const z0 = Math.max(0, Math.floor(box[2] - MARGIN));
  const x1 = Math.min(W - 1, Math.ceil(box[3] + MARGIN));
  const y1 = Math.min(H - 1, Math.ceil(box[4] + MARGIN));
  const z1 = Math.min(D - 1, Math.ceil(box[5] + MARGIN));
  for (let z = z0; z <= z1; z++)
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        const i = x + W * (y + H * z);
        sdf[i] = Math.min(sdf[i], d(x, y, z));
      }
}

export function addSphere(sdf: Float32Array, W: number, H: number, D: number, cx: number, cy: number, cz: number, r: number) {
  stamp3(sdf, W, H, D, [cx - r, cy - r, cz - r, cx + r, cy + r, cz + r], (x, y, z) => Math.hypot(x - cx, y - cy, z - cz) - r);
}

/** Box of half-extents (hx, hy, hz) rotated counter-clockwise by angleZ (radians) about the z axis through its centre. */
export function addBox3(sdf: Float32Array, W: number, H: number, D: number, cx: number, cy: number, cz: number, hx: number, hy: number, hz: number, angleZ = 0) {
  const c = Math.cos(angleZ);
  const s = Math.sin(angleZ);
  const R = Math.hypot(hx, hy);
  stamp3(sdf, W, H, D, [cx - R, cy - R, cz - hz, cx + R, cy + R, cz + hz], (x, y, z) => {
    const px = (x - cx) * c + (y - cy) * s;
    const py = -(x - cx) * s + (y - cy) * c;
    const qx = Math.abs(px) - hx;
    const qy = Math.abs(py) - hy;
    const qz = Math.abs(z - cz) - hz;
    return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
  });
}

/** Cylinder of radius r along z, spanning the whole grid depth. */
export function addCylinderZ(sdf: Float32Array, W: number, H: number, D: number, cx: number, cy: number, r: number) {
  stamp3(sdf, W, H, D, [cx - r, cy - r, 0, cx + r, cy + r, D - 1], (x, y) => Math.hypot(x - cx, y - cy) - r);
}

function polygonDistance(poly: [number, number][], x: number, y: number): number {
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
}

/** NACA0012 wing of the given chord and span with flat tips, centred at (cx, cy, cz) and pitched nose-up by angleZ about its mid-chord. */
export function addWing(sdf: Float32Array, W: number, H: number, D: number, cx: number, cy: number, cz: number, chord: number, span: number, angleZ = 0) {
  // nacaPolygon rotates by -angleZ about the leading edge, which puts the mid-chord at chord/2 (cos a, -sin a).
  const ox = cx - (chord / 2) * Math.cos(angleZ);
  const oy = cy + (chord / 2) * Math.sin(angleZ);
  const poly = nacaPolygon(0, 0, chord, 0.12, angleZ, 200).map(([x, y]): [number, number] => [x + ox, y + oy]);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of poly) {
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const section = new Float32Array(W * H).fill(NaN);
  stamp3(sdf, W, H, D, [minX, minY, cz - span / 2, maxX, maxY, cz + span / 2], (x, y, z) => {
    const k = x + W * y;
    if (Number.isNaN(section[k])) section[k] = polygonDistance(poly, x, y);
    const d2 = section[k];
    const dz = Math.abs(z - cz) - span / 2;
    return d2 < 0 || dz < 0 ? Math.max(d2, dz) : Math.hypot(d2, dz);
  });
}

/** Frontal reference area in cells² for the force coefficients; size is the diameter, edge or chord in cells. */
export function referenceArea(obstacle: Obstacle3, size: number, D: number): number {
  switch (obstacle) {
    case 'sphere': return (Math.PI * size * size) / 4;
    case 'cube': return size * size;
    case 'cylinder': return size * D;
    case 'wing': return size * 0.6 * D;
    case 'none': return 0;
  }
}
