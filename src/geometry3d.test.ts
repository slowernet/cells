import { test, expect } from 'vitest';
import { FAR } from './geometry';
import { emptySdf3, addSphere, addBox3, addCylinderZ, addWing, referenceArea } from './geometry3d';

const [W, H, D] = [32, 24, 20];
const at = (sdf: Float32Array, x: number, y: number, z: number) => sdf[x + W * (y + H * z)];

function trilinear(sdf: Float32Array, x: number, y: number, z: number) {
  const x0 = Math.floor(x), y0 = Math.floor(y), z0 = Math.floor(z);
  const tx = x - x0, ty = y - y0, tz = z - z0;
  let v = 0;
  for (let dz = 0; dz < 2; dz++)
    for (let dy = 0; dy < 2; dy++)
      for (let dx = 0; dx < 2; dx++)
        v += (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty) * (dz ? tz : 1 - tz) * at(sdf, x0 + dx, y0 + dy, z0 + dz);
  return v;
}

test('sphere sign and distance', () => {
  const sdf = emptySdf3(W, H, D);
  addSphere(sdf, W, H, D, 16, 12, 10, 4);
  expect(at(sdf, 16, 12, 10)).toBeCloseTo(-4, 6);
  expect(at(sdf, 22, 12, 10)).toBeCloseTo(2, 6);
  expect(at(sdf, 29, 12, 10)).toBe(FAR);
});

test('box rotation', () => {
  const sdf = emptySdf3(W, H, D);
  const a = Math.PI / 6;
  addBox3(sdf, W, H, D, 16, 12, 10, 8, 1, 1, a);
  const px = 16 + 6 * Math.cos(a), py = 12 + 6 * Math.sin(a);
  const mx = 16 + 6 * Math.cos(-a), my = 12 + 6 * Math.sin(-a);
  expect(trilinear(sdf, px, py, 10)).toBeLessThan(0);
  expect(trilinear(sdf, mx, my, 10)).toBeGreaterThan(0);
});

test('cylinder spans z', () => {
  const sdf = emptySdf3(W, H, D);
  addCylinderZ(sdf, W, H, D, 16, 12, 3);
  expect(at(sdf, 16, 12, 0)).toBeCloseTo(-3, 6);
  expect(at(sdf, 16, 12, D - 1)).toBeCloseTo(-3, 6);
});

test('wing pivots about mid-chord', () => {
  const sdf = emptySdf3(W, H, D);
  const [cx, cy, cz, chord, span] = [16, 12, 10, 19.2, 12];
  const a = (20 * Math.PI) / 180;
  addWing(sdf, W, H, D, cx, cy, cz, chord, span, a);
  // The trailing edge is thinner than a cell, so check the interior nodes' extent along the chord line and their distance from it.
  const d = [Math.cos(a), -Math.sin(a)];
  let lo = Infinity, hi = -Infinity, off = 0;
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      if (at(sdf, x, y, cz) >= 0) continue;
      const along = (x - cx) * d[0] + (y - cy) * d[1];
      lo = Math.min(lo, along);
      hi = Math.max(hi, along);
      off = Math.max(off, Math.abs(-(x - cx) * d[1] + (y - cy) * d[0]));
    }
  expect(trilinear(sdf, cx, cy, cz)).toBeLessThan(0);
  expect(lo).toBeGreaterThanOrEqual(-chord / 2);
  expect(lo).toBeLessThan(-chord / 2 + 1.5);
  expect(hi).toBeLessThanOrEqual(chord / 2);
  expect(hi).toBeGreaterThan(chord / 2 - 3);
  expect(off).toBeLessThan(0.06 * chord + 1);
});

test('wing tips are flat', () => {
  const sdf = emptySdf3(W, H, D);
  const [cx, cy, cz, chord, span] = [16, 12, 10, 16, 12];
  addWing(sdf, W, H, D, cx, cy, cz, chord, span, 0);
  // NACA00xx is thickest at 30% chord; the leading edge sits at cx - chord/2.
  const x = Math.round(cx - chord / 2 + 0.3 * chord);
  expect(at(sdf, x, cy, cz + span / 2 + 1)).toBeCloseTo(1, 1);
  expect(at(sdf, x, cy, cz - span / 2 - 1)).toBeCloseTo(1, 1);
  expect(at(sdf, x, cy, cz)).toBeLessThan(0);
});

test('referenceArea', () => {
  expect(referenceArea('sphere', 16, 96)).toBeCloseTo(201.06, 2);
  expect(referenceArea('cube', 16, 96)).toBe(256);
  expect(referenceArea('cylinder', 16, 96)).toBe(1536);
  expect(referenceArea('wing', 16, 96)).toBeCloseTo(921.6, 6);
  expect(referenceArea('none', 16, 96)).toBe(0);
});
