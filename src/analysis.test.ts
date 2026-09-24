import { describe, it, expect } from 'vitest';
import { periodFromCrossings, streamFunction, primaryVortex } from './analysis';
import { deriveTau, maxReynolds } from './units';
import { emptySdf, addCircle, addPolygon, nacaPolygon } from './geometry';
import { equilibrium } from './lattice';

describe('periodFromCrossings', () => {
  it('recovers the period of a sampled sine', () => {
    const t = Array.from({ length: 5000 }, (_, k) => k * 0.7);
    const v = t.map((s) => 0.3 + Math.sin((2 * Math.PI * s) / 123.4));
    expect(periodFromCrossings(t, v)!.period).toBeCloseTo(123.4, 2);
  });
  it('returns null without enough crossings', () => {
    expect(periodFromCrossings([0, 1, 2], [1, 2, 3])).toBeNull();
  });
});

describe('stream function', () => {
  it('locates the centre of a solid-body vortex', () => {
    const W = 64, H = 64, cx = 30.3, cy = 35.6;
    const ux = new Float32Array(W * H), uy = new Float32Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const r2 = (x - cx) ** 2 + (y - cy) ** 2;
        const g = Math.exp(-r2 / 200);
        ux[y * W + x] = -(y - cy) * g;
        uy[y * W + x] = (x - cx) * g;
      }
    const v = primaryVortex(streamFunction(ux, uy, W, H), W, H);
    expect(v.x).toBeCloseTo(cx, 0);
    expect(v.y).toBeCloseTo(cy, 0);
  });
});

describe('units', () => {
  it('derives tau from Re and clamps', () => {
    const d = deriveTau(100, 0.1, 40);
    expect(d.tau).toBeCloseTo(0.62, 10);
    expect(d.clamped).toBe(false);
    const c = deriveTau(1e5, 0.1, 40);
    expect(c.clamped).toBe(true);
    expect(c.tau).toBe(0.51);
    expect(c.effectiveRe).toBeCloseTo(maxReynolds(0.1, 40), 6);
  });
});

describe('geometry', () => {
  it('circle sdf is the signed distance', () => {
    const s = emptySdf(20, 20);
    addCircle(s, 20, 20, 10, 10, 4);
    expect(s[10 * 20 + 10]).toBeCloseTo(-4);
    expect(s[10 * 20 + 16]).toBeCloseTo(2);
  });
  it('naca polygon is inside-negative and 12% thick', () => {
    const W = 200, H = 60, s = emptySdf(W, H);
    const poly = nacaPolygon(50, 30, 100, 0.12, 0);
    addPolygon(s, W, H, poly);
    expect(s[30 * W + 80]).toBeLessThan(-5);
    expect(s[30 * W + 40]).toBeGreaterThan(9);
    const thick = Math.max(...poly.map((p) => p[1])) - Math.min(...poly.map((p) => p[1]));
    expect(thick).toBeCloseTo(12, 1);
  });
});

describe('equilibrium', () => {
  it('has the right moments', () => {
    const f = equilibrium(1.1, 0.05, -0.02) as number[];
    const CXs = [0, 1, 0, -1, 0, 1, -1, -1, 1], CYs = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    expect(f.reduce((a, b) => a + b)).toBeCloseTo(1.1, 12);
    expect(f.reduce((a, b, i) => a + b * CXs[i], 0)).toBeCloseTo(1.1 * 0.05, 12);
    expect(f.reduce((a, b, i) => a + b * CYs[i], 0)).toBeCloseTo(1.1 * -0.02, 12);
  });
});
