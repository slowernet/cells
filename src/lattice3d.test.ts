import { test, expect } from 'vitest';
import { CX, CY, CZ, W, OPP, MIRROR_Y, MIRROR_Z, PAIRS, Q3, FLAG_SOLID, FLAG_INLET, FLAG_OUTLET, equilibrium3 } from './lattice3d';

const C = [CX, CY, CZ];

test('weights and isotropy', () => {
  expect(CX.length).toBe(Q3);
  expect(W.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
  for (let a = 0; a < 3; a++) {
    let m1 = 0;
    for (let i = 0; i < Q3; i++) m1 += W[i] * C[a][i];
    expect(m1).toBeCloseTo(0, 12);
    for (let b = 0; b < 3; b++) {
      let m2 = 0;
      for (let i = 0; i < Q3; i++) m2 += W[i] * C[a][i] * C[b][i];
      expect(m2).toBeCloseTo(a === b ? 1 / 3 : 0, 12);
    }
  }
});

test('OPP and mirrors', () => {
  for (let i = 0; i < Q3; i++) {
    expect(OPP[OPP[i]]).toBe(i);
    for (let a = 0; a < 3; a++) expect(C[a][OPP[i]]).toBe(-C[a][i] || 0);
    expect(MIRROR_Y[MIRROR_Y[i]]).toBe(i);
    expect([CX[MIRROR_Y[i]], CY[MIRROR_Y[i]], CZ[MIRROR_Y[i]]]).toEqual([CX[i], -CY[i] || 0, CZ[i]]);
    expect(MIRROR_Z[MIRROR_Z[i]]).toBe(i);
    expect([CX[MIRROR_Z[i]], CY[MIRROR_Z[i]], CZ[MIRROR_Z[i]]]).toEqual([CX[i], CY[i], -CZ[i] || 0]);
  }
  const seen = new Set<number>();
  for (const [a, b] of PAIRS) {
    expect(b).toBe(OPP[a]);
    expect(a).toBeLessThan(b);
    seen.add(a).add(b);
  }
  expect([...seen].sort((x, y) => x - y)).toEqual(Array.from({ length: 18 }, (_, k) => k + 1));
});

test('equilibrium3 moments', () => {
  for (const [rho, ux, uy, uz] of [
    [1, 0, 0, 0],
    [1.02, 0.05, -0.03, 0.01],
    [0.97, -0.1, 0.02, 0.04],
  ]) {
    const f = equilibrium3(rho, ux, uy, uz);
    let s = 0;
    const j = [0, 0, 0];
    for (let i = 0; i < Q3; i++) {
      s += f[i];
      for (let a = 0; a < 3; a++) j[a] += f[i] * C[a][i];
    }
    expect(s).toBeCloseTo(rho, 6);
    expect(j[0]).toBeCloseTo(rho * ux, 6);
    expect(j[1]).toBeCloseTo(rho * uy, 6);
    expect(j[2]).toBeCloseTo(rho * uz, 6);
  }
});

test('flag bits', () => {
  const dirs = ((1 << 19) - 1) & ~1;
  const flags = [FLAG_SOLID, FLAG_INLET, FLAG_OUTLET];
  for (const f of flags) {
    expect(f & dirs).toBe(0);
    expect(f).toBeLessThan(2 ** 32);
    expect(f).toBeGreaterThan(0);
  }
  expect(FLAG_SOLID & FLAG_INLET).toBe(0);
  expect(FLAG_SOLID & FLAG_OUTLET).toBe(0);
  expect(FLAG_INLET & FLAG_OUTLET).toBe(0);
});
