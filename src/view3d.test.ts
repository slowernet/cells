import { test, expect } from 'vitest';
import { packView3, VIEW3_BYTES, VIEW3_OFFSETS, rakeSeeds, tracerSubsteps, TRACER_COUNT, type View3 } from './view3d';

const view: View3 = {
  viewProj: new Float32Array(16).map((_, i) => i + 1),
  invViewProj: new Float32Array(16).map((_, i) => 100 + i),
  eye: [201, 202, 203],
  sliceAxis: 2,
  slicePos: 0.25,
  mode: 1,
  uRef: 0.1,
  boxMin: [301, 302, 303],
  boxMax: [401, 402, 403],
  hasBody: true,
  tracers: true,
  steps: 12,
  count: 16384,
  frame: 7,
};

test('packView3 layout', () => {
  const buf = packView3(view);
  expect(buf.byteLength).toBe(VIEW3_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const at = (k: keyof typeof VIEW3_OFFSETS) => VIEW3_OFFSETS[k] / 4;
  expect(f[at('viewProj')]).toBe(1);
  expect(f[at('viewProj') + 15]).toBe(16);
  expect(f[at('invViewProj') + 15]).toBe(115);
  expect([...f.slice(at('eye'), at('eye') + 3)]).toEqual([201, 202, 203]);
  expect(u[at('sliceAxis')]).toBe(2);
  expect([...f.slice(at('boxMin'), at('boxMin') + 3)]).toEqual([301, 302, 303]);
  expect(f[at('slicePos')]).toBe(0.25);
  expect([...f.slice(at('boxMax'), at('boxMax') + 3)]).toEqual([401, 402, 403]);
  expect(f[at('uRef')]).toBeCloseTo(0.1, 7);
  expect(u[at('mode')]).toBe(1);
  expect(u[at('hasBody')]).toBe(1);
  expect(f[at('steps')]).toBe(12);
  expect(u[at('count')]).toBe(16384);
  expect(u[at('frame')]).toBe(7);
  expect(u[at('tracers')]).toBe(1);
});

test('rakeSeeds', () => {
  const [W, H, D] = [192, 96, 96];
  const s = rakeSeeds(W, H, D);
  expect(s.length).toBe(TRACER_COUNT * 4);
  expect(TRACER_COUNT).toBe(16384);
  const step = H / 2 / 128;
  let minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (let i = 0; i < TRACER_COUNT; i++) {
    expect(s[i * 4]).toBeCloseTo(0.1 * W, 4);
    minY = Math.min(minY, s[i * 4 + 1]); maxY = Math.max(maxY, s[i * 4 + 1]);
    minZ = Math.min(minZ, s[i * 4 + 2]); maxZ = Math.max(maxZ, s[i * 4 + 2]);
  }
  expect(minY).toBeCloseTo(H / 4 + step / 2, 5);
  expect(maxY).toBeCloseTo((3 * H) / 4 - step / 2, 5);
  expect(minZ).toBeCloseTo(D / 4 + step / 2, 5);
  expect(maxZ).toBeCloseTo((3 * D) / 4 - step / 2, 5);
});

test('tracerSubsteps', () => {
  expect(tracerSubsteps(0.1, 10)).toBe(1);
  expect(tracerSubsteps(0.1, 20)).toBe(1);
  expect(tracerSubsteps(0.1, 34)).toBe(4);
  expect(tracerSubsteps(0.1, 0)).toBe(1);
});
