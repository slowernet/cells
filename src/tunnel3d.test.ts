import { test, expect } from 'vitest';
import { FAR } from './geometry';
import { presetsThatFit, fallbackPreset, nextSmaller, buildBody, coefficientScale, nextStepsPerFrame, tuneStepsPerFrame, FRAME_BUDGET_MS, hasDiverged, fieldDiverged } from './tunnel3d';

const small = { maxStorageBufferBindingSize: 128 * 2 ** 20, maxBufferSize: 256 * 2 ** 20 };
const tiny = { maxStorageBufferBindingSize: 2 ** 20, maxBufferSize: 2 ** 20 };

test('presets and fallback', () => {
  expect(presetsThatFit('fp16', small)).toEqual(['low', 'medium']);
  expect(presetsThatFit('fp32', small)).toEqual(['low']);
  expect(fallbackPreset('medium', 'fp32', small)).toBe('low');
  expect(fallbackPreset('high', 'fp16', small)).toBe('medium');
  expect(fallbackPreset('medium', 'fp16', small)).toBe('medium');
  expect(fallbackPreset('medium', 'fp16', tiny)).toBeNull();
  expect(nextSmaller('high')).toBe('medium');
  expect(nextSmaller('low')).toBeNull();
});

test('buildBody sphere', () => {
  const [W, H, D] = [192, 96, 96];
  const b = buildBody({ obstacle: 'sphere', sizeFraction: 0.2, angleDeg: 0 }, W, H, D);
  expect(b.sdf[W / 4 + W * (H / 2 + H * (D / 2))]).toBeLessThan(0);
  expect(b.lRef).toBeCloseTo(19.2, 6);
  expect(b.area).toBeCloseTo((Math.PI * 19.2 * 19.2) / 4, 6);
  const [lo, hi] = b.bounds!;
  expect(lo[0]).toBeCloseTo(W / 4 - 9.6 - 2, 6);
  expect(hi[1]).toBeCloseTo(H / 2 + 9.6 + 2, 6);
  expect(lo[2]).toBeCloseTo(D / 2 - 9.6 - 2, 6);
});

test('buildBody cube bounds grow with rotation', () => {
  const [W, H, D] = [192, 96, 96];
  const flat = buildBody({ obstacle: 'cube', sizeFraction: 0.2, angleDeg: 0 }, W, H, D).bounds!;
  const turned = buildBody({ obstacle: 'cube', sizeFraction: 0.2, angleDeg: 20 }, W, H, D).bounds!;
  expect(turned[1][0] - turned[0][0]).toBeGreaterThan(flat[1][0] - flat[0][0]);
  expect(turned[1][2] - turned[0][2]).toBeCloseTo(flat[1][2] - flat[0][2], 6);
});

test('buildBody none', () => {
  const b = buildBody({ obstacle: 'none', sizeFraction: 0.2, angleDeg: 0 }, 16, 8, 8);
  // Tau comes from the size setting even without a body; L = 1 put tau at the stability clamp.
  expect(b.lRef).toBeCloseTo(0.2 * 8, 12);
  expect(b.area).toBe(0);
  expect(b.bounds).toBeNull();
  expect(b.sdf.every((v) => v === FAR)).toBe(true);
});

test('coefficientScale', () => {
  expect(coefficientScale(0.1, 100)).toBeCloseTo(2, 12);
  expect(coefficientScale(0.1, 0)).toBe(0);
});

test('steps per frame can grow from small values', () => {
  // round(3 * 1.1) is 3, so a pure multiplicative rule sticks at small values.
  expect(nextStepsPerFrame(3, 2)).toBe(4);
  expect(nextStepsPerFrame(1, 2)).toBe(2);
  expect(nextStepsPerFrame(20, 2)).toBe(22);
  expect(nextStepsPerFrame(20, 0.5)).toBe(16);
  expect(nextStepsPerFrame(20, 1)).toBe(20);
  expect(nextStepsPerFrame(400, 2)).toBe(400);
  expect(nextStepsPerFrame(1, 0.1)).toBe(1);
});

test('tuner targets a fixed budget of 0.85 x 16.7 ms', () => {
  expect(FRAME_BUDGET_MS).toBeCloseTo(14.195, 6);
  expect(tuneStepsPerFrame(1, 0.5)).toBe(2);
  expect(tuneStepsPerFrame(10, 7)).toBe(11);
  expect(tuneStepsPerFrame(10, 14.195)).toBe(10);
  expect(tuneStepsPerFrame(20, 28.39)).toBe(16);
});

test('divergence is any non-finite force sample', () => {
  expect(hasDiverged([])).toBe(false);
  expect(hasDiverged([{ fx: 1, fy: -2 }, { fx: 1e30, fy: 0, fz: 3 }])).toBe(false);
  expect(hasDiverged([{ fx: 1, fy: 2 }, { fx: NaN, fy: 0 }])).toBe(true);
  expect(hasDiverged([{ fx: 1, fy: Infinity }])).toBe(true);
  expect(hasDiverged([{ fx: 1, fy: 2, fz: -Infinity }])).toBe(true);
});

test('a field row with any non-finite value has diverged', () => {
  expect(fieldDiverged(new Float32Array([1, 0.1, 0, 0, 1, 0.1, 0, 0]))).toBe(false);
  expect(fieldDiverged(new Float32Array([1, 0.1, NaN, 0]))).toBe(true);
  expect(fieldDiverged(new Float32Array([Infinity, 0, 0, 0]))).toBe(true);
});
