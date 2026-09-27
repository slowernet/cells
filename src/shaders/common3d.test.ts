import { test, expect } from 'vitest';
import { PARAMS3_WGSL, PARAMS3_BYTES, codecWgsl, storageDecl, fitsLimits, PRESETS3, Precision } from './common3d';

test('Params3 layout', () => {
  const body = PARAMS3_WGSL.slice(PARAMS3_WGSL.indexOf('{') + 1, PARAMS3_WGSL.indexOf('}'));
  const fields = body.split(',').map((f) => f.trim()).filter(Boolean).map((f) => f.split(':').map((s) => s.trim()));
  expect(fields.length).toBe(16);
  for (const [, t] of fields) expect(['u32', 'f32']).toContain(t);
  expect(fields.length * 4).toBe(PARAMS3_BYTES);
  expect(fields[8][0]).toBe('tau');
  expect(fields[11][0]).toBe('uIn');
});

test('codec access', () => {
  for (const p of ['fp16', 'fp32'] as Precision[]) {
    const r = codecWgsl(p, 'x', 'read');
    expect(r).toContain('fn load_x(');
    expect(r).not.toContain('store_x');
    const rw = codecWgsl(p, 'x', 'read_write');
    expect(rw).toContain('fn load_x(');
    expect(rw).toContain('fn store_x(');
  }
});

test('codec', () => {
  const h = codecWgsl('fp16', 'f', 'read_write');
  expect(h).toContain('32768.0');
  expect(h).toContain('1.0 / 32768.0');
  expect(h).toContain('clamp(v, -1.99, 1.99)');
  expect(storageDecl('fp16').startsWith('enable f16;')).toBe(true);
  const s = codecWgsl('fp32', 'f', 'read_write');
  expect(s).not.toContain('32768');
  expect(s).not.toContain('clamp');
  expect(storageDecl('fp32')).toBe('');
});

test('fitsLimits', () => {
  const small = { maxStorageBufferBindingSize: 128 * 2 ** 20, maxBufferSize: 256 * 2 ** 20 };
  const big = { maxStorageBufferBindingSize: 4 * 2 ** 30, maxBufferSize: 4 * 2 ** 30 };
  const fits = (k: keyof typeof PRESETS3, p: Precision, l = small) => {
    const [w, h, d] = PRESETS3[k];
    return fitsLimits(w, h, d, p, l);
  };
  expect(fits('low', 'fp16')).toBe(true);
  expect(fits('low', 'fp32')).toBe(true);
  expect(fits('medium', 'fp16')).toBe(true);
  expect(fits('medium', 'fp32')).toBe(false);
  expect(fits('high', 'fp16')).toBe(false);
  expect(fits('high', 'fp32')).toBe(false);
  for (const k of ['low', 'medium', 'high'] as const) for (const p of ['fp16', 'fp32'] as Precision[]) expect(fits(k, p, big)).toBe(true);
  // Both limits apply.
  expect(fitsLimits(...PRESETS3.low, 'fp32', { maxStorageBufferBindingSize: 4 * 2 ** 30, maxBufferSize: 2 ** 20 })).toBe(false);
});
