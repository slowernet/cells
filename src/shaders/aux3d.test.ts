import { test, expect } from 'vitest';
import { flags3dShader, init3dShader, macro3dShader, reduce3dShader } from './aux3d';
import type { Precision } from './common3d';

const both: Precision[] = ['fp16', 'fp32'];
const count = (s: string, sub: string) => s.split(sub).length - 1;

test('macro marks solids with rho 0', () => {
  for (const p of both) {
    const code = macro3dShader(p);
    expect(code).toMatch(/FLAG_SOLID\) != 0u\) \{ mac\[idx\] = vec4f\(0\.0\); return; \}/);
    expect(code).toMatch(/mac\[idx\] = vec4f\(j \/ rho, rho\)/);
  }
});

test('precision header', () => {
  for (const gen of [flags3dShader, init3dShader, macro3dShader]) {
    expect(gen('fp16').startsWith('enable f16;')).toBe(true);
    expect(gen('fp16')).toContain('array<f16>');
    expect(gen('fp32')).not.toContain('enable');
    expect(gen('fp32')).not.toContain('f16');
  }
});

test('no store to read-only f', () => {
  for (const p of both) expect(macro3dShader(p)).not.toContain('store_f');
});

test('flags binds at most 8 storage buffers', () => {
  for (const p of both) {
    const code = flags3dShader(p);
    expect(count(code, 'var<storage')).toBe(7);
    expect(count(code, 'var<uniform>')).toBe(1);
  }
  expect(reduce3dShader()).toContain('array<vec4f>');
});

test('slots are counted only for body links', () => {
  // select() evaluates both arms, so an atomicAdd inside it would give every cell a slot.
  for (const p of both) expect(flags3dShader(p)).toMatch(/if \(touchesBody\) \{ sl = atomicAdd/);
});
