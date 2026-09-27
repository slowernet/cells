import { test, expect } from 'vitest';
import { step3dShader } from './step3d';
import type { Precision } from './common3d';

const both: Precision[] = ['fp16', 'fp32'];
const count = (s: string, sub: string) => s.split(sub).length - 1;

test('fast path loads only its pulls', () => {
  for (const p of both) {
    const code = step3dShader(p);
    const fast = code.slice(code.indexOf('// fast path begin'), code.indexOf('// fast path end'));
    expect(fast.length).toBeGreaterThan(0);
    expect(count(fast, 'load_src(')).toBe(19);
    for (const other of ['src[', 'sdf[', 'slot[', 'flags[']) expect(fast).not.toContain(other);
    const main = code.slice(code.indexOf('fn main('));
    expect(count(main, 'flags[')).toBe(1);
  }
});

test('precision header', () => {
  const h = step3dShader('fp16');
  expect(h.startsWith('enable f16;')).toBe(true);
  expect(h).toMatch(/var<storage, read> src: array<f16>/);
  expect(h).toMatch(/var<storage, read_write> dst: array<f16>/);
  const s = step3dShader('fp32');
  expect(s).not.toContain('enable');
  expect(s).toMatch(/var<storage, read> src: array<f32>/);
  expect(s).toMatch(/var<storage, read_write> dst: array<f32>/);
});

test('no store to read-only src', () => {
  for (const p of both) expect(step3dShader(p)).not.toContain('store_src');
});

test('bindings', () => {
  for (const p of both) {
    const code = step3dShader(p);
    for (let b = 0; b <= 6; b++) expect(count(code, `@binding(${b})`)).toBe(1);
    expect(code).not.toContain('@binding(7)');
    expect(count(code, 'var<storage')).toBeLessThanOrEqual(8);
  }
});

test('equilibria are shifted', () => {
  for (const p of both) {
    const code = step3dShader(p);
    expect(count(code, 'fn feq(')).toBe(1);
    expect(code).toMatch(/fn feq\([^)]*\) -> f32 \{[^}]*rho - 1\.0/);
    expect(code).not.toMatch(/-\s*WT\[/);
  }
});
