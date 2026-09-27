import { test, expect } from 'vitest';
import { VIEW3_WGSL, outlineShader, obstacleShader, sliceShader, tracerLineShader } from './render3d';
import { VIEW3_BYTES, VIEW3_OFFSETS } from '../view3d';

const modules = () => [outlineShader(), obstacleShader(), sliceShader(), tracerLineShader()];

test('vertex stages read storage read-only', () => {
  for (const code of modules()) expect(code).not.toContain('var<storage, read_write>');
});

test('obstacle writes depth and clips to the box', () => {
  const code = obstacleShader();
  expect(code).toContain('@builtin(frag_depth)');
  expect(code).toContain('V.boxMin');
  expect(code).toContain('V.boxMax');
  const clamp = code.match(/clamp\(d, ([\d.]+), ([\d.]+)\)/);
  expect(clamp).not.toBeNull();
  expect(Number(clamp![2])).toBeLessThanOrEqual(1.0);
});

test('outline uses the domain size', () => {
  const code = outlineShader();
  for (const f of ['P.W', 'P.H', 'P.D']) expect(code).toContain(f);
  expect(code).not.toContain('V.boxMin');
  expect(code).not.toContain('V.boxMax');
});

// WGSL uniform layout: scalars 4/4, vec3f size 12 align 16, mat4x4f size 64 align 16, struct size rounded to 16.
const LAYOUT: Record<string, [number, number]> = { f32: [4, 4], u32: [4, 4], vec3f: [12, 16], 'mat4x4f': [64, 16] };

test('struct matches packing', () => {
  const body = VIEW3_WGSL.slice(VIEW3_WGSL.indexOf('{') + 1, VIEW3_WGSL.lastIndexOf('}'));
  const fields = body.split(',').map((s) => s.trim()).filter(Boolean).map((s) => s.split(':').map((t) => t.trim()));
  let offset = 0;
  let maxAlign = 4;
  const computed: Record<string, number> = {};
  for (const [name, type] of fields) {
    const [size, align] = LAYOUT[type];
    offset = Math.ceil(offset / align) * align;
    computed[name] = offset;
    offset += size;
    maxAlign = Math.max(maxAlign, align);
  }
  const size = Math.ceil(offset / maxAlign) * maxAlign;
  expect(computed).toEqual({ ...VIEW3_OFFSETS });
  expect(size).toBe(VIEW3_BYTES);
});

test('line module binds particles read-only', () => {
  const code = tracerLineShader();
  expect(code).toMatch(/@binding\(4\) var<storage, read> particles/);
  expect(code).toContain('@binding(1)');
  expect(code).not.toContain('@binding(0)');
});
