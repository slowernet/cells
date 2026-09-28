import { test, expect } from 'vitest';
import { tracerShaders } from './render';

test('tracers whose position went NaN respawn', () => {
  // WGSL may fold x != x away, so the shader tests the exponent bits, which also catches Inf.
  expect(tracerShaders()).toContain('(bitcast<vec2u>(q) & vec2u(0x7f800000u)) == vec2u(0x7f800000u)');
});
