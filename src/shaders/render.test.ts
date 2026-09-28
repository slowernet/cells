import { test, expect } from 'vitest';
import { tracerShaders } from './render';

test('tracers whose position went NaN respawn', () => {
  // Every comparison with NaN is false, so without an explicit check a NaN streakline never respawns.
  expect(tracerShaders()).toMatch(/q\.x != q\.x \|\| q\.y != q\.y/);
});
