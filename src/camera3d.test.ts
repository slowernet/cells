import { test, expect } from 'vitest';
import { OrbitCamera, multiply, invert, transformPoint, type Vec3 } from './camera3d';

const target: Vec3 = [95.5, 47.5, 47.5];

test('target projects to the centre', () => {
  const cam = new OrbitCamera(target, 300);
  const p = transformPoint(cam.viewProj(1.5), target);
  expect(p[0]).toBeCloseTo(0, 6);
  expect(p[1]).toBeCloseTo(0, 6);
  expect(p[2]).toBeGreaterThan(0);
  expect(p[2]).toBeLessThan(1);
  // A point nearer the eye has smaller depth.
  const e = cam.eye();
  const near = transformPoint(cam.viewProj(1.5), [(e[0] + target[0]) / 2, (e[1] + target[1]) / 2, (e[2] + target[2]) / 2]);
  expect(near[2]).toBeLessThan(p[2]);
});

test('invert round-trips', () => {
  const m = new OrbitCamera(target, 300).viewProj(1.5);
  const id = multiply(m, invert(m));
  for (let i = 0; i < 16; i++) expect(Math.abs(id[i] - (i % 5 === 0 ? 1 : 0))).toBeLessThan(1e-5);
});

test('rotate keeps distance and clamps pitch', () => {
  const cam = new OrbitCamera(target, 300);
  cam.rotate(1234, 5000);
  const e = cam.eye();
  expect(Math.hypot(e[0] - target[0], e[1] - target[1], e[2] - target[2])).toBeCloseTo(300, 6);
  expect(Math.abs(cam.pitch)).toBeLessThanOrEqual(1.45);
  cam.rotate(0, -10000);
  expect(cam.pitch).toBeCloseTo(-1.45, 10);
});

test('zoom clamps', () => {
  const cam = new OrbitCamera(target, 300);
  for (let i = 0; i < 10; i++) cam.zoom(0.1);
  expect(cam.distance).toBeCloseTo(90, 6);
  for (let i = 0; i < 10; i++) cam.zoom(10);
  expect(cam.distance).toBeCloseTo(1200, 6);
});
