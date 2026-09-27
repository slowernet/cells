/** The View3 uniform shared by the 3D render modules, and tracer helpers. */
import type { Mat4, Vec3 } from './camera3d';

export interface View3 {
  viewProj: Mat4;
  invViewProj: Mat4;
  eye: Vec3;
  sliceAxis: 0 | 1 | 2;
  /** Slice position along its axis, 0 to 1. */
  slicePos: number;
  /** 0 speed, 1 vorticity magnitude. */
  mode: 0 | 1;
  /** Target inflow speed for colour scales and tracer lengths. */
  uRef: number;
  /** Obstacle march bounds; ignored when hasBody is false. */
  boxMin: Vec3;
  boxMax: Vec3;
  hasBody: boolean;
  tracers: boolean;
  /** Lattice steps advanced this frame. */
  steps: number;
  /** Particle count; Renderer3D.encode fills it. */
  count: number;
  frame: number;
}

/** Byte offsets of each View3 field in the WGSL uniform (see VIEW3_WGSL). */
export const VIEW3_OFFSETS = {
  viewProj: 0,
  invViewProj: 64,
  eye: 128,
  sliceAxis: 140,
  boxMin: 144,
  slicePos: 156,
  boxMax: 160,
  uRef: 172,
  mode: 176,
  hasBody: 180,
  steps: 184,
  count: 188,
  frame: 192,
  tracers: 196,
} as const;

export const VIEW3_BYTES = 208;

export function packView3(v: View3): ArrayBuffer {
  const buf = new ArrayBuffer(VIEW3_BYTES);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  const o = VIEW3_OFFSETS;
  f.set(v.viewProj, o.viewProj / 4);
  f.set(v.invViewProj, o.invViewProj / 4);
  f.set(v.eye, o.eye / 4);
  u[o.sliceAxis / 4] = v.sliceAxis;
  f.set(v.boxMin, o.boxMin / 4);
  f[o.slicePos / 4] = v.slicePos;
  f.set(v.boxMax, o.boxMax / 4);
  f[o.uRef / 4] = v.uRef;
  u[o.mode / 4] = v.mode;
  u[o.hasBody / 4] = v.hasBody ? 1 : 0;
  f[o.steps / 4] = v.steps;
  u[o.count / 4] = v.count;
  u[o.frame / 4] = v.frame;
  u[o.tracers / 4] = v.tracers ? 1 : 0;
  return buf;
}

export const TRACER_COUNT = 16384;
export const RAKE = 128;

/** Seed points (x, y, z, 0) on a RAKE x RAKE grid at x = 0.1 W across the middle half of y and z. */
export function rakeSeeds(W: number, H: number, D: number): Float32Array {
  const s = new Float32Array(TRACER_COUNT * 4);
  for (let i = 0; i < TRACER_COUNT; i++) {
    const y = H / 4 + (((i % RAKE) + 0.5) / RAKE) * (H / 2);
    const z = D / 4 + ((Math.floor(i / RAKE) + 0.5) / RAKE) * (D / 2);
    s.set([0.1 * W, y, z, 0], i * 4);
  }
  return s;
}

/** A frame's tracer travel (cells at uTarget) above which advection is split into sub-steps of at most one cell. */
export const SUBSTEP_TRAVEL = 2;

/** Sub-steps for one frame of tracer advection; the advect shader applies the same rule. */
export function tracerSubsteps(uTarget: number, steps: number): number {
  const travel = uTarget * steps;
  return travel > SUBSTEP_TRAVEL ? Math.ceil(travel) : 1;
}
