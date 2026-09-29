import { PARAMS3_WGSL } from './common3d';
import { COLOR_WGSL } from './render';
import { SUBSTEP_TRAVEL } from '../view3d';

/** Field order and types must match VIEW3_OFFSETS in view3d.ts. */
export const VIEW3_WGSL = /* wgsl */ `
struct View3 {
  viewProj: mat4x4f,
  invViewProj: mat4x4f,
  eye: vec3f,
  sliceAxis: u32,
  boxMin: vec3f,
  slicePos: f32,
  boxMax: vec3f,
  uRef: f32,
  mode: u32,
  hasBody: u32,
  steps: f32,
  count: u32,
  frame: u32,
  tracers: u32,
};
`;

const header = /* wgsl */ `
${PARAMS3_WGSL}
${VIEW3_WGSL}
@group(0) @binding(0) var<uniform> P: Params3;
@group(0) @binding(1) var<uniform> V: View3;
`;

/** Trilinear velocity and nearest-node solid test over the macro buffer (ux, uy, uz, rho; rho = 0 solid). */
export const MACRO_SAMPLE_WGSL = /* wgsl */ `
fn macAt(q: vec3i) -> vec4f {
  let c = clamp(q, vec3i(0), vec3i(i32(P.W) - 1, i32(P.H) - 1, i32(P.D) - 1));
  return mac[u32(c.x) + P.W * (u32(c.y) + P.H * u32(c.z))];
}

fn velocity(p: vec3f) -> vec3f {
  let b = vec3i(floor(p));
  let t = p - floor(p);
  let x00 = mix(macAt(b).xyz, macAt(b + vec3i(1, 0, 0)).xyz, t.x);
  let x10 = mix(macAt(b + vec3i(0, 1, 0)).xyz, macAt(b + vec3i(1, 1, 0)).xyz, t.x);
  let x01 = mix(macAt(b + vec3i(0, 0, 1)).xyz, macAt(b + vec3i(1, 0, 1)).xyz, t.x);
  let x11 = mix(macAt(b + vec3i(0, 1, 1)).xyz, macAt(b + vec3i(1, 1, 1)).xyz, t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

fn solidAt(p: vec3f) -> bool { return macAt(vec3i(round(p))).w == 0.0; }

// |curl u| from central differences of trilinear velocity at ±1 cell.
fn vorticity(p: vec3f) -> f32 {
  let dx = velocity(p + vec3f(1.0, 0.0, 0.0)) - velocity(p - vec3f(1.0, 0.0, 0.0));
  let dy = velocity(p + vec3f(0.0, 1.0, 0.0)) - velocity(p - vec3f(0.0, 1.0, 0.0));
  let dz = velocity(p + vec3f(0.0, 0.0, 1.0)) - velocity(p - vec3f(0.0, 0.0, 1.0));
  return length(0.5 * vec3f(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x));
}

// The View field as a 0..1 colour-map input, shared by the slice and the tracers.
fn fieldValue(p: vec3f, u: vec3f) -> f32 {
  if (V.mode == 0u) { return length(u) / (1.6 * V.uRef); }
  return vorticity(p) / (0.3 * V.uRef);
}
`;

/** The 12 edges of the domain box as a line list (24 vertices). */
export function outlineShader(): string {
  return /* wgsl */ `${header}
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let e = i / 2u;
  let end = f32(i & 1u);
  let axis = e / 4u;
  let a = f32(e & 1u);
  let b = f32((e >> 1u) & 1u);
  var t = vec3f(a, b, end);
  if (axis == 0u) { t = vec3f(end, a, b); } else if (axis == 1u) { t = vec3f(a, end, b); }
  let lo = vec3f(-0.5);
  let hi = vec3f(f32(P.W), f32(P.H), f32(P.D)) - 0.5;
  return V.viewProj * vec4f(mix(lo, hi, t), 1.0);
}

@fragment fn fs() -> @location(0) vec4f { return vec4f(0.42, 0.42, 0.46, 1.0); }
`;
}

/** Sphere-traces the obstacle through the SDF inside its bounding box and writes depth. */
export function obstacleShader(): string {
  return /* wgsl */ `${header}
@group(0) @binding(2) var<storage, read> sdf: array<f32>;

fn sdfAt(q: vec3i) -> f32 {
  let c = clamp(q, vec3i(0), vec3i(i32(P.W) - 1, i32(P.H) - 1, i32(P.D) - 1));
  // Builders stamp true distance within 3 cells of a body and FAR beyond.
  return min(sdf[u32(c.x) + P.W * (u32(c.y) + P.H * u32(c.z))], 3.0);
}

fn dist(p: vec3f) -> f32 {
  let b = vec3i(floor(p));
  let t = p - floor(p);
  let x00 = mix(sdfAt(b), sdfAt(b + vec3i(1, 0, 0)), t.x);
  let x10 = mix(sdfAt(b + vec3i(0, 1, 0)), sdfAt(b + vec3i(1, 1, 0)), t.x);
  let x01 = mix(sdfAt(b + vec3i(0, 0, 1)), sdfAt(b + vec3i(1, 0, 1)), t.x);
  let x11 = mix(sdfAt(b + vec3i(0, 1, 1)), sdfAt(b + vec3i(1, 1, 1)), t.x);
  return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

struct VsOut { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
  var o: VsOut;
  o.pos = vec4f(p, 0.0, 1.0);
  o.ndc = p;
  return o;
}

struct FsOut { @location(0) color: vec4f, @builtin(frag_depth) depth: f32 };

@fragment fn fs(in: VsOut) -> FsOut {
  if (V.hasBody == 0u) { discard; }
  let a = V.invViewProj * vec4f(in.ndc, 0.0, 1.0);
  let b = V.invViewProj * vec4f(in.ndc, 1.0, 1.0);
  let ro = a.xyz / a.w;
  let rd = normalize(b.xyz / b.w - ro);
  let inv = 1.0 / rd;
  let s0 = (V.boxMin - ro) * inv;
  let s1 = (V.boxMax - ro) * inv;
  let tEntry = max(max(min(s0.x, s1.x), min(s0.y, s1.y)), min(s0.z, s1.z));
  let tExit = min(min(max(s0.x, s1.x), max(s0.y, s1.y)), max(s0.z, s1.z));
  if (tExit < max(tEntry, 0.0)) { discard; }
  var t = max(tEntry, 0.0);
  let steps = u32(ceil(length(V.boxMax - V.boxMin))) + 2u;
  var hit = false;
  var p = ro;
  for (var k = 0u; k < steps; k++) {
    p = ro + t * rd;
    let d = dist(p);
    if (d < 0.02) { hit = true; break; }
    t += clamp(d, 0.05, 1.0);
    if (t > tExit) { break; }
  }
  if (!hit) { discard; }
  let e = 0.5;
  let n = normalize(vec3f(
    dist(p + vec3f(e, 0.0, 0.0)) - dist(p - vec3f(e, 0.0, 0.0)),
    dist(p + vec3f(0.0, e, 0.0)) - dist(p - vec3f(0.0, e, 0.0)),
    dist(p + vec3f(0.0, 0.0, e)) - dist(p - vec3f(0.0, 0.0, e))));
  let light = normalize(V.eye - p);
  let shade = 0.25 + 0.75 * max(dot(n, light), 0.0);
  let clip = V.viewProj * vec4f(p, 1.0);
  var o: FsOut;
  o.color = vec4f(vec3f(0.86, 0.84, 0.78) * shade, 1.0);
  o.depth = clip.z / clip.w;
  return o;
}
`;
}

/** An axis-aligned slice through the domain coloured by speed or vorticity magnitude. */
export function sliceShader(): string {
  return /* wgsl */ `${header}
@group(0) @binding(2) var<storage, read> mac: array<vec4f>;
${COLOR_WGSL}
${MACRO_SAMPLE_WGSL}

struct VsOut { @builtin(position) pos: vec4f, @location(0) world: vec3f };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let corners = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
  let c = corners[i];
  let dims = vec3f(f32(P.W), f32(P.H), f32(P.D));
  let lo = vec3f(-0.5);
  let hi = dims - 0.5;
  let at = V.slicePos * (dims - 1.0);
  var w: vec3f;
  if (V.sliceAxis == 0u) { w = vec3f(at.x, mix(lo.y, hi.y, c.x), mix(lo.z, hi.z, c.y)); }
  else if (V.sliceAxis == 1u) { w = vec3f(mix(lo.x, hi.x, c.x), at.y, mix(lo.z, hi.z, c.y)); }
  else { w = vec3f(mix(lo.x, hi.x, c.x), mix(lo.y, hi.y, c.y), at.z); }
  var o: VsOut;
  o.pos = V.viewProj * vec4f(w, 1.0);
  o.world = w;
  return o;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  let p = in.world;
  if (solidAt(p)) { return vec4f(vec3f(0.35), 1.0); }
  return vec4f(viridis(fieldValue(p, velocity(p))), 1.0);
}
`;
}

export const TRACER_WG = 64;

/** Moves tracers through the macro field; two vec4 per particle (position, velocity). */
export function advectShader(): string {
  return /* wgsl */ `${header}
@group(0) @binding(2) var<storage, read> mac: array<vec4f>;
@group(0) @binding(3) var<storage, read> seeds: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> particles: array<vec4f>;
${MACRO_SAMPLE_WGSL}

fn inside(p: vec3f) -> bool {
  let hi = vec3f(f32(P.W), f32(P.H), f32(P.D)) - 0.5;
  return all(p >= vec3f(-0.5)) && all(p <= hi);
}

@compute @workgroup_size(${TRACER_WG})
fn advect(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= V.count) { return; }
  var p = particles[2u * i].xyz;
  // Midpoint rule over the lattice steps taken this frame, split so no sub-step moves more than a cell.
  let travel = V.uRef * V.steps;
  let n = select(1u, u32(ceil(travel)), travel > ${SUBSTEP_TRAVEL.toFixed(1)});
  let h = V.steps / f32(n);
  for (var k = 0u; k < n; k++) {
    let v1 = velocity(p);
    p += h * velocity(p + 0.5 * h * v1);
    if (!inside(p) || solidAt(p)) {
      p = seeds[i].xyz;
      break;
    }
  }
  particles[2u * i] = vec4f(p, 0.0);
  particles[2u * i + 1u] = vec4f(velocity(p), 0.0);
}
`;
}

/** Draws each tracer as a segment from p back along its velocity, coloured by the View field; vertex stages may only read storage. */
export function tracerLineShader(): string {
  return /* wgsl */ `${header}
@group(0) @binding(2) var<storage, read> mac: array<vec4f>;
@group(0) @binding(4) var<storage, read> particles: array<vec4f>;
${COLOR_WGSL}
${MACRO_SAMPLE_WGSL}

struct VsOut { @builtin(position) pos: vec4f, @location(0) value: f32 };

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  let k = vi / 2u;
  let p = particles[2u * k].xyz;
  let u = particles[2u * k + 1u].xyz;
  let end = select(p, p - (3.0 / V.uRef) * u, (vi & 1u) == 1u);
  var o: VsOut;
  o.pos = V.viewProj * vec4f(end, 1.0);
  o.value = fieldValue(p, u);
  return o;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4f {
  return vec4f(viridis(in.value), 0.8);
}
`;
}
