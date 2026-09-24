import { PARAMS_WGSL } from './common';

export const VIEW_BYTES = 48;
export const TRACER_WG = 64;

const VIEW_WGSL = /* wgsl */ `
struct View {
  size: vec2f, mode: u32, contrast: f32,
  uRef: f32, trailFade: f32, tracerMode: u32, frame: u32,
  steps: f32, count: u32, _a: u32, _b: u32,
};
`;

const COLOR_WGSL = /* wgsl */ `
fn viridis(t0: f32) -> vec3f {
  let t = clamp(t0, 0.0, 1.0);
  let c0 = vec3f(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  let c1 = vec3f(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  let c2 = vec3f(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  let c3 = vec3f(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  let c4 = vec3f(6.228269936347081, 14.17993336680509, 56.69055260068105);
  let c5 = vec3f(4.776384997670288, -13.74514537774601, -65.35303263337234);
  let c6 = vec3f(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}

// Blue-white-red diverging map, symmetric about zero.
fn diverging(t0: f32) -> vec3f {
  let t = clamp(t0, -1.0, 1.0);
  let a = pow(abs(t), 0.8);
  let blue = vec3f(0.130, 0.400, 0.745);
  let red = vec3f(0.745, 0.160, 0.170);
  let white = vec3f(0.97, 0.97, 0.96);
  return select(mix(white, blue, a), mix(white, red, a), t > 0.0);
}
`;

export function fieldShader(): string {
  return /* wgsl */ `
${PARAMS_WGSL}
${VIEW_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var<storage, read> mac: array<vec4f>;
${COLOR_WGSL}

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn at(x: i32, y: i32) -> vec4f {
  let cx = u32(clamp(x, 0, i32(P.W) - 1));
  let cy = u32(clamp(y, 0, i32(P.H) - 1));
  return mac[cy * P.W + cx];
}

fn scalar(x: i32, y: i32) -> f32 {
  switch V.mode {
    case 0u: {
      let dvy = at(x + 1, y).z - at(x - 1, y).z;
      let dux = at(x, y + 1).y - at(x, y - 1).y;
      return 0.5 * (dvy - dux);
    }
    case 1u: { return length(at(x, y).yz); }
    case 2u: { return at(x, y).x - 1.0; }
    default: {
      let gx = at(x + 1, y).x - at(x - 1, y).x;
      let gy = at(x, y + 1).x - at(x, y - 1).x;
      return 0.5 * length(vec2f(gx, gy));
    }
  }
}

@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let g = vec2f(pos.x / V.size.x * f32(P.W) - 0.5, (1.0 - pos.y / V.size.y) * f32(P.H) - 0.5);
  let n = vec2i(round(g));
  if (at(n.x, n.y).w > 0.5) { return vec4f(0.33, 0.35, 0.38, 1.0); }
  let b = vec2i(floor(g));
  let t = g - floor(g);
  let v = mix(mix(scalar(b.x, b.y), scalar(b.x + 1, b.y), t.x),
              mix(scalar(b.x, b.y + 1), scalar(b.x + 1, b.y + 1), t.x), t.y);
  // Each scale is a typical magnitude for a body about 1/10 of the grid height at the reference speed.
  let k = V.contrast;
  var c: vec3f;
  switch V.mode {
    case 0u: { c = diverging(v * k * f32(P.H) / (0.6 * V.uRef * 40.0)); }
    case 1u: { c = viridis(v * k / (1.6 * V.uRef)); }
    case 2u: { c = diverging(v * k / (1.5 * V.uRef * V.uRef)); }
    default: { c = vec3f(exp(-v * k * 400.0 / (V.uRef * V.uRef))); }
  }
  return vec4f(c, 1.0);
}
`;
}

export function tracerShaders(): string {
  return /* wgsl */ `
${PARAMS_WGSL}
${VIEW_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<uniform> V: View;
@group(0) @binding(2) var<storage, read> mac: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> particles: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> ages: array<f32>;

fn hash(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return f32((x >> 22u) ^ x) / 4294967295.0;
}

fn node(x: i32, y: i32) -> vec4f {
  return mac[u32(clamp(y, 0, i32(P.H) - 1)) * P.W + u32(clamp(x, 0, i32(P.W) - 1))];
}

fn velocity(p: vec2f) -> vec3f {
  let b = vec2i(floor(p));
  let t = p - floor(p);
  let a = mix(mix(node(b.x, b.y), node(b.x + 1, b.y), t.x), mix(node(b.x, b.y + 1), node(b.x + 1, b.y + 1), t.x), t.y);
  return vec3f(a.yz, a.w);
}

@compute @workgroup_size(${TRACER_WG})
fn advect(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= V.count) { return; }
  var p = particles[i].xy;
  var age = ages[i] + 1.0;
  let seed = i * 9781u + V.frame * 6271u;
  // Midpoint rule over the steps taken this frame.
  let v1 = velocity(p);
  let v2 = velocity(p + 0.5 * V.steps * v1.xy);
  let q = p + V.steps * v2.xy;
  let maxAge = select(80.0 + 80.0 * hash(seed + 7u), 1e9, V.tracerMode == 2u);
  let out = q.x < 0.0 || q.y < 0.0 || q.x > f32(P.W) - 1.0 || q.y > f32(P.H) - 1.0;
  if (out || v2.z > 0.5 || age > maxAge) {
    if (V.tracerMode == 2u) {
      // Streaklines: release from a rake of fixed seed points near the inlet.
      let rakes = 24u;
      p = vec2f(2.0, (f32(i % rakes) + 0.5) / f32(rakes) * f32(P.H));
    } else {
      p = vec2f(hash(seed) * f32(P.W), hash(seed + 1u) * f32(P.H));
    }
    particles[i] = vec4f(p, p);
    ages[i] = 0.0;
    return;
  }
  particles[i] = vec4f(q, p);
  ages[i] = age;
}

`;
}

// Vertex stages may only read storage buffers, so line drawing gets its own module.
export const LINE_WGSL = /* wgsl */ `
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(3) var<storage, read> particles: array<vec4f>;

struct LineOut { @builtin(position) pos: vec4f };

@vertex fn lineVs(@builtin(vertex_index) vi: u32) -> LineOut {
  let pt = particles[vi / 2u];
  let p = select(pt.zw, pt.xy, (vi & 1u) == 1u);
  var o: LineOut;
  o.pos = vec4f((p.x + 0.5) / f32(P.W) * 2.0 - 1.0, (p.y + 0.5) / f32(P.H) * 2.0 - 1.0, 0.0, 1.0);
  return o;
}

@fragment fn lineFs() -> @location(0) vec4f { return vec4f(1.0); }
`;

export const TRAIL_WGSL = /* wgsl */ `
${VIEW_WGSL}
@group(0) @binding(0) var<uniform> V: View;
@group(0) @binding(1) var tex: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fade(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let a = min(textureLoad(tex, vec2i(pos.xy), 0).r, 1.0) * V.trailFade;
  return vec4f(select(a, 0.0, a < 0.02));
}

// Dark trails over the pale curl map, light trails over the others.
@fragment fn composite(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let a = min(textureLoad(tex, vec2i(pos.xy), 0).r, 1.0);
  let c = select(vec3f(1.0), vec3f(0.05), V.mode == 0u || V.mode == 3u);
  return vec4f(c, a * 0.5);
}
`;
