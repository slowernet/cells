import { CX, CY, Q } from '../lattice';
import { PARAMS_WGSL, LATTICE_WGSL } from './common';

export const HISTORY_LEN = 8192;
export const REDUCE_WG = 256;

const header = /* wgsl */ `
override WG: u32 = 128u;
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
`;

/** Equilibrium minus its rest weight, matching the shifted storage of the distribution buffers. */
const FEQ_WGSL = /* wgsl */ `
fn feq(i: u32, rho: f32, ux: f32, uy: f32) -> f32 {
  let cu = 3.0 * (f32(CX[i]) * ux + f32(CY[i]) * uy);
  return WT[i] * (rho - 1.0 + rho * (cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy)));
}
`;

/** Rebuilds per-cell flags from the SDF, assigns force slots, and refills cells a moved obstacle uncovered. */
export function flagsShader(): string {
  return /* wgsl */ `${header}
@group(0) @binding(1) var<storage, read> sdf: array<f32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> fA: array<f32>;
@group(0) @binding(4) var<storage, read_write> fB: array<f32>;
@group(0) @binding(5) var<storage, read> mac: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> counter: array<atomic<u32>>;
${LATTICE_WGSL}
${FEQ_WGSL}

// Open boundary columns stay fluid even when an obstacle is drawn over them.
fn solidAt(x: u32, y: u32) -> bool {
  let open = (P.left == X_OPEN && x == 0u) || (P.right == X_OPEN && x == P.W - 1u);
  return !open && sdf[y * P.W + x] < 0.0;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = cellIndex(wg, li);
  if (idx >= P.N) { return; }
  let y = idx / P.W;
  let x = idx - y * P.W;
  let old = flags[idx];
  if (solidAt(x, y)) { flags[idx] = FLAG_SOLID; return; }

  var bits = 0u;
  if (P.left == X_OPEN && x == 0u) { bits = FLAG_INLET; }
  if (P.right == X_OPEN && x == P.W - 1u) { bits = FLAG_OUTLET; }
  var touchesBody = false;
  for (var i = 1u; i < 9u; i++) {
    let sx = i32(x) - CX[i];
    let sy = i32(y) - CY[i];
    if (sx < 0 || sy < 0 || sx >= i32(P.W) || sy >= i32(P.H)) {
      bits |= 1u << i;
    } else if (solidAt(u32(sx), u32(sy))) {
      bits |= 1u << i;
      touchesBody = true;
    }
  }
  if (touchesBody) {
    let slot = atomicAdd(&counter[0], 1u) + 1u;
    if (slot < (1u << (32u - BIDX_SHIFT))) { bits |= slot << BIDX_SHIFT; }
  }
  flags[idx] = bits;

  if ((old & FLAG_SOLID) != 0u) {
    var rho = 0.0;
    var u = vec2f(0.0);
    var n = 0.0;
    for (var i = 1u; i < 9u; i++) {
      let sx = i32(x) + CX[i];
      let sy = i32(y) + CY[i];
      if (sx < 0 || sy < 0 || sx >= i32(P.W) || sy >= i32(P.H)) { continue; }
      let m = mac[u32(sy) * P.W + u32(sx)];
      if (m.w == 0.0) { rho += m.x; u += m.yz; n += 1.0; }
    }
    if (n > 0.0) { rho /= n; u /= n; } else { rho = 1.0; }
    for (var i = 0u; i < 9u; i++) {
      let v = feq(i, rho, u.x, u.y);
      fA[i * P.N + idx] = v;
      fB[i * P.N + idx] = v;
    }
  }
}
`;
}

/** Sets both distribution buffers to equilibrium at the macro field (rho, ux, uy). */
export function initShader(): string {
  return /* wgsl */ `${header}
@group(0) @binding(1) var<storage, read> mac: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> fA: array<f32>;
@group(0) @binding(3) var<storage, read_write> fB: array<f32>;
${LATTICE_WGSL}
${FEQ_WGSL}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = cellIndex(wg, li);
  if (idx >= P.N) { return; }
  let m = mac[idx];
  for (var i = 0u; i < 9u; i++) {
    let v = feq(i, m.x, m.y, m.z);
    fA[i * P.N + idx] = v;
    fB[i * P.N + idx] = v;
  }
}
`;
}

/** Density and velocity per cell for rendering, tracers and validation; w = 1 marks solid. */
export function macroShader(): string {
  const sum = (coef: (i: number) => number) =>
    Array.from({ length: Q }, (_, i) => coef(i))
      .map((c, i) => (c === 0 ? '' : `${c > 0 ? '+' : '-'} f[${i}u * P.N + idx]`))
      .filter(Boolean)
      .join(' ')
      .replace(/^\+ /, '');
  return /* wgsl */ `${header}
@group(0) @binding(1) var<storage, read> f: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> mac: array<vec4f>;
${LATTICE_WGSL}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = cellIndex(wg, li);
  if (idx >= P.N) { return; }
  if ((flags[idx] & FLAG_SOLID) != 0u) { mac[idx] = vec4f(1.0, 0.0, 0.0, 1.0); return; }
  let rho = 1.0 + ${sum(() => 1)};
  var jx = ${sum((i) => CX[i])};
  var jy = ${sum((i) => CY[i])};
  // Stored populations are post-collision, which already carry the full body-force impulse.
  if (P.forcing != 0u) { jx -= 0.5 * P.fx; jy -= 0.5 * P.fy; }
  mac[idx] = vec4f(rho, jx / rho, jy / rho, 0.0);
}
`;
}

/** Sums per-link momentum-exchange forces into one sample of the force history ring. */
export function reduceShader(): string {
  return /* wgsl */ `
const HISTORY_LEN: u32 = ${HISTORY_LEN}u;
@group(0) @binding(0) var<storage, read> cellForce: array<vec2f>;
@group(0) @binding(1) var<storage, read> counter: array<u32>;
@group(0) @binding(2) var<storage, read_write> history: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> state: array<u32>;

var<workgroup> partial: array<vec2f, ${REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(local_invocation_index) li: u32) {
  let n = counter[0];
  var acc = vec2f(0.0);
  for (var k = li; k < n; k += ${REDUCE_WG}u) { acc += cellForce[k]; }
  partial[li] = acc;
  workgroupBarrier();
  for (var s = ${REDUCE_WG / 2}u; s > 0u; s >>= 1u) {
    if (li < s) { partial[li] += partial[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) {
    let sample = state[0];
    history[sample % HISTORY_LEN] = partial[0];
    state[0] = sample + 1u;
  }
}
`;
}

export const BRUSH_BYTES = 32;

/** Unions (mode 0) or subtracts (mode 1) a disc from the SDF, inside a bounding box. */
export function brushShader(): string {
  return /* wgsl */ `
struct Brush { cx: f32, cy: f32, r: f32, mode: u32, x0: u32, y0: u32, w: u32, h: u32 };
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> B: Brush;
@group(0) @binding(1) var<uniform> D: Params;
@group(0) @binding(2) var<storage, read_write> sdf: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= B.w || g.y >= B.h) { return; }
  let x = B.x0 + g.x;
  let y = B.y0 + g.y;
  if (x >= D.W || y >= D.H) { return; }
  let idx = y * D.W + x;
  let d = distance(vec2f(f32(x), f32(y)), vec2f(B.cx, B.cy)) - B.r;
  sdf[idx] = select(max(sdf[idx], -d), min(sdf[idx], d), B.mode == 0u);
}
`;
}
