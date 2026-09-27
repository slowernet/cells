import { Q3 } from '../lattice3d';
import { REDUCE_WG } from './aux';
import { PARAMS3_WGSL, LATTICE3_WGSL, Precision, storageDecl, fElem, codecWgsl } from './common3d';

export const HISTORY3_LEN = 8192;

const header = (p: Precision) => `${storageDecl(p)}override WG: u32 = 128u;
${PARAMS3_WGSL}
@group(0) @binding(0) var<uniform> P: Params3;
`;

const coords = /* wgsl */ `
  let WH = P.W * P.H;
  let z = idx / WH;
  let y = (idx - z * WH) / P.W;
  let x = idx - z * WH - y * P.W;
  let p = vec3i(i32(x), i32(y), i32(z));`;

/** Rebuilds per-cell flags and force slots from the SDF, and refills cells a moved obstacle uncovered. */
export function flags3dShader(p: Precision): string {
  const E = fElem(p);
  return /* wgsl */ `${header(p)}
@group(0) @binding(1) var<storage, read> sdf: array<f32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> slot: array<u32>;
@group(0) @binding(4) var<storage, read_write> fA: array<${E}>;
@group(0) @binding(5) var<storage, read_write> fB: array<${E}>;
@group(0) @binding(6) var<storage, read> mac: array<vec4f>;
@group(0) @binding(7) var<storage, read_write> counter: array<atomic<u32>>;
${LATTICE3_WGSL}
${codecWgsl(p, 'fA', 'read_write')}${codecWgsl(p, 'fB', 'read_write')}
fn inRange(q: vec3i) -> bool {
  return q.x >= 0 && q.y >= 0 && q.z >= 0 && q.x < i32(P.W) && q.y < i32(P.H) && q.z < i32(P.D);
}

// The inlet and outlet planes stay fluid even when an obstacle reaches them.
fn solidAt(q: vec3i) -> bool {
  if (q.x == 0 || q.x == i32(P.W) - 1) { return false; }
  return sdf[at(q.x, q.y, q.z)] < 0.0;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = cellIndex(wg, li);
  if (idx >= P.N) { return; }
${coords}
  let old = flags[idx];
  if (solidAt(p)) {
    flags[idx] = FLAG_SOLID;
    slot[idx] = 0u;
    return;
  }

  var bits = 0u;
  if (x == 0u) { bits = FLAG_INLET; }
  if (x == P.W - 1u) { bits = FLAG_OUTLET; }
  var touchesBody = false;
  for (var i = 1u; i < ${Q3}u; i++) {
    let s = p - vec3i(CX[i], CY[i], CZ[i]);
    if (inRange(s)) {
      if (solidAt(s)) {
        bits |= 1u << i;
        touchesBody = true;
      }
      continue;
    }
    bits |= 1u << i;
    // A slip face mirrors from the reflected node, which is a body link when that node is solid.
    let outX = s.x < 0 || s.x >= i32(P.W);
    let outY = s.y < 0 || s.y >= i32(P.H);
    let outZ = s.z < 0 || s.z >= i32(P.D);
    if (!outX && outY != outZ) {
      var r = s;
      if (outY) { r.y = p.y; } else { r.z = p.z; }
      if (solidAt(r)) { touchesBody = true; }
    }
  }
  flags[idx] = bits;
  var sl = 0u;
  if (touchesBody) { sl = atomicAdd(&counter[0], 1u) + 1u; }
  slot[idx] = sl;

  if ((old & FLAG_SOLID) != 0u) {
    var rho = 0.0;
    var u = vec3f(0.0);
    var n = 0.0;
    for (var i = 1u; i < ${Q3}u; i++) {
      let q = p + vec3i(CX[i], CY[i], CZ[i]);
      if (!inRange(q)) { continue; }
      let m = mac[at(q.x, q.y, q.z)];
      if (m.w != 0.0) { rho += m.w; u += m.xyz; n += 1.0; }
    }
    if (n > 0.0) { rho /= n; u /= n; } else { rho = 1.0; u = vec3f(P.uIn, 0.0, 0.0); }
    for (var i = 0u; i < ${Q3}u; i++) {
      let v = feq(i, rho, u.x, u.y, u.z);
      store_fA(i * P.N + idx, v);
      store_fB(i * P.N + idx, v);
    }
  }
}
`;
}

/** Sets both distribution buffers to the inflow equilibrium in fluid cells and to rest in solids. */
export function init3dShader(p: Precision): string {
  const E = fElem(p);
  return /* wgsl */ `${header(p)}
@group(0) @binding(1) var<storage, read_write> fA: array<${E}>;
@group(0) @binding(2) var<storage, read_write> fB: array<${E}>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
${LATTICE3_WGSL}
${codecWgsl(p, 'fA', 'read_write')}${codecWgsl(p, 'fB', 'read_write')}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = cellIndex(wg, li);
  if (idx >= P.N) { return; }
  let solid = (flags[idx] & FLAG_SOLID) != 0u;
  for (var i = 0u; i < ${Q3}u; i++) {
    let v = select(feq(i, 1.0, P.uIn, 0.0, 0.0), 0.0, solid);
    store_fA(i * P.N + idx, v);
    store_fB(i * P.N + idx, v);
  }
}
`;
}

/** Velocity and density per cell as (ux, uy, uz, rho); rho = 0 marks a solid cell. */
export function macro3dShader(p: Precision): string {
  const E = fElem(p);
  return /* wgsl */ `${header(p)}
@group(0) @binding(1) var<storage, read> f: array<${E}>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var<storage, read_write> mac: array<vec4f>;
${LATTICE3_WGSL}
${codecWgsl(p, 'f', 'read')}
@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = cellIndex(wg, li);
  if (idx >= P.N) { return; }
  if ((flags[idx] & FLAG_SOLID) != 0u) { mac[idx] = vec4f(0.0); return; }
  var drho = 0.0;
  var j = vec3f(0.0);
  for (var i = 0u; i < ${Q3}u; i++) {
    let v = load_f(i * P.N + idx);
    drho += v;
    j += cvec(i) * v;
  }
  let rho = 1.0 + drho;
  mac[idx] = vec4f(j / rho, rho);
}
`;
}

/** Sums per-link momentum-exchange forces into one sample of the force history ring. */
export function reduce3dShader(): string {
  return /* wgsl */ `
const HISTORY_LEN: u32 = ${HISTORY3_LEN}u;
@group(0) @binding(0) var<storage, read> cellForce: array<vec4f>;
@group(0) @binding(1) var<storage, read> counter: array<u32>;
@group(0) @binding(2) var<storage, read_write> history: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> state: array<u32>;

var<workgroup> partial: array<vec4f, ${REDUCE_WG}>;

@compute @workgroup_size(${REDUCE_WG})
fn main(@builtin(local_invocation_index) li: u32) {
  let n = counter[0];
  var acc = vec4f(0.0);
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
