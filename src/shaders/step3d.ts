import { CX, CY, CZ, W, PAIRS, Q3 } from '../lattice3d';
import { wgslFloat as fl } from './common';
import { PARAMS3_WGSL, LATTICE3_WGSL, Precision, storageDecl, fElem, codecWgsl } from './common3d';

const C = [CX, CY, CZ];
const U = ['ux', 'uy', 'uz'];

/** Signed sum of f_i weighted by coef(i), skipping zero terms. */
function sum(coef: (i: number) => number, term: (i: number) => string = (i) => `f${i}`) {
  const parts = Array.from({ length: Q3 }, (_, i) => [coef(i), term(i)] as const).filter(([c]) => c !== 0);
  return parts.map(([c, t], k) => `${c > 0 ? (k ? ' + ' : '') : k ? ' - ' : '-'}${Math.abs(c) === 1 ? t : `${Math.abs(c)}.0 * ${t}`}`).join('');
}

/** c_i . u as WGSL, e.g. "ux - uz". */
function cDotU(i: number) {
  return C.map((c, a) => (c[i] === 0 ? '' : `${c[i] > 0 ? '+' : '-'}${U[a]}`)).join(' ').trim().replace(/^\+/, '');
}

/**
 * Fused pull-stream + boundary + TRT collide kernel for D3Q19. One dispatch advances the lattice one step.
 * Interior cells (flags == 0) take a branch-free unrolled path; every boundary rule lives in pullSlow3.
 * Buffers store f_i - w_i, in FP16 (scaled by 2^15) or FP32.
 */
export function step3dShader(p: Precision): string {
  const vars = Array.from({ length: Q3 }, (_, i) => `f${i}`);
  const fastPull = Array.from({ length: Q3 }, (_, i) => {
    if (i === 0) return `    f0 = load_src(idx);`;
    const shift = [
      CZ[i] === 0 ? '' : CZ[i] > 0 ? ' - WH' : ' + WH',
      CY[i] === 0 ? '' : CY[i] > 0 ? ' - P.W' : ' + P.W',
      CX[i] === 0 ? '' : CX[i] > 0 ? ' - 1u' : ' + 1u',
    ].join('');
    return `    f${i} = load_src(${i}u * N + idx${shift});`;
  }).join('\n');

  const slowPull = Array.from({ length: Q3 - 1 }, (_, k) => `    f${k + 1} = pullSlow3(${k + 1}u, xi, yi, zi, idx, fl, outlet, &force);`).join('\n');

  const moments = `
  let drho = ${vars.join(' + ')};
  let rho = 1.0 + drho;
  let invRho = 1.0 / rho;
  let ux = (${sum((i) => CX[i])}) * invRho;
  let uy = (${sum((i) => CY[i])}) * invRho;
  let uz = (${sum((i) => CZ[i])}) * invRho;
  let usq = 1.5 * (ux * ux + uy * uy + uz * uz);`;

  const eqTerms = PAIRS.map(([a]) => `  let cu${a} = 3.0 * (${cDotU(a)});`).join('\n');
  // Symmetric and antisymmetric equilibrium halves for each TRT pair (shifted by w_i).
  const eqPairs = PAIRS.map(
    ([a]) => `  let eqS${a} = ${fl(W[a])} * (drho + rho * (0.5 * cu${a} * cu${a} - usq));
  let eqA${a} = ${fl(W[a])} * rho * cu${a};`,
  ).join('\n');

  const nDecl = PAIRS.map(
    ([a, b]) => `    let n${a} = f${a} - (eqS${a} + eqA${a});
    let n${b} = f${b} - (eqS${a} - eqA${a});`,
  ).join('\n');
  const pi = (x: number, y: number) => sum((i) => C[x][i] * C[y][i], (i) => `n${i}`);
  const smag = `
  var omP = omegaBase;
  if (P.smagC2 > 0.0) {
${nDecl}
    let pxx = ${pi(0, 0)};
    let pyy = ${pi(1, 1)};
    let pzz = ${pi(2, 2)};
    let pxy = ${pi(0, 1)};
    let pxz = ${pi(0, 2)};
    let pyz = ${pi(1, 2)};
    let pi = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
    let t0 = 1.0 / omegaBase;
    let tEff = 0.5 * (t0 + sqrt(t0 * t0 + ${fl(18 * Math.SQRT2)} * P.smagC2 * pi * invRho));
    omP = 1.0 / tEff;
  }
  let omM = 1.0 / (0.5 + P.lambda / (1.0 / omP - 0.5));`;

  const absorb = `
  var s0 = 0.0;
${PAIRS.map(([a]) => `  var sS${a} = 0.0;
  var sA${a} = 0.0;`).join('\n')}
  if (sigma > 0.0) {
    // Absorbing layer: pull the local equilibrium toward the inflow state so sound and vortices leave without reflecting.
    let ut = P.uIn;
    let dq = 1.5 * rho * (ut * ut - ux * ux - uy * uy - uz * uz);
    s0 += sigma * ${fl(W[0])} * -dq;
${PAIRS.map(([a]) => `    {
      let cut = 3.0 * ${CX[a]}.0 * ut;
      sS${a} += sigma * ${fl(W[a])} * (0.5 * rho * (cut * cut - cu${a} * cu${a}) - dq);
      sA${a} += sigma * ${fl(W[a])} * rho * (cut - cu${a});
    }`).join('\n')}
  }`;

  const collide = [
    `  store_dst(idx, f0 - omP * (f0 - ${fl(W[0])} * (drho - rho * usq)) + s0);`,
    ...PAIRS.map(
      ([a, b]) => `  {
    let sym = omP * (0.5 * (f${a} + f${b}) - eqS${a});
    let asym = omM * (0.5 * (f${a} - f${b}) - eqA${a});
    store_dst(${a}u * N + idx, f${a} - sym - asym + sS${a} + sA${a});
    store_dst(${b}u * N + idx, f${b} - sym + asym + sS${a} - sA${a});
  }`,
    ),
  ].join('\n');

  const E = fElem(p);
  return `${storageDecl(p)}override WG: u32 = 128u;
${PARAMS3_WGSL}
@group(0) @binding(0) var<uniform> P: Params3;
@group(0) @binding(1) var<storage, read> src: array<${E}>;
@group(0) @binding(2) var<storage, read_write> dst: array<${E}>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read> sdf: array<f32>;
@group(0) @binding(5) var<storage, read> slot: array<u32>;
@group(0) @binding(6) var<storage, read_write> cellForce: array<vec4f>;
${LATTICE3_WGSL}
${codecWgsl(p, 'src', 'read')}${codecWgsl(p, 'dst', 'read_write')}
fn at(x: i32, y: i32, z: i32) -> u32 { return u32(x + i32(P.W) * (y + i32(P.H) * z)); }
fn isSolid(s: u32) -> bool { return (flags[s] & FLAG_SOLID) != 0u; }
fn cvec(i: u32) -> vec3f { return vec3f(f32(CX[i]), f32(CY[i]), f32(CZ[i])); }

/** Equilibrium minus its rest weight, matching the shifted storage of the distribution buffers. */
fn feq(i: u32, rho: f32, ux: f32, uy: f32, uz: f32) -> f32 {
  let cu = 3.0 * (f32(CX[i]) * ux + f32(CY[i]) * uy + f32(CZ[i]) * uz);
  return WT[i] * (rho - 1.0 + rho * (cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz)));
}

// Population arriving at (x, y, z) along direction i when its upstream node x - c_i is not a plain fluid node.
fn pullSlow3(i: u32, x: i32, y: i32, z: i32, idx: u32, fl: u32, outlet: bool, force: ptr<function, vec3f>) -> f32 {
  let N = P.N;
  let j = OPP[i];
  let sx = x - CX[i];
  let sy = y - CY[i];
  let sz = z - CZ[i];
  if ((fl & (1u << i)) == 0u) { return load_src(i * N + at(sx, sy, sz)); }
  let outX = sx < 0 || sx >= i32(P.W);
  let outY = sy < 0 || sy >= i32(P.H);
  let outZ = sz < 0 || sz >= i32(P.D);
  // Outlet: zero-gradient copy from the plane behind.
  if (outX && outlet) { return load_src(i * N + at(i32(P.W) - 2, y, z)); }
  // Edge links outside a y face and a z face: the double mirror equals bounce-back.
  if (outY && outZ) { return load_src(j * N + idx); }
  if (outY || outZ) {
    // Slip: specular reflection across the face, or bounce-back where a body meets the face.
    var r = vec3i(sx, sy, sz);
    var m = MIRROR_Y[i];
    if (outY) { r.y = y; } else { r.z = z; m = MIRROR_Z[i]; }
    let rs = at(r.x, r.y, r.z);
    if (isSolid(rs)) {
      let fj = load_src(j * N + idx);
      *force += cvec(j) * (2.0 * fj);
      return fj;
    }
    return load_src(m * N + rs);
  }
  if (outX) { return load_src(j * N + idx); }
  let s = at(sx, sy, sz);
  if (!isSolid(s)) { return load_src(i * N + s); }
  // Bouzidi linear interpolated bounce-back; q is the fluid fraction of the link, from the signed distance field.
  let fj = load_src(j * N + idx);
  let dF = sdf[idx];
  let dS = sdf[s];
  let q = clamp(dF / max(dF - dS, 1e-6), 1e-3, 1.0);
  var fi = fj;
  if (q < 0.5) {
    let nx = x + CX[i];
    let ny = y + CY[i];
    let nz = z + CZ[i];
    if (nx >= 0 && nx < i32(P.W) && ny >= 0 && ny < i32(P.H) && nz >= 0 && nz < i32(P.D)) {
      let n = at(nx, ny, nz);
      if (!isSolid(n)) { fi = 2.0 * q * fj + (1.0 - 2.0 * q) * load_src(j * N + n); }
    }
  } else {
    fi = fj / (2.0 * q) + (2.0 * q - 1.0) / (2.0 * q) * load_src(i * N + idx);
  }
  *force += cvec(j) * (fj + fi);
  return fi;
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = cellIndex(wg, li);
  let N = P.N;
  if (idx >= N) { return; }
  let fl = flags[idx];
  let WH = P.W * P.H;
  let z = idx / WH;
  let rem = idx - z * WH;
  let y = rem / P.W;
  let x = rem - y * P.W;

  ${vars.map((v) => `var ${v}: f32;`).join(' ')}
  var outlet = false;
  if (fl == 0u) {
    // fast path begin
${fastPull}
    // fast path end
  } else {
    if ((fl & FLAG_SOLID) != 0u) { return; }
    if ((fl & FLAG_INLET) != 0u) {
      for (var i = 0u; i < ${Q3}u; i++) { store_dst(i * N + idx, feq(i, 1.0, P.uIn, 0.0, 0.0)); }
      return;
    }
    outlet = (fl & FLAG_OUTLET) != 0u;
    let xi = i32(x);
    let yi = i32(y);
    let zi = i32(z);
    var force = vec3f(0.0);
    f0 = load_src(idx);
${slowPull}
    let sl = slot[idx];
    if (sl != 0u) { cellForce[sl - 1u] = vec4f(force, 0.0); }
  }
${moments}
  if (outlet) {
    // Pressure outlet: equilibrium at rho = 1 with the node's own velocity.
    for (var i = 0u; i < ${Q3}u; i++) { store_dst(i * N + idx, feq(i, 1.0, ux, uy, uz)); }
    return;
  }
${eqTerms}
${eqPairs}

  var omegaBase = 1.0 / P.tau;
  var sOut = 0.0;
  if (P.spongeStart < f32(P.W - 1u)) {
    sOut = smoothstep(P.spongeStart, f32(P.W - 1u), f32(x));
    omegaBase = 1.0 / mix(P.tau, P.tauSponge, sOut);
  }
  let sigma = P.absorb * sOut * sOut;
${smag}
${absorb}
${collide}
}
`;
}
