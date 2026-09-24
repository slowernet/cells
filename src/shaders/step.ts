import { CX, CY, W, PAIRS, Q } from '../lattice';
import { PARAMS_WGSL, LATTICE_WGSL, wgslFloat as fl } from './common';

/**
 * Fused pull-stream + boundary + TRT collide kernel. One dispatch advances the lattice one step.
 * Interior cells (flags == 0) take a branch-free unrolled path; every boundary rule lives in pullSlow.
 * Buffers store f_i - w_i (rest state removed) so float32 keeps precision for small deviations.
 */
export function stepShader(): string {
  const idx = (i: number) => `${i}u * N + idx`;
  const fastPull = Array.from({ length: Q }, (_, i) => {
    if (i === 0) return `    f0 = src[idx];`;
    const shift = `${CY[i] === 0 ? '' : CY[i] > 0 ? '- P.W' : '+ P.W'}${CX[i] === 0 ? '' : CX[i] > 0 ? ' - 1u' : ' + 1u'}`;
    return `    f${i} = src[${idx(i)} ${shift}];`;
  }).join('\n');

  const slowPull = Array.from({ length: Q - 1 }, (_, k) => {
    const i = k + 1;
    return `    f${i} = pullSlow(${i}u, x, y, idx, fl, &force);`;
  }).join('\n');

  const moments = `
  let drho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8;
  let rho = 1.0 + drho;
  let invRho = 1.0 / rho;
  var ux = (f1 - f3 + f5 - f6 - f7 + f8) * invRho;
  var uy = (f2 - f4 + f5 + f6 - f7 - f8) * invRho;
  if (P.forcing != 0u) {
    ux += 0.5 * P.fx * invRho;
    uy += 0.5 * P.fy * invRho;
  }
  let usq = 1.5 * (ux * ux + uy * uy);`;

  const eqTerms = PAIRS.map(([i]) => {
    const cu = [CX[i] !== 0 ? `${CX[i] > 0 ? '' : '-'}ux` : '', CY[i] !== 0 ? `${CY[i] > 0 ? '+' : '-'}uy` : '']
        .join(' ')
        .trim()
        .replace(/^\+/, '');
    return `  let cu${i} = 3.0 * (${cu});`;
  }).join('\n');

  // Symmetric and antisymmetric equilibrium halves for each TRT pair.
  const eqPairs = PAIRS.map(
    ([a]) => `  let eqS${a} = ${fl(W[a])} * (drho + rho * (0.5 * cu${a} * cu${a} - usq));
  let eqA${a} = ${fl(W[a])} * rho * cu${a};`,
  ).join('\n');

  const smag = `
  var omP = omegaBase;
  if (P.smagC2 > 0.0) {
${PAIRS.map(
  ([a, b]) => `    let n${a} = f${a} - (eqS${a} + eqA${a});
    let n${b} = f${b} - (eqS${a} - eqA${a});`,
).join('\n')}
    let pxx = ${[1, 3, 5, 6, 7, 8].map((i) => `n${i}`).join(' + ')};
    let pyy = ${[2, 4, 5, 6, 7, 8].map((i) => `n${i}`).join(' + ')};
    let pxy = n5 - n6 + n7 - n8;
    let pi = sqrt(pxx * pxx + pyy * pyy + 2.0 * pxy * pxy);
    let t0 = 1.0 / omegaBase;
    let tEff = 0.5 * (t0 + sqrt(t0 * t0 + ${fl(18 * Math.SQRT2)} * P.smagC2 * pi * invRho));
    omP = 1.0 / tEff;
  }
  let omM = 1.0 / (0.5 + P.lambda / (1.0 / omP - 0.5));`;

  const forcing = `
  var s0 = 0.0;
${PAIRS.map(([a]) => `  var sS${a} = 0.0;
  var sA${a} = 0.0;`).join('\n')}
  if (sigma > 0.0) {
    // Absorbing layer: pull the local equilibrium toward the inflow state so sound and vortices leave without reflecting.
    let ut = inletVelocity(y);
    let dq = 1.5 * rho * (ut * ut - ux * ux - uy * uy);
    s0 += sigma * ${fl(W[0])} * -dq;
${PAIRS.map(([a]) => {
  const ct = `${CX[a]}.0 * ut`;
  return `    {
      let cut = 3.0 * ${ct};
      sS${a} += sigma * ${fl(W[a])} * (0.5 * rho * (cut * cut - cu${a} * cu${a}) - dq);
      sA${a} += sigma * ${fl(W[a])} * rho * (cut - cu${a});
    }`;
}).join('\n')}
  }
  if (P.forcing != 0u) {
    let uF = ux * P.fx + uy * P.fy;
    s0 += (1.0 - 0.5 * omP) * ${fl(W[0])} * (-3.0 * uF);
${PAIRS.map(([a]) => {
  const cF = `(${CX[a]}.0 * P.fx + ${CY[a]}.0 * P.fy)`;
  return `    sS${a} += (1.0 - 0.5 * omP) * ${fl(W[a])} * (-3.0 * uF + 9.0 * (cu${a} / 3.0) * ${cF});
    sA${a} += (1.0 - 0.5 * omM) * ${fl(W[a])} * 3.0 * ${cF};`;
}).join('\n')}
  }`;

  const collide = [
    `  dst[idx] = f0 - omP * (f0 - ${fl(W[0])} * (drho - rho * usq)) + s0;`,
    ...PAIRS.map(
      ([a, b]) => `  {
    let sym = omP * (0.5 * (f${a} + f${b}) - eqS${a});
    let asym = omM * (0.5 * (f${a} - f${b}) - eqA${a});
    dst[${idx(a)}] = f${a} - sym - asym + sS${a} + sA${a};
    dst[${idx(b)}] = f${b} - sym + asym + sS${a} - sA${a};
  }`,
    ),
  ].join('\n');

  return /* wgsl */ `
override WG: u32 = 128u;
${PARAMS_WGSL}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read> sdf: array<f32>;
@group(0) @binding(5) var<storage, read_write> cellForce: array<vec2f>;
${LATTICE_WGSL}

fn isSolid(s: u32) -> bool { return (flags[s] & FLAG_SOLID) != 0u; }

// Population arriving at (x, y) along direction i when its upstream node x - c_i is not a plain fluid node.
fn pullSlow(i: u32, x: u32, y: u32, idx: u32, fl: u32, force: ptr<function, vec2f>) -> f32 {
  let N = P.N;
  let W = i32(P.W);
  let H = i32(P.H);
  let j = OPP[i];
  var sx = i32(x) - CX[i];
  var sy = i32(y) - CY[i];
  if ((fl & (1u << i)) == 0u) {
    return src[i * N + u32(sy * W + sx)];
  }
  let ws = wrapPeriodic(vec2i(sx, sy));
  sx = ws.x;
  sy = ws.y;
  let inX = sx >= 0 && sx < W;
  let inY = sy >= 0 && sy < H;

  if (inX && inY) {
    let s = u32(sy * W + sx);
    if (!isSolid(s)) { return src[i * N + s]; }
    // Bouzidi linear interpolated bounce-back; q is the fluid fraction of the link, from the signed distance field.
    let fj = src[j * N + idx];
    let dF = sdf[idx];
    let dS = sdf[s];
    let q = clamp(dF / max(dF - dS, 1e-6), 1e-3, 1.0);
    var fi = fj;
    if (q < 0.5) {
      let np = wrapPeriodic(vec2i(i32(x) + CX[i], i32(y) + CY[i]));
      let nx = np.x;
      let ny = np.y;
      if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
        let n = u32(ny * W + nx);
        if (!isSolid(n)) {
          fi = 2.0 * q * fj + (1.0 - 2.0 * q) * src[j * N + n];
        }
      }
    } else {
      fi = fj / (2.0 * q) + (2.0 * q - 1.0) / (2.0 * q) * src[i * N + idx];
    }
    *force += vec2f(f32(CX[j]), f32(CY[j])) * (fj + fi);
    return fi;
  }

  if (!inY) {
    let mode = select(P.top, P.bottom, sy < 0);
    if (mode == Y_SLIP) {
      let rx = u32(clamp(sx, 0, W - 1));
      return src[MIRROR_Y[i] * N + y * P.W + rx];
    }
    if (mode == Y_MOVING) {
      return src[j * N + idx] + 6.0 * WT[i] * f32(CX[i]) * P.uLid;
    }
    return src[j * N + idx];
  }

  // Walls, and placeholders that the open-boundary fix-up in main overwrites.
  return src[j * N + idx];
}

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = cellIndex(wg, li);
  let N = P.N;
  if (idx >= N) { return; }
  let fl = flags[idx];
  let y = idx / P.W;
  let x = idx - y * P.W;

  var f0: f32; var f1: f32; var f2: f32; var f3: f32; var f4: f32;
  var f5: f32; var f6: f32; var f7: f32; var f8: f32;
  if (fl == 0u) {
${fastPull}
  } else {
    if ((fl & FLAG_SOLID) != 0u) { return; }
    var force = vec2f(0.0);
    f0 = src[idx];
${slowPull}
    let bidx = fl >> BIDX_SHIFT;
    if (bidx != 0u) { cellForce[bidx - 1u] = force; }
    // Zou-He: rebuild the populations entering from outside the domain (stored shifted by w_i).
    if ((fl & FLAG_INLET) != 0u) {
      let u = inletVelocity(y);
      let r = (1.0 + f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u);
      f1 = f3 + ${fl(2 / 3)} * r * u;
      f5 = f7 - 0.5 * (f2 - f4) + ${fl(1 / 6)} * r * u;
      f8 = f6 + 0.5 * (f2 - f4) + ${fl(1 / 6)} * r * u;
    } else if ((fl & FLAG_OUTLET) != 0u) {
      let u = f0 + f2 + f4 + 2.0 * (f1 + f5 + f8);
      f3 = f1 - ${fl(2 / 3)} * u;
      f7 = f5 + 0.5 * (f2 - f4) - ${fl(1 / 6)} * u;
      f6 = f8 - 0.5 * (f2 - f4) - ${fl(1 / 6)} * u;
    }
  }
${moments}
${eqTerms}
${eqPairs}

  var omegaBase = 1.0 / P.tau;
  var sOut = 0.0;
  if (P.spongeStart < f32(P.W - 1u)) {
    sOut = smoothstep(P.spongeStart, f32(P.W - 1u), f32(x));
    omegaBase = 1.0 / mix(P.tau, P.tauSponge, sOut);
  }
  var sIn = 0.0;
  if (P.inletLayer > 0.0) { sIn = 1.0 - smoothstep(0.0, P.inletLayer, f32(x)); }
  let sigma = P.absorb * max(sOut * sOut, sIn * sIn);
${smag}
${forcing}
${collide}
}
`;
}
