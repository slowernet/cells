import { CX, CY, CZ, W, OPP, MIRROR_Y, MIRROR_Z, Q3, FLAG_SOLID, FLAG_INLET, FLAG_OUTLET } from '../lattice3d';
import { wgslFloat as f } from './common';

export type Precision = 'fp16' | 'fp32';

/** Byte size of the Params3 uniform; keep in sync with PARAMS3_WGSL and Solver3D.writeParams. */
export const PARAMS3_BYTES = 64;

export const PARAMS3_WGSL = /* wgsl */ `
struct Params3 {
  W: u32, H: u32, D: u32, N: u32,
  groupsX: u32, _p0: u32, _p1: u32, _p2: u32,
  tau: f32, lambda: f32, smagC2: f32, uIn: f32,
  spongeStart: f32, tauSponge: f32, absorb: f32, _p3: f32,
};
`;

export const LATTICE3_WGSL = /* wgsl */ `
const CX = array<i32, ${Q3}>(${CX.join(', ')});
const CY = array<i32, ${Q3}>(${CY.join(', ')});
const CZ = array<i32, ${Q3}>(${CZ.join(', ')});
const WT = array<f32, ${Q3}>(${W.map((w) => f(w)).join(', ')});
const OPP = array<u32, ${Q3}>(${OPP.map((o) => `${o}u`).join(', ')});
const MIRROR_Y = array<u32, ${Q3}>(${MIRROR_Y.map((o) => `${o}u`).join(', ')});
const MIRROR_Z = array<u32, ${Q3}>(${MIRROR_Z.map((o) => `${o}u`).join(', ')});
const FLAG_SOLID: u32 = ${FLAG_SOLID}u;
const FLAG_INLET: u32 = ${FLAG_INLET}u;
const FLAG_OUTLET: u32 = ${FLAG_OUTLET}u;

fn cellIndex(wg: vec3u, li: u32) -> u32 {
  return (wg.y * P.groupsX + wg.x) * WG + li;
}

fn at(x: i32, y: i32, z: i32) -> u32 { return u32(x + i32(P.W) * (y + i32(P.H) * z)); }
fn cvec(i: u32) -> vec3f { return vec3f(f32(CX[i]), f32(CY[i]), f32(CZ[i])); }

// Equilibrium minus its rest weight, matching the shifted storage of the distribution buffers.
fn feq(i: u32, rho: f32, ux: f32, uy: f32, uz: f32) -> f32 {
  let cu = 3.0 * (f32(CX[i]) * ux + f32(CY[i]) * uy + f32(CZ[i]) * uz);
  return WT[i] * (rho - 1.0 + rho * (cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz)));
}
`;

/** Module-level directive the precision needs; must come first in the module. */
export const storageDecl = (p: Precision) => (p === 'fp16' ? 'enable f16;\n' : '');

export const fElem = (p: Precision): 'f16' | 'f32' => (p === 'fp16' ? 'f16' : 'f32');

export const bytesPerPopulation = (p: Precision) => (p === 'fp16' ? 2 : 4);

/** load_/store_ accessors for a distribution buffer; FP16 is scaled by 2^15 so shifted populations stay clear of subnormals. */
export function codecWgsl(p: Precision, name: string, access: 'read' | 'read_write'): string {
  const load =
    p === 'fp16'
      ? `fn load_${name}(k: u32) -> f32 { return f32(${name}[k]) * (1.0 / 32768.0); }`
      : `fn load_${name}(k: u32) -> f32 { return ${name}[k]; }`;
  if (access === 'read') return load + '\n';
  const store =
    p === 'fp16'
      ? `fn store_${name}(k: u32, v: f32) { ${name}[k] = f16(clamp(v, -1.99, 1.99) * 32768.0); }`
      : `fn store_${name}(k: u32, v: f32) { ${name}[k] = v; }`;
  return `${load}\n${store}\n`;
}

export const PRESETS3 = { low: [128, 64, 64], medium: [192, 96, 96], high: [256, 128, 128] } as const;

export function fitsLimits(W: number, H: number, D: number, p: Precision, limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }): boolean {
  const bytes = 19 * bytesPerPopulation(p) * W * H * D;
  return bytes <= limits.maxStorageBufferBindingSize && bytes <= limits.maxBufferSize;
}
