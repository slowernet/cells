import { CX, CY, W, OPP, MIRROR_Y, FLAG_SOLID, FLAG_INLET, FLAG_OUTLET, BIDX_SHIFT } from '../lattice';

/** A number as a WGSL float literal. */
export const wgslFloat = (x: number) => (Number.isInteger(x) ? `${x}.0` : `${x}`);
const f = wgslFloat;

/** Byte size of the Params uniform; keep in sync with PARAMS_WGSL and Solver.writeParams. */
export const PARAMS_BYTES = 96;

export const PARAMS_WGSL = /* wgsl */ `
struct Params {
  W: u32, H: u32, N: u32, groupsX: u32,
  left: u32, right: u32, bottom: u32, top: u32,
  tau: f32, lambda: f32, smagC2: f32, uIn: f32,
  parabolic: u32, spongeStart: f32, tauSponge: f32, uLid: f32,
  fx: f32, fy: f32, forcing: u32, absorb: f32,
  inletLayer: f32, _p2: u32, _p3: u32, _p4: u32,
};
`;

export const LATTICE_WGSL = /* wgsl */ `
const CX = array<i32, 9>(${CX.join(', ')});
const CY = array<i32, 9>(${CY.join(', ')});
const WT = array<f32, 9>(${W.map((w) => f(w)).join(', ')});
const OPP = array<u32, 9>(${OPP.map((o) => `${o}u`).join(', ')});
const MIRROR_Y = array<u32, 9>(${MIRROR_Y.map((o) => `${o}u`).join(', ')});
const FLAG_SOLID: u32 = ${FLAG_SOLID}u;
const FLAG_INLET: u32 = ${FLAG_INLET}u;
const FLAG_OUTLET: u32 = ${FLAG_OUTLET}u;
const BIDX_SHIFT: u32 = ${BIDX_SHIFT}u;
const X_PERIODIC: u32 = 0u;
const X_WALL: u32 = 1u;
const X_OPEN: u32 = 2u;
const Y_PERIODIC: u32 = 0u;
const Y_NOSLIP: u32 = 1u;
const Y_SLIP: u32 = 2u;
const Y_MOVING: u32 = 3u;

fn inletVelocity(y: u32) -> f32 {
  if (P.parabolic == 0u) { return P.uIn; }
  // Walls sit halfway outside the first and last rows, so the channel height is H.
  let h = f32(P.H);
  let s = f32(y) + 0.5;
  return 4.0 * P.uIn * s * (h - s) / (h * h);
}

fn cellIndex(wg: vec3u, li: u32) -> u32 {
  return (wg.y * P.groupsX + wg.x) * WG + li;
}
`;

