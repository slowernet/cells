# 3D wind tunnel, plan 1: headless solver

Spec: docs/dietpowers/2026-09-27-3d-tunnel-spec.md @ 054dcc2
Base: main
Commits: approved

**Goal:** build the D3Q19 TRT solver with FP16 or FP32 storage, its SDF geometry, the `sphereFp16` check and the D3Q19 benchmark rows, all testable without a canvas.

**Architecture:** a separate 3D stack copies the 2D patterns file for file. `lattice3d.ts` imitates `lattice.ts`, `shaders/common3d.ts` imitates `shaders/common.ts`, `shaders/step3d.ts` imitates `shaders/step.ts`, `shaders/aux3d.ts` imitates `shaders/aux.ts`, `solver3d.ts` imitates `solver.ts`, `geometry3d.ts` imitates `geometry.ts`, and `cases3d.ts` imitates `cases.ts`. The WGSL for each precision is generated from TS with the storage codec baked in. The harness learns to skip a case, and the benchmark gains D3Q19 rows. Plan 2 (the 3D page) is out of this plan.

## Global Constraints

- Lattice: D3Q19, with weights 1/3 (rest), 1/18 (6 axis directions) and 1/36 (12 edge diagonals), and c_s² = 1/3.
- Collision:
  - TRT with Λ = 3/16 (`MAGIC_LAMBDA`).
  - Optional Smagorinsky, C_s from 0.10 to 0.17, default 0.16 when enabled.
  - τ is clamped at `TAU_MIN` = 0.51.
- Storage: structure-of-arrays `f[i * N + cell]`, two buffers A and B that swap every step, pull streaming. Buffers hold f_i − w_i and density is `1 + Σ f`.
- Equilibria: every equilibrium written to a buffer (init, inlet, outlet, collision) is computed directly in shifted form, `w_i · (ρ − 1 + ρ · (cu + 0.5·cu² − 1.5·u²))`, as `feq` in `src/shaders/aux.ts` does. It is never formed as f_eq − w by subtraction.
- Precision: `'fp16' | 'fp32'`, chosen when the solver is created and baked into the generated WGSL.
  - FP32: `array<f32>`.
  - FP16: `enable f16;` and `array<f16>`. A store writes `f16(clamp(v, -1.99, 1.99) * 32768.0)` and a load reads `f32(x) * (1.0 / 32768.0)`. Arithmetic stays in FP32.
- Performance:
  - One dispatch per step, with all steps for a frame in one compute pass.
  - No per-step readbacks.
  - The interior fast path loads only the 19 pulled populations and the flag word.
- Throughput floor on the reference M5, fixed now: `npm run bench` reports at least 1,200 MLUPS for `medium` FP16 and at least 600 for `medium` FP32 on the empty-domain rows, which is 75% of FluidX3D's native M5 figures. The page at defaults sustains at least 480 lattice steps per second at any display refresh rate.
- A step shader binds at most 8 storage buffers.
- Grid presets (W×H×D, with x as the flow direction):
  - `low` 128×64×64;
  - `medium` 192×96×96 (the default);
  - `high` 256×128×128.
- A preset is offered only when one distribution buffer, `19 · b · W·H·D` bytes with b = 2 for FP16 and 4 for FP32, is at most both `device.limits.maxStorageBufferBindingSize` and `device.limits.maxBufferSize`.
- Flow:
  - Target inflow speed u_target = 0.1, ramped from 0 over 3000 steps with smoothstep t²(3 − 2t), as `app.ts` and `runRamped` do. The ramped value goes only to `Params3.u_in`. Force coefficients, Re → τ and tracer respawn use u_target, never the ramped value.
  - The outlet sponge covers the last 15% of x, with `absorb` 0.02.
  - No inlet layer.
- Obstacles sit at x = W/4, centered in y and z.
- Adaptive steps per frame fill 0.85 of the display frame interval, capped at 400, as `app.ts` does. Forces are sampled every 4 steps.
- Tracers: 16,384 particles.
- FP16 check (`sphereFp16`) acceptance, fixed now, before the first run: |C_D,FP16 − C_D,FP32| / C_D,FP32 ≤ 1%.
- No runtime dependencies. TypeScript and Vite, as today.

## References

- **Existing code copied or reused.** The line numbers are as of the spec commit:
  - `src/shaders/step.ts`: the unrolled generator. It has a fast path for flags == 0 and `pullSlow` for everything else. Bouzidi linear interpolation uses q = clamp(dF / max(dF − dS, 1e-6), 1e-3, 1): for q < 0.5, `fi = 2q·fj + (1 − 2q)·src[j][n]` with n = x + c_i when n is in range and fluid (otherwise `fi = fj`); for q ≥ 0.5, `fi = fj/(2q) + (2q − 1)/(2q)·src[i][idx]`. The force is `force += c_j·(fj + fi)` with j = OPP[i]. It also holds the TRT collide with `omM = 1/(0.5 + λ/(1/omP − 0.5))`, the Smagorinsky closed form `tEff = 0.5·(t0 + sqrt(t0² + 18√2·smagC2·|Π|/ρ))`, the sponge `smoothstep(spongeStart, W−1, x)` mixing τ toward `tauSponge`, and the absorbing source terms with `sigma = absorb·sOut²` (lines 60–229).
  - `src/shaders/aux.ts`: `feq` in shifted form (`WT[i]·(ρ − 1 + ρ·(cu + 0.5cu² − 1.5u²))`); `flagsShader`, which refills a formerly solid cell from the average of its non-solid neighbours in `mac`; `initShader`; `macroShader`; `reduceShader` (a workgroup of 256, a history ring of `HISTORY_LEN` = 8192, a counter at `counter[0]`).
  - `src/solver.ts`: buffer creation, `buildPipelines` with `layout: 'auto'` and bind groups by binding number, the parity swap, `encodeSteps`/`encodeMacro`/`run`, `readForces` with the `forceGeneration` guard against a reset overtaking a readback, `rebuildFlags`, which encodes macro before flags so refill reads current state, and `destroy`.
  - `src/solver.test.ts`: the fake-device pattern for testing `Solver` in vitest without a GPU.
  - `src/geometry.ts`: `FAR = 1e6`, `stamp` with `MARGIN = 3`, `addBox`'s rotated-box distance, and `nacaPolygon(x0, y0, chord, thickness, alpha)`.
  - `src/cases.ts`: `Metric`, `CaseResult`, `metric(name, value, ref, accept)`, `runRamped`'s smoothstep ramp and 10 s MLUPS heartbeat, and `runCase` (lines 437–442).
  - `src/validate.ts`: `formatResult` and the `window.caseNames`/`runCase` hooks.
  - `src/bench.ts`: `measure`, the row format, and `BYTES_PER_CELL`.
  - `tests/validate.spec.ts`, `tests/progress.ts`: the case list and the progress log.
  - `src/units.ts`: `deriveTau(re, uLat, length)` gives `{ tau, clamped, effectiveRe }`.
- **FP16S** (Lehmann et al., arXiv:2112.08926): shifted populations obey |f − w| ≲ 0.34ρ. Scaling by 2^15 fills the range ±2 and keeps values clear of subnormals, and f − w shifting is crucial for 16-bit storage.
- **WGSL**: `enable f16;` must be the first directive of the module and needs the device feature `shader-f16`. `array<f16>` is allowed in storage buffers at 2-byte alignment (WGSL §14.4.1). The f32 → f16 rounding mode is unspecified (§15.7). Atomics are u32/i32 only.
- **Pipelines**: pipelines use `layout: 'auto'`, so bind groups must use the shader's own binding numbers and only bindings the shader uses (AGENTS.md). `macro` is a reserved word in WGSL, so the buffer variable is named `mac`.
- **Published sphere drag at Re 100**: C_D ≈ 1.09 (Johnson & Patel 1999). It is unverified and reported only.

## Conventions for every task

- **Cell index**: `idx = x + W·(y + H·z)`. **Workgroup dispatch**: `cellIndex(wg, li) = (wg.y·groupsX + wg.x)·WG + li`, with `groups = [min(total, maxComputeWorkgroupsPerDimension), ceil(total / gx)]`, as in `solver.ts`.
- **D3Q19 direction order**, with OPP[i] = i + 1 for odd i and i − 1 for even i:

  | i | c | i | c | i | c |
  |---|---|---|---|---|---|
  | 0 | (0,0,0) | 7 | (1,1,0) | 13 | (1,−1,0) |
  | 1 | (1,0,0) | 8 | (−1,−1,0) | 14 | (−1,1,0) |
  | 2 | (−1,0,0) | 9 | (1,0,1) | 15 | (1,0,−1) |
  | 3 | (0,1,0) | 10 | (−1,0,−1) | 16 | (−1,0,1) |
  | 4 | (0,−1,0) | 11 | (0,1,1) | 17 | (0,1,−1) |
  | 5 | (0,0,1) | 12 | (0,−1,−1) | 18 | (0,−1,1) |
  | 6 | (0,0,−1) | | | | |

- **Macro layout**: `vec4f(ux, uy, uz, ρ)` per cell, with ρ = 0 exactly for a solid cell.
- **Unit tests** run with `npm test`. **GPU checks** run with `npm run test:gpu` or `npm run bench`, which build to `dist-test/`; follow them with `tail -f test-results/progress.log`. `npm run typecheck` must stay clean after every task.

## Tasks

### - [x] Task 1: D3Q19 lattice constants

- **Files**: create `src/lattice3d.ts` and `src/lattice3d.test.ts`.
- **Interfaces produced**:
  - `CX`, `CY`, `CZ`, `W`: readonly arrays of length 19 in the order above.
  - `OPP`, `MIRROR_Y`, `MIRROR_Z`: readonly arrays of length 19.
  - `PAIRS: ReadonlyArray<readonly [number, number]>`, which is `[1,2],[3,4],…,[17,18]`.
  - `Q3 = 19`.
  - `FLAG_SOLID = 1 << 19`, `FLAG_INLET = 1 << 20`, `FLAG_OUTLET = 1 << 21`.
  - `equilibrium3(rho, ux, uy, uz, out?: Float32Array | number[]): Float32Array | number[]`, which returns the raw f_eq.
- **Context**: imitate `src/lattice.ts`. The weights are 1/3 for rest, 1/18 for i = 1…6 and 1/36 for i = 7…18. `MIRROR_Y[i]` is the direction whose c equals c_i with c_y negated; `MIRROR_Z[i]` negates c_z.
- **Behavior**: constants only, plus `equilibrium3` written in the same form as `equilibrium` in `lattice.ts`, extended to three components.
- **Tests** (`src/lattice3d.test.ts`):
  - `weights and isotropy`: Σw = 1, Σw·c = 0 per axis, and Σw·c_α·c_β = δ_αβ/3 for all nine α, β pairs, each to 1e-12. A wrong weight or a wrong velocity row fails it.
  - `OPP and mirrors`:
    - `OPP` is an involution with c_OPP[i] = −c_i;
    - `MIRROR_Y` is an involution that negates only c_y, and likewise `MIRROR_Z` for c_z;
    - every `PAIRS` entry is `[i, OPP[i]]` with i < OPP[i], and together the pairs cover 1…18 exactly once.

    A swapped table entry fails it.
  - `equilibrium3 moments`: for (ρ, u) in {(1, 0, 0, 0), (1.02, 0.05, −0.03, 0.01), (0.97, −0.1, 0.02, 0.04)}, Σf = ρ and Σf·c = ρu, both to 1e-6. A wrong cu or usq term fails it.
  - `flag bits`: bits 1…18 are free, `FLAG_SOLID`, `FLAG_INLET` and `FLAG_OUTLET` don't overlap them or each other, and all bits sit below 2^32. An overlapping constant fails it.
- **Command**: `npm test -- lattice3d`.

### - [x] Task 2: 3D SDF builders and reference areas

- **Files**: create `src/geometry3d.ts` and `src/geometry3d.test.ts`.
- **Interfaces produced**:
  - `emptySdf3(W, H, D): Float32Array`, filled with `FAR`.
  - `addSphere(sdf, W, H, D, cx, cy, cz, r)`.
  - `addBox3(sdf, W, H, D, cx, cy, cz, hx, hy, hz, angleZ)`.
  - `addCylinderZ(sdf, W, H, D, cx, cy, r)`.
  - `addWing(sdf, W, H, D, cx, cy, cz, chord, span, angleZ)`.
  - `type Obstacle3 = 'sphere' | 'cube' | 'cylinder' | 'wing' | 'none'`.
  - `referenceArea(obstacle: Obstacle3, size: number, D: number): number`.
- **Context**: imitate `src/geometry.ts`. Import `FAR` and `nacaPolygon` from it without changing it. The SDF is in cells and negative inside, with node (x, y, z) at integer coordinates and index `x + W·(y + H·z)`. Each builder stamps true distance within its bounding box grown by `MARGIN = 3` cells and clipped to the grid, and unions with `min`.
- **Behavior**:
  - `addSphere` stamps the distance to the center minus r.
  - `addBox3` rotates about the z axis through (cx, cy) by `angleZ` radians counter-clockwise, then stamps the exact box distance with half-extents (hx, hy, hz), extending `addBox` to 3D.
  - `addCylinderZ` stamps the disc distance in x-y for every z from 0 to D − 1.
  - `addWing` builds `nacaPolygon(0, 0, chord, 0.12, angleZ)`, then translates every point by (cx, cy) − chord/2·(cos α, −sin α), with α = angleZ. `nacaPolygon` rotates by −α (`src/geometry.ts:54-56`), so chord/2·(cos α, −sin α) is where it puts the mid-chord, and the translation lands the mid-chord at (cx, cy). The wing then pivots about its mid-chord, as the cube pivots about its centre, and stays centred at any angle. In x-y it takes the signed polygon distance d2, as `addPolygon` computes it. In z it takes dz = |z − cz| − span/2. The stamped value is `max(d2, dz)` when either is negative, and `hypot(max(d2, 0), max(dz, 0))` otherwise, which gives flat tips.
  - `referenceArea(obstacle, size, D)`, with `size` in cells, returns:
    - sphere: π·size²/4;
    - cube: size²;
    - cylinder: size·D;
    - wing: size·(0.6·D);
    - none: 0.
  - The app's wing span is 0.6·D. Callers pass `span = 0.6·D` to `addWing`.
- **Tests** (`src/geometry3d.test.ts`), on a 32×24×20 grid:
  - `sphere sign and distance`: the center reads −r; a node 2 cells outside the surface along +x reads 2 ± 1e-6; a node beyond the margin reads `FAR`. Dropping the "− r" fails it.
  - `box rotation`: a box with hx = 8, hy = hz = 1 at angleZ = π/6 has the point 6 cells from the centre along the +30° ray inside, and the mirrored point on the −30° ray outside. A clockwise rotation fails it.
  - `cylinder spans z`: the axis node reads −r at z = 0 and at z = D − 1.
  - `wing pivots about mid-chord`: at angleZ = 20°, chord 19.2, and with d = (cos 20°, −sin 20°), sampling the SDF by trilinear interpolation at z = cz:
    - (cx, cy), (cx, cy) + (chord/2 − 1.5)·d and (cx, cy) − (chord/2 − 1.5)·d read negative;
    - (cx, cy) + (chord/2 + 1.5)·d and (cx, cy) − (chord/2 + 1.5)·d read positive.

    A leading-edge pivot (about 3.3 cells off) or the wrong sine sign fails it.
  - `wing tips are flat`: at cz ± (span/2 + 1), a node at the section's thickest point reads 1 ± 0.05. Inside the span, the same x-y point is negative.
  - `referenceArea`: the five values above for size = 16 and D = 96, for example sphere 201.06 ± 0.01. Passing a fraction of H instead of cells is a caller bug that this test doesn't catch; `sphereFp16` passes cells.
- **Command**: `npm test -- geometry3d`.
- Departure: `wing pivots about mid-chord` checks the interior nodes at z = cz instead of trilinear samples 1.5 cells inside each end. A NACA0012 is about 0.2 cells thick 1.5 cells from its trailing edge, so trilinear sampling reads a correct build as outside. The test now requires the centre sample to be negative, the interior nodes' extent along d to lie within [−chord/2, −chord/2 + 1.5] and [chord/2 − 3, chord/2], and every interior node to lie within 0.06·chord + 1 of the chord line. A mutation check confirmed that a wrong sine sign and a leading-edge pivot both fail it.

### - [x] Task 3: Params3, the storage codec and fitsLimits

- **Files**: create `src/shaders/common3d.ts` and `src/shaders/common3d.test.ts`.
- **Interfaces produced**:
  - `type Precision = 'fp16' | 'fp32'`, exported from `common3d.ts`.
  - `PARAMS3_BYTES = 64` and `PARAMS3_WGSL`. The struct `Params3` is `{ W: u32, H: u32, D: u32, N: u32, groupsX: u32, _p0: u32, _p1: u32, _p2: u32, tau: f32, lambda: f32, smagC2: f32, uIn: f32, spongeStart: f32, tauSponge: f32, absorb: f32, _p3: f32 }`.
  - `LATTICE3_WGSL`: const arrays `CX`, `CY`, `CZ` (i32), `WT` (f32), `OPP`, `MIRROR_Y` and `MIRROR_Z` (u32), the flag constants, and `fn cellIndex(wg: vec3u, li: u32) -> u32` as in `common.ts`.
  - `storageDecl(precision): string`, which returns `'enable f16;\n'` for fp16 and `''` for fp32. The caller puts it first in the module.
  - `fElem(precision): 'f16' | 'f32'`.
  - `codecWgsl(precision, bufferName, access: 'read' | 'read_write'): string`, which defines `fn load_<bufferName>(k: u32) -> f32` and, only for `'read_write'`, `fn store_<bufferName>(k: u32, v: f32)`, over the buffer's flat index k = i·N + cell. WGSL rejects any assignment to a `var<storage, read>`, even in a function that is never called, so a read-only buffer must get no `store_`.
    - fp32: plain access.
    - fp16: `store` writes `f16(clamp(v, -1.99, 1.99) * 32768.0)` and `load` returns `f32(x) * (1.0 / 32768.0)`.
  - `bytesPerPopulation(precision)`, which returns 2 or 4.
  - `fitsLimits(W, H, D, precision, limits: { maxStorageBufferBindingSize: number; maxBufferSize: number }): boolean`, exported from `common3d.ts`, which reads no GPU globals, so unit tests import it directly. Task 6's `solver3d.ts` re-exports it, as the spec names it there. It is true when `19 · bytesPerPopulation · W·H·D` is at most both limits.
  - `PRESETS3 = { low: [128, 64, 64], medium: [192, 96, 96], high: [256, 128, 128] } as const`, exported from `common3d.ts` and re-exported by `solver3d.ts`.
- **Context**: imitate `src/shaders/common.ts`. There is no `inletVelocity` function, because the 3D inflow is uniform (`P.uIn`, 0, 0).
- **Tests** (`src/shaders/common3d.test.ts`):
  - `Params3 layout`: parse the field list out of `PARAMS3_WGSL` and check it has 16 four-byte fields, 16 × 4 = `PARAMS3_BYTES`, and `tau` at field 8 and `uIn` at field 11. Moving a field without updating the constant fails it.
  - `codec access`: `codecWgsl(p, 'x', 'read')` defines `load_x` and no `store_x` in both precisions, and `'read_write'` defines both. Emitting `store_` for a read buffer fails it.
  - `codec`: the fp16 codec text contains `32768.0`, `1.0 / 32768.0` and `clamp(v, -1.99, 1.99)`, and `storageDecl('fp16')` starts with `enable f16;`. The fp32 codec contains none of these, and `storageDecl('fp32')` is empty. Dropping the scale fails it.
  - `fitsLimits`: against 128 MiB:
    - low fits in both precisions;
    - medium fits in fp16 (67,239,936 B) and not in fp32 (134,479,872 B);
    - high fits in neither.

    Against 4 GiB, every preset fits in both precisions. An off-by-precision byte count fails it.
- **Command**: `npm test -- common3d`.

### - [ ] Task 4: The step kernel generator

- **Files**: create `src/shaders/step3d.ts` and `src/shaders/step3d.test.ts`.
- **Interfaces produced**:
  - `step3dShader(precision: Precision): string`, a WGSL module with entry point `main` and `override WG: u32 = 128u`.
  - Bindings:
    - 0: `P: Params3` (uniform)
    - 1: `src: array<fElem>` (read)
    - 2: `dst: array<fElem>` (read_write)
    - 3: `flags: array<u32>` (read)
    - 4: `sdf: array<f32>` (read)
    - 5: `slot: array<u32>` (read)
    - 6: `cellForce: array<vec4f>` (read_write)
- **Context**: imitate `src/shaders/step.ts`, generating unrolled code from `lattice3d.ts`. Reuse the Bouzidi, TRT, Smagorinsky, sponge and absorbing-layer math listed under References, extended to 19 directions and 9 pairs. In 3D, |Π| = sqrt(Π_xx² + Π_yy² + Π_zz² + 2(Π_xy² + Π_xz² + Π_yz²)), where Π_αβ = Σ_i c_iα·c_iβ·n_i over the non-equilibrium parts n_i. The absorbing layer relaxes toward the equilibrium at the local ρ and the inflow velocity (`P.uIn`, 0, 0). This kernel never gets `sIn`, because there is no inlet layer.
- **Behavior**:
  - `main`:
    1. Computes idx and returns if idx ≥ N.
    2. Reads `flags[idx]` once and derives x, y, z.
    3. **Fast path** (flags == 0), between the comments `// fast path begin` and `// fast path end`: exactly 19 `load_src(...)` calls, one per direction, from the neighbour idx − c_i with index offsets folded in as `± 1u`, `± P.W` and `± P.W * P.H`, and nothing else loaded.
    4. **Slow path**:
       - A solid cell returns at once.
       - An inlet cell (`FLAG_INLET`) writes `store_dst` of `feq(i, 1.0, P.uIn, 0, 0)` for all 19 directions and returns, with no collision.
       - An outlet cell (`FLAG_OUTLET`) pulls every direction through `pullSlow3`, computes ρ and u from the pulled values, writes `feq(i, 1.0, u)` for all 19 directions and returns, with no collision.
       - Any other cell pulls every direction through `pullSlow3`, accumulates the momentum-exchange force, writes `cellForce[slot[idx] − 1] = vec4f(force, 0.0)` when `slot[idx] != 0`, then collides.
    5. **Collide**: TRT over the 9 pairs, with Smagorinsky when `P.smagC2 > 0`, and the sponge and absorbing layer when `P.spongeStart < f32(P.W − 1)`, written through `store_dst`.
  - `feq(i, rho, ux, uy, uz)` returns the shifted equilibrium in the form `WT[i]·(ρ − 1 + ρ·(cu + 0.5cu² − 1.5u²))`, with cu = 3·(c_i·u), as the spec's Equilibria constraint requires.
  - `pullSlow3(i, x, y, z, idx, fl, force)`, with source s = (x − cx, y − cy, z − cz) and j = OPP[i]:
    1. If flag bit i is clear, return `load_src(i·N + index(s))`.
    2. outX = s.x ∉ [0, W), outY = s.y ∉ [0, H), outZ = s.z ∉ [0, D).
    3. If outX and this is an outlet cell: return `load_src(i·N + index(W − 2, y, z))`.
    4. If outY and outZ (edge links, c_x = 0): return `load_src(j·N + idx)`.
    5. If exactly one of outY or outZ: set r = s with the out-of-range component replaced by the node's own y (or z), and the mirrored direction m = `MIRROR_Y[i]` (or `MIRROR_Z[i]`). If r is solid: fj = `load_src(j·N + idx)`, `force += c_j·(fj + fj)`, return fj. Otherwise return `load_src(m·N + index(r))`.
    6. If s is in range and solid: Bouzidi with the SDF, as in 2D, with neighbour n = (x + cx, y + cy, z + cz) used only when in range and fluid, plus `force += c_j·(fj + fi)`. Return fi.
    7. Otherwise return `load_src(i·N + index(s))`.
- **Tests** (`src/shaders/step3d.test.ts`):
  - `fast path loads only its pulls`: in both precisions, the text between the fast-path comments contains exactly 19 `load_src(` calls and no other `src`, `sdf`, `slot` or `flags` access, and `main` contains exactly one `flags[` read. Adding a load to the fast path fails it.
  - `precision header`: the fp16 module starts with `enable f16;` and declares `array<f16>` for src and dst; the fp32 module has no `enable` and uses `array<f32>`. A missing header fails it.
  - `no store to read-only src`: neither module contains `store_src`. Generating the codec for `src` with `'read_write'` fails it.
  - `bindings`: bindings 0 through 6 each appear exactly once and binding 7 does not, and at most 8 `var<storage` declarations exist. A seventh storage buffer or a duplicate binding fails it.
  - `equilibria are shifted`: every `feq` call site goes through the single shifted-form function, and the text contains no `- WT[` subtraction applied to an equilibrium. Computing f_eq − w by subtraction fails it.
- **Command**: `npm test -- step3d`. The GPU correctness check is `sphereFp16` in Task 8.

### - [ ] Task 5: The auxiliary kernels

- **Files**: create `src/shaders/aux3d.ts` and `src/shaders/aux3d.test.ts`.
- **Interfaces produced**: all four generators take a `Precision`, bind `P: Params3` at binding 0, and use entry point `main` and `override WG`.
  - `flags3dShader(precision)`. Bindings:
    - 1: `sdf` (read)
    - 2: `flags` (read_write)
    - 3: `slot` (read_write)
    - 4: `fA` (read_write)
    - 5: `fB` (read_write)
    - 6: `mac` (read)
    - 7: `counter: array<atomic<u32>>`
  - `init3dShader(precision)`. Bindings 1: `fA`, 2: `fB`, 3: `flags` (read).
  - `macro3dShader(precision)`. Bindings 1: `f` (read), 2: `flags` (read), 3: `mac` (read_write).
  - `reduce3dShader()`. Bindings 0: `cellForce` (read), 1: `counter` (read), 2: `history: array<vec4f>`, 3: `state`. It has no params.
  - `HISTORY3_LEN = 8192`.
- **Context**: imitate `src/shaders/aux.ts`, and reuse the shifted `feq` form from Task 4. A cell is solid when `sdf < 0`, except on the inlet plane (x = 0) and the outlet plane (x = W − 1), which are always fluid.
- **Behavior**:
  - `flags3dShader`:
    - For a solid cell: `flags = FLAG_SOLID`, `slot = 0`.
    - For a fluid cell: the INLET or OUTLET bit for its plane, plus bit i for each direction whose source s is out of range or solid.
    - It sets `touchesBody` when any s is solid, or when the side-face rule of Task 4, step 5, would read a solid r. A `touchesBody` cell gets `slot = atomicAdd(&counter[0], 1u) + 1u`; any other cell gets `slot = 0`.
    - **Refill**: a cell whose old flags had `FLAG_SOLID` and that is now fluid gets the equilibrium, in both fA and fB, at the average ρ and u of its in-range neighbours whose `mac.w != 0`. With none, it uses ρ = 1 and u = (`P.uIn`, 0, 0).
  - `init3dShader`: every fluid cell gets `feq(i, 1.0, P.uIn, 0, 0)` in fA and fB. Every solid cell gets 0.
  - `macro3dShader`: solid cells get `vec4f(0.0)`. Fluid cells get `vec4f(jx/ρ, jy/ρ, jz/ρ, ρ)` with ρ = 1 + Σf.
  - `reduce3dShader`: `reduceShader` with vec4f in place of vec2f.
- **Tests** (`src/shaders/aux3d.test.ts`):
  - `macro marks solids with rho 0`: the solid branch writes `vec4f(0.0)`, and the fluid branch writes ρ into `.w`. Writing ρ into `.x`, as 2D does, fails it.
  - `precision header` for each of `flags3dShader`, `init3dShader` and `macro3dShader`, as in Task 4.
  - `no store to read-only f`: `macro3dShader` contains no `store_f`, in both precisions.
  - `flags binds at most 8 storage buffers`: 7 storage buffers plus the uniform.
- **Command**: `npm test -- aux3d`. The GPU behavior is checked by `sphereFp16` in Task 8.

### - [ ] Task 6: Solver3D and the f16 device option

- **Files**: create `src/solver3d.ts` and `src/solver3d.test.ts`; modify `src/gpu.ts`.
- **Interfaces produced**:
  - `Solver3DConfig`: `{ width, height, depth, precision: Precision, tau, lambda?, smagorinsky?, uIn?, spongeFraction?, spongeTau?, absorb?, workgroupSize?, forceEvery? }`. The defaults are λ = `MAGIC_LAMBDA`, smagorinsky 0, uIn 0, spongeFraction 0, spongeTau 1, absorb 0, workgroupSize 128 and forceEvery 1.
  - `ForceSample3 { step: number; fx: number; fy: number; fz: number }`.
  - `class Solver3D`:
    - Fields: `W`, `H`, `D`, `N`, `cfg`, `step`, and the buffers `macro`, `sdf`, `flags` and `params`.
    - Methods:
      - `static create(device, cfg): Promise<Solver3D>`
      - `setSdf(sdf: Float32Array)`
      - `initField()`
      - `setTau(tau)`
      - `setInlet(u)`
      - `encodeSteps(pass, n)`
      - `encodeMacro(pass)`
      - `run(n, chunk = 500): Promise<void>`
      - `readMacro(): Promise<Float32Array>`
      - `readForces(): Promise<ForceSample3[]>`
      - `resetForces()`
      - `destroy()`
  - `initGpu(opts?: { f16?: boolean }): Promise<Gpu & { f16: boolean }>`.
  - `solver3d.ts` re-exports `fitsLimits`, `PRESETS3` and `Precision` from `./shaders/common3d`.
- **Context**: imitate `src/solver.ts` in full, including the `forceGeneration` guard and `rebuildFlags` encoding macro before flags. Imitate `src/solver.test.ts` for the fake device.
- **Behavior**:
  - `create`:
    1. For `precision === 'fp16'` without `device.features.has('shader-f16')`, it throws `Error('shader-f16 is not available on this device')` before creating any buffer.
    2. It pushes the error scopes `'out-of-memory'` and `'validation'`, then allocates:
       - params (`PARAMS3_BYTES`);
       - f A and B (`19 · bytesPerPopulation · N` each);
       - flags, slot and sdf (4·N each, with sdf filled with `FAR`);
       - macro (16·N);
       - cellForce (16·N);
       - counter (4 B);
       - history (`HISTORY3_LEN` · 16);
       - histState (4 B).
    3. It pops both scopes. On an error from either, it destroys every buffer it allocated and throws `Error('Solver3D allocation failed: <message>')`.
    4. Only then does it build the pipelines and run `rebuildFlags`. If either throws, it destroys every buffer it allocated and rethrows the original error unchanged, so only allocation errors carry the `Solver3D allocation failed` prefix.
  - `writeParams` fills `Params3`:
    - `spongeStart = spongeFraction > 0 ? (W − 1)·(1 − spongeFraction) : 1e9`;
    - `tauSponge = max(spongeTau, tau)`;
    - `smagC2 = smagorinsky²`.
  - `setSdf` writes the SDF and runs `rebuildFlags`: macro, then clear the counter, then flags, in one submit.
  - `initField` dispatches init, then sets parity = 0 and step = 0 and calls `resetForces()`.
  - `setTau` and `setInlet` update the config and rewrite the params.
  - `encodeSteps` works as in 2D: one dispatch per step with the parity swap, plus a reduce dispatch every `forceEvery` steps.
  - `readForces` works as in 2D, over vec4 history entries.
  - `destroy` destroys every buffer.
  - `initGpu({ f16: true })` adds `'shader-f16'` to `requiredFeatures` when `adapter.features.has('shader-f16')` and returns `f16` accordingly. `initGpu()` with no argument requests exactly what it requests today.
- **Tests** (`src/solver3d.test.ts`), using a fake device extended with `features`, `pushErrorScope`/`popErrorScope`, `createShaderModule`, `createComputePipelineAsync` (whose `getBindGroupLayout` returns a stub), `createBindGroup` and `beginComputePass` stubs:
  - `fp16 without the feature throws before allocating`: rejects with the message above and creates zero buffers. Checking the feature after allocation fails it.
  - `allocation error destroys everything`: `popErrorScope` returns an error for `'out-of-memory'`; `create` rejects with `Solver3D allocation failed`, and every buffer it created had `destroy()` called. Leaking one buffer fails it.
  - `pipeline error destroys everything and keeps its message`: `createComputePipelineAsync` rejects with `Error('bad wgsl')`; `create` rejects with exactly that message (no allocation prefix), and every created buffer had `destroy()` called. A leak, or wrapping the message as an allocation failure, fails it.
  - `params layout`: after `create` with W, H, D = 8, 6, 4, tau 0.6, uIn 0, spongeFraction 0.15, the params buffer holds W, H, D, N as u32 at offsets 0–12, tau at f32 index 8, and spongeStart = 7·0.85 at f32 index 12. After `setInlet(0.05)`, index 11 reads 0.05. A field written to the wrong offset fails it.
  - `buffer sizes`: fp16 f buffers are 19·2·N bytes and fp32 ones are 19·4·N, and cellForce is 16·N. Sizing cellForce from a slot estimate fails it.
  - `readForces discards a read overtaken by resetForces`: ported from `src/solver.test.ts` for vec4 samples.
- **Command**: `npm test -- solver3d`.

### - [ ] Task 7: Harness support for skipped cases

- **Files**: modify `src/cases.ts`, `src/validate.ts` and `tests/validate.spec.ts`; create `src/cases.test.ts`.
- **Interfaces produced**:
  - `CaseResult.skipped?: string`.
  - `runCase(device, key, log)` resolves `key` in `{ ...CASES, ...CASES3D }` and returns `pass: !r.skipped && r.metrics.every((m) => m.pass)`.
  - `CASES3D` is imported from `./cases3d`. Task 8 creates that module; for this task, create `src/cases3d.ts` exporting `CASES3D: Record<string, Case> = {}`, and export the `Case` type and `function metric` from `cases.ts`.
- **Context**: `src/cases.ts:437-442` and `src/validate.ts`. `cases3d.ts` imports `metric` and the types from `cases.ts`, and `cases.ts` imports `CASES3D` from `cases3d.ts`. That cycle is safe because neither module reads the other's bindings while it evaluates: `CASES3D` is only read inside `runCase`, and `metric` only inside a case.
- **Behavior**:
  - `validate.ts`:
    - It calls `initGpu({ f16: true })`.
    - `window.caseNames = [...Object.keys(CASES), ...Object.keys(CASES3D)]`, and the "all" button iterates `caseNames`.
    - `formatResult` prints `SKIP  <name>: <reason>` on its first line when `skipped` is set, followed by any notes.
  - `tests/validate.spec.ts`: the default list gains `'sphereFp16'`. After `runCase`, a result with `skipped` calls `test.skip(true, r.skipped)` before the pass assertion.
- **Tests** (`src/cases.test.ts`). `cases.ts` imports `solver.ts`, which reads `GPUBufferUsage` at load, so the test first stubs `globalThis.GPUBufferUsage` and `GPUMapMode` and then loads `./cases` and `./cases3d` with dynamic `import()`, exactly as `src/solver.test.ts:2-3` does:
  - `a skipped case never passes`: register a temporary key in `CASES3D` whose case returns `{ name: 'x', metrics: [], skipped: 'reason', steps: 0, cells: 0, notes: [] }`. `runCase` then returns `pass === false` and `skipped === 'reason'`. The old `[].every()` rule fails it.
  - `a case with passing metrics and no skip passes`: guards against inverting the rule.
- **Command**: `npm test -- cases`. The GPU harness is exercised in Task 8.

### - [ ] Task 8: The sphereFp16 case

- **Files**: modify `src/cases3d.ts`.
- **Interfaces produced**: `CASES3D.sphereFp16: Case`.
- **Context**:
  - Global Constraints: the FP16 acceptance range, the smoothstep ramp, the sponge and the presets.
  - The runRamped heartbeat in `src/cases.ts`.
  - Task 2's `referenceArea` and `addSphere`, and Task 6's `Solver3D`.
- **Behavior**:
  1. W, H, D = `PRESETS3.medium`. The sphere has diameter 16 at (W/4, H/2, D/2). u_target = 0.1, and τ = `deriveTau(100, 0.1, 16).tau` (0.548). The config sets spongeFraction 0.15, spongeTau 1, absorb 0.02 and forceEvery 4.
  2. If `!device.features.has('shader-f16')`, return `skipped: 'shader-f16 not available on this device'` without running.
  3. For precision in [`'fp32'`, `'fp16'`]:
     1. Create the solver. If `create` rejects with a message starting `Solver3D allocation failed`, destroy the other precision's solver if one is live and return `skipped: <error message>`. Any other error propagates, so the case fails; a shader or pipeline bug must never read as a skip.
     2. Call `setSdf`, then `initField`.
     3. Advance in chunks of 400 steps, calling `setInlet(0.1 · smoothstep(min(1, step/3000)))` before each chunk and collecting `readForces()`. Log `step <n>, <MLUPS> MLUPS` at most every 10 s, as `runRamped` does.
     4. Every 2,000 steps, which is one block, compute the block's C_D = mean(fx over the block) · 2/(0.1² · π·16²/4) and log `check k/15 C_D=<value>` through the case's log.
     5. Stop after block k when k ≥ 5 and |C_D,k − C_D,k−1| / |C_D,k−1| < 1e-4, or when k = 15. If the stop came at k = 15 without convergence, add the note `<precision> reached 15 blocks without converging`.
     6. Record the last block's C_D and destroy the solver.
  4. **Metric**: `metric('C_D relative difference FP16 vs FP32', |C16 − C32| / C32, [0, 0], [0, 0.01])`. A NaN value fails the range test.
  5. **Notes**: `C_D FP32 <value>, FP16 <value>` and `published sphere C_D at Re 100 ≈ 1.09 (Johnson & Patel 1999, unverified), not gated`.
  6. **Result**: steps is the total across both runs, and cells = N.
- **Tests**: `CASES=sphereFp16 npm run test:gpu`. On the M5 it must pass the 1% range, and the output lists both C_D values. If it misses, find the cause and don't widen the range (AGENTS.md). Also run `npm run test:gpu` with no filter, and every 2D case must still pass (spec criterion 3).
- **Command**: `CASES=sphereFp16 npm run test:gpu`, then `npm run test:gpu`. Follow along with `tail -f test-results/progress.log`.

### - [ ] Task 9: D3Q19 benchmark rows

- **Files**: modify `src/bench.ts`.
- **Interfaces produced**: none new. `runBenchmark` appends rows with `scenario` set to `'d3q19 empty fp32'`, `'d3q19 empty fp16'`, `'d3q19 sphere fp32'` or `'d3q19 sphere fp16'`, with `height` holding `H×D` in the log line.
- **Context**: `src/bench.ts` `measure` and the row format. The spec's floor is Global Constraints, throughput floor.
- **Behavior**:
  - `runBenchmark` calls `initGpu({ f16: true })`. After the 2D rows, for each precision (fp16 only when `f16` is true), for each scenario in {empty, sphere} and for each workgroup size in [64, 128, 256]:
    - create a `Solver3D` at `medium` with tau 0.56, uIn 0.1, spongeFraction 0.15, absorb 0.02, and forceEvery 0 for empty or 4 for sphere;
    - for sphere, add the `sphereFp16` sphere;
    - call `initField`, warm up with 50 steps, then measure `steps = max(100, round(4e8 / N))`;
    - log a row in the 2D format, where `gbps` uses `BYTES_PER_CELL_3D = 19 · b · 2 + 4`.
  - After the D3Q19 rows, log one line per precision: `d3q19 floor (empty, wg 128): <MLUPS> vs <floor> MLUPS: met|missed`, with floors of 1,200 for fp16 and 600 for fp32. The floor applies to the workgroup-128 empty rows, because 128 is the solver's default workgroup, which the page runs.
- **Tests**: `npm run bench` on the M5 prints the D3Q19 rows and both floor lines, and both say `met` (spec criterion 4). This benchmark needs the GPU to itself: close any tunnel tab and don't run validation at the same time.
- **Command**: `npm run bench`, then `tail -f test-results/progress.log`.
