# 3D wind tunnel spec

## Goal

Add an interactive 3D wind tunnel next to the 2D one: a D3Q19 TRT lattice Boltzmann solver in WebGPU with FP16 or FP32 distribution storage, preset obstacles, an orbit view with a slice plane and tracers, and a live C_D/C_L readout, running smoothly on an Apple M5 laptop. Frame rate and visual clarity come before validated accuracy. The 2D tunnel, its solver and its validation suite stay unchanged.

Delivery is one spec and two plans, each its own PR:

- **Plan 1, headless solver**: lattice, geometry, kernels, solver, benchmark rows, the FP16 check and unit tests.
- **Plan 2, 3D page**: renderer, UI, smoke test and links.

Plan 2 builds on the interfaces this spec fixes for plan 1 and doesn't change them.

## Constraints

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

## Design

A separate 3D stack sits beside the 2D one and copies its patterns. The 2D solver, kernels, renderer and app aren't touched. These existing files change, only as described below:

- `src/gpu.ts`: the `f16` option;
- `src/cases.ts`: `CaseResult.skipped` and `runCase`;
- `src/validate.ts`: f16 request, `caseNames` and `SKIP`;
- `src/bench.ts`: f16 request and D3Q19 rows;
- `tests/validate.spec.ts`: the case list and skip handling;
- `vite.config.ts` and `index.html`: the new page (plan 2).

### Plan 1: headless solver

- `src/lattice3d.ts`:
  - D3Q19 constants `CX`, `CY`, `CZ`, `W`, `OPP`, `PAIRS` (9 TRT pairs), `Q3 = 19`, and `MIRROR_Y` and `MIRROR_Z` (the direction with its y or z component negated).
  - `equilibrium3(rho, ux, uy, uz, out)`, which returns raw f_eq for tests.
  - Flag bits:
    - bit i, for i from 1 to 18, marks a direction whose upstream node needs the slow path (bit 0 is unused);
    - bit 19 is `FLAG_SOLID`, bit 20 `FLAG_INLET` and bit 21 `FLAG_OUTLET`.

    A flag word of 0 means an interior fluid cell on the fast path.
- `src/shaders/common3d.ts`:
  - The `Params3` uniform struct, with `PARAMS3_BYTES` kept in sync with `Solver3D.writeParams`. It holds W, H, D, N, the dispatch grouping, τ, λ, smagC2, u_in (already ramped by the CPU), spongeStart, tauSponge and absorb.
  - Shared WGSL: the lattice constants, cell indexing, and the storage codec `load(i, cell)` / `store(i, cell, v)` for the chosen precision.
- `src/shaders/step3d.ts`: `step3dShader(precision)` generates the fused pull-stream, boundary and TRT (+ Smagorinsky) collide kernel, unrolled from TS as in `step.ts`.
  - Bindings:
    - 0: params (uniform)
    - 1: `src`
    - 2: `dst`
    - 3: `flags`
    - 4: `sdf` (f32 per cell)
    - 5: `slot` (u32 per cell: force slot + 1, or 0 for none)
    - 6: `cellForce` (vec4f per slot, xyz used)

    That makes six storage buffers.
  - Boundary rules in the slow path:
    - **Inlet plane x = 0**: write the equilibrium at ρ = 1 and u = (u_in, 0, 0) to every direction, with no collision.
    - **Outlet plane x = W − 1**: a population whose upstream node lies past x = W − 1 takes the same direction's post-collision value from node (W − 2, y, z) in `src`. The node then computes its own u from the pulled populations and writes the equilibrium at ρ = 1 with that u, with no collision.
    - **Side faces y and z (slip)**: a population whose upstream node lies outside exactly one side face takes `MIRROR_Y[i]` or `MIRROR_Z[i]` from the upstream node reflected back inside across that face, as the 2D `Y_SLIP` rule does. A population whose upstream node lies outside both a y face and a z face (edge links, c_x = 0) takes `OPP[i]` from the node itself. For those links, `OPP[i]` equals the double mirror. When the source node a side-face rule would read is solid, which happens where the spanwise cylinder or wing meets a z face, the population is halfway bounce-back instead: `OPP[i]` post-collision from the node itself. That link is counted in momentum exchange like any body link, and the flag build gives the node a force slot.
    - **Obstacles**: Bouzidi interpolated bounce-back, with q from the SDF along the link, and momentum exchange accumulated into `cellForce[slot − 1]`, as in 2D.
    - **Absorbing layer**: over the last 15% of x, τ rises toward `tauSponge` and populations relax toward the inflow equilibrium at rate `absorb`, as in 2D.
- `src/shaders/aux3d.ts`:
  - flag and slot build from the SDF, with an atomic slot counter; inlet and outlet planes stay fluid. A node gets a slot when any of its links has a solid upstream node, directly or through the side-face source rule. The same kernel refills a cell that was solid and is now fluid with the equilibrium at the average ρ and u of its non-solid neighbours in `macro`, or at ρ = 1 and the inflow u if there are none, as the 2D `flagsShader` does;
  - init of every fluid cell to the inflow equilibrium at the current u_in;
  - the macro kernel, which writes vec4f (ux, uy, uz, ρ) per cell, with ρ = 0 marking a solid cell;
  - force reduction of `cellForce` into a vec3 history ring, as in 2D.
- `src/solver3d.ts`:
  - `Solver3D.create(device, cfg: Solver3DConfig): Promise<Solver3D>`, where `Solver3DConfig` holds the grid W, H and D, `precision`, `tau`, `smagorinsky`, `uIn`, `spongeFraction`, `spongeTau`, `absorb`, `workgroupSize` and `forceEvery`.
  - Methods: `setSdf(Float32Array)`, `initField()`, `setTau(tau)`, `setSmagorinsky(cs)`, `setInlet(u)`, `encodeSteps(pass, n)`, `encodeMacro(pass)`, `readForces(): Promise<ForceSample3[]>` (`{ step, fx, fy, fz }`), `readMacro()`, `run(n)` for tests, and `destroy()`.
  - It exposes the `macro`, `sdf` and `flags` buffers to the renderer.
  - `fitsLimits(W, H, D, precision, limits): boolean` is exported pure.
  - `cellForce` holds N entries (16·N bytes), as the 2D solver sizes it per cell, so the slot count can't overflow it.
  - `setSdf` uploads the SDF and reruns the flag build with refill. The flow continues and nothing is reallocated.
  - Change handling, as the 2D app does:
    - obstacle, size and angle changes call `setSdf`, re-derive τ from the new reference length, and clear the force history and chart, as `resetBody` does (`src/app.ts:256-264`); the flow continues;
    - Re changes call `setTau` only;
    - Smagorinsky changes call `setSmagorinsky` only;
    - grid and precision changes rebuild the solver: the app destroys the old solver before creating the new one, and holds a rebuild token so that a rebuild overtaken by a later one destroys its own solver and returns (`src/app.ts:138-173`).
  > **Changed 2026-09-27:** `Solver3D` gains `setSmagorinsky(cs)`, and Smagorinsky changes call it only (from no runtime path for the Smagorinsky input to a setter like `setTau`). Why: the page's Smagorinsky control needs a way to change C_s without a rebuild, which plan 1's interface lacked. Approved by the partner in the plan 2 review, finding 8.
- `src/gpu.ts` (changed): `initGpu(opts?: { f16?: boolean })`. With `f16: true`, it adds `'shader-f16'` to `requiredFeatures` when `adapter.features` has it. It returns `f16: boolean`, which says whether the device has the feature. `app.ts` passes nothing and behaves exactly as today. `validate.ts`, `bench.ts` and `app3d.ts` pass `{ f16: true }`.
- `src/geometry3d.ts`: SDF builders on a W×H×D `Float32Array`, in cells and negative inside, with nodes at integer coordinates. They stamp true distances within a margin of the body and `FAR` elsewhere, as `geometry.ts` does. Union is `min`.
  - `emptySdf3(W, H, D)`
  - `addSphere(cx, cy, cz, r)`
  - `addBox3(cx, cy, cz, hx, hy, hz, angleZ)`
  - `addCylinderZ(cx, cy, r)`, spanning the full z extent
  - `addWing(cx, cy, cz, chord, span, angleZ)`: a NACA0012 section from `nacaPolygon`, extruded over `span` and centered in z, with flat tips.
  - `referenceArea(obstacle, size, D)`, with `size` in cells (the UI's fraction × H): sphere π·size²/4, cube size², cylinder size·D, wing size·span with span = 0.6·D. With obstacle `none`, the app hides the chart and coefficient readouts.
- `src/cases3d.ts`: `CASES3D` with `sphereFp16`. It imports only types and helpers from `src/cases.ts`.
- Harness changes:
  - `CaseResult` in `src/cases.ts` gains an optional `skipped?: string`.
  - `runCase` in `src/cases.ts` looks keys up in `{...CASES, ...CASES3D}` and sets `pass = false` whenever `skipped` is set.
  - `caseNames` in `src/validate.ts` covers both registries, and `formatResult` prints `SKIP  <name>: <reason>` for a skipped case.
  - `tests/validate.spec.ts` adds `sphereFp16` to its default list and calls `test.skip(true, reason)` when a result has `skipped`, so a skip never counts as a pass or a failure.
- `sphereFp16`:
  - Setup: the `medium` grid, a sphere of diameter 16 at (W/4, H/2, D/2), Re 100 on the diameter, and u = 0.1, which gives τ = 0.548.
  - It runs FP32 first, then FP16, with identical settings.
  - Each run advances in blocks of 2,000 steps. It stops once C_D changes by less than 1e-4 relative between blocks, after at least 5 blocks and at most 15. C_D is the mean over the last block. If a run reaches 15 blocks without converging, the result gets a note, and the 1% gate still applies to the last-block means.
  - It gates the acceptance range above and reports both C_D values and the FP32 C_D against the published Re-100 sphere drag, without gating those.
  - It logs `check k/max` progress through `tests/progress.ts`.
  - Without `shader-f16`, it returns `skipped` and never passes.
- `src/bench.ts`: D3Q19 rows for FP32 and FP16 at `medium`, empty and with the `sphereFp16` sphere. They use `BYTES_PER_CELL_3D = 19 · b · 2 + 4`.

### Plan 2: 3D page

- `src/render3d.ts` + `src/shaders/render3d.ts`: `Renderer3D.draw(solver, view)`.
  - **Camera**: an orbit camera. Drag rotates, the wheel zooms, and a view-projection matrix goes in a uniform.
  - **Domain outline**: drawn as lines.
  - **Obstacle**: sphere-traced through the SDF buffer with trilinear sampling. Each ray is clipped to the obstacle's bounding box plus 2 cells, which is known on the CPU. Rays that miss the box skip the march. Each march step is at most 1 cell, and the obstacle is shaded by the SDF-gradient normal. The fragment writes `frag_depth`.
  - **Slice plane**: axis x, y or z, position 0 to 1. It samples `macro` and shows either speed (viridis) or vorticity magnitude (central differences, sequential map). Solid cells are grey.
  - **Tracers**:
    - A compute pass advects the particles with trilinear velocity from `macro`, once per frame, as the 2D tracers do (`src/shaders/render.ts:125-151`).
    - The integrator takes one midpoint step with dt = the lattice steps advanced this frame. When u_target · dt exceeds 2 cells, it splits dt into equal sub-steps of at most 1 cell of travel at u_target.
    - Seeds sit on a regular rake grid at x = 0.1·W, covering the middle half of y and z.
    - A particle respawns at its seed when it leaves the domain or enters a solid cell. There is no age limit, as in the 2D streakline mode.
    - Each is drawn as a depth-tested segment from p to p − k·u, with k = 3 / u_target, so that the target speed maps to 3 cells.
  - Vertex stages bind storage read-only.
- `3d.html` + `src/app3d.ts`: the controls in the table below, the C_D/C_L chart through `ForceChart`, MLUPS and steps-per-second readouts, and adaptive steps per frame as in `app.ts`. The coefficients are C_D = 2F_x/(u_target² · A) and C_L = 2F_y/(u_target² · A), with A = `referenceArea(...)`. `index.html` links to `3d.html` and back, and `vite.config.ts` adds `3d.html` as a build input.

### Data flow

1. The UI settings produce a `Solver3DConfig` and an SDF.
2. `Solver3D` builds flags and slots from the SDF, then inits.
3. Each frame encodes n step dispatches, a force reduction every 4 steps, and one macro dispatch into one compute pass.
4. The tracer advect pass and `Renderer3D` read `macro` and `sdf` in the same frame.
5. Force history is read back asynchronously, as in 2D.

## Inputs and failure behavior

| Input | Allowed values | Default |
|---|---|---|
| Obstacle | sphere, cube, cylinder (spanwise), wing, none | sphere |
| Size | 0.05 to 0.4 of H (diameter, edge or chord) | 0.2 |
| Angle | −20° to 20° about z; cube and wing only | 0 |
| Re | 1 to 1,000,000; τ is clamped above `maxReynolds(u, size·H)` | 100 |
| Grid | presets that pass `fitsLimits` | medium, or the largest that fits |
| Precision | FP16 (if `shader-f16`), FP32 | FP16 if available |
| Smagorinsky | off, or on with C_s 0.10 to 0.17 | off |
| View | speed, vorticity | speed |
| Slice axis / position | x, y, z / 0 to 1 | z / 0.5 |
| Tracers | on, off | on |

- **No WebGPU or no adapter**: the page shows the 2D page's message and draws nothing.
- **No `shader-f16`**: the precision control is disabled at FP32, with a note saying FP16 isn't available on this device.
- **Precision changed to one the selected grid doesn't fit**: the grid drops to the largest preset that fits the new precision, and a note says so.
- **Preset doesn't fit**: it is disabled in the selector, with a tooltip giving the required and available bytes. `low` in FP32 needs 19 · 4 · 524,288 = 39.8 MB per buffer, under the 128 MiB spec default, so at least one preset is always available.
- **Re above `maxReynolds`**: the input is kept, τ is clamped at 0.51, and the UI shows the 2D warning text with the effective Re.
- **Buffer allocation fails**: `Solver3D.create` wraps allocation in `pushErrorScope('out-of-memory')` and `pushErrorScope('validation')`. On an error it destroys what it allocated and throws. `app3d.ts` then tries the next smaller preset and tells the user. If `low` also fails, the page shows the error text. `sphereFp16` returns `skipped` with the error text and never changes grid.
- **Device lost**: the error is logged to the console, with no recovery, as in 2D.
- **Setting changed**: handled as described under `solver3d.ts`. Obstacle changes keep the flow running. Overlapping grid or precision rebuilds resolve to the latest settings, with no leaked solver. Rebuilding with unchanged settings gives the same state.

## Success criteria

1. `npm run typecheck` is clean and `npm test` passes, including new unit tests that check:
   - that the D3Q19 weights sum to 1, Σ w·c = 0, Σ w·c_α·c_β = δ_αβ/3, `OPP` is an involution with c_OPP[i] = −c_i, and `MIRROR_Y` and `MIRROR_Z` negate exactly one component;
   - that `equilibrium3` reproduces ρ and ρu to 1e-6 for sample inputs;
   - `fitsLimits` true and false cases for each preset and precision against 128 MiB and 4 GiB limits;
   - each SDF builder's sign and distance at sample points;
   - `referenceArea` for each obstacle.
2. On a device with `shader-f16`, `CASES=sphereFp16 npm run test:gpu` passes the FP16 acceptance range. It reports C_D for both precisions and the FP32 C_D against the published value. On a device without the feature, it reports `SKIP` with the reason.
3. The existing 2D validation cases still pass.
4. `npm run bench` on the reference M5 reports at least 1,200 MLUPS for `medium` FP16 and at least 600 for `medium` FP32, on the empty-domain rows.
5. On the reference M5, the page at defaults sustains at least 480 lattice steps per second at any display refresh rate, measured over 10 s after the ramp and shown by the steps-per-second readout.
6. A Playwright smoke test loads `3d.html` in headless Chrome at the defaults and checks that:
   - the C_D readout is finite and positive after 5 s;
   - there are no console errors;
   - the canvas isn't blank (some pixel differs from the clear color).

## Assumptions

- The reference machine is the partner's Apple M5 (10 GPU cores, 153 GB/s). Low-end and phone GPUs are out of scope.
- D3Q19 over D3Q27: it costs 1/1.42 of the bandwidth, and its rotational-invariance error appears only above about Re 250 around round bodies. Without Smagorinsky, the demo grids cap Re near 500 to 700 anyway.
- Equilibrium inlet and equilibrium-at-ρ = 1 outlet instead of 3D Zou-He, because they need no edge or corner rules and are robust for an interactive tunnel. They are first order at the boundary, which is acceptable for a demo.
- The sphere sits 3 diameters from the inlet in `sphereFp16`, which confines it more than the 2D references allow. That biases both precisions equally, so the FP16-to-FP32 comparison is still valid. The comparison with published drag is reported, not gated, for this reason.
- The published sphere drag at Re 100 is C_D ≈ 1.09 (Johnson & Patel 1999). **Unverified**: the research didn't retrieve it, and it is reported only.
- WGSL leaves the f32 → f16 rounding mode unspecified. If the M5's backend truncates, FP16 accuracy may be worse than Lehmann's figures. `sphereFp16` measures the combined effect.
- The UI follows the 2D page's layout and style.

## References

- `docs/research/3d-lattice-boltzmann-webgpu.md` (round 2 research, 2026-09-27), and the sources it cites:
  - M5: 10-core GPU, 153 GB/s ([Apple newsroom](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)).
  - FluidX3D D3Q19 on M5: 800 MLUPS in FP32 and 1,596 in FP16S ([FluidX3D README](https://github.com/ProjectPhysX/FluidX3D)).
  - FP16S: scale by 2^15; f − w shifting is crucial for 16-bit accuracy ([Lehmann et al., arXiv:2112.08926](https://arxiv.org/abs/2112.08926)).
  - WGSL: `array<f16>` is legal in storage buffers with `enable f16`, at 2-byte alignment ([WGSL §14.4.1](https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size)). The f32 → f16 rounding mode is unspecified ([WGSL §15.7](https://gpuweb.github.io/gpuweb/wgsl/#floating-point-conversion)).
  - `shader-f16` availability: 99.85% of macOS reports ([web3dsurvey](https://web3dsurvey.com/webgpu/features/shader-f16)).
  - D3Q19's rotational-invariance error at Re ≥ 250 (White & Chong 2011, via the report).
  - Hecht-Harting D3Q19 Zou-He ([arXiv:0811.4593](https://arxiv.org/abs/0811.4593)).
  - Esoteric Pull saves memory rather than bandwidth, and it races with Bouzidi's neighbour read (Lehmann 2022, via the report).
- This repo's D2Q9 kernel reaches about 1,700 MLUPS on grids larger than cache on the M5, about 84% of peak bandwidth (PR #1 description, "Performance (Apple M5)").
- Ginzburg et al., Λ = 3/16, and Bouzidi et al. 2001 (via `docs/research/lattice-boltzmann-browser-wind-tunnel.md`).
- Existing code this design copies or reuses:
  - `src/gpu.ts`: it already requests the adapter's max binding and buffer sizes;
  - `src/units.ts`: `deriveTau`, `maxReynolds`, `TAU_MIN`;
  - `src/shaders/aux.ts`: `feq` in shifted form, flag and slot build, force reduction;
  - `src/shaders/step.ts`: the unrolled generator, the `Y_SLIP` mirror rule, Bouzidi and momentum exchange;
  - `src/geometry.ts`: `nacaPolygon`, stamping with a margin, `FAR`;
  - `src/cases.ts` and `src/validate.ts`: `Metric`, `CaseResult`, `runCase`, `caseNames`;
  - `src/chart.ts`, `src/bench.ts`, `tests/progress.ts`, `tests/validate.spec.ts`;
  - `AGENTS.md`: solver conventions and working rules.
- WebGPU spec default limits: 128 MiB per storage binding and 8 storage buffers per stage (via the 2D research doc).

## Out of scope (follow-ups, each its own spec)

- A Q-criterion isosurface (marching cubes with `atomicAdd` and `drawIndirect`).
- Esoteric Pull in-place streaming.
- D3Q27 as a generator switch, the Bauer-Rüde D3Q19 equilibrium, and cumulant or regularized collision.
- A full 3D validation suite: square-duct Poiseuille, 3D Taylor-Green, cavity, and gated sphere drag.
- Hecht-Harting Zou-He inlet and outlet faces for validation runs.
- Drawing obstacles with a brush, and mesh upload.
- Density and Schlieren views, and volume ray marching.
- A shared lattice-parametric generator for 2D and 3D.
- README minimum specs for the 3D mode: the lowest GPU and browser that run each preset smoothly, mapped to recent Apple, NVIDIA, AMD and Intel GPUs. Base it on measured MLUPS (the M5 reference: 708 FP32, 1,311 FP16 at `medium`) and each preset's memory need, and list the `shader-f16` requirement for FP16. Requested by the partner on 2026-09-27.
- The deep-research gaps: measured WebGPU D3Q19 throughput, f16 rounding per backend, and `array<f16>` against `pack2x16float`.
