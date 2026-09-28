# 3D wind tunnel, plan 2: the 3D page

Spec: docs/dietpowers/2026-09-27-3d-tunnel-spec.md @ e94bba3 (Tasks 7-8 follow the 2026-09-27 overlay-menu change; see the spec's Changed notes)
Base: main (plan 1 is in origin/main at a71677c; local main may lag, so fetch before diffing)
Branch: feature/3d-wind-tunnel, as the partner chose; it already contains plan 1
Commits: approved

**Goal:** build the interactive `3d.html` page on top of plan 1's `Solver3D`: an orbit view with a sphere-traced obstacle, a slice plane and rake tracers, the controls from the spec's Inputs table, the C_D/C_L chart, and MLUPS and steps-per-second readouts. It meets spec criteria 5 (at least 480 steps/s) and 6 (the smoke test).

**Architecture:** the page copies the 2D page file for file. `3d.html` imitates `index.html`, `src/app3d.ts` imitates `src/app.ts`, `src/render3d.ts` imitates `src/render.ts`, and `src/shaders/render3d.ts` imitates `src/shaders/render.ts`. Pure logic that tests can reach without a GPU lives in three modules with unit tests:
- `src/camera3d.ts`: orbit camera and matrices;
- `src/view3d.ts`: the View3 uniform layout, rake seeds, tracer sub-steps and obstacle bounds;
- `src/tunnel3d.ts`: preset fallback, obstacle SDF and reference quantities, coefficient scale.

One render pass per frame draws the outline, obstacle, slice and tracers against a depth buffer. A Playwright spec covers the smoke test and the throughput criterion.

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
- Adaptive steps per frame target a fixed 0.85 × 16.7 ms (14.2 ms) of simulation time per frame, whatever the display's refresh rate. The time is measured with GPU timestamps, or as frame time when those are missing. They are capped at 400, on both pages. Forces are sampled every 4 steps.
  > See the spec's Changed note on adaptive steps per frame (2026-09-27).
- Tracers: 16,384 particles.
- FP16 check (`sphereFp16`) acceptance, fixed now, before the first run: |C_D,FP16 − C_D,FP32| / C_D,FP32 ≤ 1%.
- No runtime dependencies. TypeScript and Vite, as today.

## References

- **Plan 1** (merged in PR #3):
  - `Solver3D` exposes `params` (Params3 uniform), `macro` (`vec4f(ux, uy, uz, ρ)`, ρ = 0 solid), `sdf` (f32 per cell, `FAR` = 1e6 beyond 3 cells of a body), `flags`, `W`, `H`, `D`, `N`, `step` and `cfg`.
  - Methods: `create`, `setSdf`, `initField`, `setTau`, `setInlet`, `encodeSteps`, `encodeMacro`, `readForces` (`{ step, fx, fy, fz }`), `resetForces` and `destroy`.
  - `create` throws `Solver3D allocation failed: …` when the grid doesn't fit, and `shader-f16 is not available on this device` for FP16 without the feature.
  - `PRESETS3`, `fitsLimits` and `Precision` are re-exported from `src/solver3d.ts`.
  - `referenceArea`, `addSphere`, `addBox3`, `addCylinderZ` and `addWing` are in `src/geometry3d.ts`.
  - `initGpu({ f16: true })` returns `{ device, adapter, timestamps, f16 }`.
- **2D page patterns** (`src/app.ts`):
  - `settings()` reads the controls, and `physics()`/`applyPhysics()` derive τ with `deriveTau(re, uRef, L)` and write the notes. The clamped-Re note text is at `app.ts:97-98`.
  - `rebuild()` uses a token: it destroys the old solver first, and a rebuild overtaken by a later one destroys its own solver (`app.ts:138-173`).
  - `resetBody()` calls `setSdf`, then `applyPhysics()`, then `resetForces()` (`app.ts:256-264`).
  - `tuneSteps()` fills `FRAME_BUDGET = 0.85` of the display frame interval, capped at `MAX_SPF = 400`, using GPU timestamps when available (`app.ts:287-305`).
  - The frame loop ramps u_in with smoothstep over `RAMP_STEPS = 3000` from `rampFrom`, reads forces every 6 frames into `ForceChart`, and computes means over the last half of the chart (`app.ts:307-410`).
  - The no-WebGPU message is at `app.ts:28-34`, and the canvas is configured with `alphaMode: 'opaque'`.
- **2D renderer patterns** (`src/render.ts`, `src/shaders/render.ts`): a fullscreen-triangle vertex shader, `viridis()` and `diverging()` colour maps in `COLOR_WGSL`, tracer advection as a compute pass, and line drawing in a separate module because vertex stages can't bind `read_write` storage (AGENTS.md). Pipelines use `layout: 'auto'`, so bind groups use the shader's own binding numbers.
- **WGSL/WebGPU**:
  - A fragment shader may write `@builtin(frag_depth)`.
  - Storage buffers may be bound `read` in vertex and fragment stages.
  - `context.configure({ usage: RENDER_ATTACHMENT | COPY_SRC })` allows copying the current canvas texture.

## Conventions for every task

- **World space is lattice space**: node (x, y, z) sits at integer coordinates, the domain box is [−0.5, W − 0.5] × [−0.5, H − 0.5] × [−0.5, D − 0.5], x is the flow direction and y is up on screen.
- **Macro reads**: trilinear velocity from `macro` clamps indices to the grid, and a sample is solid when the nearest node has `w == 0`.
- **Tests**: unit tests in `src/*.test.ts` run with `npm test`. The page tests in `tests/tunnel3d.spec.ts` run with `npm run test:gpu` or `npx playwright test tests/tunnel3d.spec.ts`. `npm run typecheck` stays clean after every task.
- **Shader compile check** (Tasks 3 and 4): with `npm run dev` running, use the chrome-devtools MCP to evaluate a script on any page that requests a device, imports `/src/shaders/render3d.ts`, calls `createShaderModule` for each generator's output and awaits `getCompilationInfo()`. Any message of type `error` fails the task.
- **Visual check** (after Task 5): open `/3d.html` with the chrome-devtools MCP and take a screenshot, and check the console, before committing.

## Tasks

### - [x] Task 1: Orbit camera

- **Files**: create `src/camera3d.ts` and `src/camera3d.test.ts`.
- **Interfaces produced**:
  - `type Mat4 = Float32Array` (16 values, column-major, as WGSL `mat4x4f` expects).
  - `perspective(fovY, aspect, near, far): Mat4`. It uses WebGPU clip depth 0 to 1.
  - `lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4`.
  - `multiply(a, b): Mat4`.
  - `invert(m): Mat4`.
  - `transformPoint(m, p: Vec3): Vec3`, with a perspective divide.
  - `type Vec3 = [number, number, number]`.
  - `class OrbitCamera`:
    - Constructor: `(target: Vec3, distance: number)`, with `yaw = −0.6` and `pitch = 0.35` radians by default.
    - `rotate(dxPixels, dyPixels)`: yaw and pitch change by 0.005 rad per pixel, and pitch is clamped to ±1.45.
    - `zoom(factor)`: distance is multiplied by `factor` and clamped to [0.3, 4] × the initial distance.
    - `eye(): Vec3`.
    - `viewProj(aspect): Mat4`, with fovY 45°, near = distance / 100 and far = distance × 10.
- **Behavior**: `eye = target + distance · (cos pitch · sin yaw, sin pitch, cos pitch · cos yaw)`, with up = (0, 1, 0).
- **Tests** (`src/camera3d.test.ts`):
  - `target projects to the centre`: `transformPoint(viewProj, target)` gives NDC x and y within 1e-6 of 0, and z in (0, 1). A wrong lookAt sign or depth range fails it.
  - `invert round-trips`: `multiply(m, invert(m))` is the identity to 1e-5 for a viewProj.
  - `rotate keeps distance and clamps pitch`: after large rotations, |eye − target| = distance to 1e-6 and |pitch| ≤ 1.45.
  - `zoom clamps`: repeated `zoom(0.1)` stops at 0.3 × the initial distance.
- **Command**: `npm test -- camera3d`.

### - [x] Task 2: Page logic helpers and the Smagorinsky setter

- **Files**: create `src/tunnel3d.ts` and `src/tunnel3d.test.ts`; modify `src/solver3d.ts` and `src/solver3d.test.ts`.
- **Interfaces produced**:
  - `type PresetName = 'low' | 'medium' | 'high'`.
  - `presetsThatFit(precision, limits): PresetName[]`, in `low`, `medium`, `high` order, using `fitsLimits`.
  - `fallbackPreset(preferred: PresetName, precision, limits): PresetName | null`: the largest fitting preset no larger than `preferred`, or null.
  - `nextSmaller(p: PresetName): PresetName | null`.
  - `interface Body3 { obstacle: Obstacle3; sizeFraction: number; angleDeg: number }`.
  - `buildBody(body, W, H, D): { sdf: Float32Array; lRef: number; area: number; bounds: [Vec3, Vec3] | null }`:
    - L = sizeFraction · H cells, and the centre is (W/4, H/2, D/2), as in `sphereFp16`.
    - sphere: `addSphere` with r = L/2.
    - cube: `addBox3` with half-extent L/2 in each axis, angle −angleDeg (nose up for positive angles, as the 2D square does).
    - cylinder: `addCylinderZ` with r = L/2.
    - wing: `addWing` with chord L, span 0.6·D, angle +angleDeg in radians.
    - none: an empty SDF, lRef = 1, area = 0, bounds = null.
    - `area = referenceArea(obstacle, L, D)`.
    - `bounds` is the body's axis-aligned box grown by 2 cells and clipped to the domain. The unclipped half-extents around the centre are:
      - sphere: L/2 on every axis;
      - cube rotated by a about z: (L/2)·(|cos a| + |sin a|) in x and y, and L/2 in z;
      - cylinder: L/2 in x and y, the full depth in z;
      - wing: chord/2 in x and y, and 0.3·D in z.
  - `coefficientScale(uTarget, area): number` = `2 / (uTarget² · area)`, or 0 when area is 0.
  - `Solver3D.setSmagorinsky(cs: number)`: sets `cfg.smagorinsky` and rewrites the params.
- **Context**: spec Inputs table; `src/geometry3d.ts`; `src/shaders/common3d.ts` `fitsLimits`/`PRESETS3`.
- **Tests**:
  - `src/tunnel3d.test.ts`:
    - `presets and fallback`: against 128 MiB binding and 256 MiB buffer limits:
      - fp16 fits `low` and `medium`;
      - fp32 fits only `low`;
      - `fallbackPreset('medium', 'fp32', …)` is `low`;
      - `fallbackPreset('high', 'fp16', …)` is `medium`;
      - with 1 MiB limits, the fallback is null.
    - `buildBody sphere`: the centre node is solid, lRef = 19.2 for fraction 0.2 at H = 96, area = π·19.2²/4, and the bounds contain the sphere with 2 cells of margin.
    - `buildBody none`: lRef 1, area 0, bounds null, and every SDF value is `FAR`.
    - `coefficientScale`: 2/(0.01·100) = 2 for u = 0.1, A = 100, and 0 for A = 0.
  - `src/solver3d.test.ts`, `setSmagorinsky writes smagC2`: after `setSmagorinsky(0.16)`, params f32 index 10 reads 0.0256.
- **Command**: `npm test -- tunnel3d solver3d`.

### - [x] Task 3: The renderer: outline, obstacle and slice

- **Files**: create `src/view3d.ts`, `src/view3d.test.ts`, `src/shaders/render3d.ts`, `src/shaders/render3d.test.ts` and `src/render3d.ts`.
- **Interfaces produced**:
  - `src/view3d.ts`:
    - `VIEW3_BYTES = 208`.
    - `interface View3 { viewProj: Mat4; invViewProj: Mat4; eye: Vec3; sliceAxis: 0 | 1 | 2; slicePos: number; mode: 0 | 1; uRef: number; boxMin: Vec3; boxMax: Vec3; hasBody: boolean; tracers: boolean; steps: number; count: number; frame: number }`. `mode` is 0 for speed and 1 for vorticity. `slicePos` runs from 0 to 1. The app fills every field except `count`, which `Renderer3D.encode` sets from its own particle count.
    - `VIEW3_OFFSETS`, exported: viewProj 0, invViewProj 64, eye (vec3f) 128, sliceAxis (u32) 140, boxMin (vec3f) 144, slicePos (f32) 156, boxMax (vec3f) 160, uRef (f32) 172, mode (u32) 176, hasBody (u32) 180, steps (f32) 184, count (u32) 188, frame (u32) 192, tracers (u32) 196. The struct size, rounded to 16, is 208.
    - `packView3(v: View3): ArrayBuffer` writes each field at its `VIEW3_OFFSETS` entry into a `VIEW3_BYTES` buffer.
  - `src/shaders/render3d.ts`: `VIEW3_WGSL` (the struct, with fields in `VIEW3_OFFSETS` order and types), `outlineShader()`, `obstacleShader()` and `sliceShader()`. Each module binds `P: Params3` at 0, `V: View3` at 1, and its read-only storage at 2 onward.
  - `src/render3d.ts`, `class Renderer3D`:
    - `constructor(device, context, format)`, `init()`.
    - `attach(solver: Solver3D)`: rebuilds the bind groups.
    - `resize(w, h)`: recreates the `depth24plus` texture.
    - `encode(enc, view: View3)`: sets `view.count`, writes the uniform, then records one render pass that clears the colour to (0.07, 0.07, 0.08) and depth to 1 and draws the outline, obstacle, slice and tracers in that order (tracers arrive in Task 4). With `view.tracers` false, it skips both the advect compute pass and the tracer draw.
    - `requestPixelCount(): Promise<number>`: on the next `encode`, copies the canvas texture to a buffer and resolves with the number of pixels that differ from the clear colour by more than 8/255 in any channel.
- **Behavior**:
  - **Outline**: the 12 edges of the domain box [−0.5, W − 0.5] × [−0.5, H − 0.5] × [−0.5, D − 0.5], computed from `P.W`, `P.H` and `P.D` (so the module uses binding 0), as a `line-list` from vertex_index, grey, depth-tested. `V.boxMin`/`V.boxMax` are only the obstacle's march bounds. For obstacle `none` (bounds null), the app writes a zero box with `hasBody` false.
  - **Obstacle**: a fullscreen triangle.
    1. The fragment builds a ray from `invViewProj` (near and far NDC points).
    2. It intersects the ray with [boxMin, boxMax] and discards on a miss or when `hasBody` is false.
    3. It marches from max(tEntry, 0), so an eye inside the box starts at the eye, by `clamp(d, 0.05, 1.0)` cells, where d is the trilinear SDF sample clamped to ≤ 3. It stops on a hit at d < 0.02, on leaving the box, or after `ceil(length(boxMax − boxMin)) + 2` steps.
    4. On a hit, it shades Lambert plus ambient with the normal from central differences of the SDF (±0.5 cells) and a light from the camera direction, and writes `frag_depth` = the hit's clip z / w.
    5. Otherwise it discards.
  - **Slice**: a quad covering the domain cross-section at coordinate `slicePos · (dim − 1)` on the chosen axis, depth-tested and opaque. The fragment samples `macro`:
    - solid (nearest node `w == 0`) is grey 0.35;
    - speed: `viridis(|u| / (1.6 · uRef))`;
    - vorticity: `viridis(|curl u| / (0.3 · uRef))`, with curl from central differences of trilinear velocity at ±1 cell.
  - `requestPixelCount` relies on the canvas having `COPY_SRC` usage, which Task 5 configures.
- **Tests**:
  - `src/view3d.test.ts`, `packView3 layout`: pack a View3 with distinct values and read each field back at its `VIEW3_OFFSETS` entry; the buffer length is `VIEW3_BYTES`. Moving a field fails it.
  - `src/shaders/render3d.test.ts`:
    - `vertex stages read storage read-only`: no module declares `var<storage, read_write>`.
    - `outline uses the domain size`: `outlineShader()` reads `P.W`, `P.H` and `P.D`, and does not read `boxMin` or `boxMax`.
    - `obstacle writes depth and clips to the box`: `obstacleShader()` contains `@builtin(frag_depth)`, `boxMin` and `boxMax`, and its step clamp is at most `1.0`.
    - `struct matches packing`: parse `VIEW3_WGSL`'s fields and types, compute each offset with WGSL's uniform layout rules (f32/u32 size and alignment 4, vec3f size 12 and alignment 16, mat4x4f size 64 and alignment 16, struct size rounded up to 16), and compare the offsets with `VIEW3_OFFSETS` and the size with `VIEW3_BYTES`. Drift in either fails it.
- **Command**: `npm test -- view3d render3d`, then the shader compile check from Conventions for the outline, obstacle and slice modules.
- Departure: the chrome-devtools MCP couldn't start its browser, because its profile was locked by a browser another session had open. The compile check ran instead through `.dietpowers/compile-check.mjs`, a gitignored Playwright script using headless Chrome with the GPU-test flags. It does the same thing: it imports the module from the dev server, calls `createShaderModule`, and reads the `getCompilationInfo` errors. Outline, obstacle and slice: no errors.
- Departure: `Renderer3D` gains `afterSubmit()`, which the app calls after `device.queue.submit`. A readback buffer can't be mapped until the commands that write it are submitted, so `requestPixelCount` records its copy in `encode` and maps the buffer in `afterSubmit`.
- Departure: `COLOR_WGSL` in `src/shaders/render.ts` is now exported, so the slice reuses the 2D `viridis`. The 2D output is unchanged.

### - [x] Task 4: Tracers

- **Files**: modify `src/view3d.ts`, `src/view3d.test.ts`, `src/shaders/render3d.ts`, `src/shaders/render3d.test.ts` and `src/render3d.ts`.
- **Interfaces produced**:
  - `TRACER_COUNT = 16384` and `RAKE = 128` (a 128 × 128 seed grid).
  - `rakeSeeds(W, H, D): Float32Array`: 4 floats per particle, (x, y, z, 0) with x = 0.1·W, y = H/4 + (i % 128 + 0.5)/128 · H/2 and z = D/4 + (⌊i / 128⌋ + 0.5)/128 · D/2.
  - `tracerSubsteps(uTarget, steps): number` = `uTarget · steps > 2 ? ceil(uTarget · steps) : 1`.
  - `advectShader()`: a compute pass with workgroup 64. Bindings: P, V, `mac` (read), `seeds` (read), `particles: array<vec4f>` (read_write, 2 per particle: position and velocity).
  - `tracerLineShader()`: vertex and fragment. Bindings: `V` at 1 and `particles` (read) at 4 only, with no `P`. `attach` builds its bind group with those explicit binding numbers, as `lineBG` in `src/render.ts` does, because `layout: 'auto'` drops unused bindings.
  - `Renderer3D.attach` creates the seed and particle buffers. Each initial position is the seed with x replaced by `0.1·W + random · 0.8·W`, spread downstream so the tracers don't start as one sheet.
- **Behavior**:
  - **Advect**:
    1. n = `tracerSubsteps(uRef, steps)`, h = `steps / n`.
    2. Repeat n times: v1 = u(p), p += h · u(p + 0.5·h·v1).
    3. If p leaves the domain box or the nearest node is solid, reset p to its seed.
    4. Store the position and u(p).
  - When `steps` is 0 (paused), nothing moves.
  - **Draw**: a `line-list` with 2 vertices per particle, from p to p − (3/uRef)·u. Colour is `viridis(|u|/(1.6·uRef))` at alpha 0.8, blended over, depth-tested, and not writing depth.
- **Tests**:
  - `src/view3d.test.ts`:
    - `rakeSeeds`: 16384 × 4 floats; every x is 0.1·W; y spans (H/4, 3H/4) and z spans (D/4, 3D/4); the first and last seeds sit half a spacing inside those bounds.
    - `tracerSubsteps`: 1 for (0.1, 10) (1 cell), 1 for (0.1, 20) (exactly 2 cells), and 4 for (0.1, 34).
  - `src/shaders/render3d.test.ts`, `line module binds particles read-only`: `tracerLineShader()` declares `particles` as `var<storage, read>` at binding 4, declares binding 1, and declares no binding 0.
- **Command**: `npm test -- view3d render3d`, then the shader compile check from Conventions for the advect and line modules.
- Departure: the compile check ran through `.dietpowers/compile-check.mjs` again (see Task 3), for all five modules. None had errors.

### - [x] Task 5: The 3D page

- **Files**: create `3d.html` and `src/app3d.ts`; modify `vite.config.ts` (input `tunnel3d: '3d.html'`) and `index.html` (a "3D tunnel" link beside the Validation and Benchmark links). `3d.html`'s note line links back to the 2D tunnel (`index.html`), beside Validation and Benchmark.
- **Interfaces produced**:
  - `window.tunnel3d = { ready: boolean; step(): number; cd(): number; stepsPerSecond(): number; pixelCount(): Promise<number> }` for the page tests.
- **Context**: `src/app.ts` in full, which this imitates, and `src/app.css`, which `3d.html` reuses. It also uses Tasks 1 to 4 and the spec's Inputs and failure behavior section.
- **Behavior**:
  - **Controls**, as in the spec's Inputs table, reusing `index.html`'s markup and classes:
    - obstacle select: sphere (default), cube, cylinder, wing, none;
    - size range 0.05–0.4 (step 0.01, default 0.2), with a readout in cells;
    - angle range −20 to 20 (default 0), enabled for cube and wing only;
    - Re number input 1–1e6 (default 100);
    - grid select;
    - precision select: FP16, FP32;
    - Smagorinsky checkbox with C_s range 0.10–0.17 (default 0.16);
    - view select: speed, vorticity;
    - slice axis select: x, y, z (default z), and slice position range 0–1 (default 0.5);
    - tracers checkbox (default on);
    - Pause and Reset flow buttons.
  - **Readouts**: C_D mean and C_L mean over the last half of the chart, the step, `<MLUPS> MLUPS · <n> steps/frame`, and `<k> steps/s`. The steps/s value is the step delta over the last 1 s of wall time, updated every 10 frames. The page also shows the τ note and the Re warning note.
  - **Start**:
    1. Call `initGpu({ f16: true })`. On failure, show the 2D page's no-WebGPU message. Configure the context with `{ device, format, alphaMode: 'opaque', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC }`, so `requestPixelCount` can copy the canvas.
    2. Disable FP16 with the note "FP16 isn't available on this device" when the device lacks it. Then run `refreshPresets(precision)`: it enables exactly the presets in `presetsThatFit(precision, device.limits)` and disables the others, with a `title` giving required and available bytes.
    3. Start at `medium` (or the fallback) with FP16 if available.
  - **Changes**:
    - Obstacle, size and angle: `setSdf(buildBody(...).sdf)`, then re-derive τ with `deriveTau(re, uTarget, lRef)` and `setTau`, update the renderer's box, then `resetForces()` and `chart.clear()`.
    - Re: `setTau` only.
    - Smagorinsky: `setSmagorinsky(on ? cs : 0)`.
    - Grid: rebuild with the 2D token rule.
    - Precision: run `refreshPresets(newPrecision)` first, then move the grid to `fallbackPreset(currentGrid, newPrecision, limits)` with a note if the grid changed, then rebuild with the token rule.
  - **Rebuild failure**: on `Solver3D allocation failed`, try `nextSmaller` and show "Grid <name> didn't fit (<message>); using <smaller>". If `low` fails, show the error text in the no-WebGPU box.
  - **Frame loop**: as `app.ts`:
    - ramp u_in with smoothstep over 3000 steps from `rampFrom`;
    - `tuneSteps` with timestamps;
    - one compute pass (`encodeSteps(n)`, `encodeMacro`) and one render pass;
    - forces every 6 frames into the chart, scaled by `coefficientScale(uTarget, area)`, pushing C_D from fx and C_L from fy;
    - hide the chart and coefficient readouts for `none`.
  - **Camera**: an `OrbitCamera` with target at the domain centre ((W − 1)/2, (H − 1)/2, (D − 1)/2) and distance 1.6·W, re-created on every grid rebuild. Pointer drag calls `camera.rotate`, the wheel calls `camera.zoom(exp(deltaY · 0.001))`, and the canvas fills the stage at `devicePixelRatio`.
  - **Test hook**: `window.tunnel3d` is set once the first frame is submitted.
- **Tests**: none new in this task. Tasks 1 to 4 cover the logic, and Task 6 covers the page. Before committing, check with `npm run dev` and the chrome-devtools MCP:
  - a screenshot shows the sphere, the slice and moving tracers;
  - the console has no errors;
  - obstacle, grid and precision switches work.

  Record anything that departs from this plan as a `Departure:` line.
- **Command**: `npm run typecheck && npm test`, plus the manual check above.
- Departure: the step tuner copied from `app.ts` sticks at small values. It scales `spf` by at most ×1.1 and then rounds, so `round(3 × 1.1) = 3`. After a slow frame the page sat at 3 steps per frame and 180 steps/s. The growth rule is now `nextStepsPerFrame` in `src/tunnel3d.ts`, which gains at least one step whenever the budget has room, with a unit test. With it, defaults run at about 575–600 steps/s on the M5 (about 10 steps per frame, about 1,200 MLUPS), with WindowServer holding about 22% of the GPU. The 2D `tuneSteps` has the same trap. It rarely bites there because 2D runs at 20 or more steps per frame, and it is left unchanged for a separate fix.
- Departure: the visual check ran through `.dietpowers/page-check.mjs` and `.dietpowers/perf-check.mjs`, gitignored Playwright scripts against the dev server, because the chrome-devtools MCP browser was locked. They took screenshots of the sphere; the wing with the vorticity view on a y slice; the cylinder on the low grid in FP32, where a vortex street is visible; and `none`. Every run had no console errors or warnings.

### - [x] Task 6: Page tests and docs

- **Files**: create `tests/tunnel3d.spec.ts`; modify `AGENTS.md` (the Layout section, and the commands note that `npm run test:gpu` now includes the 3D page tests).
- **Interfaces produced**: none.
- **Context**: `tests/validate.spec.ts` and `tests/progress.ts` for the pattern; spec criteria 5 and 6.
- **Behavior**:
  - `smoke` test:
    1. `streamProgress(page, 'tunnel3d smoke')` and collect `console` messages of type `error`, plus `pageerror`s.
    2. Go to `/3d.html` and wait for `window.tunnel3d?.ready`.
    3. Wait 5 s, then assert `cd()` is finite and > 0, `pixelCount()` > 1000, and no errors were collected.
  - `throughput` test, which runs only with `THROUGHPUT=1` (`test.skip(!process.env.THROUGHPUT, 'set THROUGHPUT=1 on the reference machine with the GPU idle')`):
    1. Wait until `step()` ≥ 3000 (the ramp) with a 60 s timeout.
    2. Read `step()`, wait 10 s, and read it again.
    3. Log `steps/s <value>` and assert that (Δstep / 10) ≥ 480, which is spec criterion 5. This one needs the GPU otherwise idle, like the benchmark.
- **Tests**: both of the above.
- **Command**: `npx playwright test tests/tunnel3d.spec.ts`, then `THROUGHPUT=1 npx playwright test tests/tunnel3d.spec.ts` with the GPU idle, then the full `npm run test:gpu`.
- **AGENTS.md**: under Commands, `npm run test:gpu` runs every spec in `tests/`: the validation cases (filtered by `CASES`), the 2D benchmark and the 3D page smoke test. `THROUGHPUT=1 npx playwright test tests/tunnel3d.spec.ts` checks the page's steps/s criterion and needs the GPU otherwise idle.

### - [x] Task 7: Icons and the overlay layout CSS

- **Files**: create `src/icons.ts` and `src/icons.test.ts`; rewrite `src/app.css`.
- **Interfaces produced**:
  - `icon(name: IconName, size = 18): string` returns an inline `<svg>` string using Lucide's 24×24 viewBox, `stroke="currentColor"`, `stroke-width="2"`, round caps and joins, and `aria-hidden="true"`.
  - `IconName` covers: `menu`, `x`, `play`, `pause`, `rotate-ccw`, `pencil`, `eraser`, `crosshair`, `box`, `shapes`, `wind`, `eye`, `gauge`, `chart-line`, `link` and `chevron-down`.
  - The file header carries Lucide's ISC licence notice.
  - `src/app.css` classes, shared by both pages:
    - `#view`: the canvas;
    - `.menu-button`;
    - `#panel` plus the `.open` state;
    - `.group`: a `<details>` whose `<summary>` holds an icon, a title and the chevron;
    - `.toolbar`;
    - `.hud`.
- **Behavior**: the CSS implements the spec's page layout:
  - the canvas is fixed, full viewport, with `touch-action: none`;
  - the panel is fixed at the left, `min(360px, 100vw)` wide and `100dvh` tall, with `overflow-y: auto`, `overscroll-behavior: contain`, `touch-action: pan-y`, `contain: layout paint style` and safe-area padding;
  - when closed, the panel has `transform: translateX(-100%)`, `visibility: hidden` and `content-visibility: hidden`;
  - only `transform` transitions, so the closed panel stays out of rendering and hit-testing;
  - the toolbar and readout box are fixed overlays with solid rgba backgrounds and no `backdrop-filter`, and the readout values use `font-variant-numeric: tabular-nums` with fixed min-widths;
  - `prefers-reduced-motion` removes the transition.
- **Tests** (`src/icons.test.ts`):
  - `every icon renders an svg`: each name returns markup with `viewBox="0 0 24 24"`, `stroke="currentColor"` and `aria-hidden="true"`.
  - `no backdrop-filter over the canvas`: `src/app.css`, read with `node:fs`, contains no `backdrop-filter`.
- **Command**: `npm test -- icons`.

### - [x] Task 8: Both pages on the overlay layout

- **Files**: modify `index.html`, `3d.html`, `src/app.ts` and `src/app3d.ts`; create `src/menu.ts` and `tests/menu.spec.ts`.
- **Interfaces produced**:
  - `initMenu(): void`, in `src/menu.ts`. It fills every `[data-icon]` element with `icon(name)`, and wires `.menu-button` to toggle `#panel.open`, `aria-expanded` and the button icon. Escape closes the panel. Opening moves focus into the panel, and closing returns focus to the button.
  - Both pages keep every control id, and gain `.menu-button`, `.toolbar` and `.hud`.
  - `app3d.ts` and `app.ts` size the canvas from a `ResizeObserver`, using `devicePixelContentBoxSize` when it exists and `contentBoxSize × devicePixelRatio` otherwise. The 2D page letterboxes the canvas to the grid's aspect ratio inside the viewport.
- **Behavior**:
  - The menu groups follow the spec's table. The Forces group holds the chart.
  - The readout box holds C_D, C_L, steps/s and MLUPS, plus Strouhal on the 2D page. The 3D page hides C_D and C_L when the obstacle is `none`.
  - The toolbar holds the play/pause button (`#pause`, which swaps the `play` and `pause` icons) and `#resetFlow`. The 2D page's toolbar adds the draw, erase and probe radio buttons and the brush size.
- **Tests** (`tests/menu.spec.ts`): for each of `/index.html` and `/3d.html`, at a 1400×800 viewport and at a 390×844 viewport with touch:
  - `menu opens, closes and scrolls`:
    1. Record the canvas's pixel width and height.
    2. Click `.menu-button` and assert `#panel` is visible.
    3. Assert `#panel`'s `scrollHeight` exceeds its `clientHeight` (all groups are opened first on the desktop viewport, so the content is tall enough), then scroll the panel and assert `scrollTop > 0`.
    4. Press Escape and assert the panel is hidden.
    5. Assert the canvas's pixel size is unchanged.
  - `controls still work`: open the menu, change `#obstacle`, and assert the page's readouts still update, meaning the step count increases.
- **Command**: `npx playwright test tests/menu.spec.ts tests/tunnel3d.spec.ts`, then a visual check with `.dietpowers/page-check.mjs` on both pages at both viewports.
- Departure: `ForceChart` gains `invalidate()`, and both pages draw the chart only while it is visible (`isShown` in `src/menu.ts`). The chart sizes itself from `clientWidth`, which is 0 inside a closed panel.
- Departure: the menu test waits for the 180 ms slide-in before scrolling. A gesture sent mid-slide landed on the canvas behind the panel, and `elementFromPoint` confirmed that. The phone scroll uses CDP `Input.synthesizeScrollGesture` at fixed viewport coordinates.
- Departure: the 3D pause test opens the menu before changing `#obstacle`, because that control now sits in the closed panel.
- Departure: the panel is opaque (`--surface-1`), not 94% translucent. The screenshots showed the readout box and toolbar showing through, and an opaque panel also skips blending over the canvas.
- Checked: screenshots of both pages at 1400×800 and 390×844 (phone, 3× scale), with the menu open and closed, from `.dietpowers/layout-shots.mjs`. There were no console warnings or errors. There is one verbose-level Chrome message per load about rendering in a `content-visibility` subtree.

### - [x] Task 9: Menu at the top right, controls and stats in the panel header, 2D/3D switch

- **Files**: modify `index.html`, `3d.html`, `src/app.css`, `src/app3d.ts`, `tests/menu.spec.ts` and `tests/tunnel3d.spec.ts`.
- **Behavior**, following the spec's second 2026-09-27 change:
  - `.menu-button` is fixed at the top right, and `#panel` sits at the right, sliding in from `translateX(100%)`.
  - The panel header is, in order: the title; a `.mode-switch` segmented control with `2D` → `index.html` and `3D` → `3d.html`, the current page carrying `aria-current="page"`; the `.controls` row with the former toolbar's buttons and tools, all ids kept; and the `#stats` `<dl>`. The Pages group keeps only the validation and benchmark links.
  - The floating `.toolbar` and `.hud` are removed.
  - `app3d.ts` toggles pause on Space when focus isn't in a form field, as `app.ts` does.
- **Tests**:
  - `tests/menu.spec.ts` gains `mode switch links to the other page` on both pages: the current segment has `aria-current="page"` and the other links to the other page.
  - Every test that clicks `#pause` opens the menu first.
  - `menu opens, closes and scrolls` also checks that the menu button's box sits in the right half of the viewport.
- **Command**: `npx playwright test tests/menu.spec.ts tests/tunnel3d.spec.ts`, then screenshots with `.dietpowers/layout-shots.mjs`.
- Departure: when the menu opens, focus moves to the panel itself (`tabindex="-1"`) instead of its first control. Both pages ignore Space only when it targets a real control (input, select, textarea, button, link or summary), so Space pauses with the menu open or closed, as the spec says. Before this, the 2D page only paused when focus was on `<body>`.
- Checked: all 14 page tests pass (menu on both pages at both viewports, the mode switch, controls, and the 3D smoke and pause tests), plus screenshots of both pages at desktop and phone sizes with no console warnings.

### - [x] Task 10: Clicking off the menu closes it

- **Files**: modify `src/menu.ts`, `src/app.ts`, `src/app3d.ts` and `tests/menu.spec.ts`.
- **Behavior**, following the spec's third 2026-09-27 change: a `pointerdown` outside the open panel and the menu button, caught in the capture phase on `window`, closes the menu. That pointer's later `pointermove`, `pointerup` and `pointercancel` events are stopped before they reach the canvas, so the press doesn't draw in 2D or orbit in 3D.
- **Tests**: `clicking off the menu closes it without acting on the canvas` runs on both pages, at desktop (a drag) and phone (a tap). It uses the test hook `window.interactionState()`: the 2D drawn box, or the 3D camera angles. With the swallowing removed, three of the four fail; the 3D phone tap has no movement, so it can't orbit.
- **Checked**: 18 page tests and 64 unit tests pass.

### - [x] Task 11: Slice axis "off"

- **Files**: modify `3d.html`, `src/view3d.ts`, `src/view3d.test.ts`, `src/render3d.ts`, `src/app3d.ts` and `tests/tunnel3d.spec.ts`.
- **Behavior**, following the spec's slice-off change: `#sliceAxis` gains `off`. View3 gains a CPU-side `slice` flag that isn't packed into the uniform, and `Renderer3D.encode` skips the slice draw when it is false. The slice position slider is disabled while the slice is off.
- **Tests**: `slice axis off hides the slice`. With tracers off, the drawn pixel count falls below half: it measured 123,849 with the slice and 6,299 without.

### - [x] Task 12: Fixed compute budget for the step tuner (both pages)

- **Files**: modify `src/tunnel3d.ts`, `src/tunnel3d.test.ts`, `src/app.ts` and `src/app3d.ts`.
- **Interfaces produced**:
  - `FRAME_BUDGET_MS = 0.85 × 16.7`.
  - `tuneStepsPerFrame(spf, costMs): number` returns `nextStepsPerFrame(spf, FRAME_BUDGET_MS / max(0.5, costMs))`.
  - `FrameInterval` and the pages' `state.interval` are removed.
- **Behavior**: both pages call `tuneStepsPerFrame` each running frame (not while paused, and not in the 2D manual modes). The cost is the smoothed GPU compute time when a timestamp reading exists, and `frameMs − 3` otherwise. The slow-frame branch that compared a frame against 1.5 × the interval estimate is removed; an over-budget cost already shrinks steps by up to ×0.8 per frame.
- **Tests**:
  - unit tests for `tuneStepsPerFrame`: it grows from 1 under budget, shrinks over budget and holds at the budget;
  - `tests/tuner.spec.ts` passes 20 of 20 repeats;
  - the menu and 3D page suites pass;
  - the throughput test holds ≥ 480 steps/s.
- Checked: `tuner.spec` passes 20 of 20; the menu and 3D suites pass (19); throughput measured 536 and 551 steps/s; 65 unit tests pass. A 12 ms budget measured 404–457 steps/s and was dropped, as recorded in the spec's Changed note.
