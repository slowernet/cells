# AGENTS.md

Browser wind tunnel: D2Q9 (2D) and D3Q19 (3D) TRT lattice Boltzmann solvers in WebGPU compute shaders, written in TypeScript with Vite. There is no framework and there are no runtime dependencies.

- The 2D design follows `docs/research/lattice-boltzmann-browser-wind-tunnel.md`.
- The 3D design follows `docs/research/3d-lattice-boltzmann-webgpu.md` and the spec `docs/dietpowers/2026-09-27-3d-tunnel-spec.md`, which has plans 1 and 2 beside it.

## Commands

```sh
npm run dev        # dev server; pages: / (2D tunnel), /3d.html (3D tunnel), /validate.html, /bench.html
npm run typecheck  # tsc, must be clean
npm test           # vitest unit tests (src/**/*.test.ts), no GPU needed
npm run test:gpu   # Playwright + headless Chrome WebGPU: every spec in tests/
CASES=poiseuille,taylorGreen npm run test:gpu   # run a subset of the validation cases
THROUGHPUT=1 npx playwright test tests/tunnel3d.spec.ts   # 3D page smoke + steps/s check (GPU idle)
npm run bench      # MLUPS benchmark sweep
tail -f test-results/progress.log   # live progress of a running GPU test or benchmark
```

`npm run test:gpu` runs every spec in `tests/`: the validation cases (which `CASES` filters), the 2D benchmark and the 3D page smoke test. The 3D page's throughput check (≥ 480 steps/s at defaults) runs only with `THROUGHPUT=1` and needs the GPU to itself.

GPU tests build to `dist-test/` and serve it on port 5179 with `vite preview`. They use a static build because the dev server's hot reload restarts the page when a source file changes, which aborts long runs. Don't point Playwright at the dev server.

Long cases log a timestamped line after each convergence check (`check k/max`, and they stop early once converged), plus a throughput heartbeat every 10 s. `tests/progress.ts` writes these lines to stdout and to `test-results/progress.log`. When you pipe test output through a filter, keep it line-buffered (`rg --line-buffered`); otherwise nothing shows until the run ends.

## Layout

- `src/lattice.ts`: D2Q9 constants, boundary mode enums, and the flag bit layout.
- `src/shaders/common.ts`: the `Params` uniform struct and shared WGSL. `PARAMS_WGSL`, `PARAMS_BYTES` and `Solver.writeParams` must stay in sync.
- `src/shaders/step.ts`: generates the fused stream + boundary + collide kernel, unrolled from TS.
- `src/shaders/aux.ts`: kernels for flags/refill, init, macro fields, force reduction and the brush.
- `src/solver.ts`: buffers, pipelines, stepping, readbacks.
- `src/render.ts`: field views and GPU tracers; its WGSL lives in `src/shaders/render.ts`.
- `src/chart.ts`: the C_D/C_L chart.
- `src/app.css`: the layout both pages share. The canvas fills the viewport, with a slide-in menu panel, a toolbar and a readout box over it. Overlays never use `backdrop-filter`, the panel animates only `transform`, and a closed panel is `content-visibility: hidden`.
- `src/menu.ts`: the menu toggle, Escape and focus handling, and `isShown`, which tells whether the chart is visible. `src/icons.ts`: inline Lucide icons, with their licence notices.
- `src/app.ts`: the UI.
- `src/cases.ts`: validation cases.
- `src/bench.ts`: the benchmark.
- `src/geometry.ts`: SDF builders.
- `src/units.ts`: Re → τ conversion.
- `src/analysis.ts`: Strouhal and stream-function helpers.
- `src/gpu.ts`: `initGpu` (requests the adapter's buffer limits, plus `shader-f16` with `{ f16: true }`) and `readBuffer`.
- `src/validate.ts`: the validation page, the `runCase` hook and the PASS/FAIL/SKIP output.
- 3D (D3Q19, `3d.html`):
  - `src/lattice3d.ts`: D3Q19 constants and flag bits.
  - `src/shaders/common3d.ts`: the `Params3` struct, the FP16/FP32 storage codec, shared WGSL and `fitsLimits`.
  - `src/shaders/step3d.ts`: the step kernel generator.
  - `src/shaders/aux3d.ts`: the flag/refill, init, macro and force-reduction kernels.
  - `src/solver3d.ts`: `Solver3D`.
  - `src/geometry3d.ts`: 3D SDF builders and reference areas.
  - `src/cases3d.ts`: the `sphereFp16` validation case.
  - `src/app3d.ts`: the 3D page.
  - `src/render3d.ts` + `src/shaders/render3d.ts`: the outline, sphere-traced obstacle, slice and tracers.
  - `src/camera3d.ts`: the orbit camera.
  - `src/view3d.ts`: the View3 uniform and tracer helpers.
  - `src/tunnel3d.ts`: page logic, including presets, obstacle bodies and the step tuner.
- `tests/`: the Playwright specs `validate.spec.ts`, `bench.spec.ts`, `tunnel3d.spec.ts` and `menu.spec.ts` (both pages, desktop and phone viewports), and `progress.ts` for the progress log.
- Unit tests sit next to their modules as `src/**/*.test.ts`. Tests of modules that import a solver stub `GPUBufferUsage` first and import dynamically, as `src/solver.test.ts` does.

## Solver conventions

These apply to both solvers unless a line says otherwise.


- Distribution buffers store **f_i − w_i**, not f_i. Every equilibrium written to them must subtract the weight as well, including init, refill and boundaries. Density is `1 + Σ f`.
- Storage is SoA with `f[i * N + cell]`. Each step reads buffer A and writes buffer B, and they swap every step (`Solver.parity` tracks which one holds the latest post-collision populations).
- A flag word of 0 means an interior fluid cell and takes the unrolled fast path.
  - 2D: bits 0–8 mark directions whose upstream node needs special handling, bits 9–11 are SOLID, INLET and OUTLET, and bits from 12 up hold the force slot index + 1.
  - 3D: bits 1–18 mark directions and bits 19–21 are SOLID, INLET and OUTLET. Force slots live in a separate `slot` buffer, and `cellForce` holds one entry per cell.
- Obstacles are a signed distance field in cells (negative inside) with nodes at integer coordinates. Bouzidi link fractions come from the SDF, and halfway walls sit at −0.5 and H − 0.5.
- 2D open boundaries: a Zou-He velocity inlet and a Zou-He pressure outlet at ρ = 1. The absorbing layers (`absorb`, `inletLayerFraction`) stop acoustic resonance. Without them, the lift on a cylinder grows without bound.
- 3D open boundaries: the inlet plane is written as the equilibrium at the ramped inflow speed. The outlet plane copies unknown populations from the plane behind it, then writes the equilibrium at ρ = 1 with its own velocity. The side faces are slip, with bounce-back where the reflected source node is solid. There is an outlet sponge and absorbing layer, but no inlet layer.
- 3D storage precision is `'fp16' | 'fp32'`, baked into the generated WGSL. FP16 is scaled by 2^15 and clamped before every store. `sphereFp16` gates FP16 against FP32 at 1%.
- Keep the absorbing layers weak (`absorb` ≈ 0.02) and bodies well downstream. The inlet fixes the inflow speed, so it confines a nearby body: a cylinder 8D from the inlet read St 3% high and C_D 5% high. At 16D from the inlet in a 48D-wide domain it matched the references. A strong layer (0.1) or an inlet layer adds a few percent more.

## WGSL gotchas

- `macro` is a reserved word, which is why the macro buffer is called `mac`.
- Vertex stages can't bind `read_write` storage. Line drawing uses its own module that binds the buffer read-only.
- Pipelines use `layout: 'auto'`, which keeps only the bindings the shader actually uses. Bind groups must use the shader's own binding numbers.
- Any write to a `var<storage, read>` buffer is a shader-creation error, even inside a function that is never called. `codecWgsl(..., 'read')` emits no `store_` for that reason.
- `select()` evaluates both arms, so never put a side effect such as `atomicAdd` inside it.
- `enable f16;` must be the module's first directive and needs the device feature `shader-f16`.

## Working rules

- Validation acceptance ranges are fixed before a case runs. If a result misses, find the cause. Don't widen the range to pass. Any change to a range has to be stated and justified in the commit message.
- Performance numbers need the GPU to itself. Don't run the benchmark alongside validation or an open tunnel tab. Background GPU users count too: a video call or a mirrored display pulled D3Q19 FP16 from about 1,300 to 1,124 MLUPS. To check the load first, run `ioreg -r -d 1 -c IOAccelerator | rg 'Device Utilization'`.
- Performance-sensitive paths:
  - Use one dispatch per step, with every step for a frame in a single compute pass.
  - Keep the hot kernel free of per-step readbacks.
  - Don't add loads to the fast path.
