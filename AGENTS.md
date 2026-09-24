# AGENTS.md

Browser wind tunnel: a D2Q9 TRT lattice Boltzmann solver in WebGPU compute shaders, written in TypeScript with Vite. There is no framework and there are no runtime dependencies. The design follows `docs/research/lattice-boltzmann-browser-wind-tunnel.md`.

## Commands

```sh
npm run dev        # dev server; pages: / (tunnel), /validate.html, /bench.html
npm run typecheck  # tsc, must be clean
npm test           # vitest unit tests (src/**/*.test.ts), no GPU needed
npm run test:gpu   # Playwright + headless Chrome WebGPU: physics validation suite
CASES=poiseuille,taylorGreen npm run test:gpu   # run a subset
npm run bench      # MLUPS benchmark sweep
tail -f test-results/progress.log   # live progress of a running GPU test or benchmark
```

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
- `src/app.ts`: the UI.
- `src/cases.ts`: validation cases.
- `src/bench.ts`: the benchmark.
- `src/geometry.ts`: SDF builders.
- `src/units.ts`: Re → τ conversion.
- `src/analysis.ts`: Strouhal and stream-function helpers.

## Solver conventions

- Distribution buffers store **f_i − w_i**, not f_i. Every equilibrium written to them must subtract the weight as well, including init, refill and boundaries. Density is `1 + Σ f`.
- Storage is SoA with `f[i * N + cell]`. Each step reads buffer A and writes buffer B, and they swap every step (`Solver.parity` tracks which one holds the latest post-collision populations).
- A flag word of 0 means an interior fluid cell and takes the unrolled fast path. Bits 0–8 mark directions whose upstream node needs special handling. The higher bits are SOLID, INLET and OUTLET, and bits from 12 up hold the force slot index + 1.
- Obstacles are a signed distance field in cells (negative inside) with nodes at integer coordinates. Bouzidi link fractions come from the SDF, and halfway walls sit at −0.5 and H − 0.5.
- The open boundaries are a Zou-He velocity inlet and a Zou-He pressure outlet at ρ = 1. The absorbing layers (`absorb`, `inletLayerFraction`) stop acoustic resonance. Without them, the lift on a cylinder grows without bound.
- Keep the absorbing layers weak (`absorb` ≈ 0.02) and bodies well downstream. The inlet fixes the inflow speed, so it confines a nearby body: a cylinder 8D from the inlet read St 3% high and C_D 5% high. At 16D from the inlet in a 48D-wide domain it matched the references. A strong layer (0.1) or an inlet layer adds a few percent more.

## WGSL gotchas

- `macro` is a reserved word, which is why the macro buffer is called `mac`.
- Vertex stages can't bind `read_write` storage. Line drawing uses its own module that binds the buffer read-only.
- Pipelines use `layout: 'auto'`, which keeps only the bindings the shader actually uses. Bind groups must use the shader's own binding numbers.

## Working rules

- Validation acceptance ranges are fixed before a case runs. If a result misses, find the cause. Don't widen the range to pass. Any change to a range has to be stated and justified in the commit message.
- Performance numbers need the GPU to itself. Don't run the benchmark alongside validation or an open tunnel tab.
- Performance-sensitive paths:
  - Use one dispatch per step, with every step for a frame in a single compute pass.
  - Keep the hot kernel free of per-step readbacks.
  - Don't add loads to the fast path.
