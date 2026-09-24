# Lattice Boltzmann wind tunnel

A 2D wind tunnel in the browser: D2Q9 lattice Boltzmann with TRT collision, running as WebGPU compute shaders. It implements the recommendations in [docs/research/lattice-boltzmann-browser-wind-tunnel.md](docs/research/lattice-boltzmann-browser-wind-tunnel.md).

```sh
npm install
npm run dev          # http://localhost:5173 — tunnel, /validate.html, /bench.html
npm test             # unit tests (vitest)
npm run test:gpu     # validation benchmarks in headless Chrome (WebGPU)
npm run bench        # MLUPS benchmark in headless Chrome
tail -f test-results/progress.log   # follow a running GPU test or benchmark
```

Needs a browser with WebGPU: Chrome/Edge 113+, Safari 26, or Firefox on Windows or Apple Silicon.

## What it does

- **Solver** (`src/shaders/step.ts`): one fused pull-stream, boundary and collide kernel per time step. All steps for a frame go into one compute pass. Distributions are stored structure-of-arrays as `f_i − w_i`, which keeps float32 precision for small deviations from rest.
- **Collision**: TRT with Λ = 3/16, plus an optional Smagorinsky subgrid model in closed form from the non-equilibrium momentum flux. τ is clamped at 0.51.
- **Boundaries**:
  - Zou-He velocity inlet, ramped over 3000 steps, and a Zou-He pressure outlet (ρ = 1).
  - Absorbing layers: a viscosity sponge plus relaxation toward the inflow state over the last 15% of the domain, and a thin layer after the inlet.
  - Walls can be slip, no-slip, moving or periodic.
- **Obstacles**: stored as a signed distance field. Bouzidi interpolated bounce-back uses link fractions derived from the SDF. Drawn shapes are unions of discs, so they get the same curved-wall treatment.
- **Forces**: computed by momentum exchange inside the step kernel and summed on the GPU. They are shown as C_D and C_L traces, and the Strouhal number comes from zero crossings of C_L.
- **Visualization**: vorticity, speed, density and Schlieren views, computed in the fragment shader from the macro buffer. GPU tracer particles are drawn as line segments into fading trail textures, in wind or streakline mode.

## Validation

`/validate.html` (or `npm run test:gpu`) runs the benchmark suite from the research doc: Poiseuille, Taylor-Green, Schäfer-Turek 2D-1 and 2D-2, the unconfined cylinder at Re 100, the Ghia cavity at Re 1000, and NACA0012 at Re 500. Each metric prints the published range and the acceptance range the solver has to meet.
