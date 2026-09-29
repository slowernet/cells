# Lattice Boltzmann wind tunnel

A lattice Boltzmann 2D/3D wind tunnel with TRT collision, running as WebGPU compute shaders.

- **2D** (D2Q9, `index.html`) implements the recommendations in [docs/research/lattice-boltzmann-browser-wind-tunnel.md](docs/research/lattice-boltzmann-browser-wind-tunnel.md).
- **3D** (D3Q19, `3d.html`) follows [docs/research/3d-lattice-boltzmann-webgpu.md](docs/research/3d-lattice-boltzmann-webgpu.md) and the [3D spec](docs/dietpowers/2026-09-27-3d-tunnel-spec.md).

![Speed field and wind particles behind an inclined flat plate, showing alternating shed vortices](docs/images/flat-plate-wake.jpg)

**Demo: https://slowernet.github.io/cells/** (2D) and [3d.html](https://slowernet.github.io/cells/3d.html) (3D).

```sh
npm install
npm run dev          # http://localhost:5173 — 2D tunnel, /3d.html, /validate.html, /bench.html
npm test             # unit tests (vitest)
npm run test:gpu     # headless Chrome (WebGPU): validation cases, 2D and 3D benchmark, page tests
npm run bench        # MLUPS benchmark in headless Chrome
tail -f test-results/progress.log   # follow a running GPU test or benchmark
```

Needs a browser with WebGPU: Chrome/Edge 113+, Safari 26, or Firefox on Windows or Apple Silicon. The 3D tunnel stores distributions in FP16 when the device has the `shader-f16` feature, and falls back to FP32 otherwise.

## What it does: 2D

- **Solver** (`src/shaders/step.ts`): one fused pull-stream, boundary and collide kernel per time step. All steps for a frame go into one compute pass. Distributions are stored structure-of-arrays as `f_i − w_i`, which keeps float32 precision for small deviations from rest.
- **Collision**: TRT with Λ = 3/16, plus an optional Smagorinsky subgrid model in closed form from the non-equilibrium momentum flux. τ is clamped at 0.51.
- **Boundaries**:
  - Zou-He velocity inlet, ramped over 3000 steps, and a Zou-He pressure outlet (ρ = 1).
  - Absorbing layers: a viscosity sponge plus relaxation toward the inflow state over the last 15% of the domain. The solver also supports a thin layer after the inlet (`inletLayerFraction`), which only the NACA validation case uses.
  - Walls can be slip, no-slip, moving or periodic.
- **Obstacles**: stored as a signed distance field. Bouzidi interpolated bounce-back uses link fractions derived from the SDF. Drawn shapes are unions of discs, so they get the same curved-wall treatment.
- **Forces**: computed by momentum exchange inside the step kernel and summed on the GPU. They are shown as C_D and C_L traces, and the Strouhal number comes from zero crossings of C_L.
- **Visualization**: vorticity, speed, density and Schlieren views, computed in the fragment shader from the macro buffer. GPU tracer particles are drawn as line segments into fading trail textures, in wind or streakline mode.

## What it does: 3D

- **Solver** (`src/shaders/step3d.ts`): a D3Q19 version of the same fused kernel, with TRT, optional Smagorinsky, and τ clamped at 0.51. Λ is 3/16 down to τ ≈ 0.56 and shrinks toward BGK below that, as min(3/16, 50·(τ − ½)²): with Λ fixed, the inlet drove the flow unstable from τ ≈ 0.535 (Re ≈ 200 at the default sphere).
  - Distributions are stored as `f_i − w_i`. FP16 storage is scaled by 2^15 and roughly doubles throughput; FP32 is the reference.
  - Grids are 128×64×64, 192×96×96 (the default) and 256×128×128. A grid is offered only if its buffers fit the device's limits.
- **Boundaries**:
  - The inlet plane is set to the equilibrium at the ramped inflow speed.
  - The outlet plane copies the missing populations from the plane behind it, then relaxes to the equilibrium at ρ = 1.
  - The same outlet sponge and absorbing layer as 2D.
  - Slip walls on the four sides. Where a spanwise body meets a side face, the wall bounces back.
- **Obstacles**: a sphere, a cube, a spanwise cylinder or a NACA0012 wing, as a signed distance field with Bouzidi walls and momentum-exchange forces. Changing the obstacle swaps it in without restarting the flow.
- **Visualization**:
  - an orbit camera;
  - the obstacle, sphere-traced through its SDF;
  - one slice plane on any axis;
  - 16,384 tracers released from a rake near the inlet;
  - a "Colour by" choice of speed or vorticity magnitude, which applies to both the slice and the tracers. The slice shows wake structure best.
- **Throughput on an Apple M5**, medium grid: about 1,400 MLUPS with FP16 and 750 with FP32 (`npm run bench`). The page runs about 640 lattice steps per second at defaults.

## Accuracy and its limits

The tunnels are built for looking at flows, and the numbers they print are approximations. The 2D solver reproduces its benchmark suite within fixed ranges, as described under Validation. The 3D solver has no absolute accuracy validation: its only gated check compares FP16 against FP32. Treat 3D coefficients as qualitative. These are the main limits, roughly in order of how much they matter.

**Both tunnels**

- **Compressibility.** Lattice Boltzmann is weakly compressible, with an error that grows as Mach². Both tunnels run the inflow at lattice speed 0.1, which is Mach 0.17, so the error is a few percent.
- **Reynolds ceiling.** Stability needs τ ≥ 0.51, which caps Re at about 30 × the body size in cells at the default speed. Past that, τ is clamped and the page shows the Re it actually runs. If a flow blows up anyway, the page resets it, keeps the obstacle and settings, and says so; a second blow-up with nothing changed pauses it. The optional Smagorinsky model lets you go further, but only as a stabilizer: it keeps a coarse grid from blowing up, and it doesn't model real turbulence.
- **Open boundaries and confinement.** The inlet fixes the inflow speed, and the side walls are slip. A body close to either feels them. In the 2D validation, a cylinder 8 diameters from the inlet read the Strouhal number 3% high and C_D 5% high, and matched the references only at 16 diameters from the inlet in a domain 48 diameters wide. The absorbing layers at the outlet reduce reflected pressure waves but don't remove them.
- **Resolution.** Bodies are typically 15 to 50 cells across. Bouzidi walls are second-order accurate on curved surfaces, but thin features such as a wing's trailing edge span only a cell or two.

**2D only**

- **2D is not 3D.** Real wakes turn three-dimensional above Re ≈ 200, and 2D turbulence sends energy to large scales, where 3D sends it to small ones. Above Re ≈ 200 the 2D tunnel still shows plausible vortex streets, but its coefficients no longer describe a real cylinder or airfoil; the page labels such runs qualitative. Even at Re 20, a 3D channel gives C_D of 6.05 to 6.25 where 2D gives 5.57 to 5.59 ([research doc](docs/research/lattice-boltzmann-browser-wind-tunnel.md)).

**3D only**

- **Coarse grids and a short tunnel.** At the default grid (192×96×96), the sphere is about 19 cells across, sits only about 2.5 diameters behind the inlet, and has about 5 diameters of width around it. That is far smaller than a reference domain. In the validation case at Re 100, FP32 C_D read 1.156 against the published ≈ 1.09, about 6% high. That fits the confinement expected from a tunnel this short, though nothing in the suite separates confinement from other errors.
- **D3Q19 artifacts.** D3Q19 has fewer velocity directions than D3Q27 and isn't fully rotationally invariant. Around round bodies above Re ≈ 250, results can depend on how the flow lines up with the lattice ([3D research doc](docs/research/3d-lattice-boltzmann-webgpu.md)). This is exactly where a sphere's wake becomes interesting: it stays steady and axisymmetric below Re ≈ 210, and sheds hairpin vortices from roughly 270–300 (Johnson & Patel 1999). D3Q27 would remove the artifact, at 1.42 times the memory traffic.
- **Sphere wake regimes.** In FP32 at the default grid, the tunnel reproduces the transitions: at Re 250 the wake is steady but leans to one side (C_L 0.063; published ≈ 0.062), and at Re 300 it sheds with St 0.135 (published 0.137). C_D reads 12 to 13% high at both, which fits the confinement. Shedding takes about 30,000 steps to develop from rest, a minute or two on the page. In FP16 the sphere does not shed at Re 300 ([#8](https://github.com/slowernet/cells/issues/8)), so use FP32 to see it.
- **FP16 storage.** Storing populations in 16 bits roughly doubles speed. On the validation sphere it read C_D 0.72% below FP32, and the gate allows 1%. FP16 also adds faint noise in quiet, low-vorticity regions. WGSL doesn't specify how f32 rounds to f16, so the error can differ between GPUs and browsers. Stored values are clamped to about ±2, which keeps a diverging run from producing NaNs but hides the divergence: the blow-up check never fires, and the page shows noise ([#9](https://github.com/slowernet/cells/issues/9)). Switch the Tunnel group to FP32 when you need the reference numbers.
- **Simple open boundaries.** The inlet and outlet planes are set to equilibrium states. That's robust at edges and corners but only first-order accurate. Pinning the outlet density also reflects pressure waves, which the outlet sponge only partly absorbs.
- **Averaging window.** The C_D and C_L readouts average the second half of the force history, which includes transients after the 3000-step ramp or after any change. Let the flow settle before reading them. C_L is the y component of the side force only; a 3D wake can lean along z instead, so a steady or shedding wake may barely show in C_L. The Forces chart and a vorticity slice show it better.

## Validation

`/validate.html` (or `npm run test:gpu`) runs the validation suite. Each metric prints the published range and the acceptance range the solver has to meet.

- **2D**, the benchmark suite from the research doc: Poiseuille, Taylor-Green, Schäfer-Turek 2D-1 and 2D-2, the unconfined cylinder at Re 100, the Ghia cavity at Re 1000 and NACA0012 at Re 500. A periodic-seam check confirms that forces don't change when a body straddles a periodic boundary.
- **3D**: `sphereFp16` runs a sphere at Re 100 in FP32 and FP16 and requires FP16's C_D to be within 1% of FP32's. It reports C_D against the published value without gating it, because the sphere sits close to the inlet. It is skipped on devices without `shader-f16`.

## GPU requirements

Any GPU with WebGPU runs both tunnels. Memory bandwidth decides how fast they run, because lattice Boltzmann is bandwidth-bound: each step streams every cell's populations in and out. The 2D tunnel is light enough that bandwidth only changes how many steps fit in a frame. The 3D tunnel is the demanding one.

**Browsers.** WebGPU is on by default in:
- Chrome and Edge 113+ on Windows, macOS and ChromeOS;
- Chrome 121+ on Android 12+;
- Safari 26 on macOS and iPadOS;
- Firefox on Windows and on Apple Silicon Macs.

On Linux, Chrome enables it for Intel Gen12+ and for NVIDIA on Wayland; other setups, and Firefox on Linux, need flags ([implementation status](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status)). The 3D tunnel uses FP16 storage when the browser exposes `shader-f16`, which about 94% of WebGPU devices do (99.9% on macOS, 71% on Linux; [web3dsurvey](https://web3dsurvey.com/webgpu/features/shader-f16)). Otherwise it falls back to FP32, which moves twice the bytes per step.

**Memory per 3D grid:**

| Grid | Cells | FP16 total | FP32 total | Largest buffer (FP32) |
|---|---|---|---|---|
| low 128×64×64 | 0.52M | 63 MB | 103 MB | 40 MB |
| medium 192×96×96 | 1.77M | 212 MB | 347 MB | 134.5 MB |
| high 256×128×128 | 4.19M | 503 MB | 822 MB | 319 MB |

WebGPU's default limit for one storage buffer is 128 MiB, so medium FP32 and high need the larger limits most GPUs report. The page offers only the grids that fit the device.

**Smooth 3D.** The target is 480 lattice steps per second at the page's defaults (medium grid), so that the flow crosses the tunnel in a few seconds. On an Apple M5 (153 GB/s), the page ran about 600 steps/s with FP16 and the benchmark about 1,330 MLUPS. Scaling that by bandwidth gives these thresholds for 480 steps/s:

| Grid | FP16 | FP32 |
|---|---|---|
| low | ≈ 36 GB/s | ≈ 68 GB/s |
| medium (default) | ≈ 120 GB/s | ≈ 230 GB/s |
| high | ≈ 290 GB/s | ≈ 550 GB/s |

The table below applies the same scaling to recent GPUs. Only the M5 row is measured. The others are estimates that assume every GPU turns bandwidth into lattice updates as efficiently as the M5; real results vary with the driver and architecture. For integrated GPUs, bandwidth depends on the memory the laptop maker fitted, so the table uses the fastest memory each chip supports.

| GPU | Bandwidth | Est. steps/s, medium FP16 | Smooth at |
|---|---|---|---|
| Apple M1 | 68 GB/s | ≈ 270 | low |
| Apple M2, M3 | 100 GB/s ([M2](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/), [M3](https://support.apple.com/en-us/118551)) | ≈ 390 | low |
| Apple M4 | 120 GB/s ([Apple](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)) | ≈ 470 | medium, just |
| **Apple M5** | 153 GB/s ([Apple](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/)) | **≈ 600, measured** | medium |
| Apple M4 Pro, M5 Pro | 273, 307 GB/s ([M5 Pro/Max](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)) | ≈ 1,070, 1,200 | medium; high on M5 Pro |
| Apple M4 Max, M5 Max | up to 546, 614 GB/s | ≈ 2,100+ | high, including FP32 on M5 Max |
| Intel Iris Xe (Tiger/Alder Lake) | 51–83 GB/s ([ARK](https://www.intel.com/content/www/us/en/products/sku/226254/intel-core-i71260p-processor-18m-cache-up-to-4-70-ghz/specifications.html)) | ≈ 200–330 | low |
| Intel Arc iGPU (Meteor Lake) | 90–120 GB/s ([ARK](https://www.intel.com/content/www/us/en/products/sku/236847/intel-core-ultra-7-processor-155h-24m-cache-up-to-4-80-ghz/specifications.html)) | ≈ 350–470 | low; medium just short even with LPDDR5X |
| Intel Arc 140V (Lunar Lake) | 137 GB/s ([ARK](https://www.intel.com/content/www/us/en/products/sku/240957/intel-core-ultra-7-processor-258v-12m-cache-up-to-4-80-ghz/specifications.html)) | ≈ 540 | medium |
| AMD Radeon 780M, 890M | 90–128 GB/s ([780M](https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-7-7840u.html), [890M](https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-9-hx-370.html)) | ≈ 350–500 | low; medium only on the 890M with LPDDR5X-8000 |
| AMD Radeon 8060S (Strix Halo) | 256 GB/s ([AMD](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html)) | ≈ 1,000 | medium, FP32 too |
| NVIDIA RTX 3050 / 4050 laptop | 192 GB/s ([TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-4050-mobile.c3953)) | ≈ 750 | medium |
| NVIDIA RTX 4060 (laptop, desktop), AMD RX 7600 | 256–288 GB/s ([TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-4060.c4107)) | ≈ 1,000–1,130 | medium, FP32 too |
| NVIDIA RTX 5060, 4070; Intel Arc B580, A750 | 448–512 GB/s ([TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5060.c4219)) | ≈ 1,750–2,000 | high |
| NVIDIA RTX 5070 | 672 GB/s ([TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5070.c4218)) | ≈ 2,600 | high, FP32 too |
| Snapdragon 8 Gen 3, 8 Elite Gen 5 phones | 77–85 GB/s ([Qualcomm](https://docs.qualcomm.com/bundle/publicresource/87-71408-1_REV_C_Snapdragon_8_gen_3_Mobile_Platform_Product_Brief.pdf)) | ≈ 300–330 | low |

The discrete and integrated x86 figures are computed from each part's memory speed and bus width. On machines below the medium threshold, the page still runs at medium, just with the flow developing more slowly; the low grid restores the pace.

## Acknowledgement

Claude Opus 5.5 (Anthropic) researched and implemented this project, working in Claude Code under the direction of [@slowernet](https://github.com/slowernet). The research docs, both solvers, the validation suite, the benchmark, the demo and this README all came out of those sessions. The 2D implementation session cost about $11.61 in API usage. Two subagent runs, a reference check and an adversarial code review, account for roughly 200,000 of its tokens. The 2D research doc was written in an earlier session, and the 3D mode in later ones; neither cost is included in that figure.
