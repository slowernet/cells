import { initGpu, readBuffer } from './gpu';
import { Solver, SolverConfig } from './solver';
import { XMode, YMode } from './lattice';
import { emptySdf, addCircle } from './geometry';

export interface BenchRow {
  scenario: string;
  width: number;
  height: number;
  workgroup: number;
  forceEvery: number;
  steps: number;
  ms: number;
  mlups: number;
  gbps: number;
  timer: 'gpu-timestamps' | 'wall-clock';
}

/** Bytes moved per cell update: nine loads and nine stores of f32 plus the flag word. */
const BYTES_PER_CELL = 9 * 4 * 2 + 4;

type Encode = (pass: GPUComputePassEncoder, steps: number) => void;

async function measure(device: GPUDevice, encode: Encode, steps: number, timestamps: boolean): Promise<{ ms: number; timer: BenchRow['timer'] }> {
  if (!timestamps) {
    const t0 = performance.now();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    encode(pass, steps);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return { ms: performance.now() - t0, timer: 'wall-clock' };
  }
  const qs = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
  encode(pass, steps);
  pass.end();
  enc.resolveQuerySet(qs, 0, 2, resolve, 0);
  device.queue.submit([enc.finish()]);
  const t = new BigUint64Array(await readBuffer(device, resolve, 0, 16));
  qs.destroy();
  resolve.destroy();
  return { ms: Number(t[1] - t[0]) / 1e6, timer: 'gpu-timestamps' };
}

/** Copies nine SoA floats and reads a flag word per cell: the memory traffic of one step with no arithmetic. */
async function copyRoofline(device: GPUDevice, W: number, H: number, wg: number, steps: number, timestamps: boolean) {
  const N = W * H;
  const code = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
const N: u32 = ${N}u;
const GX: u32 = ${Math.min(Math.ceil(N / wg), 65535)}u;
@compute @workgroup_size(${wg})
fn main(@builtin(workgroup_id) g: vec3u, @builtin(local_invocation_index) li: u32) {
  let idx = (g.y * GX + g.x) * ${wg}u + li;
  if (idx >= N) { return; }
  let k = f32(flags[idx]);
${Array.from({ length: 9 }, (_, i) => `  dst[${i}u * N + idx] = src[${i}u * N + idx] + k;`).join('\n')}
}`;
  const pipe = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const mk = (size: number) => device.createBuffer({ size, usage: GPUBufferUsage.STORAGE });
  const [a, b, fl] = [mk(N * 36), mk(N * 36), mk(N * 4)];
  const bgs = [[a, b], [b, a]].map(([x, y]) =>
    device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [x, y, fl].map((buffer, binding) => ({ binding, resource: { buffer } })) }),
  );
  const total = Math.ceil(N / wg);
  const groups: [number, number] = [Math.min(total, 65535), Math.ceil(total / Math.min(total, 65535))];
  const encode: Encode = (pass, n) => {
    pass.setPipeline(pipe);
    for (let k = 0; k < n; k++) {
      pass.setBindGroup(0, bgs[k & 1]);
      pass.dispatchWorkgroups(...groups);
    }
  };
  await measure(device, encode, 50, timestamps);
  const r = await measure(device, encode, steps, timestamps);
  for (const x of [a, b, fl]) x.destroy();
  return r;
}

export async function runBenchmark(log: (s: string) => void = () => {}): Promise<{ adapter: string; rows: BenchRow[] }> {
  const { device, adapter, timestamps } = await initGpu();
  const info = adapter.info;
  const adapterName = `${info.vendor} ${info.architecture} ${info.description}`.trim();
  log(`adapter: ${adapterName}; timer: ${timestamps ? 'GPU timestamps' : 'wall clock'}`);
  const rows: BenchRow[] = [];
  const sizes: [number, number][] = [[512, 256], [1024, 512], [2048, 1024], [4096, 2048]];
  const scenarios: { name: string; cfg: Partial<SolverConfig>; body: boolean; forceEvery: number[] }[] = [
    { name: 'periodic', cfg: { left: XMode.Periodic, right: XMode.Periodic, bottom: YMode.Periodic, top: YMode.Periodic }, body: false, forceEvery: [0] },
    { name: 'tunnel+cylinder', cfg: { left: XMode.Open, right: XMode.Open, bottom: YMode.Slip, top: YMode.Slip, uIn: 0.1, spongeFraction: 0.15 }, body: true, forceEvery: [0, 1] },
  ];
  for (const [W, H] of sizes) {
    const steps = Math.max(100, Math.round(4e8 / (W * H)));
    for (const wg of [64, 128, 256]) {
      const { ms, timer } = await copyRoofline(device, W, H, wg, steps, timestamps);
      const mlups = (W * H * steps) / (ms * 1e3);
      const row: BenchRow = { scenario: 'copy roofline', width: W, height: H, workgroup: wg, forceEvery: 0, steps, ms, mlups, gbps: (mlups * 1e6 * BYTES_PER_CELL) / 1e9, timer };
      rows.push(row);
      log(`${row.scenario.padEnd(16)} ${`${W}x${H}`.padEnd(10)} wg ${String(wg).padEnd(4)} ${''.padEnd(16)} ${mlups.toFixed(0).padStart(6)} MLUPS  ${row.gbps.toFixed(0).padStart(4)} GB/s`);
    }
    for (const sc of scenarios)
      for (const wg of [64, 128, 256])
        for (const fe of sc.forceEvery) {
          const s = await Solver.create(device, { width: W, height: H, tau: 0.56, workgroupSize: wg, forceEvery: fe, ...sc.cfg } as SolverConfig);
          if (sc.body) {
            const sdf = emptySdf(W, H);
            addCircle(sdf, W, H, W / 4, H / 2, H / 10);
            s.setSdf(sdf);
          }
          s.initField();
          const encode: Encode = (pass, n) => s.encodeSteps(pass, n);
          await measure(device, encode, 50, timestamps);
          const { ms, timer } = await measure(device, encode, steps, timestamps);
          const mlups = (W * H * steps) / (ms * 1e3);
          const row: BenchRow = { scenario: sc.name, width: W, height: H, workgroup: wg, forceEvery: fe, steps, ms, mlups, gbps: (mlups * 1e6 * BYTES_PER_CELL) / 1e9, timer };
          rows.push(row);
          log(`${sc.name.padEnd(16)} ${`${W}x${H}`.padEnd(10)} wg ${String(wg).padEnd(4)} force ${fe ? 'every step' : 'off       '} ${mlups.toFixed(0).padStart(6)} MLUPS  ${row.gbps.toFixed(0).padStart(4)} GB/s  (${steps} steps, ${ms.toFixed(1)} ms)`);
          s.destroy();
        }
  }
  return { adapter: adapterName, rows };
}

const out = document.getElementById('out')!;
const log = (s: string) => {
  out.textContent += s + '\n';
  console.log(s);
};
const w = window as unknown as Record<string, unknown>;
w.runBenchmark = () => runBenchmark(log);
w.benchReady = true;
document.getElementById('run')!.onclick = () => runBenchmark(log);
