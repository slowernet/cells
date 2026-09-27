import { test, expect } from 'vitest';
(globalThis as any).GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, MAP_READ: 1, QUERY_RESOLVE: 512 };
(globalThis as any).GPUMapMode = { READ: 1 };

interface FakeOpts {
  f16?: boolean;
  scopeError?: 'out-of-memory' | 'validation';
  pipelineError?: string;
  maxBinding?: number;
}

// Just enough of GPUDevice for Solver3D: buffers are ArrayBuffers, copies run at submit, compute passes are no-ops.
function fakeDevice(opts: FakeOpts = {}) {
  const bufs = new Map<any, ArrayBuffer>();
  const created: any[] = [];
  const pendingMaps: (() => void)[] = [];
  const scopes: string[] = [];
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  const device: any = {
    limits: { maxComputeWorkgroupsPerDimension: 65535, maxStorageBufferBindingSize: opts.maxBinding ?? 2 ** 31, maxBufferSize: 2 ** 32 },
    features: new Set(opts.f16 ? ['shader-f16'] : []),
    createBuffer({ size }: any) {
      const b: any = { size, destroyed: false, mapState: 'unmapped' };
      b.destroy = () => (b.destroyed = true);
      bufs.set(b, new ArrayBuffer(size));
      b.mapAsync = () => new Promise<void>((r) => pendingMaps.push(r));
      b.getMappedRange = () => bufs.get(b)!;
      created.push(b);
      return b;
    },
    pushErrorScope(filter: string) {
      scopes.push(filter);
    },
    async popErrorScope() {
      const filter = scopes.pop();
      return filter === opts.scopeError ? { message: `fake ${filter}` } : null;
    },
    createShaderModule: ({ code }: any) => ({ code }),
    async createComputePipelineAsync() {
      if (opts.pipelineError) throw new Error(opts.pipelineError);
      return { getBindGroupLayout: () => ({}) };
    },
    createBindGroup: () => ({}),
    createCommandEncoder() {
      const ops: (() => void)[] = [];
      return {
        beginComputePass: () => pass,
        clearBuffer(b: any) {
          ops.push(() => new Uint8Array(bufs.get(b)!).fill(0));
        },
        copyBufferToBuffer(src: any, so: number, dst: any, dO: number, size: number) {
          ops.push(() => new Uint8Array(bufs.get(dst)!, dO, size).set(new Uint8Array(bufs.get(src)!, so, size)));
        },
        finish: () => ops,
      };
    },
    queue: {
      submit(list: any[]) {
        for (const ops of list) for (const op of ops) op();
      },
      writeBuffer(b: any, off: number, data: any) {
        const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        new Uint8Array(bufs.get(b)!, off, src.byteLength).set(src);
      },
      onSubmittedWorkDone: async () => {},
    },
  };
  return { device, bufs, created, pendingMaps };
}

const cfg = { width: 8, height: 6, depth: 4, tau: 0.6 };

test('fp16 without the feature throws before allocating', async () => {
  const { Solver3D } = await import('./solver3d');
  const { device, created } = fakeDevice();
  await expect(Solver3D.create(device, { ...cfg, precision: 'fp16' })).rejects.toThrow('shader-f16 is not available on this device');
  expect(created).toHaveLength(0);
});

test('allocation error destroys everything', async () => {
  const { Solver3D } = await import('./solver3d');
  const { device, created } = fakeDevice({ scopeError: 'out-of-memory' });
  await expect(Solver3D.create(device, { ...cfg, precision: 'fp32' })).rejects.toThrow(/^Solver3D allocation failed: fake out-of-memory/);
  expect(created.length).toBeGreaterThan(0);
  expect(created.every((b) => b.destroyed)).toBe(true);
});

test('a buffer over the binding limit fails as an allocation error before allocating', async () => {
  const { Solver3D } = await import('./solver3d');
  const N = 8 * 6 * 4;
  const { device, created } = fakeDevice({ maxBinding: 19 * 4 * N - 1 });
  const err = (await Solver3D.create(device, { ...cfg, precision: 'fp32' }).catch((e: Error) => e)) as Error;
  expect(err.message).toBe(`Solver3D allocation failed: one distribution buffer needs ${19 * 4 * N} bytes; this device allows ${19 * 4 * N - 1} per storage binding and ${2 ** 32} per buffer`);
  expect(created).toHaveLength(0);
});

test('error scopes are popped together', async () => {
  const { Solver3D } = await import('./solver3d');
  const { device } = fakeDevice();
  const pops: number[] = [];
  let pending = 0;
  const pop = device.popErrorScope;
  device.popErrorScope = () => {
    pops.push(pending++);
    return pop().then((r: unknown) => (pending--, r));
  };
  await Solver3D.create(device, { ...cfg, precision: 'fp32' });
  // Both pops are issued before either resolves, so the second sees the first still pending.
  expect(pops).toEqual([0, 1]);
});

test('pipeline error destroys everything and keeps its message', async () => {
  const { Solver3D } = await import('./solver3d');
  const { device, created } = fakeDevice({ pipelineError: 'bad wgsl' });
  const err = await Solver3D.create(device, { ...cfg, precision: 'fp32' }).catch((e: Error) => e);
  expect((err as Error).message).toBe('bad wgsl');
  expect(created.length).toBeGreaterThan(0);
  expect(created.every((b) => b.destroyed)).toBe(true);
});

test('params layout', async () => {
  const { Solver3D } = await import('./solver3d');
  const { device, bufs } = fakeDevice();
  const s = await Solver3D.create(device, { ...cfg, precision: 'fp32', uIn: 0, spongeFraction: 0.15 });
  const u = new Uint32Array(bufs.get(s.params)!);
  const f = new Float32Array(bufs.get(s.params)!);
  expect([...u.slice(0, 4)]).toEqual([8, 6, 4, 192]);
  expect(f[8]).toBeCloseTo(0.6, 6);
  expect(f[12]).toBeCloseTo(7 * 0.85, 5);
  s.setInlet(0.05);
  expect(f[11]).toBeCloseTo(0.05, 6);
  s.setTau(0.7);
  expect(f[8]).toBeCloseTo(0.7, 6);
});

test('buffer sizes', async () => {
  const { Solver3D } = await import('./solver3d');
  const N = 8 * 6 * 4;
  for (const [precision, b] of [['fp16', 2], ['fp32', 4]] as const) {
    const { device } = fakeDevice({ f16: true });
    const s: any = await Solver3D.create(device, { ...cfg, precision });
    expect(s.f[0].size).toBe(19 * b * N);
    expect(s.f[1].size).toBe(19 * b * N);
    expect(s.cellForce.size).toBe(16 * N);
  }
});

test('readForces discards a read overtaken by resetForces', async () => {
  const { Solver3D } = await import('./solver3d');
  const { device, bufs, pendingMaps } = fakeDevice();
  const s: any = Object.create(Solver3D.prototype);
  Object.assign(s, { device, step: 0, parity: 0, forceSamplesRead: 0, forceSteps: [], forceGeneration: 0, cfg: { forceEvery: 4 } });
  s.history = device.createBuffer({ size: 8192 * 16 });
  s.histState = device.createBuffer({ size: 4 });
  const gpuState = () => new Uint32Array(bufs.get(s.histState)!);
  const hist = () => new Float32Array(bufs.get(s.history)!);
  // Stands in for the GPU reduce pass: one sample every 4 steps.
  const sim = (steps: number, fx: number) => {
    for (let k = 0; k < steps; k++) {
      s.step++;
      if (s.step % 4 === 0) {
        s.forceSteps.push(s.step);
        const n = gpuState()[0];
        hist().set([fx, 2 * fx, 3 * fx, 0], (n % 8192) * 4);
        gpuState()[0] = n + 1;
      }
    }
  };
  const release = () => pendingMaps.splice(0).forEach((r) => r());

  sim(4000, 99);
  const stale = s.readForces();
  s.step = 0;
  s.resetForces();
  sim(40, 1);
  release();
  expect(await stale).toEqual([]);

  sim(400, 2);
  const next = s.readForces();
  release();
  const samples = await next;
  expect(samples).toHaveLength(110);
  expect(samples[0]).toMatchObject({ step: 4, fx: 1, fy: 2, fz: 3 });
  expect(samples[109]).toMatchObject({ step: 440, fx: 2, fy: 4, fz: 6 });
});
