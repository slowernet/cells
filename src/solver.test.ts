import { test, expect } from 'vitest';
(globalThis as any).GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, MAP_READ: 1, QUERY_RESOLVE: 512 };
(globalThis as any).GPUMapMode = { READ: 1 };

// Just enough of GPUDevice for readForces: copies run at submit, and mapAsync resolves when the test releases it.
function fakeDevice() {
  const bufs = new Map<any, ArrayBuffer>();
  const pendingMaps: (() => void)[] = [];
  const device: any = {
    limits: { maxComputeWorkgroupsPerDimension: 65535 },
    createBuffer({ size }: any) {
      const b: any = { size, destroy() {} , mapState: 'unmapped' };
      bufs.set(b, new ArrayBuffer(size));
      b.mapAsync = () => new Promise<void>((r) => pendingMaps.push(r));
      b.getMappedRange = () => bufs.get(b)!;
      return b;
    },
    createCommandEncoder() {
      const ops: (() => void)[] = [];
      return {
        copyBufferToBuffer(src: any, so: number, dst: any, dO: number, size: number) {
          ops.push(() => new Uint8Array(bufs.get(dst)!, dO, size).set(new Uint8Array(bufs.get(src)!, so, size)));
        },
        finish: () => ops,
      };
    },
    queue: {
      submit(list: any[]) { for (const ops of list) for (const op of ops) op(); },
      writeBuffer(b: any, off: number, data: any) {
        const src = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        new Uint8Array(bufs.get(b)!, off, src.byteLength).set(src);
      },
    },
  };
  return { device, bufs, pendingMaps };
}

test('a force read in flight across resetForces returns nothing and keeps later samples aligned', async () => {
  const { Solver } = await import('./solver');
  const { device, bufs, pendingMaps } = fakeDevice();
  const s: any = Object.create(Solver.prototype);
  Object.assign(s, { device, step: 0, parity: 0, forceSamplesRead: 0, forceSteps: [], forceGeneration: 0, cfg: { forceEvery: 4 } });
  s.history = device.createBuffer({ size: 8192 * 8 });
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
        hist()[(n % 8192) * 2] = fx;
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
  expect(samples[0]).toMatchObject({ step: 4, fx: 1 });
  expect(samples[109]).toMatchObject({ step: 440, fx: 2 });

  sim(400, 3);
  const third = s.readForces();
  release();
  expect((await third)[0]).toMatchObject({ step: 444, fx: 3 });
});
