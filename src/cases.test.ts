import { test, expect } from 'vitest';
(globalThis as any).GPUBufferUsage = { STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, MAP_READ: 1, QUERY_RESOLVE: 512 };
(globalThis as any).GPUMapMode = { READ: 1 };

const device = {} as GPUDevice;

test('a skipped case never passes', async () => {
  const { runCase } = await import('./cases');
  const { CASES3D } = await import('./cases3d');
  CASES3D.__skip = async () => ({ name: 'x', metrics: [], skipped: 'reason', steps: 0, cells: 0, notes: [] });
  try {
    const r = await runCase(device, '__skip');
    expect(r.pass).toBe(false);
    expect(r.skipped).toBe('reason');
  } finally {
    delete CASES3D.__skip;
  }
});

test('a case with passing metrics and no skip passes', async () => {
  const { runCase, metric } = await import('./cases');
  const { CASES3D } = await import('./cases3d');
  CASES3D.__ok = async () => ({ name: 'y', metrics: [metric('m', 1, [1, 1], [0, 2])], steps: 0, cells: 0, notes: [] });
  try {
    const r = await runCase(device, '__ok');
    expect(r.pass).toBe(true);
    expect(r.skipped).toBeUndefined();
  } finally {
    delete CASES3D.__ok;
  }
});
