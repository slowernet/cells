export interface Gpu {
  device: GPUDevice;
  adapter: GPUAdapter;
  timestamps: boolean;
}

/** With f16, requests 'shader-f16' when the adapter has it; the result's f16 says whether the device got it. */
export async function initGpu(opts: { f16?: boolean } = {}): Promise<Gpu & { f16: boolean }> {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found.');
  const timestamps = adapter.features.has('timestamp-query');
  const f16 = !!opts.f16 && adapter.features.has('shader-f16');
  const requiredFeatures: GPUFeatureName[] = [];
  if (timestamps) requiredFeatures.push('timestamp-query');
  if (f16) requiredFeatures.push('shader-f16');
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  device.lost.then((info) => console.error(`WebGPU device lost: ${info.message}`));
  device.addEventListener('uncapturederror', (e) => console.error((e as GPUUncapturedErrorEvent).error.message));
  return { device, adapter, timestamps, f16 };
}

export async function readBuffer(device: GPUDevice, src: GPUBuffer, offset: number, size: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, offset, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.destroy();
  return out;
}
