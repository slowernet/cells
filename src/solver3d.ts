import { MAGIC_LAMBDA } from './lattice';
import { Q3 } from './lattice3d';
import { FAR } from './geometry';
import { PARAMS3_BYTES, Precision, bytesPerPopulation, fitsLimits } from './shaders/common3d';
import { step3dShader } from './shaders/step3d';
import { flags3dShader, init3dShader, macro3dShader, reduce3dShader, HISTORY3_LEN } from './shaders/aux3d';
import { readBuffer } from './gpu';

export { fitsLimits, PRESETS3 } from './shaders/common3d';
export type { Precision } from './shaders/common3d';

export interface Solver3DConfig {
  width: number;
  height: number;
  depth: number;
  precision: Precision;
  tau: number;
  lambda?: number;
  /** Smagorinsky constant C_s; 0 disables the subgrid model. */
  smagorinsky?: number;
  /** Inflow speed, already ramped by the caller. */
  uIn?: number;
  /** Fraction of the domain length, ending at the outlet, over which tau rises to spongeTau. */
  spongeFraction?: number;
  spongeTau?: number;
  /** Peak relaxation rate of the outlet absorbing layer toward the inflow state; 0 disables it. */
  absorb?: number;
  workgroupSize?: number;
  /** Sample the obstacle force every n steps; 0 disables force sampling. */
  forceEvery?: number;
}

export interface ForceSample3 {
  step: number;
  fx: number;
  fy: number;
  fz: number;
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class Solver3D {
  readonly W: number;
  readonly H: number;
  readonly D: number;
  readonly N: number;
  readonly wg: number;
  readonly groups: [number, number];
  readonly cfg: Required<Solver3DConfig>;
  step = 0;
  private parity = 0;
  private forceSamplesRead = 0;
  private forceSteps: number[] = [];
  private forceGeneration = 0;

  readonly params: GPUBuffer;
  readonly f: [GPUBuffer, GPUBuffer];
  readonly flags: GPUBuffer;
  readonly slot: GPUBuffer;
  readonly sdf: GPUBuffer;
  readonly macro: GPUBuffer;
  private readonly cellForce: GPUBuffer;
  private readonly counter: GPUBuffer;
  private readonly history: GPUBuffer;
  private readonly histState: GPUBuffer;

  private stepPipe!: GPUComputePipeline;
  private flagsPipe!: GPUComputePipeline;
  private initPipe!: GPUComputePipeline;
  private macroPipe!: GPUComputePipeline;
  private reducePipe!: GPUComputePipeline;
  private stepBG!: [GPUBindGroup, GPUBindGroup];
  private macroBG!: [GPUBindGroup, GPUBindGroup];
  private flagsBG!: GPUBindGroup;
  private initBG!: GPUBindGroup;
  private reduceBG!: GPUBindGroup;

  /** Throws 'Solver3D allocation failed: ...' when the buffers don't fit; any other error is rethrown unchanged. */
  static async create(device: GPUDevice, config: Solver3DConfig): Promise<Solver3D> {
    if (config.precision === 'fp16' && !device.features.has('shader-f16')) throw new Error('shader-f16 is not available on this device');
    const { width: W, height: H, depth: D, precision } = config;
    // A buffer over the binding limit still allocates; only the later bind group fails, outside the error scopes.
    if (!fitsLimits(W, H, D, precision, device.limits)) {
      const bytes = Q3 * bytesPerPopulation(precision) * W * H * D;
      throw new Error(`Solver3D allocation failed: one distribution buffer needs ${bytes} bytes; this device allows ${device.limits.maxStorageBufferBindingSize} per storage binding and ${device.limits.maxBufferSize} per buffer`);
    }
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    const s = new Solver3D(device, config);
    const [validation, oom] = await Promise.all([device.popErrorScope(), device.popErrorScope()]);
    const err = oom ?? validation;
    if (err) {
      s.destroy();
      throw new Error(`Solver3D allocation failed: ${err.message}`);
    }
    try {
      await s.buildPipelines();
      s.rebuildFlags();
    } catch (e) {
      s.destroy();
      throw e;
    }
    return s;
  }

  private constructor(
    readonly device: GPUDevice,
    config: Solver3DConfig,
  ) {
    this.cfg = {
      lambda: MAGIC_LAMBDA,
      smagorinsky: 0,
      uIn: 0,
      spongeFraction: 0,
      spongeTau: 1,
      absorb: 0,
      workgroupSize: 128,
      forceEvery: 1,
      ...config,
    };
    this.W = config.width;
    this.H = config.height;
    this.D = config.depth;
    this.N = this.W * this.H * this.D;
    this.wg = this.cfg.workgroupSize;
    const total = Math.ceil(this.N / this.wg);
    const gx = Math.min(total, device.limits.maxComputeWorkgroupsPerDimension);
    this.groups = [gx, Math.ceil(total / gx)];

    const buf = (size: number, usage = STORAGE) => device.createBuffer({ size: Math.max(size, 16), usage });
    this.params = buf(PARAMS3_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const fBytes = Q3 * bytesPerPopulation(config.precision) * this.N;
    this.f = [buf(fBytes), buf(fBytes)];
    this.flags = buf(this.N * 4);
    this.slot = buf(this.N * 4);
    this.sdf = buf(this.N * 4);
    this.macro = buf(this.N * 16);
    this.cellForce = buf(this.N * 16);
    this.counter = buf(4);
    this.history = buf(HISTORY3_LEN * 16);
    this.histState = buf(4);
    this.writeParams();
    device.queue.writeBuffer(this.sdf, 0, new Float32Array(this.N).fill(FAR));
  }

  private async buildPipelines() {
    const d = this.device;
    const p = this.cfg.precision;
    const constants = { WG: this.wg };
    const make = (code: string) =>
      d.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: d.createShaderModule({ code }), entryPoint: 'main', constants },
      });
    const makePlain = (code: string) =>
      d.createComputePipelineAsync({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: 'main' } });
    [this.stepPipe, this.flagsPipe, this.initPipe, this.macroPipe, this.reducePipe] = await Promise.all([
      make(step3dShader(p)),
      make(flags3dShader(p)),
      make(init3dShader(p)),
      make(macro3dShader(p)),
      makePlain(reduce3dShader()),
    ]);
    const bg = (pipe: GPUComputePipeline, buffers: GPUBuffer[]) =>
      d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    const [A, B] = this.f;
    this.stepBG = [
      bg(this.stepPipe, [this.params, A, B, this.flags, this.sdf, this.slot, this.cellForce]),
      bg(this.stepPipe, [this.params, B, A, this.flags, this.sdf, this.slot, this.cellForce]),
    ];
    this.macroBG = [
      bg(this.macroPipe, [this.params, A, this.flags, this.macro]),
      bg(this.macroPipe, [this.params, B, this.flags, this.macro]),
    ];
    this.flagsBG = bg(this.flagsPipe, [this.params, this.sdf, this.flags, this.slot, A, B, this.macro, this.counter]);
    this.initBG = bg(this.initPipe, [this.params, A, B, this.flags]);
    this.reduceBG = bg(this.reducePipe, [this.cellForce, this.counter, this.history, this.histState]);
  }

  setTau(tau: number) {
    this.cfg.tau = tau;
    this.writeParams();
  }

  setInlet(u: number) {
    this.cfg.uIn = u;
    this.writeParams();
  }

  private writeParams() {
    const c = this.cfg;
    const buf = new ArrayBuffer(PARAMS3_BYTES);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u.set([this.W, this.H, this.D, this.N, this.groups[0]]);
    f.set([c.tau, c.lambda, c.smagorinsky * c.smagorinsky, c.uIn], 8);
    f[12] = c.spongeFraction > 0 ? (this.W - 1) * (1 - c.spongeFraction) : 1e9;
    f[13] = Math.max(c.spongeTau, c.tau);
    f[14] = c.absorb;
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Replaces the obstacle signed distance field (cells, negative inside), rebuilds flags and refills uncovered cells. */
  setSdf(sdf: Float32Array) {
    this.device.queue.writeBuffer(this.sdf, 0, sdf);
    this.rebuildFlags();
  }

  private rebuildFlags() {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.encodeMacro(pass);
    pass.end();
    enc.clearBuffer(this.counter);
    const pass2 = enc.beginComputePass();
    pass2.setPipeline(this.flagsPipe);
    pass2.setBindGroup(0, this.flagsBG);
    pass2.dispatchWorkgroups(...this.groups);
    pass2.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Sets every fluid cell to the equilibrium at the current inflow speed. */
  initField() {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.initPipe);
    pass.setBindGroup(0, this.initBG);
    pass.dispatchWorkgroups(...this.groups);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.parity = 0;
    this.step = 0;
    this.resetForces();
  }

  resetForces() {
    this.device.queue.writeBuffer(this.histState, 0, new Uint32Array(1));
    this.forceSamplesRead = 0;
    this.forceSteps = [];
    this.forceGeneration++;
  }

  /** Encodes n time steps into one compute pass, one dispatch per step plus force reductions. */
  encodeSteps(pass: GPUComputePassEncoder, n: number) {
    const every = this.cfg.forceEvery;
    for (let k = 0; k < n; k++) {
      pass.setPipeline(this.stepPipe);
      pass.setBindGroup(0, this.stepBG[this.parity]);
      pass.dispatchWorkgroups(...this.groups);
      this.parity ^= 1;
      this.step++;
      if (every > 0 && this.step % every === 0) {
        pass.setPipeline(this.reducePipe);
        pass.setBindGroup(0, this.reduceBG);
        pass.dispatchWorkgroups(1);
        this.forceSteps.push(this.step);
      }
    }
  }

  encodeMacro(pass: GPUComputePassEncoder) {
    pass.setPipeline(this.macroPipe);
    pass.setBindGroup(0, this.macroBG[this.parity]);
    pass.dispatchWorkgroups(...this.groups);
  }

  /** Runs n steps and refreshes the macro buffer; resolves when the GPU has finished. */
  async run(n: number, chunk = 500) {
    for (let done = 0; done < n; ) {
      const k = Math.min(chunk, n - done);
      const enc = this.device.createCommandEncoder();
      const pass = enc.beginComputePass();
      this.encodeSteps(pass, k);
      this.encodeMacro(pass);
      pass.end();
      this.device.queue.submit([enc.finish()]);
      done += k;
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  async readMacro(): Promise<Float32Array> {
    return new Float32Array(await readBuffer(this.device, this.macro, 0, this.N * 16));
  }

  /** Force samples produced since the last call, oldest first. Older samples beyond the ring are dropped. */
  async readForces(): Promise<ForceSample3[]> {
    if (this.forceSteps.length === 0) return [];
    const generation = this.forceGeneration;
    const [hist, state] = await Promise.all([
      readBuffer(this.device, this.history, 0, HISTORY3_LEN * 16),
      readBuffer(this.device, this.histState, 0, 4),
    ]);
    // A reset while the copy was in flight makes these samples stale.
    if (generation !== this.forceGeneration) return [];
    const written = new Uint32Array(state)[0];
    const h = new Float32Array(hist);
    const firstAvailable = Math.max(this.forceSamplesRead, written - HISTORY3_LEN);
    const out: ForceSample3[] = [];
    for (let k = firstAvailable; k < written; k++) {
      const s = (k % HISTORY3_LEN) * 4;
      out.push({ step: this.forceSteps[k - this.forceSamplesRead], fx: h[s], fy: h[s + 1], fz: h[s + 2] });
    }
    this.forceSteps.splice(0, written - this.forceSamplesRead);
    this.forceSamplesRead = written;
    return out;
  }

  destroy() {
    for (const b of [this.params, ...this.f, this.flags, this.slot, this.sdf, this.macro, this.cellForce, this.counter, this.history, this.histState]) b?.destroy();
  }
}
