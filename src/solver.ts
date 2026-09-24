import { XMode, YMode, MAGIC_LAMBDA, Q } from './lattice';
import { PARAMS_BYTES } from './shaders/common';
import { stepShader } from './shaders/step';
import {
  flagsShader,
  initShader,
  macroShader,
  reduceShader,
  brushShader,
  HISTORY_LEN,
  BRUSH_BYTES,
} from './shaders/aux';
import { readBuffer } from './gpu';

export interface SolverConfig {
  width: number;
  height: number;
  left: XMode;
  right: XMode;
  bottom: YMode;
  top: YMode;
  tau: number;
  lambda?: number;
  /** Smagorinsky constant C_s; 0 disables the subgrid model. */
  smagorinsky?: number;
  uIn?: number;
  parabolic?: boolean;
  uLid?: number;
  /** Fraction of the domain length, ending at the outlet, over which tau rises to spongeTau. */
  spongeFraction?: number;
  spongeTau?: number;
  bodyForce?: [number, number];
  /** Peak relaxation rate of the absorbing layers toward the inflow state; 0 disables them. */
  absorb?: number;
  /** Fraction of the domain length, starting at the inlet, covered by an absorbing layer. */
  inletLayerFraction?: number;
  workgroupSize?: number;
  /** Sample the obstacle force every n steps; 0 disables force sampling. */
  forceEvery?: number;
}

export interface ForceSample {
  step: number;
  fx: number;
  fy: number;
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export class Solver {
  readonly W: number;
  readonly H: number;
  readonly N: number;
  readonly wg: number;
  readonly groups: [number, number];
  readonly cfg: Required<SolverConfig>;
  step = 0;
  private parity = 0;
  private forceSamplesRead = 0;
  private forceSteps: number[] = [];
  private forceGeneration = 0;

  readonly params: GPUBuffer;
  readonly f: [GPUBuffer, GPUBuffer];
  readonly flags: GPUBuffer;
  readonly sdf: GPUBuffer;
  readonly macro: GPUBuffer;
  private readonly cellForce: GPUBuffer;
  private readonly counter: GPUBuffer;
  private readonly history: GPUBuffer;
  private readonly histState: GPUBuffer;
  private readonly brushParams: GPUBuffer;

  private stepPipe!: GPUComputePipeline;
  private flagsPipe!: GPUComputePipeline;
  private initPipe!: GPUComputePipeline;
  private macroPipe!: GPUComputePipeline;
  private reducePipe!: GPUComputePipeline;
  private brushPipe!: GPUComputePipeline;
  private stepBG!: [GPUBindGroup, GPUBindGroup];
  private macroBG!: [GPUBindGroup, GPUBindGroup];
  private flagsBG!: GPUBindGroup;
  private initBG!: GPUBindGroup;
  private reduceBG!: GPUBindGroup;
  private brushBG!: GPUBindGroup;

  static async create(device: GPUDevice, config: SolverConfig): Promise<Solver> {
    const s = new Solver(device, config);
    await s.buildPipelines();
    s.rebuildFlags();
    return s;
  }

  private constructor(
    readonly device: GPUDevice,
    config: SolverConfig,
  ) {
    this.cfg = {
      lambda: MAGIC_LAMBDA,
      smagorinsky: 0,
      uIn: 0,
      parabolic: false,
      uLid: 0,
      spongeFraction: 0,
      spongeTau: 1,
      bodyForce: [0, 0],
      absorb: 0,
      inletLayerFraction: 0,
      workgroupSize: 128,
      forceEvery: 1,
      ...config,
    };
    this.W = config.width;
    this.H = config.height;
    this.N = this.W * this.H;
    this.wg = this.cfg.workgroupSize;
    const total = Math.ceil(this.N / this.wg);
    const gx = Math.min(total, device.limits.maxComputeWorkgroupsPerDimension);
    this.groups = [gx, Math.ceil(total / gx)];

    const buf = (size: number, usage = STORAGE) => device.createBuffer({ size: Math.max(size, 16), usage });
    this.params = buf(PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.f = [buf(Q * this.N * 4), buf(Q * this.N * 4)];
    this.flags = buf(this.N * 4);
    this.sdf = buf(this.N * 4);
    this.macro = buf(this.N * 16);
    this.cellForce = buf(this.N * 8);
    this.counter = buf(4);
    this.history = buf(HISTORY_LEN * 8);
    this.histState = buf(4);
    this.brushParams = buf(BRUSH_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.writeParams();
    device.queue.writeBuffer(this.sdf, 0, new Float32Array(this.N).fill(1e6));
  }

  private async buildPipelines() {
    const d = this.device;
    const constants = { WG: this.wg };
    const make = (code: string) =>
      d.createComputePipelineAsync({
        layout: 'auto',
        compute: { module: d.createShaderModule({ code }), entryPoint: 'main', constants },
      });
    const makePlain = (code: string) =>
      d.createComputePipelineAsync({ layout: 'auto', compute: { module: d.createShaderModule({ code }), entryPoint: 'main' } });
    [this.stepPipe, this.flagsPipe, this.initPipe, this.macroPipe, this.reducePipe, this.brushPipe] = await Promise.all([
      make(stepShader()),
      make(flagsShader()),
      make(initShader()),
      make(macroShader()),
      makePlain(reduceShader()),
      makePlain(brushShader()),
    ]);
    const bg = (pipe: GPUComputePipeline, buffers: GPUBuffer[]) =>
      d.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    const [A, B] = this.f;
    this.stepBG = [
      bg(this.stepPipe, [this.params, A, B, this.flags, this.sdf, this.cellForce]),
      bg(this.stepPipe, [this.params, B, A, this.flags, this.sdf, this.cellForce]),
    ];
    this.macroBG = [
      bg(this.macroPipe, [this.params, A, this.flags, this.macro]),
      bg(this.macroPipe, [this.params, B, this.flags, this.macro]),
    ];
    this.flagsBG = bg(this.flagsPipe, [this.params, this.sdf, this.flags, A, B, this.macro, this.counter]);
    this.initBG = bg(this.initPipe, [this.params, this.macro, A, B]);
    this.reduceBG = bg(this.reducePipe, [this.cellForce, this.counter, this.history, this.histState]);
    this.brushBG = bg(this.brushPipe, [this.brushParams, this.params, this.sdf]);
  }


  update(changes: Partial<SolverConfig>) {
    Object.assign(this.cfg, changes);
    this.writeParams();
  }

  private writeParams() {
    const c = this.cfg;
    const buf = new ArrayBuffer(PARAMS_BYTES);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u.set([this.W, this.H, this.N, this.groups[0], c.left, c.right, c.bottom, c.top]);
    f.set([c.tau, c.lambda, c.smagorinsky * c.smagorinsky, c.uIn], 8);
    u[12] = c.parabolic ? 1 : 0;
    f[13] = c.spongeFraction > 0 ? (this.W - 1) * (1 - c.spongeFraction) : 1e9;
    f[14] = Math.max(c.spongeTau, c.tau);
    f[15] = c.uLid;
    f[16] = c.bodyForce[0];
    f[17] = c.bodyForce[1];
    u[18] = c.bodyForce[0] !== 0 || c.bodyForce[1] !== 0 ? 1 : 0;
    f[19] = c.absorb;
    f[20] = c.inletLayerFraction * (this.W - 1);
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Replaces the obstacle signed distance field (cells, negative inside) and rebuilds flags. */
  setSdf(sdf: Float32Array) {
    this.device.queue.writeBuffer(this.sdf, 0, sdf);
    this.rebuildFlags();
  }

  rebuildFlags() {
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

  /** Sets every cell to equilibrium at the given macro field, or at rest when omitted. */
  initField(macro?: Float32Array) {
    const m = macro ?? new Float32Array(this.N * 4);
    if (!macro) for (let i = 0; i < this.N; i++) m[i * 4] = 1;
    this.device.queue.writeBuffer(this.macro, 0, m);
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

  /** Stamps discs of radius r along the given points into the SDF, then rebuilds flags once. */
  brushStroke(points: [number, number][], r: number, erase: boolean) {
    for (const [x, y] of points) this.brush(x, y, r, erase, false);
    this.rebuildFlags();
  }

  brush(cx: number, cy: number, r: number, erase: boolean, rebuild = true) {
    const pad = 2;
    const x0 = Math.max(0, Math.floor(cx - r - pad));
    const y0 = Math.max(0, Math.floor(cy - r - pad));
    const x1 = Math.min(this.W - 1, Math.ceil(cx + r + pad));
    const y1 = Math.min(this.H - 1, Math.ceil(cy + r + pad));
    if (x1 < x0 || y1 < y0) return;
    const b = new ArrayBuffer(BRUSH_BYTES);
    new Float32Array(b, 0, 3).set([cx, cy, r]);
    new Uint32Array(b, 12, 5).set([erase ? 1 : 0, x0, y0, x1 - x0 + 1, y1 - y0 + 1]);
    this.device.queue.writeBuffer(this.brushParams, 0, b);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.brushPipe);
    pass.setBindGroup(0, this.brushBG);
    pass.dispatchWorkgroups(Math.ceil((x1 - x0 + 1) / 8), Math.ceil((y1 - y0 + 1) / 8));
    pass.end();
    this.device.queue.submit([enc.finish()]);
    if (rebuild) this.rebuildFlags();
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
  async readForces(): Promise<ForceSample[]> {
    if (this.forceSteps.length === 0) return [];
    const generation = this.forceGeneration;
    const [hist, state] = await Promise.all([
      readBuffer(this.device, this.history, 0, HISTORY_LEN * 8),
      readBuffer(this.device, this.histState, 0, 4),
    ]);
    // A reset while the copy was in flight makes these samples stale.
    if (generation !== this.forceGeneration) return [];
    const written = new Uint32Array(state)[0];
    const h = new Float32Array(hist);
    const firstAvailable = Math.max(this.forceSamplesRead, written - HISTORY_LEN);
    const out: ForceSample[] = [];
    for (let k = firstAvailable; k < written; k++) {
      const slot = k % HISTORY_LEN;
      out.push({ step: this.forceSteps[k - this.forceSamplesRead], fx: h[slot * 2], fy: h[slot * 2 + 1] });
    }
    this.forceSteps.splice(0, written - this.forceSamplesRead);
    this.forceSamplesRead = written;
    return out;
  }

  destroy() {
    for (const b of [this.params, ...this.f, this.flags, this.sdf, this.macro, this.cellForce, this.counter, this.history, this.histState, this.brushParams]) b.destroy();
  }
}
