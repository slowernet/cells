import type { Solver } from './solver';
import { VIEW_BYTES, TRACER_WG, fieldShader, tracerShaders, LINE_WGSL, TRAIL_WGSL } from './shaders/render';

export const enum ViewMode {
  Curl = 0,
  Speed = 1,
  Density = 2,
  Schlieren = 3,
}

export const enum TracerMode {
  Off = 0,
  Wind = 1,
  Streaklines = 2,
}

export interface ViewSettings {
  mode: ViewMode;
  /** Multiplies the default colour-scale range; larger shows weaker features. */
  contrast: number;
  /** Reference speed for scaling speed and curl colours. */
  uRef: number;
  tracers: TracerMode;
  trailFade: number;
}

export class Renderer {
  private view: GPUBuffer;
  private fieldPipe!: GPURenderPipeline;
  private advectPipe!: GPUComputePipeline;
  private linePipe!: GPURenderPipeline;
  private fadePipe!: GPURenderPipeline;
  private compositePipe!: GPURenderPipeline;
  private fieldBG!: GPUBindGroup;
  private tracerBG!: GPUBindGroup;
  private lineBG!: GPUBindGroup;
  private trails: GPUTexture[] = [];
  private trailBG: GPUBindGroup[] = [];
  private trailIdx = 0;
  private particles!: GPUBuffer;
  private ages!: GPUBuffer;
  private frame = 0;
  count = 0;

  constructor(
    readonly device: GPUDevice,
    readonly context: GPUCanvasContext,
    readonly format: GPUTextureFormat,
  ) {
    this.view = device.createBuffer({ size: VIEW_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async init() {
    const d = this.device;
    const field = d.createShaderModule({ code: fieldShader() });
    const tracer = d.createShaderModule({ code: tracerShaders() });
    const trail = d.createShaderModule({ code: TRAIL_WGSL });
    const line = d.createShaderModule({ code: LINE_WGSL });
    const alphaBlend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
    };
    [this.fieldPipe, this.advectPipe, this.linePipe, this.fadePipe, this.compositePipe] = await Promise.all([
      d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: field, entryPoint: 'vs' },
        fragment: { module: field, entryPoint: 'fs', targets: [{ format: this.format }] },
      }),
      d.createComputePipelineAsync({ layout: 'auto', compute: { module: tracer, entryPoint: 'advect' } }),
      d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: line, entryPoint: 'lineVs' },
        fragment: { module: line, entryPoint: 'lineFs', targets: [{ format: 'r16float', blend: { color: { srcFactor: 'one', dstFactor: 'one' }, alpha: { srcFactor: 'one', dstFactor: 'one' } } }] },
        primitive: { topology: 'line-list' },
      }),
      d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: trail, entryPoint: 'vs' },
        fragment: { module: trail, entryPoint: 'fade', targets: [{ format: 'r16float' }] },
      }),
      d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module: trail, entryPoint: 'vs' },
        fragment: { module: trail, entryPoint: 'composite', targets: [{ format: this.format, blend: alphaBlend }] },
      }),
    ]);
  }

  /** Binds to a (new) solver; particle count scales with the grid area. */
  attach(solver: Solver) {
    const d = this.device;
    this.particles?.destroy();
    this.ages?.destroy();
    this.count = Math.min(1 << 15, Math.round(solver.N / 64));
    const init = new Float32Array(this.count * 4);
    for (let i = 0; i < this.count; i++) {
      const x = Math.random() * solver.W, y = Math.random() * solver.H;
      init.set([x, y, x, y], i * 4);
    }
    this.particles = d.createBuffer({ size: init.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.particles, 0, init);
    this.ages = d.createBuffer({ size: this.count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(this.ages, 0, new Float32Array(this.count).map(() => Math.random() * 80));
    const bg = (layout: GPUBindGroupLayout, res: GPUBindingResource[]) =>
      d.createBindGroup({ layout, entries: res.map((resource, binding) => ({ binding, resource })) });
    this.fieldBG = bg(this.fieldPipe.getBindGroupLayout(0), [{ buffer: solver.params }, { buffer: this.view }, { buffer: solver.macro }]);
    this.tracerBG = bg(this.advectPipe.getBindGroupLayout(0), [
      { buffer: solver.params }, { buffer: this.view }, { buffer: solver.macro }, { buffer: this.particles }, { buffer: this.ages },
    ]);
    this.lineBG = d.createBindGroup({
      layout: this.linePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: solver.params } }, { binding: 3, resource: { buffer: this.particles } }],
    });
  }

  /** Recreates the trail textures at the canvas resolution. */
  resize(width: number, height: number) {
    for (const t of this.trails) t.destroy();
    this.trails = [0, 1].map(() =>
      this.device.createTexture({
        size: [width, height],
        format: 'r16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
    );
    const layoutFade = this.fadePipe.getBindGroupLayout(0);
    const layoutComp = this.compositePipe.getBindGroupLayout(0);
    this.trailBG = [
      ...this.trails.map((t) => this.device.createBindGroup({ layout: layoutFade, entries: [{ binding: 0, resource: { buffer: this.view } }, { binding: 1, resource: t.createView() }] })),
      ...this.trails.map((t) => this.device.createBindGroup({ layout: layoutComp, entries: [{ binding: 0, resource: { buffer: this.view } }, { binding: 1, resource: t.createView() }] })),
    ];
    this.clearTrails();
  }

  clearTrails() {
    const enc = this.device.createCommandEncoder();
    for (const t of this.trails) enc.beginRenderPass({ colorAttachments: [{ view: t.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] }).end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Encodes tracer advection (compute) and all drawing for one frame; steps is the lattice steps since the last frame. */
  encode(enc: GPUCommandEncoder, v: ViewSettings, steps: number) {
    const canvas = this.context.canvas as HTMLCanvasElement;
    const buf = new ArrayBuffer(VIEW_BYTES);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f.set([canvas.width, canvas.height]);
    u[2] = v.mode;
    f[3] = v.contrast;
    f[4] = v.uRef;
    f[5] = v.trailFade;
    u[6] = v.tracers;
    u[7] = this.frame++;
    f[8] = steps;
    u[9] = this.count;
    this.device.queue.writeBuffer(this.view, 0, buf);

    const tracing = v.tracers !== TracerMode.Off && this.trails.length === 2;
    if (tracing) {
      const cp = enc.beginComputePass();
      cp.setPipeline(this.advectPipe);
      cp.setBindGroup(0, this.tracerBG);
      cp.dispatchWorkgroups(Math.ceil(this.count / TRACER_WG));
      cp.end();
      const next = this.trailIdx ^ 1;
      const tp = enc.beginRenderPass({ colorAttachments: [{ view: this.trails[next].createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0] }] });
      tp.setPipeline(this.fadePipe);
      tp.setBindGroup(0, this.trailBG[this.trailIdx]);
      tp.draw(3);
      tp.setPipeline(this.linePipe);
      tp.setBindGroup(0, this.lineBG);
      tp.draw(this.count * 2);
      tp.end();
      this.trailIdx = next;
    }

    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.fieldPipe);
    pass.setBindGroup(0, this.fieldBG);
    pass.draw(3);
    if (tracing) {
      pass.setPipeline(this.compositePipe);
      pass.setBindGroup(0, this.trailBG[2 + this.trailIdx]);
      pass.draw(3);
    }
    pass.end();
  }
}
