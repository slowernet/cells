import type { Solver3D } from './solver3d';
import { VIEW3_BYTES, packView3, type View3 } from './view3d';
import { outlineShader, obstacleShader, sliceShader } from './shaders/render3d';

const CLEAR = [0.07, 0.07, 0.08];
const DEPTH_FORMAT: GPUTextureFormat = 'depth24plus';

interface PixelRequest {
  resolve: (n: number) => void;
  buffer?: GPUBuffer;
  width?: number;
  height?: number;
  bytesPerRow?: number;
}

export class Renderer3D {
  private view: GPUBuffer;
  private outlinePipe!: GPURenderPipeline;
  private obstaclePipe!: GPURenderPipeline;
  private slicePipe!: GPURenderPipeline;
  private outlineBG!: GPUBindGroup;
  private obstacleBG!: GPUBindGroup;
  private sliceBG!: GPUBindGroup;
  private depth: GPUTexture | null = null;
  private pixelRequest: PixelRequest | null = null;
  count = 0;

  constructor(
    readonly device: GPUDevice,
    readonly context: GPUCanvasContext,
    readonly format: GPUTextureFormat,
  ) {
    this.view = device.createBuffer({ size: VIEW3_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  }

  async init() {
    const d = this.device;
    const depthStencil = (write: boolean): GPUDepthStencilState => ({ format: DEPTH_FORMAT, depthWriteEnabled: write, depthCompare: 'less' });
    const pipe = (code: string, topology: GPUPrimitiveTopology) => {
      const module = d.createShaderModule({ code });
      return d.createRenderPipelineAsync({
        layout: 'auto',
        vertex: { module, entryPoint: 'vs' },
        fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
        primitive: { topology },
        depthStencil: depthStencil(true),
      });
    };
    [this.outlinePipe, this.obstaclePipe, this.slicePipe] = await Promise.all([
      pipe(outlineShader(), 'line-list'),
      pipe(obstacleShader(), 'triangle-list'),
      pipe(sliceShader(), 'triangle-list'),
    ]);
  }

  /** Binds to a (new) solver's buffers. */
  attach(solver: Solver3D) {
    const d = this.device;
    const bg = (pipe: GPURenderPipeline, buffers: GPUBuffer[]) =>
      d.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    this.outlineBG = bg(this.outlinePipe, [solver.params, this.view]);
    this.obstacleBG = bg(this.obstaclePipe, [solver.params, this.view, solver.sdf]);
    this.sliceBG = bg(this.slicePipe, [solver.params, this.view, solver.macro]);
  }

  /** Recreates the depth buffer at the canvas resolution. */
  resize(width: number, height: number) {
    this.depth?.destroy();
    this.depth = this.device.createTexture({ size: [width, height], format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT });
  }

  /** Records one frame's drawing; call afterSubmit once the encoder is submitted. */
  encode(enc: GPUCommandEncoder, view: View3) {
    view.count = this.count;
    this.device.queue.writeBuffer(this.view, 0, packView3(view));
    const target = this.context.getCurrentTexture();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [...CLEAR, 1] }],
      depthStencilAttachment: { view: this.depth!.createView(), depthLoadOp: 'clear', depthClearValue: 1, depthStoreOp: 'store' },
    });
    pass.setPipeline(this.outlinePipe);
    pass.setBindGroup(0, this.outlineBG);
    pass.draw(24);
    pass.setPipeline(this.obstaclePipe);
    pass.setBindGroup(0, this.obstacleBG);
    pass.draw(3);
    pass.setPipeline(this.slicePipe);
    pass.setBindGroup(0, this.sliceBG);
    pass.draw(6);
    pass.end();

    const req = this.pixelRequest;
    if (req && !req.buffer) {
      const bytesPerRow = Math.ceil((target.width * 4) / 256) * 256;
      req.buffer = this.device.createBuffer({ size: bytesPerRow * target.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      req.width = target.width;
      req.height = target.height;
      req.bytesPerRow = bytesPerRow;
      enc.copyTextureToBuffer({ texture: target }, { buffer: req.buffer, bytesPerRow }, [target.width, target.height]);
    }
  }

  /** Starts any readback recorded by encode; buffers can only be mapped after their commands are submitted. */
  afterSubmit() {
    const req = this.pixelRequest;
    if (!req?.buffer) return;
    this.pixelRequest = null;
    const { buffer, width, height, bytesPerRow } = req as Required<PixelRequest>;
    buffer.mapAsync(GPUMapMode.READ).then(() => {
      const px = new Uint8Array(buffer.getMappedRange());
      const clear = CLEAR.map((c) => Math.round(c * 255));
      if (this.format.startsWith('bgra')) clear.reverse();
      let n = 0;
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
          const i = y * bytesPerRow + x * 4;
          if (Math.abs(px[i] - clear[0]) > 8 || Math.abs(px[i + 1] - clear[1]) > 8 || Math.abs(px[i + 2] - clear[2]) > 8) n++;
        }
      buffer.destroy();
      req.resolve(n);
    });
  }

  /** Resolves with the number of pixels in the next frame that differ from the clear colour. */
  requestPixelCount(): Promise<number> {
    return new Promise((resolve) => (this.pixelRequest = { resolve }));
  }
}
