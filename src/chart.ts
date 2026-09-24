/** Two stacked single-series traces (C_D above C_L) on a shared step axis, with a crosshair readout. */

const SURFACE = '#1a1a19';
const SERIES = '#3987e5';
const TEXT = '#ffffff';
const TEXT_2 = '#c3c2b7';
const GRID = 'rgba(195, 194, 183, 0.14)';
const MAX_POINTS = 6000;

export class ForceChart {
  private steps: number[] = [];
  private cd: number[] = [];
  private cl: number[] = [];
  private hoverX: number | null = null;
  private readonly ctx: CanvasRenderingContext2D;
  private dirty = true;

  constructor(
    readonly canvas: HTMLCanvasElement,
    private readonly tooltip: HTMLElement,
  ) {
    this.ctx = canvas.getContext('2d')!;
    canvas.addEventListener('pointermove', (e) => {
      this.hoverX = e.offsetX;
      this.dirty = true;
    });
    canvas.addEventListener('pointerleave', () => {
      this.hoverX = null;
      this.tooltip.hidden = true;
      this.dirty = true;
    });
  }

  clear() {
    this.steps = [];
    this.cd = [];
    this.cl = [];
    this.dirty = true;
  }

  push(step: number, cd: number, cl: number) {
    this.steps.push(step);
    this.cd.push(cd);
    this.cl.push(cl);
    if (this.steps.length > MAX_POINTS) {
      const drop = this.steps.length - MAX_POINTS;
      this.steps.splice(0, drop);
      this.cd.splice(0, drop);
      this.cl.splice(0, drop);
    }
    this.dirty = true;
  }

  get data() {
    return { steps: this.steps, cd: this.cd, cl: this.cl };
  }

  draw() {
    if (!this.dirty) return;
    this.dirty = false;
    const dpr = devicePixelRatio;
    const cssW = this.canvas.clientWidth;
    const cssH = this.canvas.clientHeight;
    if (this.canvas.width !== Math.round(cssW * dpr) || this.canvas.height !== Math.round(cssH * dpr)) {
      this.canvas.width = Math.round(cssW * dpr);
      this.canvas.height = Math.round(cssH * dpr);
    }
    const c = this.ctx;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.fillStyle = SURFACE;
    c.fillRect(0, 0, cssW, cssH);
    const left = 44, right = 8, gap = 18, top = 16;
    const panelH = (cssH - top - gap - 18) / 2;
    const n = this.steps.length;
    const x = (k: number) => left + ((this.steps[k] - this.steps[0]) / Math.max(1, this.steps[n - 1] - this.steps[0])) * (cssW - left - right);
    const panels: [string, number[], number][] = [
      ['C_D', this.cd, top],
      ['C_L', this.cl, top + panelH + gap],
    ];
    let hoverK = -1;
    if (this.hoverX !== null && n > 1) {
      const t = this.steps[0] + ((this.hoverX - left) / (cssW - left - right)) * (this.steps[n - 1] - this.steps[0]);
      let lo = 0, hi = n - 1;
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (this.steps[mid] < t) lo = mid; else hi = mid;
      }
      hoverK = Math.abs(this.steps[lo] - t) < Math.abs(this.steps[hi] - t) ? lo : hi;
    }
    c.font = '11px ui-sans-serif, system-ui, sans-serif';
    for (const [label, v, y0] of panels) {
      c.fillStyle = TEXT;
      c.textBaseline = 'bottom';
      c.textAlign = 'left';
      c.fillText(label, left, y0 - 2);
      if (n > 0) {
        c.fillStyle = TEXT_2;
        c.textAlign = 'right';
        c.fillText(v[n - 1].toFixed(3), cssW - right, y0 - 2);
      }
      if (n < 2) continue;
      // Robust range: ignore the start-up transient that would flatten the rest of the trace.
      const tail = v.slice(Math.floor(n * 0.2));
      let lo = Math.min(...tail), hi = Math.max(...tail);
      if (hi - lo < 1e-6) { lo -= 0.05; hi += 0.05; }
      const pad = (hi - lo) * 0.1;
      lo -= pad; hi += pad;
      const y = (val: number) => y0 + panelH - ((Math.min(hi, Math.max(lo, val)) - lo) / (hi - lo)) * panelH;
      c.strokeStyle = GRID;
      c.lineWidth = 1;
      c.fillStyle = TEXT_2;
      c.textBaseline = 'middle';
      c.textAlign = 'right';
      for (const g of [lo + pad, (lo + hi) / 2, hi - pad]) {
        c.beginPath();
        c.moveTo(left, Math.round(y(g)) + 0.5);
        c.lineTo(cssW - right, Math.round(y(g)) + 0.5);
        c.stroke();
        c.fillText(g.toFixed(Math.abs(hi - lo) < 0.1 ? 3 : 2), left - 6, y(g));
      }
      c.strokeStyle = SERIES;
      c.lineWidth = 2;
      c.lineJoin = 'round';
      c.beginPath();
      const stride = Math.max(1, Math.floor(n / (cssW * 2)));
      for (let k = 0; k < n; k += stride) (k === 0 ? c.moveTo : c.lineTo).call(c, x(k), y(v[k]));
      c.lineTo(x(n - 1), y(v[n - 1]));
      c.stroke();
      if (hoverK >= 0) {
        c.fillStyle = SERIES;
        c.beginPath();
        c.arc(x(hoverK), y(v[hoverK]), 4, 0, Math.PI * 2);
        c.fill();
        c.strokeStyle = SURFACE;
        c.lineWidth = 2;
        c.stroke();
      }
    }
    c.fillStyle = TEXT_2;
    c.textAlign = 'center';
    c.textBaseline = 'bottom';
    if (n > 1) c.fillText(`step ${this.steps[0]} to ${this.steps[n - 1]}`, (left + cssW) / 2, cssH - 2);
    if (hoverK >= 0) {
      c.strokeStyle = 'rgba(255,255,255,0.35)';
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(Math.round(x(hoverK)) + 0.5, top);
      c.lineTo(Math.round(x(hoverK)) + 0.5, cssH - 18);
      c.stroke();
      this.tooltip.hidden = false;
      this.tooltip.textContent = `step ${this.steps[hoverK]}   C_D ${this.cd[hoverK].toFixed(4)}   C_L ${this.cl[hoverK].toFixed(4)}`;
    }
  }
}
