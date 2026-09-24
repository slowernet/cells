import { initGpu } from './gpu';
import { CASES, runCase, CaseResult } from './cases';

const out = document.getElementById('out')!;
const log = (s: string) => {
  out.textContent += s + '\n';
  console.log(s);
};

function formatResult(r: CaseResult): string {
  const lines = [`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}  (${r.steps} steps, ${r.cells} cells, ${r.seconds.toFixed(1)} s)`];
  for (const m of r.metrics) {
    const ref = m.ref[0] === m.ref[1] ? `${m.ref[0]}` : `${m.ref[0]}..${m.ref[1]}`;
    lines.push(`  ${m.pass ? 'ok  ' : 'MISS'} ${m.name.padEnd(28)} ${m.value.toPrecision(5).padStart(11)}   ref ${ref}   accept ${m.accept[0].toPrecision(4)}..${m.accept[1].toPrecision(4)}`);
  }
  for (const n of r.notes) lines.push(`  note: ${n}`);
  return lines.join('\n');
}

const gpu = await initGpu();
const w = window as unknown as Record<string, unknown>;
w.caseNames = Object.keys(CASES);
w.runCase = async (key: string) => {
  const r = await runCase(gpu.device, key, log);
  log(formatResult(r));
  return r;
};
w.validateReady = true;

document.getElementById('all')!.onclick = async () => {
  for (const k of Object.keys(CASES)) await (w.runCase as (k: string) => Promise<CaseResult>)(k);
};

