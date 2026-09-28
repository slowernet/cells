import { icon, type IconName } from './icons';

/** Fills [data-icon] placeholders and wires the overlay menu; onVisibility runs when the panel or a group opens or closes. */
export function initMenu(onVisibility: () => void = () => {}) {
  for (const el of Array.from(document.querySelectorAll<HTMLElement>('[data-icon]'))) el.innerHTML = icon(el.dataset.icon as IconName);
  const button = document.querySelector<HTMLButtonElement>('.menu-button')!;
  const panel = document.getElementById('panel')!;
  const glyph = button.querySelector<HTMLElement>('[data-icon]')!;

  const set = (open: boolean) => {
    panel.classList.toggle('open', open);
    button.setAttribute('aria-expanded', String(open));
    button.setAttribute('aria-label', open ? 'Close settings' : 'Open settings');
    glyph.innerHTML = icon(open ? 'x' : 'menu');
    if (open) panel.focus({ preventScroll: true });
    else if (panel.contains(document.activeElement)) button.focus();
    onVisibility();
  };

  button.addEventListener('click', () => set(!panel.classList.contains('open')));
  addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && panel.classList.contains('open')) set(false);
  });
  // A press outside the open panel closes it and is swallowed until release, so it neither draws nor orbits.
  let swallowing = false;
  const swallow = (e: Event) => {
    if (!swallowing) return;
    e.stopPropagation();
    if (e.type === 'pointerup' || e.type === 'pointercancel') swallowing = false;
  };
  addEventListener(
    'pointerdown',
    (e) => {
      const t = e.target as Node;
      if (!panel.classList.contains('open') || panel.contains(t) || button.contains(t)) return;
      e.stopPropagation();
      e.preventDefault();
      swallowing = true;
      set(false);
    },
    { capture: true },
  );
  for (const type of ['pointermove', 'pointerup', 'pointercancel']) addEventListener(type, swallow, { capture: true });
  for (const d of Array.from(panel.querySelectorAll('details'))) d.addEventListener('toggle', onVisibility);
}

/** True when the element is inside an open panel and all its enclosing groups are open. */
export function isShown(el: Element): boolean {
  const panel = document.getElementById('panel');
  if (panel && panel.contains(el) && !panel.classList.contains('open')) return false;
  for (let d = el.closest('details'); d; d = d.parentElement?.closest('details') ?? null) if (!d.open) return false;
  return true;
}

let toastTimer = 0;

/** Shows a message at the top centre for 8 s, or until clicked when persist is set. */
export function showToast(text: string, persist = false) {
  let el = document.querySelector<HTMLElement>('.toast');
  if (!el) {
    el = document.createElement('div');
    el.className = 'toast';
    el.setAttribute('role', 'status');
    el.addEventListener('click', () => (el!.hidden = true));
    document.body.append(el);
  }
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  if (!persist) toastTimer = window.setTimeout(() => (el!.hidden = true), 8000);
}
