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
  for (const d of Array.from(panel.querySelectorAll('details'))) d.addEventListener('toggle', onVisibility);
}

/** True when the element is inside an open panel and all its enclosing groups are open. */
export function isShown(el: Element): boolean {
  const panel = document.getElementById('panel');
  if (panel && panel.contains(el) && !panel.classList.contains('open')) return false;
  for (let d = el.closest('details'); d; d = d.parentElement?.closest('details') ?? null) if (!d.open) return false;
  return true;
}
