const tip = document.querySelector('.tooltip');
function show(el, x, y) {
  tip.replaceChildren();
  const v = document.createElement('strong'); v.textContent = el.dataset.value;
  const l = document.createElement('span'); l.textContent = el.dataset.label;
  tip.append(v, l); tip.style.display = 'block';
  const w = tip.offsetWidth; tip.style.left = Math.min(x + 12, window.innerWidth - w - 8) + 'px'; tip.style.top = (y - 48) + 'px';
}
document.querySelectorAll('svg .hit').forEach(el => {
  el.addEventListener('pointermove', e => show(el, e.clientX, e.clientY));
  el.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
  el.addEventListener('focus', () => { const r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top + 40); });
  el.addEventListener('blur', () => { tip.style.display = 'none'; });
});
