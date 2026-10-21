// Renders pendulum.svg: three identical triple pendulums started a billionth of a radian apart.
// The animation is SMIL inside the SVG, so it plays in a README on GitHub, which strips JavaScript.
// The physics is pendulum.js, the same code the interactive pendulum.html runs.
//
//   node render-svg.mjs        (prints the divergence time, doubling time and energy drift it measured)

import { createRequire } from 'node:module';
import { writeFileSync } from 'node:fs';

const require = createRequire(import.meta.url);
const P = require('./pendulum.js');

const THETA = [1.9, 2.2, 2.4];   // starting angles, radians from straight down (a big, chaotic swing)
const EPS = 1e-9;                // the copies differ by this much in the last rod's angle
const DT = 0.0005;               // integration step, seconds
const DURATION = 30;             // seconds, then the animation loops
const FPS = 30;
const VISIBLE = 0.15;            // tips this far apart (rod lengths) count as visibly different

const COLORS = ['#58a6ff', '#f778ba', '#e3b341'];
const W = 500, H = 560, CX = 250, CY = 236, SCALE = 68;

const sys = P.system([1, 1, 1], [1, 1, 1]);
const copies = [0, EPS, -EPS].map(e => P.state([THETA[0], THETA[1], THETA[2] + e]));
const e0 = copies.map(s => P.energy(s, sys));

const frames = copies.map(() => []);
const stepsPerFrame = Math.round(1 / (FPS * DT));
const nFrames = DURATION * FPS;
let tDiverge = null, drift = 0;
const logSep = [];

for (let f = 0; f <= nFrames; f++) {
  if (f > 0) for (let k = 0; k < stepsPerFrame; k++) for (const s of copies) P.step(s, sys, DT);
  const pos = copies.map(s => P.positions(s, sys));
  pos.forEach((p, i) => frames[i].push(p));
  let sep = 0;
  for (let a = 0; a < 3; a++) for (let b = a + 1; b < 3; b++) sep = Math.max(sep, Math.hypot(pos[a][4] - pos[b][4], pos[a][5] - pos[b][5]));
  const t = f / FPS;
  logSep.push([t, Math.log(Math.max(sep, 1e-300))]);
  if (tDiverge === null && sep > VISIBLE) tDiverge = t;
  copies.forEach((s, i) => { drift = Math.max(drift, Math.abs((P.energy(s, sys) - e0[i]) / e0[i])); });
}

// Growth rate of the separation while it is small: the slope of log(separation) between 1e-7 and 1e-2.
const pts = logSep.filter(([, l]) => l > Math.log(1e-7) && l < Math.log(1e-2));
const mt = pts.reduce((a, p) => a + p[0], 0) / pts.length, ml = pts.reduce((a, p) => a + p[1], 0) / pts.length;
const lambda = pts.reduce((a, p) => a + (p[0] - mt) * (p[1] - ml), 0) / pts.reduce((a, p) => a + (p[0] - mt) ** 2, 0);

const px = (x) => (CX + SCALE * x).toFixed(1);
const py = (y) => (CY + SCALE * y).toFixed(1);
const keyTime = (f) => (f / nFrames).toFixed(4);

// One polyline per pendulum (pivot and three bobs), its points animated frame by frame.
function pendulum(i) {
  const values = frames[i].map(p => `${CX},${CY} ${px(p[0])},${py(p[1])} ${px(p[2])},${py(p[3])} ${px(p[4])},${py(p[5])}`);
  return `<polyline points="${values[0]}" fill="none" stroke="${COLORS[i]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" stroke-opacity="0.9" marker-mid="url(#bob${i})" marker-end="url(#bob${i})">
    <animate attributeName="points" dur="${DURATION}s" repeatCount="indefinite" values="${values.join(';')}"/>
  </polyline>`;
}

// The tip's path, revealed as it is travelled: the dash offset follows the cumulative arc length.
function trail(i) {
  const f = frames[i];
  const d = f.map((p, k) => `${k ? 'L' : 'M'}${px(p[4])} ${py(p[5])}`).join('');
  const cum = [0];
  for (let k = 1; k < f.length; k++) cum.push(cum[k - 1] + Math.hypot(f[k][4] - f[k - 1][4], f[k][5] - f[k - 1][5]));
  const total = cum[cum.length - 1];
  const values = cum.map(c => (1000 * (1 - c / total)).toFixed(1)).join(';');
  const keyTimes = cum.map((_, k) => keyTime(k)).join(';');
  return `<path d="${d}" pathLength="1000" fill="none" stroke="${COLORS[i]}" stroke-width="1.3" stroke-opacity="0.55" stroke-dasharray="1000 1000" stroke-dashoffset="1000">
    <animate attributeName="stroke-dashoffset" dur="${DURATION}s" repeatCount="indefinite" values="${values}" keyTimes="${keyTimes}"/>
  </path>`;
}

// Text visible from second a to second b of each loop.
function during(a, b, inner) {
  const ka = (a / DURATION).toFixed(4), kb = (b / DURATION).toFixed(4);
  const anim = a === 0
    ? `<animate attributeName="opacity" dur="${DURATION}s" repeatCount="indefinite" calcMode="discrete" values="1;0" keyTimes="0;${kb}"/>`
    : `<animate attributeName="opacity" dur="${DURATION}s" repeatCount="indefinite" calcMode="discrete" values="0;1;0" keyTimes="0;${ka};${kb}"/>`;
  return `<g opacity="${a === 0 ? 1 : 0}">${anim}${inner}</g>`;
}

const split = Math.round(tDiverge);
const clock = Array.from({ length: DURATION }, (_, s) =>
  during(s, s + 1, `<text x="22" y="34" class="muted">t = ${s} s</text>`)).join('\n  ');

const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-labelledby="title desc">
  <title id="title">Three triple pendulums started a billionth of a radian apart</title>
  <desc id="desc">Three identical triple pendulums move as one for about ${split} seconds, then follow completely different paths, while their rods and energy stay exactly the same.</desc>
  <style>
    text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; fill: #e6edf3; }
    .muted { fill: #8b949e; font-size: 14px; }
    .lead { font-size: 17px; font-weight: 600; }
    .sub { font-size: 14px; fill: #8b949e; }
  </style>
  <defs>
    ${COLORS.map((c, i) => `<marker id="bob${i}" viewBox="-1 -1 2 2" markerWidth="13" markerHeight="13" markerUnits="userSpaceOnUse" refX="0" refY="0"><circle r="1" fill="${c}"/></marker>`).join('\n    ')}
  </defs>
  <rect width="${W}" height="${H}" rx="12" fill="#0d1117"/>
  <circle cx="${CX}" cy="${CY}" r="${3 * SCALE}" fill="none" stroke="#30363d" stroke-dasharray="4 6"/>
  <text x="${CX}" y="${CY + 3 * SCALE + 18}" text-anchor="middle" class="muted">how far the tip can reach: always known</text>
  ${[0, 1, 2].map(trail).join('\n  ')}
  ${[0, 1, 2].map(pendulum).join('\n  ')}
  <circle cx="${CX}" cy="${CY}" r="4.5" fill="#e6edf3"/>
  ${clock}
  ${during(0, split, `<text x="${CX}" y="${H - 48}" text-anchor="middle" class="lead">Same rods, same physics, a billionth of a radian apart</text>
    <text x="${CX}" y="${H - 24}" text-anchor="middle" class="sub">For now the three move as one.</text>`)}
  ${during(split, DURATION, `<text x="${CX}" y="${H - 48}" text-anchor="middle" class="lead">After ${split} seconds they have nothing in common</text>
    <text x="${CX}" y="${H - 24}" text-anchor="middle" class="sub">Rods intact, energy unchanged. Only the future is gone.</text>`)}
</svg>
`;

writeFileSync(new URL('./pendulum.svg', import.meta.url), svg);
console.log(`visibly apart (${VISIBLE} rod lengths) at ${tDiverge.toFixed(1)} s`);
console.log(`separation grows e^(${lambda.toFixed(2)} t): it doubles every ${(Math.LN2 / lambda).toFixed(2)} s`);
console.log(`largest relative energy drift: ${drift.toExponential(1)}`);
console.log(`pendulum.svg: ${(svg.length / 1024).toFixed(0)} KB`);
