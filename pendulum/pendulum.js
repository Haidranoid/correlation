/*
 * A pendulum of any number of links: point masses on massless rigid rods, frictionless hinges.
 * Exact equations of motion (from the Lagrangian), integrated with classic fourth-order Runge-Kutta.
 *
 * Angles are measured from straight down; x is to the right and y is down, in rod-length units.
 * Used by pendulum.html in the browser (as window.Pendulum) and by render-svg.mjs in Node (require).
 */
(function (root) {
  'use strict';

  const G = 9.81; // m/s²

  // A system of masses m and rod lengths l. tail[i] is the mass hanging below hinge i.
  function system(m, l) {
    const n = m.length, tail = new Float64Array(n);
    let s = 0;
    for (let i = n - 1; i >= 0; i--) { s += m[i]; tail[i] = s; }
    return { m: Float64Array.from(m), l: Float64Array.from(l), tail };
  }

  function state(theta, omega) {
    return { th: Float64Array.from(theta), om: Float64Array.from(omega || theta.map(() => 0)) };
  }

  // Solves A x = b by Gaussian elimination with partial pivoting; A is an array of rows and is overwritten.
  function solve(A, b) {
    const n = b.length;
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      if (p !== c) { [A[p], A[c]] = [A[c], A[p]]; [b[p], b[c]] = [b[c], b[p]]; }
      for (let r = c + 1; r < n; r++) {
        const f = A[r][c] / A[c][c];
        for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
        b[r] -= f * b[c];
      }
    }
    const x = new Float64Array(n);
    for (let r = n - 1; r >= 0; r--) {
      let s = b[r];
      for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
      x[r] = s / A[r][r];
    }
    return x;
  }

  // Angular accelerations. For each hinge i, with mu_ij the mass below hinge max(i, j):
  //   sum_j mu_ij l_i l_j [cos(th_i - th_j) acc_j + sin(th_i - th_j) om_j²] + g mu_ii l_i sin(th_i) = 0
  function accelerations(th, om, p) {
    const n = th.length, l = p.l, tail = p.tail;
    const A = new Array(n), b = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      A[i] = new Float64Array(n);
      let rhs = -G * tail[i] * l[i] * Math.sin(th[i]);
      for (let j = 0; j < n; j++) {
        const k = tail[i > j ? i : j] * l[i] * l[j], d = th[i] - th[j];
        A[i][j] = k * Math.cos(d);
        rhs -= k * Math.sin(d) * om[j] * om[j];
      }
      b[i] = rhs;
    }
    return solve(A, b);
  }

  // One Runge-Kutta step of length dt, in place.
  function step(s, p, dt) {
    const n = s.th.length, th = s.th, om = s.om;
    const t2 = new Float64Array(n), o2 = new Float64Array(n);
    const t3 = new Float64Array(n), o3 = new Float64Array(n);
    const t4 = new Float64Array(n), o4 = new Float64Array(n);
    const a1 = accelerations(th, om, p);
    for (let i = 0; i < n; i++) { t2[i] = th[i] + 0.5 * dt * om[i]; o2[i] = om[i] + 0.5 * dt * a1[i]; }
    const a2 = accelerations(t2, o2, p);
    for (let i = 0; i < n; i++) { t3[i] = th[i] + 0.5 * dt * o2[i]; o3[i] = om[i] + 0.5 * dt * a2[i]; }
    const a3 = accelerations(t3, o3, p);
    for (let i = 0; i < n; i++) { t4[i] = th[i] + dt * o3[i]; o4[i] = om[i] + dt * a3[i]; }
    const a4 = accelerations(t4, o4, p);
    for (let i = 0; i < n; i++) {
      th[i] += (dt / 6) * (om[i] + 2 * o2[i] + 2 * o3[i] + o4[i]);
      om[i] += (dt / 6) * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]);
    }
  }

  // Bob positions [x1, y1, x2, y2, ...], with the pivot at the origin.
  function positions(s, p) {
    const n = s.th.length, out = new Float64Array(2 * n);
    let x = 0, y = 0;
    for (let i = 0; i < n; i++) {
      x += p.l[i] * Math.sin(s.th[i]);
      y += p.l[i] * Math.cos(s.th[i]);
      out[2 * i] = x; out[2 * i + 1] = y;
    }
    return out;
  }

  // Total energy, kinetic plus potential. It is constant for the true motion, so its drift measures integration error.
  function energy(s, p) {
    const n = s.th.length, l = p.l, tail = p.tail, th = s.th, om = s.om;
    let kin = 0, pot = 0;
    for (let i = 0; i < n; i++) {
      pot -= G * tail[i] * l[i] * Math.cos(th[i]);
      for (let j = 0; j < n; j++) kin += 0.5 * tail[i > j ? i : j] * l[i] * l[j] * Math.cos(th[i] - th[j]) * om[i] * om[j];
    }
    return kin + pot;
  }

  const api = { G, system, state, step, positions, energy, accelerations };
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Pendulum = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
