"""Gravity: the compiled kernels (numba).

Arrays are the panel's: lnC, lnH, lnL float64 [R, T] (NaN where the price is missing or fails the bad-print rule).
A spread of B on attractor A at lag tau is s_tau(t') = lnC_B(t') - beta * lnC_A(t' - tau). Its mean and sd (ddof 1) come
from t' = t-63..t-1 where both closes exist (>= 57 days); the gaps at t use B's close, high and low with A's close at
t - tau; the closest approach r is 0 when B's range reached the attractor (g_L <= 0 <= g_H), else min(|g_L|, |g_H|).
"""
import math

import numpy as np
from numba import njit, prange

SPR_W = 63
SPR_MIN = 57


@njit(cache=True)
def _gaps_one(lnC, lnH, lnL, b, a, beta, tau, ta, tb, gC, gR):
    """One pair (b, a) at one lag: fills gC[0..tb-ta) and gR (closest approach) for days ta..tb-1 (NaN if invalid).
    Window sums are prefix sums of x - x_ref (x_ref = the first valid value) in float64."""
    L = tb - ta + SPR_W
    t_first = ta - SPR_W
    ps = np.zeros(L + 1)
    ps2 = np.zeros(L + 1)
    pn = np.zeros(L + 1)
    xref = np.nan
    for i in range(L):
        t = t_first + i
        x = np.nan
        if t - tau >= 0 and t >= 0:
            cb = lnC[b, t]
            ca = lnC[a, t - tau]
            if not (math.isnan(cb) or math.isnan(ca)):
                x = cb - beta * ca
        if not math.isnan(x):
            if math.isnan(xref):
                xref = x
            d = x - xref
            ps[i + 1] = ps[i] + d
            ps2[i + 1] = ps2[i] + d * d
            pn[i + 1] = pn[i] + 1.0
        else:
            ps[i + 1] = ps[i]
            ps2[i + 1] = ps2[i]
            pn[i + 1] = pn[i]
    for j in range(tb - ta):
        gC[j] = np.nan
        gR[j] = np.nan
        t = ta + j
        i1 = j + SPR_W          # prefix index of t (exclusive end of the window t-63..t-1)
        i0 = j
        n = pn[i1] - pn[i0]
        if n < SPR_MIN:
            continue
        s1 = ps[i1] - ps[i0]
        s2 = ps2[i1] - ps2[i0]
        mu = s1 / n
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        if not (var > 0.0):
            continue
        sd = math.sqrt(var)
        if t - tau < 0:
            continue
        ca = lnC[a, t - tau]
        cb = lnC[b, t]
        hb = lnH[b, t]
        lb = lnL[b, t]
        if math.isnan(ca) or math.isnan(cb) or math.isnan(hb) or math.isnan(lb):
            continue
        base = beta * ca + xref + mu
        g_c = (cb - base) / sd
        g_h = (hb - base) / sd
        g_l = (lb - base) / sd
        gC[j] = g_c
        if g_l <= 0.0 <= g_h:
            gR[j] = 0.0
        else:
            gR[j] = min(abs(g_l), abs(g_h))


@njit(parallel=True, cache=True)
def gaps(lnC, lnH, lnL, Bv, Av, betav, tau, ta, tb, gC, gR):
    """Many pairs at one lag: Bv, Av int64 [P], betav float64 [P]; gC, gR float64 [P, tb - ta] (written)."""
    P = len(Bv)
    for p in prange(P):
        if Av[p] < 0 or math.isnan(betav[p]):
            for j in range(tb - ta):
                gC[p, j] = np.nan
                gR[p, j] = np.nan
            continue
        _gaps_one(lnC, lnH, lnL, Bv[p], Av[p], betav[p], tau, ta, tb, gC[p], gR[p])


@njit(parallel=True, cache=True)
def g3_accum(lnC, lnH, lnL, Bv, att, mass, betas, ta, tb, Wk, eps, ps, wmin, F, cnt):
    """G3's force for the stocks Bv [nB] on days ta..tb-1.
    att, mass, betas [nB, K]; Wk [nk, TAU_MAX + 1] kernel weights by tau (0 where unused); ps [np] exponents.
    For each attractor: acc[q, p, v] = sum over tau of Wk[q, tau] * sign(g_C,tau) / (r + eps)^p, with r the closest
    approach (v = 0) or |g_C,tau| (v = 1, closes only); wsum[q] = the weight of the taus with data. An attractor counts
    for kernel q when wsum[q] >= wmin: F[b, j, q, p, v] += -mass * acc; cnt[b, j, q] += 1.
    F [nB, nd, nk, np, 2] and cnt [nB, nd, nk] must be zeroed by the caller."""
    nB = len(Bv)
    nk = Wk.shape[0]
    npw = len(ps)
    nd = tb - ta
    taumax = Wk.shape[1] - 1
    for ib in prange(nB):
        b = Bv[ib]
        gC = np.empty(nd)
        gR = np.empty(nd)
        acc = np.zeros((nd, nk, npw, 2))
        wsum = np.zeros((nd, nk))
        for s in range(att.shape[1]):
            a = att[ib, s]
            m = mass[ib, s]
            be = betas[ib, s]
            if a < 0 or math.isnan(m) or math.isnan(be):
                continue
            acc[:] = 0.0
            wsum[:] = 0.0
            for tau in range(1, taumax + 1):
                used = False
                for q in range(nk):
                    if Wk[q, tau] > 0.0:
                        used = True
                if not used:
                    continue
                _gaps_one(lnC, lnH, lnL, b, a, be, tau, ta, tb, gC, gR)
                for j in range(nd):
                    g = gC[j]
                    if math.isnan(g):
                        continue
                    sg = 1.0 if g > 0 else (-1.0 if g < 0 else 0.0)
                    r = gR[j]
                    ag = abs(g)
                    for q in range(nk):
                        w = Wk[q, tau]
                        if w == 0.0:
                            continue
                        wsum[j, q] += w
                        for pi in range(npw):
                            acc[j, q, pi, 0] += w * sg / (r + eps) ** ps[pi]
                            acc[j, q, pi, 1] += w * sg / (ag + eps) ** ps[pi]
            for j in range(nd):
                for q in range(nk):
                    if wsum[j, q] >= wmin:
                        cnt[ib, j, q] += 1
                        for pi in range(npw):
                            F[ib, j, q, pi, 0] += -m * acc[j, q, pi, 0]
                            F[ib, j, q, pi, 1] += -m * acc[j, q, pi, 1]


@njit(parallel=True, cache=True)
def rank_center(X, M):
    """Per column (day): average ranks of X[:, t] among rows with M[:, t], centred to -0.5..+0.5
    ((rank - 1) / (n - 1) - 0.5; a single row gets 0). NaN elsewhere. X float64 [R, T], M bool [R, T]."""
    R, T = X.shape
    out = np.full((R, T), np.nan)
    for t in prange(T):
        idx = np.empty(R, np.int64)
        n = 0
        for i in range(R):
            if M[i, t]:
                idx[n] = i
                n += 1
        if n == 0:
            continue
        if n == 1:
            out[idx[0], t] = 0.0
            continue
        vals = np.empty(n)
        for k in range(n):
            vals[k] = X[idx[k], t]
        order = np.argsort(vals, kind='mergesort')
        k = 0
        while k < n:
            k2 = k
            while k2 + 1 < n and vals[order[k2 + 1]] == vals[order[k]]:
                k2 += 1
            rk = 0.5 * (k + k2)          # 0-based average rank
            for q in range(k, k2 + 1):
                out[idx[order[q]], t] = rk / (n - 1) - 0.5
            k = k2 + 1
    return out


@njit(cache=True)
def book(F, M, rets, ta, tb, frac_div):
    """The money test's book: each day t in ta..tb-1, among rows with M[:, t], long the top fifth by F (n // 5 names)
    and short the bottom fifth, equal weight, $1 a side. rets [nE, R, T]: each execution's return for a position
    decided at the close of t (NaN = missing; a leg's return is the mean over its names with a return).
    Returns gross [nE, T] (NaN where no book), turnover [T] (sum |w_t - w_{t-1}| over all rows) and names a side [T]."""
    nE, R, T = rets.shape
    gross = np.full((nE, T), np.nan)
    turn = np.full(T, np.nan)
    nside = np.zeros(T, np.int64)
    wprev = np.zeros(R)
    wnew = np.zeros(R)
    idx = np.empty(R, np.int64)
    vals = np.empty(R)
    for t in range(ta, tb):
        n = 0
        for i in range(R):
            if M[i, t]:
                idx[n] = i
                vals[n] = F[i, t]
                n += 1
        ns = n // frac_div
        for i in range(R):
            wnew[i] = 0.0
        if ns >= 1:
            order = np.argsort(vals[:n], kind='mergesort')
            for q in range(ns):
                wnew[idx[order[n - 1 - q]]] = 1.0 / ns
                wnew[idx[order[q]]] = -1.0 / ns
            for e in range(nE):
                sl, nl, ss, nsh = 0.0, 0, 0.0, 0
                for q in range(ns):
                    rl = rets[e, idx[order[n - 1 - q]], t]
                    if not math.isnan(rl):
                        sl += rl
                        nl += 1
                    rs = rets[e, idx[order[q]], t]
                    if not math.isnan(rs):
                        ss += rs
                        nsh += 1
                if nl > 0 and nsh > 0:
                    gross[e, t] = sl / nl - ss / nsh
        tv = 0.0
        for i in range(R):
            tv += abs(wnew[i] - wprev[i])
            wprev[i] = wnew[i]
        turn[t] = tv if ns >= 1 else np.nan
        nside[t] = ns
    return gross, turn, nside
