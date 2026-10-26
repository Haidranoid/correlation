"""Gravity, step 5: the measured force law (G1) and the measured time kernel (G2). Reported, not tests.

    python -I -B shape.py old|new

Force law: every (B, A) attractor pair-day of G1 (real and random attractors) falls in a bin of its closest approach r:
0, (0, 0.25], (0.25, 0.5], (0.5, 1], (1, 1.5], (1.5, 2], (2, 3], above 3. For each bin: the mean over pair-days of B's
next open-to-open return in the attractor's direction (-sign(g_C)), divided by B's sigma (through t), for real minus
random attractors, with date-block bootstrap intervals. The same with r = |g_C| (closes only) is reported. Per bin the
mean r and mean |g_C| are kept, so each law's predicted shape (m / (r + eps)^p; the spring's |g_C|) can be drawn.

Kernel: for each N, with the flat kernel's (a = 0) attractors: each day, the cross-sectional slope (with an intercept)
of z_B(t + 1) on sign(m) z_A(t + 1 - tau) over all (B, A) pairs, for tau = 1..2N-1, real minus random attractors,
averaged over days. The unsigned slope (on z_A itself, the plan's literal wording) is reported beside it.
Writes out/shape_<tag>.json / .txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
import forces  # noqa: E402
import tests  # noqa: E402
np = K.np

EDGES = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
BIN_NAMES = ['0', '(0, 0.25]', '(0.25, 0.5]', '(0.5, 1]', '(1, 1.5]', '(1.5, 2]', '(2, 3]', '> 3']
NB = 8


def bin_of(r):
    b = np.full(r.shape, -1, np.int64)
    ok = np.isfinite(r)
    b[ok & (r == 0)] = 0
    for i in range(1, len(EDGES)):
        lo = EDGES[i - 1]
        hi = EDGES[i]
        b[ok & (r > lo) & (r <= hi)] = i
    b[ok & (r > EDGES[-1])] = NB - 1
    return b


def force_law(tag, P, ta, tb, fh):
    """Per date and bin: sums and counts of y for real and random, both distance definitions; plus pooled bin means of
    r, |g_C| and the predicted pulls."""
    rets, _, _ = tests.returns(P)
    sig = np.asarray(P['sig'], np.float64)
    with np.errstate(all='ignore'):
        yv = rets[0] / sig                         # next open-to-open return in units of sigma through t
    del rets
    nd = tb - ta
    out = {}
    pe = [(p, e) for p, e in K.G1_PE if p > 0]
    for world in ('real', 'random'):
        gC = np.load(K.BIG / f'pairs_{tag}_{world}_gC.npy', mmap_mode='r')
        gR = np.load(K.BIG / f'pairs_{tag}_{world}_r.npy', mmap_mode='r')
        for dist in ('hl', 'close'):
            sums = np.zeros((nd, NB))
            cnts = np.zeros((nd, NB))
            mr = np.zeros(NB)
            mg = np.zeros(NB)
            pred = np.zeros((len(pe), NB))
            for c0 in range(ta, tb, 400):
                c1 = min(tb, c0 + 400)
                g = np.asarray(gC[:, :, c0:c1], np.float64)
                r = np.asarray(gR[:, :, c0:c1], np.float64) if dist == 'hl' else np.abs(g)
                y = -np.sign(g) * yv[:, None, c0:c1]
                ok = np.isfinite(g) & np.isfinite(y)
                bins = bin_of(np.where(ok, r, np.nan))
                for b in range(NB):
                    m = bins == b
                    sums[c0 - ta:c1 - ta, b] = np.where(m, y, 0.0).sum((0, 1))
                    cnts[c0 - ta:c1 - ta, b] = m.sum((0, 1))
                    mr[b] += r[m].sum()
                    mg[b] += np.abs(g[m]).sum()
                    for j, (p, e) in enumerate(pe):
                        pred[j, b] += (1.0 / (r[m] + e) ** p).sum()
            n = cnts.sum(0)
            out[(world, dist)] = {'sums': sums, 'cnts': cnts, 'mean_r': mr / np.maximum(n, 1), 'mean_absg': mg / np.maximum(n, 1),
                                  'pred': pred / np.maximum(n, 1)[None, :], 'n': n}
    return out, pe


def kernel_shape(tag, P, PE, ta, tb, fh):
    """Per N: daily slopes [4, nd, 2N-1]: real signed, real unsigned, random signed, random unsigned."""
    z = np.asarray(P['z'], np.float64)
    refit = np.asarray(P['refit'])
    res = {}
    for N in (5, 10, 21):
        q = K.G2_ALL.index((N, 0.0))
        L = 2 * N - 1
        sl = np.full((4, tb - ta, L), np.nan)
        for k in range(len(refit)):
            a, b = forces._span(P, k)
            a, b = max(a, ta), min(b, tb)
            if b <= a:
                continue
            sel, rows = forces._rows(P, k, None)
            for wi, (att, m) in enumerate([(PE['A2'][q, k][sel], PE['M2'][q, k][sel]), (PE['AR'][k][sel], PE['M2R'][q, k][sel])]):
                m = np.asarray(m, np.float64)
                good = (att >= 0) & np.isfinite(m)
                Y = z[rows][:, a + 1:b + 1]                                   # z_B(t + 1), [n, nd]
                for tau in range(1, L + 1):
                    ZA = z[:, a + 1 - tau:b + 1 - tau][np.maximum(att, 0)]     # z_A(t + 1 - tau), [n, 10, nd]
                    for si, signed in enumerate((True, False)):
                        x = ZA * np.sign(m)[:, :, None] if signed else ZA
                        yy = np.broadcast_to(Y[:, None, :], x.shape)
                        ok = np.isfinite(x) & np.isfinite(yy) & good[:, :, None]
                        nn = ok.sum((0, 1))
                        x0 = np.where(ok, x, 0.0)
                        y0 = np.where(ok, yy, 0.0)
                        sx, sy = x0.sum((0, 1)), y0.sum((0, 1))
                        sxx, sxy = (x0 * x0).sum((0, 1)), (x0 * y0).sum((0, 1))
                        with np.errstate(all='ignore'):
                            s = (nn * sxy - sx * sy) / (nn * sxx - sx * sx)
                        s[nn < 100] = np.nan
                        sl[2 * wi + si, a - ta:b - ta, tau - 1] = s
        res[N] = sl
        K.log(f'  kernel N={N} done', fh)
    return res


def main(tag):
    fh = open(K.OUT / f'shape_{tag}.txt', 'w')
    P = data.load_panel(tag)
    PE = dict(np.load(K.BIG / f'peers_{tag}.npz'))
    T = P['T']
    ta = tests.first_day(tag, P)
    tb = T - 2
    days = np.asarray(P['days'])[ta:tb]
    fl, pe = force_law(tag, P, ta, tb, fh)
    ks = kernel_shape(tag, P, PE, ta, tb, fh)
    res = {'bins': BIN_NAMES, 'pe': pe, 'periods': {}}
    for name, (a, b) in tests.periods_for(tag).items():
        lo = (np.datetime64(a) - np.datetime64('1970-01-01')).astype(int)
        hi = (np.datetime64(b) - np.datetime64('1970-01-01')).astype(int)
        sel = (days >= lo) & (days < hi)
        if sel.sum() < 30:
            continue
        W = K.boot_weights(int(sel.sum())).astype(np.float64)
        pr = {'first': K.c2eval.iso(days[sel][0]), 'last': K.c2eval.iso(days[sel][-1]), 'days': int(sel.sum())}
        for dist in ('hl', 'close'):
            R_, Q_ = fl[('real', dist)], fl[('random', dist)]
            est_r = R_['sums'][sel].sum(0) / R_['cnts'][sel].sum(0)
            est_q = Q_['sums'][sel].sum(0) / Q_['cnts'][sel].sum(0)
            br = (W @ R_['sums'][sel]) / (W @ R_['cnts'][sel])
            bq = (W @ Q_['sums'][sel]) / (W @ Q_['cnts'][sel])
            rows = []
            for i in range(NB):
                d = K.summarize(est_r[i] - est_q[i], br[:, i] - bq[:, i])
                rows.append({'bin': BIN_NAMES[i], 'real': float(est_r[i]), 'random': float(est_q[i]),
                             'real_se': float(np.std(br[:, i], ddof=1)), 'random_se': float(np.std(bq[:, i], ddof=1)),
                             'diff': d, 'pairs_real': int(R_['cnts'][sel].sum(0)[i]), 'pairs_random': int(Q_['cnts'][sel].sum(0)[i]),
                             'mean_r': float(R_['mean_r'][i]), 'mean_absg': float(R_['mean_absg'][i]),
                             'pred': {f'p={p} eps={e:g}': float(R_['pred'][j, i]) for j, (p, e) in enumerate(pe)}})
            pr[f'force_law_{dist}'] = rows
        kern = {}
        for N, sl in ks.items():
            s = sl[:, sel]
            out = []
            for tau in range(2 * N - 1):
                bmr = K.boot_means(W, s[:, :, tau].T)
                rec = {'tau': tau + 1}
                for j, nm in enumerate(('real_signed', 'real_unsigned', 'random_signed', 'random_unsigned')):
                    rec[nm] = float(np.nanmean(s[j, :, tau]))
                rec['diff_signed'] = K.summarize(rec['real_signed'] - rec['random_signed'], bmr[:, 0] - bmr[:, 2])
                rec['diff_unsigned'] = K.summarize(rec['real_unsigned'] - rec['random_unsigned'], bmr[:, 1] - bmr[:, 3])
                out.append(rec)
            kern[str(N)] = out
        pr['kernel'] = kern
        res['periods'][name] = pr
        K.log(f'--- {name} ({pr["first"]}..{pr["last"]})', fh)
        for row in pr['force_law_hl']:
            d = row['diff']
            K.log(f'  r {row["bin"]:>12}: real {row["real"]:+.5f} random {row["random"]:+.5f} diff {d["est"]:+.5f} '
                  f'(t {d["t"]:+.2f}; 90% {d["ci90"][0]:+.5f}..{d["ci90"][1]:+.5f}); pairs {row["pairs_real"]:,}', fh)
        for N, out in kern.items():
            K.log(f'  kernel N={N}: ' + ' '.join(f'{r["tau"]}:{r["diff_signed"]["est"]:+.4f}({r["diff_signed"]["t"]:+.1f})' for r in out), fh)
    K.jdump(res, K.OUT / f'shape_{tag}.json')
    fh.close()


if __name__ == '__main__':
    main(sys.argv[1])
