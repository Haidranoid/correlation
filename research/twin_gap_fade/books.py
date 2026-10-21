"""Daily intraday books from the rows that build.py writes (shared by tests.py, randoms.py and unseen.py).

A book on day t: $1 long, $1 short, equal weight, entered at the open, out at the close; every stock carries
-beta x its twin from the twin's open to its close (+beta for a short). Returns are simple open-to-close returns.
"""
import sys
from pathlib import Path

sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'
BIG = c2eval.WORK / 'twin_gap_fade'
P_HEAD = 10e-4    # headline entry penalty per side ($1 of each leg)
TWIN_SIDE = 2e-4  # per side on the twin legs: 4 bps a day on the fund notional (in at the open, out at the close)
MIN_N = 20        # skip a day whose ranked set has fewer stocks
FRAC = 0.10
VR_CUT = 2.0      # post hoc volume split: day's volume >= 2x its 60-day median


def load_rows(tag):
    z = np.load(BIG / f'rows_{tag}.npz')
    return {k: z[k] for k in z.files}


def day_slices(t):
    u, start = np.unique(t, return_index=True)
    end = np.append(start[1:], len(t))
    return u, start, end


def fund_cost(f, w):
    """4 bps x the sum over funds of |net fund weight| (w = signed fund weights of the positions)."""
    uf, inv = np.unique(f, return_inverse=True)
    net = np.bincount(inv, weights=w, minlength=len(uf))
    return 2 * TWIN_SIDE * np.abs(net).sum()


SIGNALS = ('twin', 'raw', 'akbas')


def books(R, sub='all', clip=False):
    """Per-day results for the three books on one universe. sub: 'all' or '500'. clip: keep the intraday bad-print
    days with the intraday return clipped at +-0.5 (the sensitivity); otherwise they are dropped.
    Returns dict of arrays over days: t, n, nl, and for each signal s: s_gross, s_tc (netted twin cost),
    s_tcu (un-netted), s_long, s_short (mean hedged return of each leg), s_idbad (legs' positions with an intraday
    bad print); plus the volume split of the twin book: vs_hi, vs_lo (gross long-short within each group, NaN if
    a group misses a leg) and the counts."""
    keep = np.ones(len(R['t']), bool)
    if sub == '500':
        keep &= R['in500']
    elif sub == '100':
        keep &= R['in100']
    if not clip:
        keep &= ~R['idbad']
    t = R['t'][keep]
    cols = {k: R[k][keep] for k in ('f', 'beta', 'z', 'g', 'a', 'rs', 'rf', 'vr', 'idbad')}
    h_all = cols['rs'] - cols['beta'].astype(np.float64) * cols['rf']
    u, st, en = day_slices(t)
    D = len(u)
    out = {'t': u, 'n': np.zeros(D, int), 'nl': np.zeros(D, int)}
    for s in SIGNALS:
        for x in ('gross', 'tc', 'tcu', 'long', 'short'):
            out[f'{s}_{x}'] = np.full(D, np.nan)
        out[f'{s}_idbad'] = np.zeros(D, int)
    for x in ('vs_hi', 'vs_lo', 'vs_hi_long', 'vs_lo_long', 'vs_hi_short', 'vs_lo_short'):
        out[x] = np.full(D, np.nan)
    out['vs_hi_n'] = np.zeros(D, int)
    out['vs_lo_n'] = np.zeros(D, int)
    for d in range(D):
        a, b = st[d], en[d]
        n = b - a
        out['n'][d] = n
        if n < MIN_N:
            continue
        nl = int(round(FRAC * n))
        out['nl'][d] = nl
        h = h_all[a:b]
        beta = cols['beta'][a:b].astype(np.float64)
        f = cols['f'][a:b]
        for s, key, fade in (('twin', 'z', True), ('raw', 'g', True), ('akbas', 'a', False)):
            o = np.argsort(cols[key][a:b], kind='stable')
            lo, hi = o[:nl], o[n - nl:]
            L, S = (lo, hi) if fade else (hi, lo)
            out[f'{s}_long'][d] = h[L].mean()
            out[f'{s}_short'][d] = h[S].mean()
            out[f'{s}_gross'][d] = h[L].mean() - h[S].mean()
            fi = np.concatenate([f[L], f[S]])
            w = np.concatenate([-beta[L] / nl, beta[S] / nl])
            out[f'{s}_tc'][d] = fund_cost(fi, w)
            out[f'{s}_tcu'][d] = 2 * TWIN_SIDE * (np.abs(beta[L]).mean() + np.abs(beta[S]).mean())
            out[f'{s}_idbad'][d] = int(cols['idbad'][a:b][L].sum() + cols['idbad'][a:b][S].sum())
            if s == 'twin':
                vr = cols['vr'][a:b]
                hiL, hiS = L[vr[L] >= VR_CUT], S[vr[S] >= VR_CUT]
                loL, loS = L[~(vr[L] >= VR_CUT)], S[~(vr[S] >= VR_CUT)]
                out['vs_hi_n'][d] = len(hiL) + len(hiS)
                out['vs_lo_n'][d] = len(loL) + len(loS)
                if len(hiL):
                    out['vs_hi_long'][d] = h[hiL].mean()
                if len(hiS):
                    out['vs_hi_short'][d] = h[hiS].mean()
                if len(loL):
                    out['vs_lo_long'][d] = h[loL].mean()
                if len(loS):
                    out['vs_lo_short'][d] = h[loS].mean()
                if len(hiL) and len(hiS):
                    out['vs_hi'][d] = h[hiL].mean() - h[hiS].mean()
                if len(loL) and len(loS):
                    out['vs_lo'][d] = h[loL].mean() - h[loS].mean()
    ok = out['nl'] > 0
    return {k: v[ok] for k, v in out.items()}


def net(B, s, p=P_HEAD, netted=True):
    """Daily book return after the entry penalty p per side and the twin-leg costs."""
    return B[f'{s}_gross'] - 2 * p - (B[f'{s}_tc'] if netted else B[f'{s}_tcu'])


def tstat(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), len(x)


def nw_t(x, lags=5):
    """Newey-West t of the mean (Bartlett kernel)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    e = x - x.mean()
    n = len(x)
    v = e @ e / n
    for L in range(1, lags + 1):
        v += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / n
    return float(x.mean() / np.sqrt(v / n))


def period_mask(days, t, a, b):
    """Days t (export indices) with a <= date < b (ISO strings; b may be None)."""
    lo = c2eval.day_index(days, a)
    hi = c2eval.day_index(days, b) if b else len(days)
    return (t >= lo) & (t < hi)
