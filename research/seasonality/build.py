"""Monthly panels, the seasonality signals and the universe at each month-end, for one export.

    python3 -I -B build.py old|new      -> WORK/panel_<tag>.npz (and start.txt for old)

Months: calendar months of the export's days. Month-end day e[i] = the export's last day in month i.
  - lc[j, i]: stock j's last export close in month i (NaN if it has no close in month i).
  - r[j, i] = log(lc[j, i] / lc[j, i-1]): the signal's monthly log return, present only when the stock has a close in
    both months.
  - g[j, i] = log(P[j, e[i]] / P[j, e[i-1]]) with P the forward-filled closes: the holding return of a stock bought at
    the close of e[i-1] and sold at the close of e[i] (0 for a month without a trade).
Signals for holding month i (formed at the close of e[i-1], held through e[i]):
  - same[j, i]: mean of r[j, i - 12k], k = 1..10, at least 5 present;
  - other[j, i]: mean of r[j, i - l], l = 1..120 not a multiple of 12 (the other months of the same ten lag-years),
    at least 55 of the 110 present;
  - other_kln[j, i]: the same skipping lags 1..11 (Keloharju et al.'s definition: years 2-10 only), at least 50 of 99.
Universe for holding month i: the judge-like liquid set (and its 500 largest) of the latest c2eval.twins calendar day
at or before e[i-1].
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (c2eval, np, WORK, HERE, EXPORTS, YEARS, MIN_YEARS, MIN_OTHER, START_MIN, log, universe,  # noqa: E402
                    month_of, month_label)


def monthly(close, days):
    """Monthly panels from a close panel [N, T] (any prefix of the export) and its days [T]."""
    mon = month_of(days)
    months = np.unique(mon)
    e = np.searchsorted(mon, months, side='right') - 1
    f = np.searchsorted(mon, months, side='left')
    c = np.asarray(close, np.float64)
    N, T = c.shape
    ok = np.isfinite(c) & (c > 0)
    idx = np.where(ok, np.arange(T, dtype=np.int32)[None, :], -1).astype(np.int32)
    np.maximum.accumulate(idx, axis=1, out=idx)
    li = idx[:, e]                                    # [N, M] last close index at or before each month-end
    del idx
    lc_ff = np.take_along_axis(c, np.maximum(li, 0), 1)
    lc_ff[li < 0] = np.nan
    lc_in = np.where(li >= f[None, :], lc_ff, np.nan)  # a close inside the month
    M = len(months)
    r = np.full((N, M), np.nan)
    g = np.full((N, M), np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(lc_in[:, 1:] / lc_in[:, :-1])
        g[:, 1:] = np.log(lc_ff[:, 1:] / lc_ff[:, :-1])
    return dict(months=months, e=e, f=f, r=r, g=g, lc_in=lc_in)


def lag_mean(r, lags, min_n):
    """[N, M] mean of r[:, i - l] over l in lags (where i - l >= 0), NaN unless at least min_n present."""
    N, M = r.shape
    s = np.zeros((N, M))
    n = np.zeros((N, M))
    for l in lags:
        x = np.full((N, M), np.nan)
        x[:, l:] = r[:, :M - l]
        fin = np.isfinite(x)
        s += np.where(fin, x, 0.0)
        n += fin
    with np.errstate(all='ignore'):
        out = s / n
    out[n < min_n] = np.nan
    return out, n


SAME_LAGS = [12 * k for k in range(1, YEARS + 1)]
OTHER_LAGS = [l for l in range(1, 12 * YEARS + 1) if l % 12]
KLN_LAGS = [l for l in OTHER_LAGS if l > 12]


def signals(r):
    same, n_same = lag_mean(r, SAME_LAGS, MIN_YEARS)
    other, _ = lag_mean(r, OTHER_LAGS, MIN_OTHER)
    other_kln, _ = lag_mean(r, KLN_LAGS, 50)
    return dict(same=same, n_same=n_same, other=other, other_kln=other_kln)


def build(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume', 'market'))
    days = E['days']
    N, T = E['close'].shape
    log(tag, 'N', N, 'T', T, c2eval.iso(days[0]), '..', c2eval.iso(days[-1]))
    P = monthly(E['close'], days)
    months, e = P['months'], P['e']
    M = len(months)
    S = signals(P['r'])
    log('months', M, month_label(months[0]), '..', month_label(months[-1]))
    tw = c2eval.twins(E)
    uni, big = universe(E, tw)
    log('universe built; rebal days', len(tw['rebal']))
    uni_f = np.zeros((M, N), bool)
    big_f = np.zeros((M, N), bool)
    kf = np.full(M, -1)
    for i in range(1, M):
        k = c2eval.rebal_index(tw, int(e[i - 1]))
        kf[i] = k
        if k >= 0:
            uni_f[i], big_f[i] = uni[k], big[k]
    has = np.isfinite(S['same']).T                    # [M, N]
    n_uni = uni_f.sum(1)
    n_sig = (uni_f & has).sum(1)
    n_sig500 = (big_f & has).sum(1)
    mk = np.asarray(E['market'], np.float64)
    mkt = np.full(M, np.nan)
    for i in range(1, M):
        mkt[i] = np.expm1(np.nansum(mk[e[i - 1] + 1:e[i] + 1]))
    np.savez(WORK / f'panel_{tag}.npz', days=days, months=months, e=e, f=P['f'], kf=kf,
             r=P['r'].astype(np.float32), g=P['g'].astype(np.float32),
             same=S['same'].astype(np.float32), n_same=S['n_same'].astype(np.int16),
             other=S['other'].astype(np.float32), other_kln=S['other_kln'].astype(np.float32),
             uni_f=uni_f, big_f=big_f, n_uni=n_uni, n_sig=n_sig, n_sig500=n_sig500, mkt=mkt)
    lines = [f'{tag}: holding month, universe size at formation, universe stocks with a signal, 500 largest with a signal']
    for i in range(1, M):
        lines.append(f'{month_label(months[i])} formed {c2eval.iso(days[e[i - 1]])}: universe {n_uni[i]:5d}  '
                     f'with signal {n_sig[i]:5d}  500-largest with signal {n_sig500[i]:4d}')
    first = np.nonzero(n_sig >= START_MIN)[0]
    if tag == 'old':
        i0 = int(first[0])
        head = (f'START MONTH (first holding month with >= {START_MIN} universe stocks holding a signal): '
                f'{month_label(months[i0])}, formed at the close of {c2eval.iso(days[e[i0 - 1]])}, '
                f'{n_sig[i0]} stocks with a signal of {n_uni[i0]} in the universe. '
                f'Recorded before any book return was computed.')
        lines.insert(0, head)
        log(head)
        (HERE / 'start.txt').write_text('\n'.join(lines) + '\n')
    else:
        (HERE / 'counts_new.txt').write_text('\n'.join(lines) + '\n')
    log('saved')


if __name__ == '__main__':
    build(sys.argv[1])
