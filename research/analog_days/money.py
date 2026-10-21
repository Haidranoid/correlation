"""The money test (the judge), a proxy for the file name: each liquid, not fund-like stock is mapped to the sector fund
it correlates with most over the trailing 252 days (on c2eval.twins' calendar), and scored with the rank of that
sector's analog forecast (1 = worst, 9 = best) + that correlation.

    python3 -I money.py old     # walk-forward 2006-07..2024-25, uniform and volatility-matched -> out/money_old.*
    python3 -I money.py new     # the unseen months, single holdout from 2025-06-30 (once, at the end) -> out/money_new.*
"""
import json
import sys as _sys, os as _os
import warnings
from collections import Counter

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

DATA_W, DATA_FRAC, MED_W, MIN_PRICE, MIN_DV = 756, 0.9, 252, 5.0, 5e6
CORR_W, CORR_FRAC, MAX_CORR = 252, 0.9, 0.999


def twins_read_only(E):
    """c2eval.twins, but only when its cache file already exists (so nothing is written outside this folder)."""
    key = c2eval.WORK / f"twins3_{E['meta']['fingerprint']}_21_252_0.9_5000000_0.75_0.95.npz"
    assert key.exists(), f'twin cache missing: {key}'
    return c2eval.twins(E)


def mapping(E, tw, P):
    """Per twins rebalance k: liquid not-fund-like rows, their best sector (0..8) and its correlation."""
    close = E['close']
    dv = E['dollar_volume']
    sect = np.log(P['c'][1:])
    rebal = tw['rebal']
    out = []
    for k, t0 in enumerate(rebal):
        if t0 - DATA_W + 1 < 0:
            out.append(None)
            continue
        c = np.asarray(close[:, t0 - DATA_W + 1:t0 + 1], np.float64)
        frac = np.isfinite(c).mean(1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mp = np.nanmedian(c[:, -MED_W:], 1)
            md = np.nanmedian(np.asarray(dv[:, t0 - MED_W + 1:t0 + 1], np.float64), 1)
        liq = (frac >= DATA_FRAC) & (np.nan_to_num(mp) >= MIN_PRICE) & (np.nan_to_num(md) >= MIN_DV) & ~tw['fund_like'][k].astype(bool)
        rows = np.nonzero(liq)[0]
        cw = c[rows, -(CORR_W + 1):]
        with np.errstate(all='ignore'):
            X = np.log(cw[:, 1:] / cw[:, :-1])
        X[~np.isfinite(X)] = np.nan
        Fr = (sect[:, t0 - CORR_W + 1:t0 + 1] - sect[:, t0 - CORR_W:t0])       # [9, 252], complete
        if not np.isfinite(Fr).all():
            out.append(None)
            continue
        M = np.isfinite(X).astype(np.float64)
        keep = M.mean(1) >= CORR_FRAC
        rows, X, M = rows[keep], X[keep], M[keep]
        X0 = np.nan_to_num(X)
        n = M.sum(1, keepdims=True)
        Sx = X0.sum(1, keepdims=True); Sxx = (X0 ** 2).sum(1, keepdims=True)
        Sf = M @ Fr.T; Sff = M @ (Fr ** 2).T; Sxf = X0 @ Fr.T
        with np.errstate(all='ignore'):
            r = (Sxf - Sx * Sf / n) / np.sqrt((Sxx - Sx ** 2 / n) * (Sff - Sf ** 2 / n))
        r[~np.isfinite(r)] = -np.inf
        sec = np.argmax(r, 1)
        best = r[np.arange(len(rows)), sec]
        ok = best > 0
        out.append(dict(rows=rows[ok], sec=sec[ok], corr=np.minimum(best[ok], MAX_CORR), n_liquid=int(liq.sum()),
                        n_scored=int(ok.sum())))
    return out


def scores(tag):
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
    P = C.build(tag)
    tw = twins_read_only(E)
    maps = mapping(E, tw, P)
    N, T = E['close'].shape
    has = P['idx'][:, 0] >= 0
    R = np.full((T, 9), np.nan)
    R[has] = np.argsort(np.argsort(P['F'][has], 1, kind='stable'), 1) + 1          # 1 = worst .. 9 = best
    S = np.full((N, T), np.nan, np.float32)
    rebal = tw['rebal']
    cov = []
    for k, t0 in enumerate(rebal):
        t1 = rebal[k + 1] if k + 1 < len(rebal) else T
        m = maps[k]
        dd = np.arange(t0, t1)
        dd = dd[has[dd]]
        if m is None or len(dd) == 0:
            continue
        S[np.ix_(m['rows'], dd)] = (R[dd][:, m['sec']].T + m['corr'][:, None]).astype(np.float32)
        cov.append((c2eval.iso(E['days'][t0]), m['n_liquid'], m['n_scored'], np.bincount(m['sec'], minlength=9).tolist(),
                    float(np.median(m['corr']))))
    name = f'analog_days/analog_proxy_{tag}'
    p = c2eval.save_scores(S, name, export=C.EXPORTS[tag])
    # a look at the picks: the judge's top 20 on a few days
    tk = E['tickers']
    lines = [f'scored days {int(np.isfinite(S).any(0).sum())}; first {c2eval.iso(E["days"][np.argmax(np.isfinite(S).any(0))])}',
             'rebalance: liquid, scored, mapped stocks per sector (' + ' '.join(C.SECT) + '), median best corr']
    lines += [f'  {a}: {b}, {c}, {d}, {e:.2f}' for a, b, c, d, e in cov[::12]]
    cnt = Counter()
    for d in np.nonzero(np.isfinite(S).any(0))[0][::63]:
        s = S[:, d]
        ok = np.isfinite(s)
        top = np.nonzero(ok)[0][np.argsort(-s[ok], kind='stable')[:20]]
        best = C.SECT[int(np.argmax(R[d]))]
        lines.append(f'  {c2eval.iso(E["days"][d])} best sector {best}: ' + ', '.join(str(tk[j]) for j in top[:10]) + ' ...')
    (C.OUT / f'picks_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:3] + lines[-5:]), flush=True)
    return p


def grade(tag, p):
    res, out = {}, []
    for rm in ('uniform', 'volatility'):
        if tag == 'old':
            txt, js = c2eval.judge(p, rebalance=5, random_match=rm)
        else:
            txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=5, random_match=rm)
        cors = [c2eval.change_over_random(js, cost_index=i) for i in range(3)]
        res[rm] = cors
        out.append(f'===== random_match={rm} =====\n{txt}')
        if tag == 'old':
            out.append('SUMMARY ' + c2eval.summary_line(js))
        else:
            rep = js['report']
            res[rm + '_report'] = dict(strategy=rep['strategy'], ls_bps_per_day=rep.get('ls_bps_per_day'), ls_t=rep.get('ls_t'),
                                       ic_ret_t=rep.get('ic_ret_t'),
                                       benchmarks=[(b.get('ticker'), b.get('return_pct')) for b in rep.get('benchmarks', [])])
        for i, cor in enumerate(cors):
            out.append(f"CHANGE OVER RANDOM ({rm}, cost level {i}): {cor['change']:+.2f} pts/yr; per year {cor['per_year']}; "
                       f"ahead {cor['ahead']}/{cor['years']}; se {cor['se']:.2f}")
    (C.OUT / f'money_{tag}.txt').write_text('\n'.join(out) + '\n')
    (C.OUT / f'money_{tag}.json').write_text(json.dumps(res, indent=1, default=float))
    print('\n'.join(l for l in out if l.startswith(('SUMMARY', 'CHANGE'))), flush=True)


if __name__ == '__main__':
    tag = _sys.argv[1]
    grade(tag, scores(tag))
