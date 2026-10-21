"""Short volume: diagnostics chosen after the walk-forward results (not tests).

1. Is it 5-day reversal? Short sellers trade against recent gains (Diether, Lee & Werner 2009), so the signal may
   proxy for the stock's own past week. Per decision day: the Spearman correlation of the oriented G with -x5, where x5
   is the past week's market-relative log return (close t-5 -> t); reversal's own IC (-x5); and the IC of G after
   removing reversal (the residual of rank(oriented G) on rank(-x5), cross-sectional OLS).
2. Long and short sides of the decile book (gross, each against the equal-weight mean of all ranked names).

    python3 -I -B posthoc.py old    # training, 2016-2019, 2020-2025-06
    python3 -I -B posthoc.py new    # the unseen months (run once, with tests.py new and money.py new)
"""
import sys
sys.dont_write_bytecode = True
import json  # noqa: E402
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import weekly as Wk  # noqa: E402
np, c2eval = C.np, C.c2eval


def main(tag):
    sign = json.loads((C.OUT / 'train_old.json').read_text())['G']['sign']
    W = Wk.load(tag, signals=('G',))
    E = W['E']
    dd = W['dd']
    close = E['close']
    c0 = np.asarray(close[:, dd], np.float64)
    cm5 = np.asarray(close[:, dd - C.H], np.float64)
    m = W['market']
    cm = np.concatenate([[0.0], np.cumsum(np.nan_to_num(m))])
    mk = cm[dd + 1] - cm[dd - C.H + 1]
    with np.errstate(all='ignore'):
        x5 = np.log(c0 / cm5) - mk[None, :]
    x5[~np.isfinite(x5)] = np.nan
    G = sign * W['S']['G']
    periods = ['train', 'test1', 'test3'] if tag == 'old' else ['unseen']
    res, lines = {}, []
    for members in ('U', 'B'):
        for p in periods:
            mask = C.in_period(W['days'], dd, p)
            if tag == 'new':
                mask &= dd + C.H <= len(W['days']) - 1
            rho, icr, icg, icres, longs, shorts = [], [], [], [], [], []
            for i in np.nonzero(mask)[0]:
                ok = W[members][:, i] & np.isfinite(G[:, i]) & np.isfinite(x5[:, i]) & np.isfinite(W['y'][:, i])
                if ok.sum() < C.MIN_NAMES:
                    continue
                g, r, y = G[ok, i], -x5[ok, i], W['y'][ok, i]
                rho.append(C.spearman(g, r))
                icr.append(C.spearman(r, y))
                icg.append(C.spearman(g, y))
                rg, rr = C.avg_rank(g), C.avg_rank(r)
                rr_c = rr - rr.mean()
                beta = (rr_c @ (rg - rg.mean())) / (rr_c @ rr_c)
                icres.append(C.spearman(rg - beta * rr, y))
                # sides of the decile book (all names with a finite G, as in the book)
                ok2 = W[members][:, i] & np.isfinite(G[:, i])
                idx = np.nonzero(ok2)[0]
                o = idx[np.argsort(G[idx, i], kind='stable')]
                k = int(round(C.DECILE * len(idx)))
                rs = W['r'][:, i]
                base = np.nanmean(rs[idx])
                longs.append(np.nanmean(rs[o[-k:]]) - base)
                shorts.append(np.nanmean(rs[o[:k]]) - base)
            out = {'corr_with_reversal': C.tstat(rho), 'ic_reversal': C.tstat(icr), 'ic_G': C.tstat(icg),
                   'ic_G_minus_reversal': C.tstat(icres), 'long_side_bps': C.tstat(np.array(longs) * 1e4),
                   'short_side_bps': C.tstat(np.array(shorts) * 1e4)}
            res[f'{p}|{members}'] = out
            lines.append(f"{members} {p:6s}: corr(G, -x5) {out['corr_with_reversal'][0]:+.3f}; IC reversal {out['ic_reversal'][0]:+.4f} "
                         f"(t {out['ic_reversal'][1]:+.2f}); IC G {out['ic_G'][0]:+.4f} (t {out['ic_G'][1]:+.2f}); IC G after reversal "
                         f"{out['ic_G_minus_reversal'][0]:+.4f} (t {out['ic_G_minus_reversal'][1]:+.2f}); long side "
                         f"{out['long_side_bps'][0]:+.1f} bps/wk (t {out['long_side_bps'][1]:+.2f}), short side "
                         f"{out['short_side_bps'][0]:+.1f} (t {out['short_side_bps'][1]:+.2f}); {out['ic_G'][2]} wk")
    (C.OUT / f'posthoc_{tag}.json').write_text(json.dumps(res, indent=1))
    (C.OUT / f'posthoc_{tag}.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1])
