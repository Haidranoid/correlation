"""Short volume: the universe, and how well FINRA's symbols map onto it (no returns are used here).

    python3 -I -B mapcheck.py old|new
Writes WORK/short_volume/<tag>/universe.npz (uni, big on the twins calendar, rebal) and the per-day volume-match
correlation vcorr.npy ([N, T] float32: corr of log FINRA TotalVolume with log export volume over the 63 days through
d, NaN with fewer than 40 common days), and out/mapcheck_<tag>.txt: per year, the share of universe stock-days with a
FINRA record, with a usable SVR (TotalVolume >= 10,000), FINRA total / export volume, and the volume correlation.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np, c2eval = C.np, C.c2eval

VC_W, VC_MIN = 63, 40
MAX_RATIO = 20.0


def rolling_corr(x, y, w, min_n):
    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.where(ok, x, 0.0), np.where(ok, y, 0.0)

    def rs(a):
        cs = np.zeros((a.shape[0], a.shape[1] + 1))
        np.cumsum(a, 1, out=cs[:, 1:])
        out = np.full(a.shape, np.nan)
        out[:, w - 1:] = cs[:, w:] - cs[:, :-w]
        return out
    n = rs(ok.astype(np.float64))
    sx, sy, sxx, syy, sxy = rs(xs), rs(ys), rs(xs * xs), rs(ys * ys), rs(xs * ys)
    with np.errstate(all='ignore'):
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        r = cov / np.sqrt(vx * vy)
    r[(n < min_n) | ~np.isfinite(r)] = np.nan
    return r.astype(np.float32)


def main(tag):
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    tw = c2eval.twins(E)
    uni, big = C.universe(E, tw)
    np.savez(C.wdir(tag) / 'universe.npz', uni=uni, big=big, rebal=tw['rebal'], fund_row=tw['fund_row'])
    C.log('universe built; median size', int(np.median(uni.sum(1)[uni.sum(1) > 0])))
    total = np.load(C.WORK / tag / 'total.npy', mmap_mode='r')
    info = np.load(C.WORK / tag / 'day_info.npz')
    d0 = int(info['d0'])
    vc = np.full((N, T), np.nan, np.float32)
    ratio_rows = []
    nbroken = np.zeros(N, np.int64)
    for a in range(0, N, 1000):
        b = min(N, a + 1000)
        with np.errstate(all='ignore'):
            vol = np.asarray(E['dollar_volume'][a:b], np.float64) / np.asarray(E['close'][a:b], np.float64)
            ft = np.asarray(total[a:b], np.float64)
            lx = np.where(ft > 0, np.log(ft), np.nan)
            ly = np.where(vol > 0, np.log(vol), np.nan)
            # a day whose FINRA (off-exchange) volume exceeds 20x the export's (consolidated, split-adjusted) volume
            # is a broken export volume print (e.g. 2026-03-03 in the new store: volumes of a few dozen shares);
            # it is left out of the correlation
            broken = ft > MAX_RATIO * vol
            lx[broken] = np.nan
            ly[broken] = np.nan
            nbroken[a:b] = broken.sum(1)
        vc[a:b] = rolling_corr(lx, ly, VC_W, VC_MIN)
        with np.errstate(all='ignore'):
            ratio_rows.append((ft / vol).astype(np.float32))
    ratio = np.concatenate(ratio_rows, 0)
    del ratio_rows
    np.save(C.wdir(tag) / 'vcorr.npy', vc)
    with np.errstate(all='ignore'):
        brk_days = (ratio > MAX_RATIO).sum(0)
    top = np.argsort(-brk_days)[:8]
    rebal = tw['rebal']
    iso = [c2eval.iso(d) for d in days]
    years = sorted(set(i[:4] for i in iso[d0:]))
    lines = [f'export {tag}: universe stock-days from {iso[d0]} (judge-like liquid set with a twin, minus excluded)',
             'year  stock-days  FINRA record  SVR usable  FINRA/export volume: median, p10, p90, share >1  '
             'volume corr (63d): median, share <0.3, share NaN',
             f'stock-days left out of the volume correlation as broken export volume (FINRA > {MAX_RATIO:.0f}x export): '
             f'{int(nbroken.sum())}; days with the most: ' + ', '.join(f'{iso[d]} {int(brk_days[d])}' for d in top)]
    for y in years:
        ds = [d for d in range(max(d0, int(rebal[0])), T) if iso[d][:4] == y]
        rec = use = n = g3 = g5 = 0
        rs, cs = [], []
        for d in ds:
            k = C.kidx(rebal, d)
            if k < 0:
                continue
            u = np.nonzero(uni[k])[0]
            n += len(u)
            t = np.asarray(total[u, d])
            rec += int(np.isfinite(t).sum())
            use += int((t >= C.MIN_TOTAL).sum())
            g3 += int(((t >= C.MIN_TOTAL) & (vc[u, d] >= 0.3)).sum())
            g5 += int(((t >= C.MIN_TOTAL) & (vc[u, d] >= 0.5)).sum())
            r = ratio[u, d]
            rs.append(r[np.isfinite(r)])
            cs.append(vc[u, d])
        if n == 0:
            continue
        r = np.concatenate(rs)
        c = np.concatenate(cs)
        cf = c[np.isfinite(c)]
        if len(r) == 0 or len(cf) == 0:
            lines.append(f'{y}  {n:>10}  {rec / n:>11.1%}  {use / n:>10.1%}  (no volume comparison)')
            continue
        lines.append(f'{y}  {n:>10}  {rec / n:>11.1%}  {use / n:>10.1%}  {np.median(r):.3f}, {np.percentile(r, 10):.3f}, '
                     f'{np.percentile(r, 90):.3f}, {(r > 1).mean():.2%}   {np.median(cf):.3f}, {(cf < 0.3).mean():.2%}, '
                     f'{(~np.isfinite(c)).mean():.2%}; usable with corr >= 0.3: {g3 / n:.1%}, >= 0.5: {g5 / n:.1%}; corr < 0.5: {(cf < 0.5).mean():.2%}')
    # the universe stocks with the lowest volume correlation, to eyeball wrong matches
    low = []
    for k in range(len(rebal)):
        d = int(rebal[k])
        if d < d0 + VC_W:
            continue
        u = np.nonzero(uni[k])[0]
        c = vc[u, d]
        for j in u[np.isfinite(c) & (c < 0.2)]:
            low.append((iso[d], str(E['tickers'][j]), float(vc[j, d]), float(ratio[j, d])))
    lines.append(f'universe stock-days on rebalance days with volume corr < 0.2: {len(low)}; e.g.')
    for x in low[:: max(1, len(low) // 40)][:40]:
        lines.append(f'  {x[0]} {x[1]} corr {x[2]:.2f} ratio {x[3]:.3f}')
    # ORF (FORF) records on the monthly sample days that belong to universe stocks
    import json
    pj = json.loads((C.OUT / f'parse_{tag}.json').read_text())
    pos = {s: i for i, s in enumerate(iso)}
    inu = [h for h in pj['orf_hits'] if C.kidx(rebal, pos[h[0]]) >= 0 and uni[C.kidx(rebal, pos[h[0]]), h[2]]]
    vol_orf = sum(h[3] for h in inu)
    vol_fac = sum(h[4] for h in inu)
    lines.append(f'ORF records of universe stocks on the {len(set(h[0] for h in pj["orf_hits"]))} monthly sample days: {len(inu)} '
                 f'(ORF volume {vol_orf:,}, their TRF+ADF volume the same day {vol_fac:,}); e.g. ' +
                 '; '.join(f'{h[0]} {h[1]} orf {h[3]} fac {h[4]}' for h in inu[:12]))
    txt = '\n'.join(lines) + '\n'
    (C.OUT / f'mapcheck_{tag}.txt').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main(sys.argv[1])
