"""Gravity: G1's measured force law at 5-minute steps on the cached Yahoo bars (reported, once, at the end).

    python -I -B intraday.py

Bars: ~/.local/share/correlation2/experiments/intraday_lead_lag/bars/5m (read only; no new fetch), with the filters of
research/intraday_lead_lag/build.py (09:30 <= time < 16:00, UTC offset -04:00, a finite close, the first of duplicate
rows). Days: the bar dates that are full days (SPY has all 78 bars) and in the new export (through 2026-10-19).
Attractors: G1's real and random attractors, masses and beta at the new export's last refit before the first bar day,
for that refit's universe; Yahoo symbols are store tickers with '.' -> '-'.
Each symbol-day is put on the store's scale with f = store close (d - 1) / Yahoo daily close (d - 1), and is used only
if its last bar's close x f is within 2% of the store's close on d and Yahoo reports no split on d - 1 or d.
At each bar b (0..76) of day d: the spread's mean and sd come from the store's daily closes d-63..d-1 (G1's
definition); g_C, g_H, g_L use B's bar close, high and low and A's bar close; r is the bar's closest approach. The
outcome is B's next-bar log return in the attractor's direction, -sign(g_C) ln(C_B,b+1 / C_B,b), divided by B's daily
sigma through d - 1 / sqrt(78). Bins of r as shape.py; real minus random; a circular bootstrap of days in 5-day blocks
(10,000 draws). Writes out/intraday.json / .txt.
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
import data  # noqa: E402
import shape  # noqa: E402
np = K.np

WORK = K.c2eval.WORK / 'intraday_lead_lag'
NBAR = 78


def read_bars(syms):
    sidx = {s: i for i, s in enumerate(syms)}
    rows = {}
    seen = set()
    counts = {'rows': 0, 'kept': 0, 'outside': 0, 'offset': 0, 'nan': 0, 'dup': 0}
    for f in sorted((WORK / 'bars' / '5m').glob('5m_*.csv')):
        with open(f, newline='') as fh:
            for r in csv.DictReader(fh):
                counts['rows'] += 1
                ts = r['ts']
                hh, mm, off = int(ts[11:13]), int(ts[14:16]), ts[19:]
                mins = hh * 60 + mm - (9 * 60 + 30)
                if not (0 <= mins < 390):
                    counts['outside'] += 1
                    continue
                if off != '-04:00':
                    counts['offset'] += 1
                    continue
                try:
                    c, h, lo = float(r['close']), float(r['high']), float(r['low'])
                except ValueError:
                    counts['nan'] += 1
                    continue
                if not (math.isfinite(c) and math.isfinite(h) and math.isfinite(lo)):
                    counts['nan'] += 1
                    continue
                if r['sym'] not in sidx:
                    continue
                key = (r['sym'], ts)
                if key in seen:
                    counts['dup'] += 1
                    continue
                seen.add(key)
                rows.setdefault(ts[:10], []).append((sidx[r['sym']], mins // 5, c, h, lo))
                counts['kept'] += 1
    dates = sorted(rows)
    C = np.full((len(syms), len(dates), NBAR), np.nan)
    H, L = C.copy(), C.copy()
    for di, d in enumerate(dates):
        a = np.array(rows[d])
        s, b = a[:, 0].astype(int), a[:, 1].astype(int)
        C[s, di, b], H[s, di, b], L[s, di, b] = a[:, 2], a[:, 3], a[:, 4]
    return dates, C, H, L, counts


def main():
    fh = open(K.OUT / 'intraday.txt', 'w')
    P = data.load_panel('new')
    PE = dict(np.load(K.BIG / 'peers_new.npz'))
    days = np.asarray(P['days'])
    tick = list(P['tickers'])
    D = np.load(WORK / 'daily.npz')
    ysyms = [str(s) for s in D['syms']]
    ydates = [str(x) for x in D['dates']]
    dates, C, H, L, counts = read_bars(ysyms)
    K.log(f'bars: {counts}; {len(dates)} dates {dates[0]}..{dates[-1]}', fh)
    spy = ysyms.index('SPY') if 'SPY' in ysyms else None
    full = [di for di, d in enumerate(dates) if spy is None or np.isfinite(C[spy, di]).all()]
    exp_days = {K.c2eval.iso(x): i for i, x in enumerate(days)}
    use = [di for di in full if dates[di] in exp_days]
    K.log(f'full days {len(full)}; in the new export (through {K.c2eval.iso(days[-1])}) {len(use)}', fh)
    first_t = exp_days[dates[use[0]]]
    refit = np.asarray(P['refit'])
    k = int(np.searchsorted(refit, first_t, side='left')) - 1          # last refit strictly before the first bar day
    u = np.asarray(P['uni'][k])
    u = u[u >= 0]
    ymap = {s: i for i, s in enumerate(ysyms)}
    yrow = np.array([ymap.get(tick[b].replace('.', '-'), -1) for b in range(len(tick))])
    covered = int((yrow[u] >= 0).sum())
    K.log(f'attractors from the refit of {K.c2eval.iso(days[refit[k]])}; universe {len(u)} stocks, {covered} with bars '
          f'({100 * covered / len(u):.1f}%)', fh)
    res = {'refit': K.c2eval.iso(days[refit[k]]), 'universe': len(u), 'with_bars': covered, 'bar_counts': counts,
           'days_used': [dates[di] for di in use]}
    if covered < 0.5 * len(u):
        res['not_run'] = f'only {covered} of {len(u)} universe stocks have 5-minute bars'
        K.log(f'NOT RUN: {res["not_run"]}', fh)
        K.jdump(res, K.OUT / 'intraday.json')
        return
    lnC = np.asarray(P['lnC'], np.float64)
    sig = np.asarray(P['sig'], np.float64)
    ycl = D['close']
    ysp = D['split']
    ydi = {d: i for i, d in enumerate(ydates)}
    nd = len(use)
    sums = {w: np.zeros((nd, shape.NB)) for w in ('real', 'random')}
    cnts = {w: np.zeros((nd, shape.NB)) for w in ('real', 'random')}
    dropped = 0
    kept_sd = 0
    for j, di in enumerate(use):
        t = exp_days[dates[di]]
        yd = ydi.get(dates[di])
        prev = ydates[yd - 1] if yd is not None and yd > 0 else None
        if prev is None or prev not in exp_days:
            continue
        tp = exp_days[prev]
        assert tp == t - 1, (dates[di], prev)
        # scale factors and the 2% / split rule per symbol-day
        lf = np.full(len(ysyms), np.nan)
        for b in u:
            s = yrow[b]
            if s < 0:
                continue
            yc0 = ycl[s, yd - 1]
            if not (np.isfinite(yc0) and yc0 > 0) or ysp[s, yd] != 0 or ysp[s, yd - 1] != 0:
                continue
            f = lnC[b, t - 1] - math.log(yc0) if np.isfinite(lnC[b, t - 1]) else np.nan
            last = C[s, di, NBAR - 1]
            if np.isfinite(f) and np.isfinite(last) and np.isfinite(lnC[b, t]) and abs(math.log(last) + f - lnC[b, t]) <= math.log(1.02):
                lf[s] = f
                kept_sd += 1
            else:
                dropped += 1
        for wi, world in enumerate(('real', 'random')):
            for i, b in enumerate(u):
                sb = yrow[b]
                if sb < 0 or not np.isfinite(lf[sb]) or not np.isfinite(sig[b, t - 1]):
                    continue
                s5 = sig[b, t - 1] / math.sqrt(NBAR)
                cB = np.log(C[sb, di]) + lf[sb]
                hB = np.log(H[sb, di]) + lf[sb]
                lB = np.log(L[sb, di]) + lf[sb]
                yv = np.full(NBAR, np.nan)
                yv[:-1] = (np.log(C[sb, di, 1:]) - np.log(C[sb, di, :-1])) / s5
                for s_ in range(K.K_ATT):
                    if world == 'real':
                        a, m, be = int(PE['A1'][k, i, s_]), float(PE['M1'][k, i, s_]), float(PE['B1'][k, i, s_])
                    else:
                        a, m, be = int(PE['AR'][k, i, s_]), float(PE['MR'][k, i, s_]), float(PE['BR'][k, i, s_])
                    if a < 0 or not np.isfinite(be) or not np.isfinite(m):
                        continue
                    sa = yrow[a]
                    if sa < 0 or not np.isfinite(lf[sa]):
                        continue
                    xs = lnC[b, t - 63:t] - be * lnC[a, t - 63:t]
                    xs = xs[np.isfinite(xs)]
                    if len(xs) < 57:
                        continue
                    mu, sd = xs.mean(), xs.std(ddof=1)
                    if not sd > 0:
                        continue
                    cA = np.log(C[sa, di]) + lf[sa]
                    gC = (cB - be * cA - mu) / sd
                    gH = (hB - be * cA - mu) / sd
                    gL = (lB - be * cA - mu) / sd
                    r = np.where((gL <= 0) & (gH >= 0), 0.0, np.minimum(np.abs(gL), np.abs(gH)))
                    y = -np.sign(gC) * yv
                    ok = np.isfinite(gC) & np.isfinite(r) & np.isfinite(y)
                    bins = shape.bin_of(np.where(ok, r, np.nan))
                    for bb in range(shape.NB):
                        mm = bins == bb
                        sums[world][j, bb] += y[mm].sum()
                        cnts[world][j, bb] += mm.sum()
    K.log(f'symbol-days kept {kept_sd:,}, dropped by the 2% or split rule {dropped:,}', fh)
    ok_days = cnts['real'].sum(1) > 0
    W = K.boot_weights(int(ok_days.sum()), block=5).astype(np.float64)
    rows = []
    for bb in range(shape.NB):
        sr, cr = sums['real'][ok_days, bb], cnts['real'][ok_days, bb]
        sq, cq = sums['random'][ok_days, bb], cnts['random'][ok_days, bb]
        er, eq = sr.sum() / cr.sum(), sq.sum() / cq.sum()
        br = (W @ sr) / (W @ cr)
        bq = (W @ sq) / (W @ cq)
        d = K.summarize(er - eq, br - bq)
        rows.append({'bin': shape.BIN_NAMES[bb], 'real': float(er), 'random': float(eq), 'diff': d,
                     'pairs_real': int(cr.sum()), 'pairs_random': int(cq.sum())})
        K.log(f'  r {shape.BIN_NAMES[bb]:>12}: real {er:+.5f} random {eq:+.5f} diff {d["est"]:+.5f} (t {d["t"]:+.2f}; '
              f'90% {d["ci90"][0]:+.5f}..{d["ci90"][1]:+.5f}); bar-pairs {int(cr.sum()):,}', fh)
    res.update({'days': int(ok_days.sum()), 'force_law_5m': rows, 'symbol_days_kept': kept_sd, 'symbol_days_dropped': dropped})
    K.jdump(res, K.OUT / 'intraday.json')
    fh.close()


if __name__ == '__main__':
    main()
