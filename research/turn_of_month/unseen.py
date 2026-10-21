"""The unseen months, run once after the walk-forward was final: 2025-06-30..2026-10-19 on the new export's calendar,
SPY from Yahoo matched to it, the new export's equal-weight market, and (reported only) French's CRSP value-weighted
market through 2026-08-31. Writes results_new.txt and results_new.json."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np, pubdata  # noqa: E402

out = []
RES = {}


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


def pct(x, k=3):
    return f'{100 * x:+.{k}f}%'


En = c2eval.load_export(c2eval.EXPORT_NEW, arrays=('market',))
dn = En['days']
mu = C.span_mask(dn, *C.UNSEEN)
days = dn[mu]
lab = C.tom_label(days, dn)
yd, px = pubdata.etf_adjclose(['SPY'])
yday = pubdata.to_days(yd[np.isfinite(px['SPY'])])
assert np.array_equal(np.intersect1d(yday, dn[C.span_mask(dn, '2025-06-27', '2026-10-19')]),
                      dn[C.span_mask(dn, '2025-06-27', '2026-10-19')])
r = C.spy_total(dn)[0][mu]
rf = pubdata.rf_daily(dn)[mu]
mk = np.asarray(En['market'], float)[mu]
assert np.isfinite(r).all() and np.isfinite(rf).all() and np.isfinite(mk).all()
ex, mkx, lr, lrf = r - rf, mk - np.log1p(rf), np.log1p(r), np.log1p(rf)
tom = lab != 0
T = len(days)
y, m = C.ymd(days)
mkey = y * 12 + m
say(f'unseen {c2eval.iso(days[0])}..{c2eval.iso(days[-1])}: {T} days, {int(tom.sum())} TOM days '
    f'({ {p: int((lab == p).sum()) for p in C.POSITIONS} }), {len(np.unique(mkey))} calendar months (partial: '
    'June 2025 = 1 day, October 2026 = 13 days)')

d, se, t = C.welch(ex[tom], ex[~tom])
dm, sem, tm = C.welch(mkx[tom], mkx[~tom])
RES['spy'] = {'tom_mean_ex': float(ex[tom].mean()), 'other_mean_ex': float(ex[~tom].mean()), 'diff': d, 'se': se,
              't': t, 'tom_mean_raw': float(r[tom].mean()), 'other_mean_raw': float(r[~tom].mean()),
              'positions': {p: {'mean_ex': float(ex[lab == p].mean()), 't': C.welch(ex[lab == p], ex[~tom])[2],
                                'n': int((lab == p).sum())} for p in C.POSITIONS}}
RES['ew'] = {'tom_mean_ex': float(mkx[tom].mean()), 'other_mean_ex': float(mkx[~tom].mean()), 'diff': dm, 'se': sem,
             't': tm, 'positions': {p: float(mkx[lab == p].mean()) for p in C.POSITIONS}}
say(f'SPY excess: TOM {pct(ex[tom].mean())} vs other {pct(ex[~tom].mean())}; diff {pct(d)}, se {pct(se)}, Welch t {t:+.2f}')
say('  positions: ' + '; '.join(f'{p:+d}: {pct(v["mean_ex"])} t {v["t"]:+.2f} (n {v["n"]})'
                                for p, v in RES['spy']['positions'].items()))
say(f'EW market (new export, excess log): TOM {pct(mkx[tom].mean())} vs other {pct(mkx[~tom].mean())}; diff {pct(dm)}, '
    f't {tm:+.2f}; positions ' + ', '.join(f'{p:+d}: {pct(v)}' for p, v in RES['ew']['positions'].items()))
xl = lr - lrf
yrs = T / C.DPY
say(f'SPY excess log per year: TOM {pct(xl[tom].sum() / yrs, 1)}, other {pct(xl[~tom].sum() / yrs, 1)}, all '
    f'{pct(xl.sum() / yrs, 1)}')


def paths_log(ind, cost):
    prev = np.concatenate([np.zeros((ind.shape[0], 1), bool), ind[:, :-1]], 1)
    nxt = np.concatenate([ind[:, 1:], np.zeros((ind.shape[0], 1), bool)], 1)
    trades = (ind & ~prev).astype(np.int8) + (ind & ~nxt).astype(np.int8)
    g = np.where(ind, lr[None, :], lrf[None, :])
    if cost:
        g = g + np.log1p(-cost) * trades
    return g


def cagr(g):
    return np.exp(g.sum(-1) * C.DPY / g.shape[-1]) - 1.0


months = np.unique(mkey)
month_idx = [np.nonzero(mkey == k)[0] for k in months]
k_per = [int(tom[i].sum()) for i in month_idx]
rng = np.random.default_rng(C.SEED)
ind = np.zeros((C.N_RANDOM, T), bool)
for idx, kk in zip(month_idx, k_per):
    pick = np.argsort(rng.random((C.N_RANDOM, len(idx))), axis=1)[:, :kk]
    ind[np.repeat(np.arange(C.N_RANDOM), kk), idx[pick.ravel()]] = True
assert (ind.sum(1) == tom.sum()).all()
for c in (0.0, C.COST):
    dr = cagr(paths_log(ind, c))
    ru = float(cagr(paths_log(tom[None, :], c))[0])
    med = float(np.median(dr))
    p = float((1 + np.sum(dr >= ru)) / (1 + len(dr)))
    RES[f'timing_{int(c * 1e4)}bp'] = {'rule': ru, 'median_random': med, 'diff_pts': 100 * (ru - med), 'p': p}
    say(f'rule vs random at {c * 1e4:.0f} bp/side: rule {pct(ru, 2)}, median random {pct(med, 2)}, diff '
        f'{100 * (ru - med):+.2f} pts, p {p:.4f}')
RES['buy_hold'] = float(np.exp(lr.sum() * C.DPY / T) - 1)
RES['cash'] = float(np.exp(lrf.sum() * C.DPY / T) - 1)
say(f'buy-and-hold SPY {pct(RES["buy_hold"], 2)}, cash {pct(RES["cash"], 2)} (annualized)')

# French's CRSP VW market over the part of the unseen months it covers (reported only)
fd, f = pubdata.ff3_daily()
fdays = pubdata.to_days(fd)
fm = C.span_mask(fdays, '2025-06-30', '2026-08-31')
flab = C.tom_label(fdays, fdays)[fm]
fx = f['Mkt-RF'][fm]
fdiff, fse, ft = C.welch(fx[flab != 0], fx[flab == 0])
RES['french_vw'] = {'end': str(fd[fm][-1]), 'diff': fdiff, 't': ft, 'n_tom': int((flab != 0).sum())}
say(f'French CRSP VW excess, 2025-06-30..{fd[fm][-1]}: TOM {pct(fx[flab != 0].mean())} vs other '
    f'{pct(fx[flab == 0].mean())}; diff {pct(fdiff)}, t {ft:+.2f} (n TOM {int((flab != 0).sum())})')
RES['seed'] = C.SEED
(C.HERE / 'results_new.txt').write_text('\n'.join(out) + '\n')
(C.HERE / 'results_new.json').write_text(json.dumps(RES, indent=1))
