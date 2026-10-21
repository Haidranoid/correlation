"""Turn of the month on the old export's span, 2006-01..2025-06 (experiments/-9.6-turn-of-month.md).

Writes results_old.txt and results_old.json. Run under the memory cap:
    python3 -I -B analyze.py
"""
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


# ---------------------------------------------------------------- data
Eo = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('market',))
En = c2eval.load_export(c2eval.EXPORT_NEW, arrays=())
do, dn = Eo['days'], En['days']
assert np.array_equal(dn[:len(do)], do)
span = C.span_mask(do, *C.SPAN)
days = do[span]
lab = C.tom_label(do, dn)[span]
r_all, _, _ = C.spy_total(do)
r = r_all[span]
rf = pubdata.rf_daily(do)[span]
mk = np.asarray(Eo['market'], float)[span]
assert np.isfinite(r).all() and np.isfinite(rf).all() and np.isfinite(mk).all()
ex = r - rf                               # SPY excess simple return
mkx = mk - np.log1p(rf)                   # EW market excess log return
lr = np.log1p(r)
lrf = np.log1p(rf)
y, m = C.ymd(days)
mkey = y * 12 + m
tom = lab != 0
T = len(days)
say(f'span {c2eval.iso(days[0])}..{c2eval.iso(days[-1])}: {T} days, {int(tom.sum())} TOM days, {len(np.unique(mkey))} months')

EXCL = {
    'all days': np.ones(T, bool),
    'without 2008': y != 2008,
    'without March 2020': ~((y == 2020) & (m == 3)),
    'without 2008 and March 2020': (y != 2008) & ~((y == 2020) & (m == 3)),
}


def pmask(lo, hi):
    return (days >= C.dnum(lo)) & (days <= C.dnum(hi))


# ---------------------------------------------------------------- daily statistics
def daily_stats(mask):
    t, o = mask & tom, mask & ~tom
    d, se, tt = C.welch(ex[t], ex[o])
    res = {'n_tom': int(t.sum()), 'n_other': int(o.sum()),
           'tom_mean_ex': float(ex[t].mean()), 'other_mean_ex': float(ex[o].mean()),
           'tom_mean_raw': float(r[t].mean()), 'other_mean_raw': float(r[o].mean()),
           'diff': d, 'se': se, 't': tt, 'sd_all': float(ex[mask].std(ddof=1)),
           'tom_sd': float(ex[t].std(ddof=1)), 'other_sd': float(ex[o].std(ddof=1))}
    pos = {}
    for p in C.POSITIONS:
        s = mask & (lab == p)
        dp, sep, tp = C.welch(ex[s], ex[o])
        pos[p] = {'n': int(s.sum()), 'mean_ex': float(ex[s].mean()), 'mean_raw': float(r[s].mean()),
                  'diff': dp, 't': tp, 't_zero': float(ex[s].mean() / (ex[s].std(ddof=1) / np.sqrt(s.sum()))),
                  'pos_share': float(np.mean(ex[s] > 0))}
    res['positions'] = pos
    dm, sem, tm = C.welch(mkx[t], mkx[o])
    res['ew'] = {'tom_mean_ex': float(mkx[t].mean()), 'other_mean_ex': float(mkx[o].mean()), 'diff': dm, 'se': sem,
                 't': tm, 'tom_mean_raw': float(mk[t].mean()), 'other_mean_raw': float(mk[o].mean()),
                 'positions': {p: {'mean_ex': float(mkx[mask & (lab == p)].mean()),
                                   't': C.welch(mkx[mask & (lab == p)], mkx[o])[2]} for p in C.POSITIONS}}
    # share of SPY's excess log return earned on TOM days
    xl = lr - lrf
    yrs = mask.sum() / C.DPY
    res['excess_log_tom_per_year'] = float(xl[t].sum() / yrs)
    res['excess_log_other_per_year'] = float(xl[o].sum() / yrs)
    res['excess_log_total_per_year'] = float(xl[mask].sum() / yrs)
    return res


def fmt_daily(name, s):
    say(f'  {name}: TOM {pct(s["tom_mean_ex"])} (n {s["n_tom"]}) vs other {pct(s["other_mean_ex"])} '
        f'(n {s["n_other"]}); diff {pct(s["diff"])}, se {pct(s["se"])}, Welch t {s["t"]:+.2f} | '
        f'EW market diff {pct(s["ew"]["diff"])} t {s["ew"]["t"]:+.2f} | SPY excess log/yr: TOM '
        f'{pct(s["excess_log_tom_per_year"], 1)}, other {pct(s["excess_log_other_per_year"], 1)}, '
        f'all {pct(s["excess_log_total_per_year"], 1)}')
    say('     positions (SPY excess mean, t vs other): ' + '; '.join(
        f'{p:+d}: {pct(v["mean_ex"])} t {v["t"]:+.2f} (n {v["n"]})' for p, v in s['positions'].items()))


# ---------------------------------------------------------------- random timings
def paths_log(ind, cost):
    """Per-day log growth [D, T] of holding SPY on days where ind is True (close before to close), cash otherwise,
    paying `cost` per side on each position change. ind: bool [D, T]."""
    ind = np.asarray(ind, bool)
    prev = np.concatenate([np.zeros((ind.shape[0], 1), bool), ind[:, :-1]], 1)
    nxt = np.concatenate([ind[:, 1:], np.zeros((ind.shape[0], 1), bool)], 1)
    trades = (ind & ~prev).astype(np.int8) + (ind & ~nxt).astype(np.int8)
    g = np.where(ind, lr[None, :], lrf[None, :])
    if cost:
        g = g + np.log1p(-cost) * trades
    return g, trades


def cagr(g, mask):
    return np.exp(g[..., mask].sum(-1) * C.DPY / mask.sum()) - 1.0


MASKS = {}
for pn, (lo, hi) in C.PERIODS.items():
    for en, em in EXCL.items():
        MASKS[(pn, en)] = pmask(lo, hi) & em
for pn, (lo, hi) in C.BLOCKS.items():
    MASKS.setdefault((pn, 'all days'), pmask(lo, hi))

months = np.unique(mkey)
rule_ind = tom[None, :]
k_per_month = {k: int(tom[mkey == k].sum()) for k in months}
assert set(k_per_month.values()) <= {3, 4}
rng = np.random.default_rng(C.SEED)
CH = 1000
rand = {c: {k: np.empty(C.N_RANDOM) for k in MASKS} for c in (0.0, C.COST)}
rand_trades = np.empty(C.N_RANDOM)
month_idx = [np.nonzero(mkey == k)[0] for k in months]
for a in range(0, C.N_RANDOM, CH):
    b = min(C.N_RANDOM, a + CH)
    ind = np.zeros((b - a, T), bool)
    for k, idx in zip(months, month_idx):
        kk = k_per_month[k]
        pick = np.argsort(rng.random((b - a, len(idx))), axis=1)[:, :kk]
        rows = np.repeat(np.arange(b - a), kk)
        ind[rows, idx[pick.ravel()]] = True
    assert (ind.sum(1) == tom.sum()).all()
    for c in (0.0, C.COST):
        g, tr = paths_log(ind, c)
        if c:
            rand_trades[a:b] = tr.sum(1)
        for key, mm in MASKS.items():
            rand[c][key][a:b] = cagr(g, mm)
    print(f'random {b}/{C.N_RANDOM}', flush=True)

rule = {}
for c in (0.0, C.COST):
    g, tr = paths_log(rule_ind, c)
    if c:
        rule_trades = int(tr.sum())
    rule[c] = {key: float(cagr(g, mm)[0]) for key, mm in MASKS.items()}
bh = {key: float(np.exp(lr[mm].sum() * C.DPY / mm.sum()) - 1) for key, mm in MASKS.items()}
cash = {key: float(np.exp(lrf[mm].sum() * C.DPY / mm.sum()) - 1) for key, mm in MASKS.items()}


def timing(key, c):
    d = rand[c][key]
    med = float(np.median(d))
    return {'rule': rule[c][key], 'median_random': med, 'diff_pts': 100 * (rule[c][key] - med),
            'p': float((1 + np.sum(d >= rule[c][key])) / (1 + len(d))),
            'random_p5': float(np.percentile(d, 5)), 'random_p95': float(np.percentile(d, 95))}


# ---------------------------------------------------------------- report
say(f'\nSeed {C.SEED}; {C.N_RANDOM} random timings; trades over the span: rule {rule_trades} '
    f'(entries + exits), random median {np.median(rand_trades):.0f}')
for pn in list(C.PERIODS) + [b for b in C.BLOCKS if b not in C.PERIODS]:
    say(f'\n=== {pn}')
    for en in EXCL:
        key = (pn, en)
        if key not in MASKS:
            continue
        mm = MASKS[key]
        s = daily_stats(mm)
        t0, t1 = timing(key, 0.0), timing(key, C.COST)
        RES[f'{pn} | {en}'] = {'daily': s, 'timing_0bp': t0, 'timing_1bp': t1, 'buy_hold': bh[key],
                               'cash': cash[key], 'days': int(mm.sum())}
        say(f' [{en}] days {int(mm.sum())}')
        fmt_daily('SPY excess', s)
        say(f'     rule vs random, 0 bps: rule {pct(t0["rule"], 2)}, median random {pct(t0["median_random"], 2)}, '
            f'diff {t0["diff_pts"]:+.2f} pts, p {t0["p"]:.4f} | 1 bp/side: rule {pct(t1["rule"], 2)}, median '
            f'{pct(t1["median_random"], 2)}, diff {t1["diff_pts"]:+.2f}, p {t1["p"]:.4f} | buy-hold '
            f'{pct(bh[key], 2)}, cash {pct(cash[key], 2)}')

# per-year TOM minus other (SPY excess)
say('\n=== by year: SPY excess TOM mean - other mean (%), EW market diff, rule - median random (0 bps, pts)')
peryear = {}
for yy in range(2006, 2026):
    mm = y == yy
    tt, oo = mm & tom, mm & ~tom
    dspy = float(ex[tt].mean() - ex[oo].mean())
    dew = float(mkx[tt].mean() - mkx[oo].mean())
    peryear[yy] = {'spy_diff': dspy, 'ew_diff': dew}
say('  ' + ' '.join(f'{yy}:{100 * v["spy_diff"]:+.3f}' for yy, v in peryear.items()))
say(f'  SPY TOM ahead in {sum(v["spy_diff"] > 0 for v in peryear.values())} of {len(peryear)} years '
    f'(2006-2019: {sum(peryear[yy]["spy_diff"] > 0 for yy in range(2006, 2020))} of 14); EW ahead in '
    f'{sum(v["ew_diff"] > 0 for v in peryear.values())} of {len(peryear)}')
RES['by_year'] = peryear

# neighbours of the window, reported only (Etula et al. 2020 comparison)
s_, e_, ks, ke = C.tom_position(do, dn)
s_, e_ = s_[span], e_[span]
nb = {}
for pn, (lo, hi) in C.PERIODS.items():
    mm = pmask(lo, hi)
    o = mm & ~tom
    row = {}
    for lbl, sel in (('-5', e_ == -5), ('-4', e_ == -4), ('-3', e_ == -3), ('-2', e_ == -2), ('+4', s_ == 4),
                     ('+5', s_ == 5)):
        q = mm & sel
        row[lbl] = {'mean_ex': float(ex[q].mean()), 't_vs_other': C.welch(ex[q], ex[o])[2], 'n': int(q.sum())}
    nb[pn] = row
    say(f'  neighbours {pn}: ' + '; '.join(f'{k}: {pct(v["mean_ex"])} t {v["t_vs_other"]:+.2f}' for k, v in row.items()))
RES['neighbours'] = nb

# Lakonishok & Smidt comparison: mean 4-day TOM compound return vs mean calendar-month return (SPY, raw total)
ls = {}
for pn, (lo, hi) in C.PERIODS.items():
    mm = pmask(lo, hi)
    win = []
    # a window = the -1 day of month k plus +1..+3 of month k+1 (complete windows inside the period)
    idx = np.nonzero(mm & (lab == -1))[0]
    for i in idx:
        if i + 3 < T and mm[i + 3] and list(lab[i:i + 4]) == [-1, 1, 2, 3]:
            win.append(float(np.prod(1 + r[i:i + 4]) - 1))
    mon = [float(np.prod(1 + r[mm & (mkey == k)]) - 1) for k in np.unique(mkey[mm])
           if (mm & (mkey == k)).sum() == (mkey == k).sum() and k != 2025 * 12 + 6]
    ls[pn] = {'tom_4day_mean': float(np.mean(win)), 'n_windows': len(win), 'month_mean': float(np.mean(mon)),
              'n_months': len(mon)}
    say(f'  L&S comparison {pn}: mean 4-day TOM return {pct(np.mean(win), 3)} ({len(win)} windows), mean month '
        f'{pct(np.mean(mon), 3)} ({len(mon)} months)')
RES['ls'] = ls

# dividend bias in the EW market after 2022 (reported only)
try:
    E2 = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
    dfix = pubdata.div_fix('old')
    i0 = int(np.searchsorted(do, C.dnum('2023-01-01')))
    i1 = len(do)
    cl = np.asarray(E2['close'][:, i0 - 1:i1], np.float32)
    have = np.isfinite(cl[:, 1:]) & np.isfinite(cl[:, :-1])
    fx = np.asarray(dfix[:, i0:i1], np.float32)
    cs = np.where(have, fx, 0).sum(0) / np.maximum(have.sum(0), 1)
    lab_all = C.tom_label(do, dn)[i0:i1]
    t_ = lab_all != 0
    RES['div_bias'] = {'tom': float(cs[t_].mean()), 'other': float(cs[~t_].mean())}
    say(f'\n  div_fix cross-sectional mean 2023-01..2025-06 (stocks with closes): TOM {100 * cs[t_].mean():.4f}%/day, '
        f'other {100 * cs[~t_].mean():.4f}%/day: missing-dividend bias on the EW TOM-minus-other difference '
        f'{100 * (cs[~t_].mean() - cs[t_].mean()):+.4f}% (positive = flatters TOM)')
    del cl, fx, have
except Exception as exc:  # reported only
    say('div bias check failed:', repr(exc))

# headline
k = ('2006-2019', 'all days')
s = RES['2006-2019 | all days']
say('\n=== TESTS')
t1a = s['daily']['t'] >= 2
t1b = s['timing_0bp']['p'] < 0.10
say(f' 1. 2006-2019: diff {pct(s["daily"]["diff"])}, Welch t {s["daily"]["t"]:+.2f} (pass t >= 2: {t1a}); planned se '
    f'0.05%, actual se {pct(s["daily"]["se"])}; rule - median random {s["timing_0bp"]["diff_pts"]:+.2f} pts, p '
    f'{s["timing_0bp"]["p"]:.4f} (pass p < 0.10: {t1b})')
sign = np.sign(s['daily']['diff'])
t2 = np.sign(s['daily']['ew']['diff']) == sign
say(f' 2. EW market 2006-2019 diff {pct(s["daily"]["ew"]["diff"])} t {s["daily"]["ew"]["t"]:+.2f}: same sign {bool(t2)}')
s3 = RES['2020-2025-06 | all days']
t3 = np.sign(s3['daily']['diff']) == sign
say(f' 3. 2020-2025-06 diff {pct(s3["daily"]["diff"])} t {s3["daily"]["t"]:+.2f}: same sign {bool(t3)} '
    '(unseen months: unseen.py)')
w = RES['2006-2025-06 | all days']['timing_0bp']
say(f' NUMBER: rule - median random, 2006-2025-06, 0 bps: {w["diff_pts"]:+.1f} points a year (p {w["p"]:.4f})')
RES['tests'] = {'t1a': bool(t1a), 't1b': bool(t1b), 't2': bool(t2), 't3_2020': bool(t3),
                'number': round(w['diff_pts'], 1)}
RES['meta'] = {'seed': C.SEED, 'n_random': C.N_RANDOM, 'rule_trades': rule_trades,
               'random_trades_median': float(np.median(rand_trades))}

(C.HERE / 'results_old.txt').write_text('\n'.join(out) + '\n')
(C.HERE / 'results_old.json').write_text(json.dumps(RES, indent=1, default=str))
np.savez_compressed(C.WORK / 'random_old.npz', **{f'{c}|{a}|{b}': v for c, d in rand.items() for (a, b), v in d.items()})
