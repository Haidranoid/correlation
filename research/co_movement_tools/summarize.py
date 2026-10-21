"""Summarize the co-movement tools backtest (crates/c2-census/examples/co_movement_backtest.rs).

Reads the TSVs the example wrote under ~/.local/share/correlation2/experiments/co_movement_tools/{raw,clean}/
and prints the tables for experiments/+5.1-co-movement-tools.md. The design is fixed in that file ("The
test"); this script only aggregates.

    python3 -I research/co_movement_tools/summarize.py > research/co_movement_tools/results.txt
"""
import csv
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np  # noqa: E402

WORK = Path.home() / '.local/share/correlation2/experiments/co_movement_tools'
UNSEEN = '2025-10-17'


def read(run, name):
    rows = list(csv.DictReader(open(WORK / run / name), delimiter='\t'))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def years_of(t):
    return sorted(set(t['as_of']))


def label(y, t):
    i = np.where(t['as_of'] == y)[0][0]
    return f"{y[:7]}..{t['next_to'][i][:7]}"


def mse(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float('nan'))


def by_year(t, value, mask=None, fn=np.mean):
    ys = years_of(t)
    m = np.ones(len(t['as_of']), bool) if mask is None else mask
    return np.array([fn(value[(t['as_of'] == y) & m]) for y in ys])


def summary(per_year, ys):
    """Overall (18 years) mean and se, the 17 seen years, and the unseen year."""
    seen = np.array([y != UNSEEN for y in ys])
    a, se = mse(per_year)
    b, se_b = mse(per_year[seen])
    return a, se, b, se_b, per_year[~seen][0], int((per_year > 0).sum())


def fmt(x, d=1, sign=True):
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def diversify(run):
    t = read(run, 'diversify.tsv')
    ys = years_of(t)
    pick = t['next_vol_pick']
    imp = {
        'uniform': 100 * (1 - pick / t['next_vol_uniform']),
        'median': 100 * (1 - pick / t['next_vol_uniform_median']),
        'matched': 100 * (1 - pick / t['next_vol_matched']),
        'spy': 100 * (1 - pick / t['next_vol_spy']),
        'nothing': 100 * (1 - pick / t['next_vol']),
    }
    bets_vs_random = t['next_bets_pick'] - t['next_bets_uniform']
    bets_vs_nothing = t['next_bets_pick'] - t['next_bets']
    hit = (pick < t['next_vol_uniform']).astype(float)
    out = [f"=== Best diversifiers ({run}) ===",
           "Per year: mean over 300 portfolios x 2 arms of 100 x (1 - next-year vol with the tool's pick / with a random pick). + = lower volatility = better.",
           f"{'year':<18}{'vs uniform':>11}{'vs median':>10}{'vs vol-matched':>15}{'vs SPY':>8}{'vs nothing':>11}{'beats random':>13}{'pick rank':>10}{'bets vs random':>15}{'bets vs nothing':>16}"]
    py = {k: by_year(t, v) for k, v in imp.items()}
    py_b = by_year(t, bets_vs_random)
    py_bn = by_year(t, bets_vs_nothing)
    py_hit = by_year(t, hit)
    py_rank = by_year(t, t['rank_pick'], fn=np.median)
    for i, y in enumerate(ys):
        out.append(f"{label(y, t):<18}{fmt(py['uniform'][i]):>11}{fmt(py['median'][i]):>10}{fmt(py['matched'][i]):>15}{fmt(py['spy'][i]):>8}"
                   f"{fmt(py['nothing'][i]):>11}{100 * py_hit[i]:>12.0f}%{py_rank[i]:>10.2f}{fmt(py_b[i], 2):>15}{fmt(py_bn[i], 2):>16}")
    res = {}
    for k in imp:
        a, se, b, se_b, last, ahead = summary(py[k], ys)
        res[k] = (a, se, b, se_b, last, ahead)
        out.append(f"vs {k:<8}: all 18 years {fmt(a)} (se {se:.1f}; better in {ahead}/18 years); 17 seen years {fmt(b)} (se {se_b:.1f}); unseen year {fmt(last)}")
    a, se, b, se_b, last, _ = summary(py_b, ys)
    out.append(f"bets vs random: 18 years {fmt(a, 2)} (se {se:.2f}); seen {fmt(b, 2)}; unseen {fmt(last, 2)}")
    a, se, b, se_b, last, _ = summary(py_bn, ys)
    out.append(f"bets vs nothing added (5 holdings): 18 years {fmt(a, 2)} (se {se:.2f}); seen {fmt(b, 2)}; unseen {fmt(last, 2)}")
    out.append(f"pick beats the uniform-random expectation: {100 * hit.mean():.0f}% of cases; median rank of the pick among eligible candidates next year {np.median(t['rank_pick']):.2f} (0 = best)")
    # By arm and portfolio set.
    out.append("By arm and set (mean of yearly values, se over 18 years; vs uniform / vs vol-matched / vs SPY; unseen year in brackets):")
    for arm in ('fund', 'stock'):
        for s in ('random', 'concentrated', None):
            m = (t['arm'] == arm) & ((t['set'] == s) if s else True)
            parts = []
            for k in ('uniform', 'matched', 'spy'):
                v = by_year(t, imp[k], m)
                a, se, _, _, last, _ = summary(v, ys)
                parts.append(f"{fmt(a)} ± {se:.1f} ({fmt(last)})")
            out.append(f"  {arm:<6}{(s or 'all'):<13}" + "   ".join(parts))
    # What the picks were.
    for arm in ('fund', 'stock'):
        m = t['arm'] == arm
        tick, cnt = np.unique(t['pick'][m], return_counts=True)
        top = sorted(zip(cnt, tick), reverse=True)[:12]
        out.append(f"most frequent {arm} picks: " + ', '.join(f"{k} {c}" for c, k in top))
    out.append(f"pick's past volatility (median): fund {np.median(t['pick_vol'][t['arm'] == 'fund']):.3f}, stock {np.median(t['pick_vol'][t['arm'] == 'stock']):.3f}; "
               f"uniform candidates' (mean of means): fund {t['past_vol_uniform'][t['arm'] == 'fund'].mean():.3f}, stock {t['past_vol_uniform'][t['arm'] == 'stock'].mean():.3f}; "
               f"vol-matched set: fund {np.median(t['past_vol_matched'][t['arm'] == 'fund']):.3f}, stock {np.median(t['past_vol_matched'][t['arm'] == 'stock']):.3f}")
    pred = 100 * (1 - t['pred_vol_after'] / t['past_vol'])
    real = 100 * (1 - pick / t['next_vol'])
    out.append(f"the tool's predicted vol cut from adding the pick (vs not adding): {pred.mean():+.1f}%; realized next year: {real.mean():+.1f}%; "
               f"corr across cases {np.corrcoef(pred, real)[0, 1]:+.2f}")
    return '\n'.join(out), res, py


def hedge(run):
    t = read(run, 'hedge.tsv')
    ys = years_of(t)
    out = [f"=== Hedge finder ({run}) ===",
           "Risk left next year = sd(stock - ratio x hedge) / sd(stock), at the past ratio. Lower = better. Per year: mean (median) over ~500 stocks.",
           f"{'year':<18}{'top tracker':>16}{'top fund':>16}{'random fund':>16}{'SPY':>16}{'predicted (top)':>16}{'top < SPY':>10}{'top < random':>13}"]
    med = lambda v: np.nanmedian(v)  # noqa: E731
    mean = lambda v: np.nanmean(v)  # noqa: E731
    cols = {k: (by_year(t, t[k], fn=mean), by_year(t, t[k], fn=med)) for k in ('top_next', 'fund_next', 'funds_next', 'spy_next', 'top_pred')}
    beats_spy = by_year(t, (t['top_next'] < t['spy_next']).astype(float))
    beats_rand = by_year(t, (t['top_next'] < t['funds_next']).astype(float))
    for i, y in enumerate(ys):
        cells = ''.join(f"{100 * cols[k][0][i]:>9.1f}% ({100 * cols[k][1][i]:.0f})" for k in ('top_next', 'fund_next', 'funds_next', 'spy_next', 'top_pred'))
        out.append(f"{label(y, t):<18}{cells}{100 * beats_spy[i]:>9.0f}%{100 * beats_rand[i]:>12.0f}%")
    res = {}
    for name, a_, b_ in (('top vs random fund', 'funds_next', 'top_next'), ('top vs SPY', 'spy_next', 'top_next'), ('top fund vs random fund', 'funds_next', 'fund_next'),
                         ('top fund vs SPY', 'spy_next', 'fund_next')):
        for stat, fn in (('mean', mean), ('median', med)):
            d = 100 * (by_year(t, t[a_], fn=fn) - by_year(t, t[b_], fn=fn))
            a, se, b, se_b, last, ahead = summary(d, ys)
            res[(name, stat)] = (a, se, last)
            out.append(f"{name:<24} {stat:<6}: risk left lower by {a:+.1f} points (se {se:.1f}; {ahead}/18 years); seen {b:+.1f}; unseen {last:+.1f}")
    for k in ('top_next', 'fund_next', 'funds_next', 'spy_next'):
        out.append(f"  {k:<11} all years: mean {100 * np.nanmean(t[k]):.1f}%, median {100 * np.nanmedian(t[k]):.1f}%, worst {100 * np.nanmax(t[k]):.0f}%, missing {int(np.isnan(t[k]).sum())}")
    bias = 100 * (t['top_next'] - t['top_pred'])
    out.append(f"top tracker: predicted risk left {100 * np.nanmean(t['top_pred']):.1f}% (median {100 * np.nanmedian(t['top_pred']):.1f}), realized {100 * np.nanmean(t['top_next']):.1f}% "
               f"(median {100 * np.nanmedian(t['top_next']):.1f}); median realized - predicted {np.nanmedian(bias):+.1f} points; corr {np.corrcoef(t['top_pred'][~np.isnan(bias)], t['top_next'][~np.isnan(bias)])[0, 1]:+.2f}")
    kind = t['top_kind']
    twins = t['top_corr'] >= 0.97
    out.append(f"top tracker is a fund in {100 * (kind == 'fund').mean():.0f}% of cases, a stock in {100 * (kind == 'stock').mean():.0f}%; past correlation >= 0.97 (share-class twins, leveraged copies) in {100 * twins.mean():.0f}%")
    out.append(f"  excluding those: top tracker realized median {100 * np.nanmedian(t['top_next'][~twins]):.1f}% vs random fund median {100 * np.nanmedian(t['funds_next'][~twins]):.1f}% vs SPY median {100 * np.nanmedian(t['spy_next'][~twins]):.1f}%")
    out.append(f"top fund's rank among all scanned funds next year (median, 0 = best): {np.nanmedian(t['fund_rank']):.3f}")
    return '\n'.join(out), res


def portfolio(run):
    t = read(run, 'portfolio.tsv')
    ys = years_of(t)
    out = [f"=== Portfolio check ({run}) ===",
           "Independent bets: predicted (last year) vs realized (next year), 300 portfolios per year.",
           f"{'year':<18}{'predicted':>10}{'realized':>10}{'corr':>7}{'corr random':>12}{'corr conc.':>11}{'bias':>7}{'MAE':>7}{'MAE flat':>9}{'vol corr':>9}"]
    rows = []
    for y in ys:
        m = t['as_of'] == y
        p, r = t['past_bets'][m], t['next_bets'][m]
        c = np.corrcoef(p, r)[0, 1]
        cr = np.corrcoef(p[t['set'][m] == 'random'], r[t['set'][m] == 'random'])[0, 1]
        cc = np.corrcoef(p[t['set'][m] == 'concentrated'], r[t['set'][m] == 'concentrated'])[0, 1]
        mae = np.abs(r - p).mean()
        flat = np.abs(r - p.mean()).mean()
        vc = np.corrcoef(t['past_vol'][m], t['next_vol'][m])[0, 1]
        rows.append((c, cr, cc, (r - p).mean(), mae, flat, vc, p.mean(), r.mean()))
        out.append(f"{label(y, t):<18}{p.mean():>10.2f}{r.mean():>10.2f}{c:>+7.2f}{cr:>+12.2f}{cc:>+11.2f}{(r - p).mean():>+7.2f}{mae:>7.2f}{flat:>9.2f}{vc:>+9.2f}")
    a = np.array(rows)
    res = {}
    for i, name in enumerate(('corr', 'corr random', 'corr concentrated', 'bias', 'MAE', 'MAE flat', 'vol corr')):
        m, se, b, se_b, last, _ = summary(a[:, i], ys)
        res[name] = (m, se, last)
        out.append(f"{name:<18}: 18 years {m:+.2f} (se {se:.2f}); seen {b:+.2f}; unseen {last:+.2f}")
    pooled = np.corrcoef(t['past_bets'], t['next_bets'])[0, 1]
    out.append(f"pooled over all 5,400 portfolios: corr {pooled:+.2f}; realized / predicted bets (median) {np.median(t['next_bets'] / t['past_bets']):.2f}")
    return '\n'.join(out), res


def groups(run):
    t = read(run, 'groups.tsv')
    ys = years_of(t)
    out = [f"=== Groups map ({run}) ===",
           "Within-group mean correlation (all within-group pairs) vs random groups of the same sizes (= mean over all pairs; sd from 200 relabelings).",
           f"{'year':<18}{'past within':>12}{'next within':>12}{'next random':>12}{'z':>8}{'kept':>7}{'next raw within':>16}{'next raw random':>16}"]
    kept = (t['next_rel_within'] - t['next_rel_all']) / (t['past_rel_within'] - t['past_rel_all'])
    z = (t['next_rel_within'] - t['next_rel_shuffle_mean']) / t['next_rel_shuffle_sd']
    for i, y in enumerate(ys):
        out.append(f"{label(y, t):<18}{t['past_rel_within'][i]:>+12.3f}{t['next_rel_within'][i]:>+12.3f}{t['next_rel_all'][i]:>+12.3f}{z[i]:>8.0f}{100 * kept[i]:>6.0f}%"
                   f"{t['next_raw_within'][i]:>+16.3f}{t['next_raw_all'][i]:>+16.3f}")
    res = {}
    for name, v in (('past within', t['past_rel_within']), ('next within', t['next_rel_within']), ('next random', t['next_rel_all']),
                    ('next within - random', t['next_rel_within'] - t['next_rel_all']), ('kept', kept),
                    ('raw next within - random', t['next_raw_within'] - t['next_raw_all']), ('z', z)):
        m, se, b, se_b, last, _ = summary(np.asarray(v, float), ys)
        res[name] = (m, se, last)
        out.append(f"{name:<26}: 18 years {m:+.3f} (se {se:.3f}); seen {b:+.3f}; unseen {last:+.3f}")
    out.append(f"check: the re-implemented market-relative correlations match the tool's to within {t['max_diff_vs_tool'].max():.1e}")
    d = read(run, 'groups_detail.tsv')
    worse = (d['next_rel'] <= 0.02).mean()
    out.append(f"groups (of 360): next-year within-group market-relative corr <= 0.02 in {100 * worse:.0f}%; median group past {np.median(d['past_rel']):+.3f} -> next {np.median(d['next_rel']):+.3f}")
    return '\n'.join(out), res


def md_table(head, rows_by_year, ys, t, seen_row, unseen_row, all_row):
    """Markdown table: one row per seen year, then the 17 seen years, the unseen year and all 18."""
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    for i, y in enumerate(ys):
        if y != UNSEEN:
            out.append(f"| {label(y, t).replace('..', '–')} | " + ' | '.join(rows_by_year[i]) + ' |')
    out.append('| **17 seen years** | ' + ' | '.join(seen_row) + ' |')
    out.append(f"| **{label(UNSEEN, t).replace('..', '–')} (unseen)** | " + ' | '.join(unseen_row) + ' |')
    out.append('| **All 18 years** | ' + ' | '.join(all_row) + ' |')
    return '\n'.join(out)


def markdown(run='raw'):
    """The four results tables for the experiment file."""
    parts = []
    # Diversifiers.
    t = read(run, 'diversify.tsv')
    ys = years_of(t)
    seen = np.array([y != UNSEEN for y in ys])
    pick = t['next_vol_pick']
    cols = [by_year(t, 100 * (1 - pick / t[k])) for k in ('next_vol_uniform', 'next_vol_matched', 'next_vol_spy', 'next_vol')]
    hit = by_year(t, (pick < t['next_vol_uniform']).astype(float))
    bets = by_year(t, t['next_bets_pick'] - t['next_bets_uniform'])
    rows = [[fmt(c[i]) for c in cols] + [f"{100 * hit[i]:.0f}%", fmt(bets[i], 2)] for i in range(len(ys))]
    agg = lambda m, se=False: [fmt(c[m].mean()) + (f" (se {mse(c[m])[1]:.1f})" if se else '') for c in cols] + [f"{100 * hit[m].mean():.0f}%", fmt(bets[m].mean(), 2) + (f" (se {mse(bets[m])[1]:.2f})" if se else '')]  # noqa: E731
    all_row = agg(np.ones(len(ys), bool), True)
    all_row[0] = f"**{all_row[0]}**"
    parts.append(md_table(['Following year', 'vs uniform random', 'vs volatility-matched', 'vs SPY', 'vs adding nothing', 'Pick beats random', 'Bets vs random'],
                          rows, ys, t, agg(seen), agg(~seen), all_row))
    # Hedges.
    t = read(run, 'hedge.tsv')
    cols = [by_year(t, t[k], fn=np.nanmean) for k in ('top_next', 'fund_next', 'funds_next', 'spy_next', 'top_pred')]
    beat = by_year(t, (t['top_next'] < t['spy_next']).astype(float))
    rows = [[f"{100 * c[i]:.1f}%" for c in cols] + [f"{100 * beat[i]:.0f}%"] for i in range(len(ys))]
    agg = lambda m: [f"{100 * c[m].mean():.1f}%" for c in cols] + [f"{100 * beat[m].mean():.0f}%"]  # noqa: E731
    parts.append(md_table(['Following year', 'Top tracker', 'Top-tracking fund', 'Random fund', 'SPY at its beta', 'Predicted (top tracker)', 'Top tracker beats SPY'],
                          rows, ys, t, agg(seen), agg(~seen), agg(np.ones(len(ys), bool))))
    # Portfolio check.
    t = read(run, 'portfolio.tsv')
    vals = []
    for y in ys:
        m = t['as_of'] == y
        p, r = t['past_bets'][m], t['next_bets'][m]
        vals.append((p.mean(), r.mean(), np.corrcoef(p, r)[0, 1], (r - p).mean(), np.abs(r - p).mean(), np.abs(r - p.mean()).mean()))
    v = np.array(vals)
    f2 = lambda x, sign=False: f"{x:+.2f}" if sign else f"{x:.2f}"  # noqa: E731
    rows = [[f2(a[0]), f2(a[1]), f2(a[2], True), f2(a[3], True), f2(a[4]), f2(a[5])] for a in v]
    agg = lambda m: [f2(v[m, 0].mean()), f2(v[m, 1].mean()), f2(v[m, 2].mean(), True), f2(v[m, 3].mean(), True), f2(v[m, 4].mean()), f2(v[m, 5].mean())]  # noqa: E731
    parts.append(md_table(['Following year', 'Predicted bets', 'Realized bets', 'Correlation', 'Bias', 'Error', 'Error if all got the average'],
                          rows, ys, t, agg(seen), agg(~seen), agg(np.ones(len(ys), bool))))
    # Groups.
    t = read(run, 'groups.tsv')
    kept = (t['next_rel_within'] - t['next_rel_all']) / (t['past_rel_within'] - t['past_rel_all'])
    cols = [t['past_rel_within'], t['next_rel_within'], t['next_rel_all'], kept, t['next_raw_within'], t['next_raw_all']]
    f3 = lambda c, i: f"{100 * c[i]:.0f}%" if c is kept else f"{c[i]:+.3f}"  # noqa: E731
    rows = [[f3(c, i) for c in cols] for i in range(len(ys))]
    agg = lambda m: [f"{100 * c[m].mean():.0f}%" if c is kept else f"{c[m].mean():+.3f}" for c in cols]  # noqa: E731
    parts.append(md_table(['Following year', 'Within groups, past year', 'Within groups, next year', 'Random groups, next year', 'Excess kept', 'Raw: within groups', 'Raw: random groups'],
                          rows, ys, t, agg(seen), agg(~seen), agg(np.ones(len(ys), bool))))
    return '\n\n'.join(parts)


def main():
    import sys
    if '--markdown' in sys.argv:
        print(markdown())
        return
    for run in ('raw', 'clean'):
        if not (WORK / run).exists():
            continue
        print(f"##### run: {run} {'(next-year daily moves over 2x up or down blanked)' if run == 'clean' else '(data as the tool reads it; the pre-registered headline)'}\n")
        for f in (diversify, hedge, portfolio, groups):
            print(f(run)[0])
            print()


if __name__ == '__main__':
    main()
