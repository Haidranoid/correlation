"""The judge: the sanity check (5-day reversal through this path, rebalance=5: must give +11.8, as
+5.4-reversal-control.md) and the money test (score = -s for trading-set stocks passing the filter, rebalance=1,
uniform and volatility-matched random accounts) on the old export's walk-forward. With 'unseen', the money test on the
new export over 2025-06-30.. (once, at the end).

    python3 -I -B money.py sanity|old|unseen      # writes out/money_<what>.txt and .json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

what = sys.argv[1]
lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    lines.append(s)
    C.log(s)


def report(js, label):
    out = {}
    if 'summary' in js:
        say(label, c2eval.summary_line(js))
    for i, cb in enumerate((0, 5, 20)):
        ch = c2eval.change_over_random(js, i)
        out[cb] = ch
        say(f'  {label} {cb:>2} bps: change over random {ch["change"]:+.2f} pts/yr (se {ch["se"]:.2f}, ahead '
            f'{ch["ahead"]}/{ch["years"]}); per year {ch["per_year"]}')
    return out


res = {}
if what == 'sanity':
    E = c2eval.load_export(c2eval.EXPORT_OLD, arrays=('close',))
    c = np.asarray(E['close'], np.float64)
    s = np.full(c.shape, np.nan, np.float32)
    with np.errstate(all='ignore'):
        s[:, 5:] = -np.log(c[:, 5:] / c[:, :-5])
    p = c2eval.save_scores(s, 'etf_stat_arb/sanity_reversal5')
    txt, js = c2eval.judge(p, rebalance=5)
    res['reversal5'] = report(js, 'reversal5 r5 uniform')
elif what == 'old':
    s = np.load(C.WORK / 'scores_s_old_export_raw_all.npy')
    sc = -s
    fin = np.isfinite(sc)
    say(f'scores: stock-days with a score {int(fin.sum())}; per day median (2006+) '
        f'{int(np.median(fin.sum(0)[c2eval.day_index(np.load(c2eval.EXPORT_OLD / "days.npy"), "2006-01-01"):]))}')
    p = c2eval.save_scores(sc, 'etf_stat_arb/money_old')
    for rm in ('uniform', 'volatility'):
        txt, js = c2eval.judge(p, rebalance=1, random_match=rm)
        res[rm] = report(js, f'money r1 {rm}')
        res[rm + '_summary'] = js.get('summary')
elif what == 'unseen':
    s = np.load(C.WORK / 'scores_s_new_export_raw_all.npy')
    p = c2eval.save_scores(-s, 'etf_stat_arb/money_new', export=c2eval.EXPORT_NEW)
    for rm in ('uniform', 'volatility'):
        txt, js = c2eval.judge(p, export=c2eval.EXPORT_NEW, walk_forward=False, start='2025-06-30', rebalance=1,
                               random_match=rm)
        res[rm] = report(js, f'unseen r1 {rm}')
        say(txt)
(C.OUT / f'money_{what}.txt').write_text('\n'.join(lines) + '\n')
(C.OUT / f'money_{what}.json').write_text(json.dumps(res, indent=1, default=str))
