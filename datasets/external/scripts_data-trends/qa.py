"""QA computations for the SOURCES report (no network). Writes qa/qa.json and prints tables."""
import glob
import json
import os
import sys
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import RAW, QA, read_request, compare, weekly_mean_of_daily, chain  # noqa: E402
from terms import CORE, TICKERS, KEY8, slug  # noqa: E402

out = {}


def meta_time(p):
    try:
        return json.load(open(p[:-4] + '.meta.json'))['fetched_utc']
    except Exception:
        return None


# 1. duplicate pulls
dup = []
for kw in KEY8:
    a = os.path.join(RAW, 'weekly_joint', slug(kw) + '.txt')
    ra, _ = read_request(a)
    for tag in ('shift1d', 'exact', 'late'):
        b = os.path.join(RAW, 'weekly_joint_repeat', f'{slug(kw)}__{tag}.txt')
        if not os.path.exists(b):
            continue
        rb, _ = read_request(b)
        ks = [k for k in ra if k in rb]
        nd = sum(1 for k in ks if ra[k] != rb[k])
        c = compare(ra, rb)
        dup.append({'term': kw, 'repeat': tag, 't_orig': meta_time(a), 't_rep': meta_time(b),
                    'weeks': len(ks), 'weeks_differing': nd, 'byte_identical': open(a).read() == open(b).read(),
                    **(c or {})})
out['duplicates'] = dup
print('\n# duplicate pulls')
for d in dup:
    print(f"{d['term']:20s} {d['repeat']:8s} {d['t_orig'][11:19]} -> {d['t_rep'][11:19]} differ {d['weeks_differing']:4d}/{d['weeks']} "
          f"corr lvl {d.get('corr_level', float('nan')):.4f} log {d.get('corr_log', float('nan')):.4f} dlog {d.get('corr_logdiff') or float('nan'):.4f}")

# 2. Google method-change level shifts (2011, 2016, 2022): median across terms of
#    mean log level in weeks 2-9 of year Y minus weeks -10..-3 before Jan 1 of Y
shifts = {}
series = {}
for kw, g, s in CORE:
    p = os.path.join(RAW, 'weekly_joint', slug(kw) + '.txt')
    if os.path.exists(p) and not s:
        r, _ = read_request(p)
        if np.mean([v <= 0 for v in r.values()]) < 0.05:
            series[kw] = r
for Y in range(2005, 2027):
    j1 = date(Y, 1, 1)
    vals = []
    for kw, r in series.items():
        aft = [np.log(max(v, 0.5)) for k, v in r.items() if j1 + timedelta(days=7) <= k < j1 + timedelta(days=63)]
        bef = [np.log(max(v, 0.5)) for k, v in r.items() if j1 - timedelta(days=70) <= k < j1 - timedelta(days=21)]
        if len(aft) >= 6 and len(bef) >= 6:
            vals.append(np.mean(aft) - np.mean(bef))
    if vals:
        shifts[Y] = {'median_logshift': float(np.median(vals)), 'n_terms': len(vals),
                     'frac_positive': float(np.mean(np.array(vals) > 0))}
out['newyear_level_shift'] = shifts
med_all = np.median([v['median_logshift'] for v in shifts.values()])
sd_all = np.std([v['median_logshift'] for v in shifts.values()])
print('\n# new-year level shift (median over', len(series), 'non-seasonal terms); typical', round(float(med_all), 3), 'sd', round(float(sd_all), 3))
for Y, v in shifts.items():
    flag = ' <== announced change' if Y in (2011, 2016, 2022) else ''
    print(Y, round(v['median_logshift'], 3), round(v['frac_positive'], 2), flag)

# 3. probe comparisons: single 5y window vs joint; daily probe block vs stitched daily
pr = {}
p1 = os.path.join(RAW, 'probe', 'debt_weekly_2004-01-01_2008-12-31.txt')
j, _ = read_request(os.path.join(RAW, 'weekly_joint', 'debt.txt'))
s1, _ = read_request(p1)
pr['debt_single5y_vs_joint'] = compare(s1, j)
for kw in ('debt', 'stock_market_crash'):
    bl = glob.glob(os.path.join(RAW, 'daily_blocks', kw, '*.txt'))
    dq = sorted(glob.glob(os.path.join(RAW, 'daily_req', kw, '*.txt')))
    if bl and dq:
        b, _ = read_request(bl[0])
        pieces = [read_request(p)[0] for p in dq]
        st, _ = chain(pieces, None)
        pr[f'{kw}_daily_probe_block_vs_stitched'] = compare(b, st)
out['probe'] = pr
print('\n# probe comparisons')
for k, v in pr.items():
    print(k, json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in (v or {}).items()}))

json.dump(out, open(os.path.join(QA, 'qa.json'), 'w'), indent=1, default=str)
