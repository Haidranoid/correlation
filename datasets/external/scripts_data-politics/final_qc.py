"""Final QC over every series in catalog_data-politics.csv: ranges, duplicates, sort, gaps, head/tail."""
import csv
import datetime as dt
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXT, SERIES, WORK

cat = list(csv.DictReader(open(os.path.join(EXT, 'catalog_data-politics.csv'))))
RANGES = {  # subcategory -> (lo, hi) plausible value range
    'policy_uncertainty_state': (0, 5000), 'policy_uncertainty_country': (0, 5000), 'policy_uncertainty_global': (0, 5000),
    'policy_uncertainty_us_category': (0, 20000), 'policy_uncertainty_us': (0, 3000),
    'policy_uncertainty_twitter': (0, 100000), 'geopolitical_risk_country': (0, 100), 'partisan_conflict': (0, 1000),
    'protest': (0, 1e6), 'violent_conflict': (0, 1e6), 'covid_policy_stringency': (0, 100),
    'presidential_approval': (0, 100), 'election_calendar': (0, 1), 'federal_control_change': (-3, 3),
    'fiscal_standoff': (0, 1), 'state_partisan_control': (-50, 50), 'election_closeness': (-100, 100),
    'election_market': (-10000, 10000), 'federal_policy_activity': (0, 5000)}
lines = []
bad = 0
for e in cat:
    sid = e['id']
    path = os.path.join(SERIES, sid + '.csv')
    with open(path) as f:
        r = csv.reader(f)
        hdr = next(r)
        rows = [(d, v) for d, v in r]
    assert hdr == ['date', 'value'], (sid, hdr)
    dates = [dt.date.fromisoformat(d) for d, _ in rows]
    vals = [float(v) for _, v in rows]
    probs = []
    if len(set(dates)) != len(dates):
        probs.append('DUP')
    if dates != sorted(dates):
        probs.append('UNSORTED')
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        probs.append('NAN')
    if any(v in (-9999.0, -999.0, 9999.0, -99.0) for v in vals):
        probs.append('SENTINEL')
    if any(s.strip() == '' or s.lower() in ('nan', 'na', 'null') for _, s in rows):
        probs.append('BLANK')
    lo, hi = RANGES.get(e['subcategory'], (-1e12, 1e12))
    if min(vals) < lo or max(vals) > hi:
        probs.append(f'RANGE[{min(vals)},{max(vals)}]')
    if e['transform'] == 'logdiff' and min(vals) <= 0:
        probs.append('LOGDIFF_NONPOS')
    if e['first'] != rows[0][0] or e['last'] != rows[-1][0] or int(e['rows']) != len(rows):
        probs.append('CATALOG_MISMATCH')
    # gaps by frequency
    gaps = []
    if e['frequency'] == 'daily':
        gaps = [(a, b, (b - a).days - 1) for a, b in zip(dates, dates[1:]) if (b - a).days > 1]
    elif e['frequency'] == 'monthly':
        gaps = [(a, b, (b.year * 12 + b.month) - (a.year * 12 + a.month) - 1) for a, b in zip(dates, dates[1:])
                if (b.year * 12 + b.month) - (a.year * 12 + a.month) > 1]
    elif e['frequency'] == 'weekly':
        gaps = [(a, b, (b - a).days // 7 - 1) for a, b in zip(dates, dates[1:]) if (b - a).days > 13]
    lg = max(gaps, key=lambda g: g[2]) if gaps else None
    nz = sum(1 for v in vals if v != 0)
    bad += bool(probs)
    lines.append(f"{sid}\t{e['frequency']}\t{e['transform']}\t{e['known_after_days']}\t{rows[0][0]}..{rows[-1][0]}\t"
                 f"n={len(rows)}\tmin={min(vals):.6g}\tmax={max(vals):.6g}\tnonzero={nz}\tgaps={len(gaps)}"
                 f"{'(largest ' + str(lg[0]) + '..' + str(lg[1]) + ' ' + str(lg[2]) + ')' if lg else ''}\t"
                 f"{'PROBLEMS:' + ','.join(probs) if probs else 'ok'}\thead={rows[:2]}\ttail={rows[-2:]}")
out = os.path.join(WORK, 'final_qc.tsv')
open(out, 'w').write('\n'.join(lines) + '\n')
print(f'{len(cat)} series checked, {bad} with problems -> {out}')
for l in lines:
    if 'PROBLEMS' in l:
        print('  ', l[:300])
