"""Write SOURCES_data-trends.md from build_info.json, qa.json, catalog and fetch log."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import EXT, OUT, QA, RAW  # noqa: E402
from terms import KEY8, WEEKLY_MR5, WEEKLY_WIN  # noqa: E402

info = json.load(open(os.path.join(QA, 'build_info.json')))
qa = json.load(open(os.path.join(QA, 'qa.json')))
cat = list(csv.DictReader(open(os.path.join(OUT, 'catalog_data-trends.csv'))))
log = [json.loads(l) for l in open(os.path.join(RAW, 'fetch_log.jsonl'))]
fails = [json.loads(l) for l in open(os.path.join(RAW, 'failures.jsonl'))] if os.path.exists(
    os.path.join(RAW, 'failures.jsonl')) else []
extra = json.load(open(os.path.join(QA, 'extra.json'))) if os.path.exists(os.path.join(QA, 'extra.json')) else {}

n_saved = sum(1 for e in log if e.get('ev') == 'saved')
n_fail = [e for e in log if e.get('ev') == 'http_fail']
n429 = sum(1 for e in n_fail if e.get('code') == 429)
t0 = min(e['t'] for e in log)
t1 = max(e['t'] for e in log)
http_total = sum(e.get('n_requests', 0) for e in log if e.get('ev') == 'runner_exit') + 5 + 3  # + probe (5) + pytrends test (3)

def fallback_text(items):
    thr = [s_['thr'] for k_, v_ in items for s_ in v_['stitch'][1:]]
    if not thr:
        return 'no junctions'
    from collections import Counter
    c = Counter('sum' if t == 0 else f'>={int(t)}' for t in thr)
    return 'thresholds used at the ' + str(len(thr)) + ' junctions: ' + ', '.join(f'{k} x{v}' for k, v in sorted(c.items()))


from build import read_request  # noqa: E402
from datetime import date as _date
qa_joint_span_min = {}
for k_, v_ in info.items():
    if v_['method'] == 'stitched':
        from terms import slug as _slug
        r_, _ = read_request(os.path.join(RAW, 'weekly_joint', _slug(v_['kw']) + '.txt'))
        sp_ = {}
        for d_, x_ in r_.items():
            if d_ >= _date(2004, 1, 1):
                sp_.setdefault(min((d_.year - 2004) // 5, 4), []).append(x_)
        qa_joint_span_min[v_['kw']] = float(min(np.median(x) for x in sp_.values()))

W = [r for r in cat if r['frequency'] == 'weekly']
D = [r for r in cat if r['frequency'] == 'daily']
L = []
a = L.append

a('# Google Trends search-interest series (agent `data-trends`)\n')
a(f'Fetched {t0[:16]}Z .. {t1[:16]}Z. Catalog: `catalog_data-trends.csv` ({len(cat)} series: '
  f'{len(W)} weekly, {len(D)} daily). Series files: `series/gt_<term>_w.csv`, `series/gt_<term>_d.csv`. '
  'Raw responses: `raw/google_trends/`.\n')

a('## What was downloaded\n')
a('Google Trends "interest over time", geography US (`geo=US`), all categories (`category=0`), web search '
  '(`property=""`), `hl=en-US`, `tz=300`. Values are Google\'s 0-100 index: the term\'s share of all US '
  'Google searches in each period, scaled so the maximum of the requested comparison is 100, rounded to '
  'integers, with "<1" reported as 0. Every published series is rescaled so its own maximum is 100.\n')
a('| set | terms | frequency | range | series |')
a('|---|---|---|---|---|')
grp = {}
for r in W:
    grp.setdefault(r['subcategory'], []).append(r)
for g, rs in grp.items():
    firsts = sorted(r['first'] for r in rs)
    a(f"| {g} | {len(rs)} | weekly | {firsts[0]} (latest start {firsts[-1]}) .. {max(r['last'] for r in rs)} | "
      f"{', '.join(r['id'] for r in rs)} |")
if D:
    a(f"| key terms | {len(D)} | daily | {min(r['first'] for r in D)} .. {max(r['last'] for r in D)} | "
      f"{', '.join(r['id'] for r in D)} |")
a('')

a('## Access route: what worked and what broke\n')
a('- Google has no free official bulk API. The Google Trends API (alpha, 2025) needs an approved application; '
  'not applied for, as instructed. Everything here uses the public, unofficial web endpoints that the '
  'Trends website itself calls. They are unsupported and can change or start blocking at any time.')
a('- Installed from PyPI into `/home/bhoult/programming/c2-wt/venv-data`: `pytrends 4.9.2` (last release '
  '2023), `trendspy 0.1.6` (pulls in `requests 2.34.2`, `pandas 3.0.6`, `numpy 2.5.4`).')
a('- **Used:** my own fetcher (`requests`), replicating trendspy 0.1.6\'s route, so that pacing, backoff and '
  'saving each raw response could be controlled:')
a('  1. `GET https://trends.google.com/trends/embed/explore/TIMESERIES?req={"comparisonItem":[{"keyword","time","geo"}...],'
  '"category":0,"property":""}&hl=en-US&tz=300` returns an HTML page with the widget JSON, including its '
  'signed `token`, inside `JSON.parse(\'...\')` (hex-escaped).')
a('  2. `GET https://trends.google.com/trends/api/widgetdata/multiline` (one time range) or `.../multirange` '
  '(several ranges) with `req=<widget.request>&token=<token>&hl&tz` returns `)]}\'`-prefixed JSON '
  '(`default.timelineData`).')
a('  - The explore page `https://trends.google.com/trends/explore?geo=US` (fetched once for a session '
  'cookie) itself answers HTTP 429 but still sets the `NID` cookie; the embed route worked from the first '
  'request.')
pt = extra.get('pytrends_test')
if pt:
    a(f"- pytrends 4.9.2 (`POST /trends/api/explore` then `widgetdata/multiline`), one test call at the end: {pt}")
else:
    a('- pytrends 4.9.2 (`POST /trends/api/explore` route) was not exercised: the embed route worked at once, '
      'and every request was budgeted for data.')
a(f'- Politeness: strictly one request at a time, 13-18 s (uniform jitter) between any two requests, '
  f'exponential backoff 60 s -> 15 min on 429. Totals: {http_total} HTTP requests (incl. 5 probe and 3 pytrends-test requests), {n_saved} saved responses, '
  f'{n429} HTTP 429s, each cleared after a single 60 s backoff. Other failures: '
  f'{len(n_fail) - n429}. Task-level failures: {len(fails)}.')
a('- Each data response is saved unmodified the moment it arrives (`raw/google_trends/<kind>/...txt`), with a '
  '`.meta.json` holding the request items, the widget request, endpoint and fetch time; the runner skips '
  'anything already saved, so it can resume. `raw/google_trends/fetch_log.jsonl` logs every request event.\n')

a('## Discovery: one multirange request = one joint scale (no stitching)\n')
a('A comparison with the same keyword over several time ranges returns a `fe_multi_range_chart` widget. Its '
  'data come back on one joint scale: the maximum over all ranges is 100, and weeks shared by two ranges hold '
  'identical values. Checked on a 2-range probe for "debt": range 1 max 39, range 2 max 100, 53 shared weeks '
  'identical. Up to 5 ranges per request are accepted, mixed lengths too (1826-day and 1020-day ranges together). '
  'So **one request gives a whole 2004-2026 weekly series without stitching** (ranges '
  + ', '.join(f'{x}..{y}' for x, y in WEEKLY_MR5) + '). Range-boundary weeks are reported whole in both '
  'ranges; their two values agree to within 1 index point, and I average them. At daily resolution it '
  'also works (5 ranges of 261 days come back daily on one scale), but there overlapping ranges are '
  'separate samples: shared days agree in scale (median ratio 1.000 on every pair) yet differ by a point or a '
  'few (see Repeat pulls). Shared days are averaged.\n')
a('The cost is integer precision. With one maximum across 22 years, eras of low volume land on 0-5, where the '
  '±0.5 rounding is ±10-100% per value. That matters for spiky terms (one crash or storm week = 100). '
  'Single windows (the spec\'s method) give each 5-year window its own maximum, so where it ran the published '
  'series uses it (next section).\n')

a('## Weekly series: method and stitching error\n')
st = [(k, v) for k, v in info.items() if v['method'] == 'stitched']
jt = [(k, v) for k, v in info.items() if v['method'] == 'joint']
a(f'- **{len(jt)} weekly series: joint multirange pull** (1 request each, no stitching). The weekly point is '
  'labelled with the Sunday that starts the week and covers Sun-Sat. The week starting 2003-12-28 holds only '
  'Jan 1-3 2004 and is dropped. The last week used is 2026-10-11..17, which was not flagged `isPartial`.')
a(f'- **{len(st)} weekly series: spec method.** Six overlapping single 5-year windows ('
  + ', '.join(f'{x}..{y}' for x, y in WEEKLY_WIN) + '). Each window is rescaled to the previous one by the median '
  'ratio over shared weeks where both are >= 5 (if fewer than 8 such weeks: ratio of sums over the overlap; ' + fallback_text(st) + '). The series switches windows at the overlap midpoint, then the whole series is rescaled to max = 100. '
  'Terms: ' + ', '.join(v['kw'] for k, v in st) + '.')
if st:
    a('- **Stitching error**, measured against the independent joint pull (a different request, so it carries '
      'its own sampling and rounding). For each window: median over weeks (joint >= 5) of stitched/joint. With '
      'perfect stitching these ratios are equal for all windows, so their spread is the accumulated chain error.\n')
    a('| series | per-window ratio stitched/joint (normalised) | sd | max abs | overlap n (thr) per junction | '
      'within-overlap sd of log ratio | weekly logdiff corr vs joint |')
    a('|---|---|---|---|---|---|---|')
    allsd = []
    for k, v in st:
        per = np.array([x for x in v['stitch_vs_joint_ratio_per_window']], float)
        ok = per[np.isfinite(per)]
        norm = ok / np.median(ok) if len(ok) else ok
        sds = [s['sd_log_ratio'] for s in v['stitch'][1:] if s['sd_log_ratio'] is not None]
        ns = [f"{s['n_overlap']}({int(s['thr']) if s['thr'] else 0})" for s in v['stitch'][1:]]
        allsd.append(v['stitch_error_sd_log'])
        vj = v.get('vs_joint') or {}
        sdt = f"{100*v['stitch_error_sd_log']:.1f}%" if v.get('stitch_error_sd_log') is not None else f"n/a ({v.get('stitch_error_n_windows', 0)} usable windows)"
        mxt = f"{100*v['stitch_error_maxabs_log']:.1f}%" if v.get('stitch_error_maxabs_log') is not None else 'n/a'
        a(f"| {k} | {' '.join(f'{x:.3f}' for x in norm)} | {sdt} | "
          f"{mxt} | {' '.join(ns)} | "
          f"{' '.join(f'{x:.3f}' for x in sds)} | {vj.get('corr_logdiff', float('nan')):.3f} |")
    clean = [(k, v) for k, v in st if min(v['span_median'].values()) >= 5 or True]
    ref_lim = []
    for k, v in st:
        j = qa_joint_span_min.get(v['kw'])
        if j is not None and j < 5:
            ref_lim.append((k, j))
    good = [(k, v) for k, v in st if k not in dict(ref_lim)]
    if ref_lim:
        a('\nNote: for ' + ', '.join(f'{k} (joint pull median {j:.0f} in its weakest 5-year span)' for k, j in ref_lim)
          + ' the spread is mostly the joint reference\'s own integer rounding, not chain error. The clean '
          'estimate comes from terms whose joint pull is precise throughout: '
          + ', '.join(f"{k} sd {100*v['stitch_error_sd_log']:.1f}% / max {100*v['stitch_error_maxabs_log']:.1f}%" for k, v in good if v.get('stitch_error_sd_log') is not None) + '.')
    gsd = [v['stitch_error_sd_log'] for k, v in good if v.get('stitch_error_sd_log') is not None]
    a(f"\nMedian per-window sd over the clean terms: {100*np.median(gsd):.1f}%. "
      'In a `logdiff` series a scale error only enters the moves that span a junction (5 junctions over 22 '
      'years); the joint pull has no junctions at all.\n')

a('## Daily series: method and stitching error\n')
dd = [(k, v) for k, v in info.items() if v['method'] == 'daily_stitched']
a('- Daily points exist only for ranges under 270 days. I used a fixed grid of 261-day windows with 45-day '
  'overlaps, laid back from 2026-10-17 (39 windows to 2004-01-01). Consecutive windows go into one multirange '
  'request (1-5 per request, joint scale) only when the weekly joint series says the group\'s peak is at '
  'most 1.5x each member\'s own peak, so grouping costs each window at most a third of its precision. Spiky '
  'stretches get single windows, calm stretches share a request.')
a('- Requests are chain-stitched over the 45-day overlaps: median ratio where both >= 5 (if fewer than 8 such days: '
  'ratio of sums over the overlap, which averages out integer rounding; ' + fallback_text(dd) + '), switch at the overlap midpoint, max -> 100. Dates come from `time` (00:00 UTC of the labelled day) and '
  'all match `formattedTime`. Day boundaries are Google\'s.')
a('- Check against the independent weekly joint pull: complete Sun-Sat weeks of each stitched daily series '
  'are averaged and compared with the weekly series. The table also shows the per-year drift of the ratio, '
  'which is where chain error would accumulate.\n')
a('| series | first..last | requests | days < 5 (own units) | median within-overlap sd | corr log level vs weekly | '
  'corr weekly logdiff | yearly drift range |')
a('|---|---|---|---|---|---|---|---|')
for k, v in dd:
    vj = v.get('vs_weekly_joint') or {}
    dr = v.get('drift_by_year') or {}
    drs = f"{min(dr.values()):.2f}..{max(dr.values()):.2f}" if dr else ''
    a(f"| {k} | {v['qa']['first']}..{v['qa']['last']} | {v['n_requests']} | {100*v['frac_lt5']:.0f}% | "
      f"{100*(v['junction_sd_log_ratio_median'] or 0):.1f}% | {vj.get('corr_log', float('nan')):.3f} | "
      f"{vj.get('corr_logdiff', float('nan')):.3f} | {drs} |")
a('\n**Sampling noise by era**, from overlapping ranges inside the same daily request (same scale, '
  'independent samples):\n')
a('| series | days shared before 2014-06 | sd log(b/a) before | 1-day logdiff corr before | days shared after | '
  'sd log(b/a) after | 1-day logdiff corr after |')
a('|---|---|---|---|---|---|---|')
for k, v in dd:
    nb = v.get('noise_by_era', {})
    pr, po = nb.get('pre2014', {}), nb.get('post2014', {})
    f_ = lambda x, pct=False: ('' if x is None else (f'{100*x:.1f}%' if pct else f'{x:.3f}'))
    a(f"| {k} | {pr.get('shared_days', '')} | {f_(pr.get('sd_log_ratio'), True)} | {f_(pr.get('corr_1day_logdiff'))} | "
      f"{po.get('shared_days', '')} | {f_(po.get('sd_log_ratio'), True)} | {f_(po.get('corr_1day_logdiff'))} |")
a('\nBefore about 2014, lower-volume terms are dominated by sampling noise at the 1-day horizon, even at '
  'similar index levels: Google sampled fewer searches for them. Their daily catalog rows carry a '
  '"NOISY BEFORE 2014" flag. For those years use moves of 5 days or more, or the weekly series.\n')
a('\nThe weekly-logdiff correlation is below 1 because the two pulls are different Google samples with '
  'different rounding. Daily series show a day-of-week pattern, with weekends about 20-30% lower than '
  'weekdays.\n')

a('## Repeat pulls (sampling noise)\n')
dup = qa.get('duplicates', [])
a('The 8 key weekly terms were pulled a second time 4-7 minutes later. To force a fresh computation, the second '
  'pull shifted every range boundary by 1 day; the weeks returned are the same. For "debt" I also sent an '
  'exact repeat.\n')
a('| term | repeat | t1 -> t2 (UTC) | weeks differing | corr level | corr log | corr weekly logdiff |')
a('|---|---|---|---|---|---|---|')
for d in dup:
    a(f"| {d['term']} | {d['repeat']} | {d['t_orig'][11:16]} -> {d['t_rep'][11:16]} | {d['weeks_differing']}/{d['weeks']} | "
      f"{d.get('corr_level', float('nan')):.4f} | {d.get('corr_log', float('nan')):.4f} | "
      f"{(d.get('corr_logdiff') or float('nan')):.4f} |")
a('')
a(extra.get('dup_text', ''))
a('')

a('## Other data-quality checks\n')
sh = qa.get('newyear_level_shift', {})
if sh:
    vals = {int(y): v['median_logshift'] for y, v in sh.items()}
    others = [v for y, v in vals.items() if y not in (2011, 2016, 2022)]
    a(f"- **Google method changes.** The Trends UI notes data-collection or geo-assignment changes from "
      f"2011-01-01, 2016-01-01 and 2022-01-01. Check: median over {list(sh.values())[0]['n_terms']} non-seasonal "
      f"core terms of the log level change from mid-Oct..Dec to Jan..Feb. In ordinary years it is "
      f"{100*np.median(others):+.1f}% (sd {100*np.std(others):.1f}%). 2011: {100*vals.get(2011, np.nan):+.1f}%, "
      f"2016: {100*vals.get(2016, np.nan):+.1f}%, 2022: {100*vals.get(2022, np.nan):+.1f}%. 2016 stands out "
      f"(a level step of roughly {100*(vals.get(2016, np.nan) - np.median(others)):+.0f}% across terms, "
      f"{(vals.get(2016, np.nan) - np.median(others))/np.std(others):.1f} sd). 2011 and 2022 are about "
      f"{(vals.get(2011, np.nan) - np.median(others))/np.std(others):.1f} and "
      f"{(vals.get(2022, np.nan) - np.median(others))/np.std(others):.1f} sd above ordinary years, too little to "
      "separate from normal year-to-year variation. In `logdiff` terms this is a one-week jump in the first week of "
      "2016, common to all terms.")
a('- Checked for every series: dates strictly increasing, no duplicate dates, no gaps (weekly = 7 days, '
  'daily = 1 day), all values finite and in [0, 100], no sentinels (Trends `-1` "no data" points are '
  'dropped before merging), `isPartial` points dropped (none were present). First and last rows were printed '
  'and inspected; see `qa_summary` below.')
a('- **Zero handling.** For `logdiff` terms, Trends "<1" (reported 0) is clamped to 0.5 in the units of the '
  'request it came from, before rescaling. So the floor is "half the smallest reportable value" of that '
  'window, and `log` is always defined. A leading stretch with no measurable volume is cut: the series starts '
  'at the first non-zero week from which the next 2 years have < 10% zeros (bitcoin, ethereum, buy bitcoin, '
  'tickers listed later, etc.). "crypto" is cut to 2017-01-01: before that it was 0-1 on the joint scale and '
  'mostly other senses of the word. Seasonal `anomaly_log1p` terms keep their zeros and are not trimmed.')
a('- **Precision flags** in `notes`: "MOSTLY ZEROS" means > 20% zeros; "LOW PRECISION" means > 50% of '
  'points below 5 in source-request units, where the ±0.5 integer rounding dominates week-to-week moves.\n')

a('## Transforms\n')
a('- `logdiff` for all non-seasonal terms (values > 0 after the 0.5 clamp).')
a('- `anomaly_log1p` for black friday, flu, coupons, hurricane and heat wave: holiday or weather-season driven, '
  'with near-zero levels off-season, so a log change would mostly measure the calendar. Caveats: weekly dates '
  'are Sundays, so day-of-year shifts by 1-2 days a year, and Black Friday moves with Thanksgiving '
  '(Nov 23-29). The same-day-of-year climatology therefore needs a ±3-day tolerance. "coupons" also has a '
  'strong 2009-2012 boom and later decline, so its anomaly carries a trend.')
seas = sorted([(v['season_r2'], v['kw']) for k, v in info.items() if v['method'] != 'daily_stitched' and
               v['transform'] == 'logdiff' and v.get('season_r2') is not None and np.isfinite(v['season_r2'])
               and v['season_r2'] > 0.25], reverse=True)
a('- Annual seasonality was measured for every weekly series as a cross-validated week-of-year R2: a median '
  'week-of-year profile of the detrended log level is fitted on odd years and scored on even years, and vice '
  'versa. These `logdiff` terms have R2 > 0.25, mostly from the school year and tax season, and are flagged in '
  '`notes`: ' + ', '.join(f'{k} {r:.2f}' for r, k in seas) + '. I kept `logdiff` for them. The spec\'s '
  '`anomaly` is a deviation from earlier years\' level, which these trending series would turn into a trend. '
  'The test should deseasonalise them or at least be wary of seasonal coincidences.\n')

a('## Timing (`known_after_days`)\n')
a('- Weekly: 7. The point dated Sunday D covers D..D+6 (Sat); it is complete after Saturday and visible on '
  'Trends by the next day or so, so D+7 (or the first trading day after) is safe.')
a('- Daily: 1, as specified. The day ends at midnight. When fetched, the last day (2026-10-17) and the last '
  'week (Oct 11-17) were already final (not `isPartial`) about 1.3 days after they ended. 2 would be the '
  'conservative choice for daily if the test wants margin.')
a('- Vintage: these are today\'s values. Google re-samples, so values seen in real time back then would '
  'differ by sampling noise (see repeat pulls). Each series is normalised by its own maximum, which can lie '
  'in the future, but that is a pure scale factor and cancels in `logdiff`. The stitch ratios use future '
  'weeks only to set the scale of the next window, so they touch only the single move that crosses each '
  'junction.\n')

a('## Terms of use\n')
a('Google Trends data are public. Google asks that reuse cite "Data source: Google Trends '
  '(https://www.google.com/trends)". The endpoints used are not an official API; automated access is subject '
  'to Google\'s Terms of Service. Kept for private research; not redistributed. No account, key or '
  'registration was used.\n')

a('## Not done, and why\n')
for line in extra.get('not_done', []):
    a(f'- {line}')
a('')

a('## Meaning / quality flags per term\n')
a('| id | first | last | rows | transform | notes |')
a('|---|---|---|---|---|---|')
for r in cat:
    a(f"| {r['id']} | {r['first']} | {r['last']} | {r['rows']} | {r['transform']} | {r['notes']} |")
a('')
a('## qa_summary (first/last rows)\n')
a('```')
for k, v in info.items():
    q = v['qa']
    a(f"{k:32s} rows={q['rows']:5d} gaps={q['n_gaps']} dups={q['dups']} finite={q['finite']} "
      f"min={q['min']:.2f} max={q['max']:.0f} head={q['head'][:2]} tail={q['tail'][-2:]}")
a('```')
p = os.path.join(OUT, 'SOURCES_data-trends.md')
open(p, 'w').write('\n'.join(L) + '\n')
print('wrote', p, len(L), 'lines')
