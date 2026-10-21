"""Write qa/extra.json (repeat-pull interpretation, daily noise, not-done list) from qa.json etc."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import QA, RAW  # noqa: E402
from terms import CORE, TICKERS, KEY8, WEEKLY_WIN, slug  # noqa: E402

qa = json.load(open(os.path.join(QA, 'qa.json')))
info = json.load(open(os.path.join(QA, 'build_info.json')))
noise = json.load(open(os.path.join(QA, 'daily_noise.json'))) if os.path.exists(os.path.join(QA, 'daily_noise.json')) else {}
p = os.path.join(QA, 'extra.json')
ex = json.load(open(p)) if os.path.exists(p) else {}

dup = qa.get('duplicates', [])
short = [d for d in dup if d['repeat'] in ('shift1d', 'exact')]
late = [d for d in dup if d['repeat'] == 'late']
same = [d['term'] for d in short if d['repeat'] == 'shift1d' and d['weeks_differing'] <= 6]
diff = [d for d in short if d['repeat'] == 'shift1d' and d['weeks_differing'] > 6]
txt = []
txt.append(f"**Minutes apart, the repeat cannot measure sampling noise.** {len(same)} of "
           f"{len([d for d in short if d['repeat'] == 'shift1d'])} terms came back identical or within 1 point in "
           f"<= 6 weeks ({', '.join(same)}). Google serves the same sample for a while, and moving the range "
           "boundaries by a day does not force a new one: the exact repeat and the shifted pull of \"debt\" were "
           "byte-identical to each other.")
from build import read_request  # noqa: E402


def peak(path):
    r, _ = read_request(path)
    return str(max(r, key=r.get))


for d in diff:
    pa = peak(os.path.join(RAW, 'weekly_joint', slug(d['term']) + '.txt'))
    pb = peak(os.path.join(RAW, 'weekly_joint_repeat', slug(d['term']) + '__shift1d.txt'))
    txt.append(f"For \"{d['term']}\" the sample did change between the first pull ({d['t_orig'][11:16]}) and the "
               f"repeats ({d['t_rep'][11:16]}): {d['weeks_differing']} of {d['weeks']} weeks differ, almost all "
               f"by 1 index point (peak week {pa} vs {pb}). Correlation: levels "
               f"{d['corr_level']:.4f}, weekly logdiffs {d['corr_logdiff']:.3f}. That is one genuine re-sample "
               "of a high-volume term.")
if late:
    lt = [f"{d['term']} {d['weeks_differing']}/{d['weeks']} weeks differ (corr level {d['corr_level']:.4f}, "
          f"weekly logdiff {(d.get('corr_logdiff') or float('nan')):.3f})" for d in late]
    hrs = np.median([(int(d['t_rep'][11:13]) * 60 + int(d['t_rep'][14:16]) - int(d['t_orig'][11:13]) * 60 -
                      int(d['t_orig'][14:16])) / 60 for d in late])
    txt.append(f"**Late repeat, about {hrs:.1f} h after the original:** " + '; '.join(lt) + '.')
    from build import compare as _cmp
    vs = []
    for d in late:
        ra, _ = read_request(os.path.join(RAW, 'weekly_joint_repeat', slug(d['term']) + '__shift1d.txt'))
        rb, _ = read_request(os.path.join(RAW, 'weekly_joint_repeat', slug(d['term']) + '__late.txt'))
        vs.append((d['term'], sum(1 for k in ra if k in rb and ra[k] != rb[k]), len(set(ra) & set(rb))))
    txt.append('Late repeat against the 5-minute repeat: ' + ', '.join(f'{t} {n}/{m}' for t, n, m in vs)
               + ' weeks differ. So the weekly sample Google served stayed fixed for at least ~1.5 h. The one '
               'genuine re-sample in this session is debt\'s very first pull (00:48), the session\'s first data '
               'request. Weekly sampling noise is therefore best judged from debt\'s re-sample and from the daily '
               'replicates below.')
if noise:
    sds = [v['sd_log_ratio'] for v in noise.values() if v.get('sd_log_ratio')]
    cs = [v['corr_1day_logdiff'] for v in noise.values() if v.get('corr_1day_logdiff')]
    noise = {k: v for k, v in noise.items()}
    txt.append(f"**Daily sampling noise, measured directly** (all years pooled; the by-era table under 'Daily series' "
               "is the one to use). Inside one daily multirange request, overlapping "
               f"ranges share the scale (median ratio 1.000 on every pair) but are separate samples. Over "
               f"{sum(v['shared_days'] for v in noise.values())} shared days in {len(noise)} terms, the two "
               f"samples differ by an sd of {100*min(sds):.1f}-{100*max(sds):.1f}% in log terms (both >= 5). "
               f"Their 1-day logdiffs correlate {min(cs):.3f}-{max(cs):.3f}. So a single daily value carries "
               f"about {100*np.median(sds)/np.sqrt(2):.1f}% sampling noise.")
ex['dup_text'] = '\n\n'.join(txt)

have_w = {v['kw'] for k, v in info.items() if v['method'] in ('joint', 'stitched')}
stitched = {v['kw'] for k, v in info.items() if v['method'] == 'stitched'}
have_d = {v['kw'] for k, v in info.items() if v['method'] == 'daily_stitched'}
nd = []
nd.append('Official Google Trends API (alpha): needs an approved application; not applied for, as instructed.')
miss_w = [kw for kw, g, s in CORE if kw not in have_w] + [t for t in TICKERS if t not in have_w]
nd.append('Weekly: ' + ('every requested term was fetched' if not miss_w else 'missing ' + ', '.join(miss_w))
          + '. The monthly all-time fallback was not needed, so none was fetched.')
queued_spec = ['dow jones', 'debt', 'stock market', 'unemployment', 's&p 500', 'file for unemployment', 'flu',
               'heat wave', 'nasdaq', 'margin call', 'short selling', 'mortgage rates', 'recession', 'buy stocks',
               'layoffs', 'stock market crash']
not_st = [k for k in queued_spec if k not in stitched]
nd.append('Spec-method single 5-year windows ran for ' + str(len(stitched)) + ' terms (' + ', '.join(sorted(stitched))
          + '). ' + ('Planned but cut by the 3-hour budget: ' + ', '.join(not_st) + '. ' if not_st else '')
          + 'Every other term is published from its joint multirange pull, which has no stitching error but '
          'coarser integer precision where volume is low (see the LOW PRECISION flags).')
nd.append('Hurricane and black friday got no single windows: in every 5-year window one storm or one Black Friday '
          'week sets 100, so single windows would not lift their off-peak precision.')
miss_d = [k for k in KEY8 if k not in have_d]
nd.append('Daily: ' + ', '.join(sorted(have_d)) + '. ' + ('Missing: ' + ', '.join(miss_d) + '. ' if miss_d else '')
          + 'No daily series for other terms (budget).')
daily_first = {v['kw']: v['qa']['first'] for k, v in info.items() if v['method'] == 'daily_stitched'}
late_start = {k: v for k, v in daily_first.items() if v > '2004-02-01'}
if late_start:
    nd.append('Daily series starting after 2004-01: ' + ', '.join(f'{k} from {v}' for k, v in sorted(late_start.items()))
              + '. Every planned older daily window was fetched (bitcoin was planned from 2011). These start later '
              'because the leading-zero rule (first day from which the next 2 years have < 10% zeros) cut sparse, '
              'zero-heavy early years. So gt_stock_market_crash_d has no 2008 crash; that period is only in its weekly '
              'series. The raw files for the cut years are kept in raw/google_trends/daily_req/, so a looser trim can '
              'be rebuilt with the scripts.')
nd.append('No other geographies, categories or Google properties (news, YouTube), and no related-queries data: '
          'not requested.')
ex['not_done'] = nd
json.dump(ex, open(p, 'w'), indent=1)
print(ex['dup_text'])
print('\n'.join(nd))
