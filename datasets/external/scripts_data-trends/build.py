"""Build normalized Google Trends series + catalog from saved raw responses (no network).

Weekly: default = joint multirange pull (5 ranges, one request, one scale; no stitching).
        If single 5-year windows exist (weekly_win/<slug>/), the series is the chain-stitched
        windows (spec method) and the joint pull is used only to measure the stitching error.
Daily:  chain-stitched daily requests (daily_req/<slug>/), median ratio over the 45-day overlaps.
Values: logdiff terms are clamped to >= 0.5 in each source window's own units (Trends reports
        '<1' as 0) before scaling; anomaly_log1p terms keep zeros. Series max = 100.
"""
import glob
import json
import os
import sys
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtparse import parse_any, median_ratio  # noqa: E402
from terms import CORE, TICKERS, KEY8, END, slug, WEEKLY_WIN  # noqa: E402

EXT = os.path.expanduser('~/.local/share/correlation2/external')
RAW = os.path.join(EXT, 'raw', 'google_trends')
OUT = os.environ.get('GT_OUT', EXT)
SER = os.path.join(OUT, 'series')
HERE = os.path.dirname(os.path.abspath(__file__))
QA = os.path.join(os.environ.get('GT_OUT', HERE), 'qa')
os.makedirs(QA, exist_ok=True)

SEASONAL = {'black friday', 'flu', 'coupons', 'hurricane', 'heat wave'}


def week_floor(d):
    return d - timedelta(days=(d.weekday() + 1) % 7)  # Sunday on/before


def read_request(path):
    """One saved response (multiline or multirange): date->value with duplicate dates (range
    boundaries / overlapping items, same joint scale) averaged; isPartial points dropped."""
    merged, cols, conf = parse_any(path)
    acc = {}
    for c in cols:
        for k, (v, part, ft) in c.items():
            if part:
                continue
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}, len(conf)


def chain(pieces, floor):
    """pieces: chronological list of dict date->raw value, each on its own scale. Each piece is
    rescaled to the previous (already rescaled) one by the median ratio over shared dates where
    both >= 5 (fallback 2, 1, then sum ratio). The output switches from one piece to the next at
    the midpoint of their overlap. floor: raw values clamped to >= floor in the piece's own units
    (after the ratio is estimated from unclamped values), or None. Output scaled to max = 100."""
    facs, stats, scaled = [], [], []
    for i, w in enumerate(pieces):
        if i == 0:
            r, n, thr, sd = 1.0, 0, None, None
        else:
            r, n, thr, sd = median_ratio(scaled[-1], w)
        if not np.isfinite(r) or r <= 0:
            raise ValueError(f'bad stitch ratio at piece {i}: {r} (n={n})')
        facs.append(r)
        scaled.append({k: v * r for k, v in w.items()})
        stats.append({'piece': i, 'factor': r, 'n_overlap': n, 'thr': thr, 'sd_log_ratio': sd,
                      'first': str(min(w)), 'last': str(max(w))})
    out, rawv = {}, {}
    for i, w in enumerate(pieces):
        lo = hi = None
        if i > 0:
            sh = sorted(set(pieces[i - 1]) & set(w))
            lo = sh[len(sh) // 2] if sh else None
        if i + 1 < len(pieces):
            sh = sorted(set(w) & set(pieces[i + 1]))
            hi = sh[len(sh) // 2] if sh else None
        for k, v in w.items():
            if (lo is not None and k < lo) or (hi is not None and k >= hi):
                continue
            out[k] = (max(v, floor) if floor is not None else v) * facs[i]
            rawv[k] = v
    m = max(out.values())
    chain.last_raw = rawv  # raw value (own request units) behind each output date
    return {k: v * 100.0 / m for k, v in out.items()}, stats


def trim_leading_zeros(ser, step_days):
    """Drop a leading stretch with no measurable volume: start at the first non-zero date from
    which the next 2 years contain < 10% zeros. Returns (series, first_kept or None)."""
    ks = sorted(ser)
    n2y = int(round(730 / step_days))
    for i, k in enumerate(ks):
        if ser[k] <= 0:
            continue
        nxt = [ser[x] for x in ks[i:i + n2y]]
        if np.mean([x <= 0 for x in nxt]) < 0.10:
            return {x: ser[x] for x in ks[i:]}, (ks[i] if i > 0 else None)
    return ser, None


def seasonality_r2(ser):
    """Cross-validated seasonal R2 of detrended log(max(v,0.5)) (centred 53-week mean removed):
    week-of-year median profile fitted on odd years, scored on even years and vice versa; the two
    R2 are averaged. ~0 = no usable annual pattern, 1 = all variance is the annual pattern."""
    ks = sorted(ser)
    y = np.log(np.maximum(np.array([ser[k] for k in ks]), 0.5))
    if len(y) < 160:
        return float('nan')
    c = np.convolve(y, np.ones(53) / 53, mode='same')
    resid = (y - c)[26:-26]
    kk = ks[26:-26]
    woy = np.array([min((k.timetuple().tm_yday - 1) // 7, 51) for k in kk])
    odd = np.array([k.year % 2 == 1 for k in kk])
    r2 = []
    for fit_h, ev_h in ((odd, ~odd), (~odd, odd)):
        prof = np.array([np.median(resid[(woy == w) & fit_h]) if np.any((woy == w) & fit_h) else 0.0
                         for w in range(52)])
        e = resid[ev_h]
        if np.var(e) == 0:
            continue
        r2.append(1 - np.var(e - prof[woy[ev_h]]) / np.var(e))
    return float(np.mean(r2)) if r2 else float('nan')


def write_series(sid, ser):
    p = os.path.join(SER, sid + '.csv')
    with open(p + '.part', 'w') as f:
        f.write('date,value\n')
        for k in sorted(ser):
            f.write(f'{k.isoformat()},{round(ser[k], 4):g}\n')
    os.replace(p + '.part', p)
    return p


def qa_series(sid, ser, step):
    ks = sorted(ser)
    v = np.array([ser[k] for k in ks])
    gaps = [(str(a), str(b)) for a, b in zip(ks, ks[1:]) if (b - a).days != step]
    return {'id': sid, 'first': str(ks[0]), 'last': str(ks[-1]), 'rows': len(ks),
            'min': float(v.min()), 'max': float(v.max()), 'median': float(np.median(v)),
            'n_gaps': len(gaps), 'gaps': gaps[:10], 'dups': len(ks) - len(set(ks)),
            'finite': bool(np.all(np.isfinite(v))),
            'head': [(str(k), round(ser[k], 3)) for k in ks[:3]],
            'tail': [(str(k), round(ser[k], 3)) for k in ks[-3:]]}


def weekly_mean_of_daily(daily):
    acc = {}
    for k, v in daily.items():
        acc.setdefault(week_floor(k), []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items() if len(v) == 7}


def compare(a, b, thr=0.0):
    """corr of levels, of log levels, of 1-step logdiffs over common dates (both > thr)."""
    ks = sorted(k for k in set(a) & set(b) if a[k] > thr and b[k] > thr)
    if len(ks) < 10:
        return None
    x = np.array([a[k] for k in ks])
    y = np.array([b[k] for k in ks])
    lx, ly = np.log(np.maximum(x, 0.5)), np.log(np.maximum(y, 0.5))
    # 1-step logdiffs only across consecutive common dates
    step = min((k2 - k1).days for k1, k2 in zip(ks, ks[1:]))
    idx = [i for i in range(1, len(ks)) if (ks[i] - ks[i - 1]).days == step]
    dx = np.array([lx[i] - lx[i - 1] for i in idx])
    dy = np.array([ly[i] - ly[i - 1] for i in idx])
    r = x / np.where(y > 0, y, np.nan)
    return {'n': len(ks), 'corr_level': float(np.corrcoef(x, y)[0, 1]),
            'corr_log': float(np.corrcoef(lx, ly)[0, 1]),
            'corr_logdiff': float(np.corrcoef(dx, dy)[0, 1]) if len(dx) > 5 else None,
            'median_abs_log_ratio': float(np.nanmedian(np.abs(np.log(r[(x > 0) & (y > 0)])))),
            'ratio_median': float(np.nanmedian(r))}


# ---------------------------------------------------------------- main build
MEANING = {
    'crypto': 'MEANING CHANGE: before ~2013 mostly cryptography/other senses; crypto-asset sense dominates from ~2017',
    'depression': 'mostly the mental-health sense, not economic depression (FEARS used it as a negative-sentiment word)',
    'derivatives': 'dominated by the calculus sense (school-calendar seasonality), not financial derivatives',
    'leverage': 'generic word (also a TV series 2008-2012)',
    'crash': 'generic word (car/computer crashes), not only market crashes',
    'bubble': 'generic word', 'sell': 'generic word', 'cash': 'generic word', 'money': 'generic word',
    'gold': 'generic word (jewelry, colour, games), not only bullion',
    'oil': 'generic word (cooking, motor oil, essential oils), not only crude',
    'credit': 'generic word (credit score, credit cards)',
    'economics': 'school-calendar seasonality (students)',
    'markets': 'generic word (farmers markets etc.)',
    'jobs': 'generic word; also Steve Jobs (spike 2011-10)',
    'portfolio': 'generic word (art/design portfolios)', 'revenue': 'mostly tax/IRS-related and business usage',
    'housing': 'generic (student/public housing)', 'profit': 'generic word', 'finance': 'generic (car finance, student finance)',
    'META': 'MEANING CHANGE: ticker META only since 2022-06 (was FB); before the 2021-10 rename "meta" is a generic word',
    'AMC': 'MEANING: mostly AMC Theatres / AMC TV network, not the ticker; meme-stock spike 2021',
    'SPY': 'MEANING: mostly the generic word "spy", not the SPY ETF',
    'BAC': 'MEANING: ambiguous - blood alcohol content as well as the Bank of America ticker',
    'GME': 'ticker search; near zero before the 2021 meme-stock episode (series starts 2020-10)',
    'GOOGL': 'MEANING: GOOGL ticker exists only since 2014-04; earlier searches are mostly misspellings of "google"',
    'ethereum': 'exists since 2015 (near zero before)', 'bitcoin': 'near zero before 2011',
    'buy bitcoin': 'near zero before 2013', 'TSLA': 'TSLA listed 2010-06; near zero before',
    'social security': 'annual pattern (COLA announcements each October)',
    'black friday': 'one-week-a-year spike (late November)', 'flu': 'winter flu season; 2009 H1N1 and 2020 COVID spikes',
    'hurricane': 'Atlantic hurricane season Jun-Nov; single-storm spikes', 'heat wave': 'summer seasonal',
    'coupons': 'holiday-season pattern', 'mortgage rates': 'tracks refinancing waves',
    'file for unemployment': 'extreme spike 2020-03 (COVID); low volume otherwise',
    'frugal': 'low volume; noisy', 'margin call': 'also a 2011 film title; low volume', 'short selling': 'low volume; noisy',
    'gdp': 'quarterly release spikes',
}
MANUAL_START = {'crypto': date(2017, 1, 1)}  # meaning change + no measurable volume before
GROUP = {kw: g for kw, g, s in CORE}
for _t in TICKERS:
    GROUP[_t] = 'ticker_da2011'
NOTE_METHOD_CHANGES = ('Google data-collection changes (Trends UI notes 2011-01-01, 2016-01-01, 2022-01-01): a common '
                       'level step of ~+8% is visible at 2016-01-01 across terms (artefact)')


def explore_url(kw, freq):
    from urllib.parse import quote
    d = '2004-01-01%20' + END
    return f'https://trends.google.com/trends/explore?date={d}&geo=US&q={quote(kw)}'


def transform_for(kw):
    return 'anomaly_log1p' if kw in SEASONAL else 'logdiff'


def build_weekly(kw):
    s = slug(kw)
    jp = os.path.join(RAW, 'weekly_joint', s + '.txt')
    if not os.path.exists(jp):
        return None
    joint, nconf = read_request(jp)
    joint = {k: v for k, v in joint.items() if k >= date(2004, 1, 1)}
    tr = transform_for(kw)
    floor = 0.5 if tr == 'logdiff' else None
    wins = [os.path.join(RAW, 'weekly_win', s, f'{a}_{b}.txt') for a, b in WEEKLY_WIN]
    info = {'kw': kw, 'transform': tr, 'joint_conflicts': nconf}
    if all(os.path.exists(p) for p in wins):
        pieces = [{k: v for k, v in read_request(p)[0].items() if k >= date(2004, 1, 1)} for p in wins]
        raw_ser, stats = chain(pieces, None)
        own_units = dict(chain.last_raw)
        ser, _ = chain(pieces, floor)
        info['method'] = 'stitched'
        info['stitch'] = stats
        # stitching error vs the independent joint pull: per-window median ratio stitched/joint
        per = []
        for i, w in enumerate(pieces):
            ks = [k for k in w if k in joint and joint[k] >= 5 and raw_ser.get(k, 0) > 0]
            if len(ks) >= 8:
                per.append(float(np.median([raw_ser[k] / joint[k] for k in ks])))
            else:
                per.append(float('nan'))
        per = np.array(per)
        ok = per[np.isfinite(per)]
        info['stitch_vs_joint_ratio_per_window'] = [round(float(x), 4) for x in per]
        info['stitch_error_n_windows'] = int(len(ok))
        info['stitch_error_sd_log'] = float(np.std(np.log(ok / np.median(ok)))) if len(ok) >= 3 else None
        info['stitch_error_maxabs_log'] = float(np.max(np.abs(np.log(ok / np.median(ok))))) if len(ok) >= 3 else None
        info['vs_joint'] = compare(raw_ser, joint)
    else:
        raw_ser = dict(joint)
        own_units = dict(joint)
        ser = {k: (max(v, floor) if floor is not None else v) for k, v in joint.items()}
        info['method'] = 'joint'
    start = None
    if tr == 'logdiff':
        trimmed, start = trim_leading_zeros(raw_ser, 7)
        if kw in MANUAL_START:
            start = max(start or date(2004, 1, 1), MANUAL_START[kw])
        if start is not None:
            ser = {k: v for k, v in ser.items() if k >= start}
            raw_ser = {k: v for k, v in raw_ser.items() if k >= start}
    v = np.array([raw_ser[k] for k in sorted(raw_ser)])
    info['trim_start'] = str(start) if start else None
    info['zero_frac'] = float(np.mean(v <= 0))
    info['frac_lt5'] = float(np.mean([own_units[k] < 5 for k in raw_ser]))  # in source-request units
    # precision in each 5-year span, on the published scale (max 100)
    spans = {}
    for k in raw_ser:
        spans.setdefault(min(max((k.year - 2004) // 5, 0), 4), []).append(raw_ser[k])
    info['span_median'] = {2004 + 5 * i: float(np.median(spans[i])) for i in sorted(spans)}
    info['season_r2'] = seasonality_r2(raw_ser)
    return ser, info


def era_noise(kw, cut=date(2014, 6, 1)):
    """Sampling noise from overlapping items inside one daily request (same scale, separate samples),
    split at `cut`: sd of log(b/a) (both >= 5) and corr of the two samples' 1-day logdiffs."""
    acc = {}
    for p in sorted(glob.glob(os.path.join(RAW, 'daily_req', slug(kw), '*.txt'))):
        merged, cols, conf = parse_any(p)
        for i in range(len(cols) - 1):
            sh = sorted(set(cols[i]) & set(cols[i + 1]))
            if not sh:
                continue
            e = acc.setdefault('pre2014' if sh[0] < cut else 'post2014', {'lr': [], 'da': [], 'db': [], 'n': 0})
            x = np.array([cols[i][k][0] for k in sh])
            y = np.array([cols[i + 1][k][0] for k in sh])
            m = (x >= 5) & (y >= 5)
            e['lr'] += np.log(y[m] / x[m]).tolist()
            e['n'] += len(sh)
            e['da'] += np.diff(np.log(np.maximum(x, 0.5))).tolist()
            e['db'] += np.diff(np.log(np.maximum(y, 0.5))).tolist()
    out = {}
    for era, e in acc.items():
        out[era] = {'shared_days': e['n'], 'sd_log_ratio': float(np.std(e['lr'])) if e['lr'] else None,
                    'corr_1day_logdiff': float(np.corrcoef(e['da'], e['db'])[0, 1]) if len(e['da']) > 10 else None}
    return out


def build_daily(kw):
    s = slug(kw)
    ps = sorted(glob.glob(os.path.join(RAW, 'daily_req', s, '*.txt')))
    if not ps:
        return None
    tr = transform_for(kw)
    floor = 0.5 if tr == 'logdiff' else None
    pieces = [read_request(p)[0] for p in ps]
    # keep only a contiguous (overlapping) chain ending at the most recent piece
    chainp = [pieces[-1]]
    for w in reversed(pieces[:-1]):
        if set(w) & set(chainp[0]):
            chainp.insert(0, w)
        else:
            break
    raw_ser, stats = chain(chainp, None)
    own_units = dict(chain.last_raw)
    ser, _ = chain(chainp, floor)
    start = None
    if tr == 'logdiff':
        trimmed, start = trim_leading_zeros(raw_ser, 1)
        if kw in MANUAL_START:
            start = max(start or date(2004, 1, 1), MANUAL_START[kw])
        if start is not None:
            ser = {k: v for k, v in ser.items() if k >= start}
            raw_ser = {k: v for k, v in raw_ser.items() if k >= start}
    info = {'kw': kw, 'transform': tr, 'method': 'daily_stitched', 'n_requests': len(chainp), 'noise_by_era': era_noise(kw),
            'n_files': len(ps), 'stitch': stats, 'trim_start': str(start) if start else None}
    v = np.array([raw_ser[k] for k in sorted(raw_ser)])
    info['zero_frac'] = float(np.mean(v <= 0))
    info['frac_lt5'] = float(np.mean([own_units[k] < 5 for k in raw_ser]))  # in source-request units
    sds = [x['sd_log_ratio'] for x in stats[1:] if x['sd_log_ratio'] is not None and np.isfinite(x['sd_log_ratio'])]
    info['junction_sd_log_ratio_median'] = float(np.median(sds)) if sds else None
    info['junction_n_overlap_min'] = int(min(x['n_overlap'] for x in stats[1:])) if len(stats) > 1 else None
    # validation vs the weekly joint pull: weekly means of the stitched daily series
    jp = os.path.join(RAW, 'weekly_joint', s + '.txt')
    if os.path.exists(jp):
        joint, _ = read_request(jp)
        wm = weekly_mean_of_daily(raw_ser)
        info['vs_weekly_joint'] = compare(wm, joint, thr=0.0)
        # drift: per-year median ratio (weekly mean of daily)/(joint), normalised
        yr = {}
        for k in wm:
            if k in joint and joint[k] >= 5 and wm[k] > 0:
                yr.setdefault(k.year, []).append(wm[k] / joint[k])
        if yr:
            med = {y: float(np.median(a)) for y, a in yr.items()}
            ref = float(np.median(list(med.values())))
            info['drift_by_year'] = {y: round(m / ref, 3) for y, m in sorted(med.items())}
    return ser, info


def catalog_row(sid, kw, freq, ser, info):
    ks = sorted(ser)
    notes = []
    if info['method'] == 'joint':
        notes.append('one multirange request (5 ranges 2004-' + END + ', one joint scale, no stitching)')
        if kw in KEY8 and glob.glob(os.path.join(RAW, 'daily_req', slug(kw), '*.txt')):
            notes.append(f'more precise weekly source from 2014-06 on: gt_{slug(kw)}_d averaged over Sun-Sat weeks '
                         '(its requests have their own peaks; this joint series rounds low values coarsely)')
    elif info['method'] == 'stitched':
        notes.append('spec method: 6 overlapping single 5-year windows chain-stitched (median ratio over shared weeks where both >=5)')
        if info.get('stitch_error_sd_log') is not None:
            notes.append(f"stitch error vs independent joint pull: sd {100*info['stitch_error_sd_log']:.1f}% "
                         f"max {100*info['stitch_error_maxabs_log']:.1f}% per window (upper bound where the joint "
                         'reference itself is coarse)')
    else:
        notes.append(f"{info['n_requests']} daily requests (261-day ranges, 1-5 per request on a joint scale) "
                     'chain-stitched over 45-day overlaps (median ratio where both >=5; ratio of sums for low-volume overlaps)')
        vj = info.get('vs_weekly_joint')
        if vj:
            notes.append(f"weekly means vs weekly pull: corr log {vj['corr_log']:.3f}, logdiff {vj['corr_logdiff']:.3f}")
        notes.append('weekend days ~20-30% lower: take 1-day moves between consecutive trading days, not calendar days')
        nb = info.get('noise_by_era', {})
        pre, post = nb.get('pre2014', {}), nb.get('post2014', {})
        if pre.get('corr_1day_logdiff') is not None and post.get('corr_1day_logdiff') is not None:
            if pre['corr_1day_logdiff'] < 0.9:
                notes.append(f"NOISY BEFORE 2014: two independent Google samples' 1-day logdiffs correlate only "
                             f"{pre['corr_1day_logdiff']:.2f} before 2014-06 (vs {post['corr_1day_logdiff']:.2f} after); "
                             "1-day moves there are mostly sampling noise - prefer >=5-day moves or the weekly series")
            else:
                notes.append(f"sampling noise check: two independent samples' 1-day logdiffs correlate "
                             f"{pre['corr_1day_logdiff']:.2f} before 2014-06, {post['corr_1day_logdiff']:.2f} after")
        notes.append('known_after_days 1 as specified; the last day was already final (not isPartial) ~1.3 days '
                     'after it ended; 2 is the conservative choice')
    if info['transform'] == 'logdiff':
        notes.append('zeros ("<1") clamped to 0.5 of the source window unit before scaling')
    else:
        notes.append('strongly seasonal: anomaly_log1p; zeros kept')
    if info.get('trim_start') and info['trim_start'] > ('2004-01-04' if freq == 'weekly' else '2004-01-01'):
        notes.append(f"leading near-zero stretch dropped (starts {info['trim_start']})")
    if info['zero_frac'] > 0.2:
        notes.append(f"MOSTLY ZEROS: {100*info['zero_frac']:.0f}% zero")
    elif info['frac_lt5'] > 0.5:
        notes.append(f"LOW PRECISION: {100*info['frac_lt5']:.0f}% of points < 5 (integer rounding noise)")
    if kw in MEANING:
        notes.append(MEANING[kw])
    if info.get('season_r2') is not None and np.isfinite(info.get('season_r2', np.nan)) and info['season_r2'] > 0.25 \
            and info['transform'] == 'logdiff':
        notes.append(f"annual seasonality present (cross-validated week-of-year R2 {info['season_r2']:.2f}"
                     + (', from the weekly series)' if freq == 'daily' else ')'))
    notes.append('sampled data: a re-sample changes values by ~1 index point; this is the 2026-10 vintage, not real-time')
    notes.append(NOTE_METHOD_CHANGES)
    name = f'Google Trends US web-search interest: "{kw}" ({freq})'
    return [sid, name, 'attention', GROUP.get(kw, 'other'), 'US', 'Google Trends', explore_url(kw, freq),
            freq, 'index 0-100 (share of US Google web searches; series max = 100)', info['transform'],
            7 if freq == 'weekly' else 1, ks[0].isoformat(), ks[-1].isoformat(), len(ks), '; '.join(notes)]


def main():
    import csv
    rows, infos = [], {}
    allkw = [kw for kw, g, s in CORE] + TICKERS
    for kw in allkw:
        r = build_weekly(kw)
        if r is None:
            continue
        ser, info = r
        sid = f'gt_{slug(kw)}_w'
        write_series(sid, ser)
        info['qa'] = qa_series(sid, ser, 7)
        infos[sid] = info
        rows.append(catalog_row(sid, kw, 'weekly', ser, info))
    for kw in allkw:
        r = build_daily(kw)
        if r is None:
            continue
        ser, info = r
        sid = f'gt_{slug(kw)}_d'
        info['season_r2'] = infos.get(f'gt_{slug(kw)}_w', {}).get('season_r2')
        write_series(sid, ser)
        info['qa'] = qa_series(sid, ser, 1)
        infos[sid] = info
        rows.append(catalog_row(sid, kw, 'daily', ser, info))
    cat = os.path.join(OUT, 'catalog_data-trends.csv')
    with open(cat + '.part', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'name', 'category', 'subcategory', 'region', 'source', 'url', 'frequency', 'units',
                    'transform', 'known_after_days', 'first', 'last', 'rows', 'notes'])
        w.writerows(rows)
    os.replace(cat + '.part', cat)
    json.dump(infos, open(os.path.join(QA, 'build_info.json'), 'w'), indent=1, default=str)
    print('wrote', len(rows), 'series;', cat)
    return infos


if __name__ == '__main__':
    main()
