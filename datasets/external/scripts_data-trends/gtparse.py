"""Parsing and stitching of saved Google Trends responses (no network)."""
import json
import os
from datetime import datetime, timezone, date

import numpy as np


def load(path):
    txt = open(path).read()
    body = txt.split('\n', 1)[1] if txt.startswith(")]}'") else txt
    return json.loads(body)


def utc_date(t):
    return datetime.fromtimestamp(int(t), timezone.utc).date()


def parse_any(path):
    """Return dict date -> (value, partial) merged over all columns; also list of per-column dicts.
    Raises if overlapping columns disagree."""
    d = load(path)
    tl = d['default']['timelineData']
    cols = []
    if tl and 'columnData' in tl[0]:
        n = len(tl[0]['columnData'])
        cols = [dict() for _ in range(n)]
        for p in tl:
            for i, c in enumerate(p['columnData']):
                if not c.get('time') or c.get('value') is None or c.get('value') == -1:
                    continue
                cols[i][utc_date(c['time'])] = (float(c['value']), bool(c.get('isPartial', False)),
                                                c.get('formattedTime', ''))
    else:
        col = {}
        for p in tl:
            # hasData:false marks zero-volume points (value 0, shown as 0 in the UI); keep them as 0,
            # as multirange responses do
            if not p.get('time') or not p.get('value'):
                continue
            col[utc_date(p['time'])] = (float(p['value'][0]), bool(p.get('isPartial', False)),
                                        p.get('formattedTime', ''))
        cols = [col]
    merged, conflicts = {}, []
    for c in cols:
        for k, v in c.items():
            if k in merged and merged[k][0] != v[0]:
                conflicts.append((k, merged[k][0], v[0]))
            merged.setdefault(k, v)
    return merged, cols, conflicts


def series_from(merged, drop_partial=True):
    ds = sorted(merged)
    out = [(k, merged[k][0]) for k in ds if not (drop_partial and merged[k][1])]
    return out


def median_ratio(base, new, thr=5.0, min_n=8):
    """ratio r such that new*r ~ base over shared dates. Spec method: median of per-date ratios over
    dates where both >= thr (needs >= min_n such dates). Fallback for low-volume overlaps (small
    integers make a median of ratios coarse): ratio of sums over all shared dates.
    Returns (r, n_used, thr_used (0 = sum ratio), sd of log per-date ratios or nan)."""
    common = sorted(set(base) & set(new))
    pairs = [(base[k], new[k]) for k in common if base[k] >= thr and new[k] >= thr]
    if len(pairs) >= min_n:
        r = np.array([x / y for x, y in pairs])
        return float(np.median(r)), len(pairs), thr, float(np.std(np.log(r)))
    a = sum(base[k] for k in common)
    b = sum(new[k] for k in common)
    pos = [(base[k], new[k]) for k in common if base[k] > 0 and new[k] > 0]
    sd = float(np.std(np.log([x / y for x, y in pos]))) if len(pos) > 2 else float('nan')
    if b > 0 and a > 0:
        return a / b, len(common), 0.0, sd
    return float('nan'), len(common), 0.0, sd


def chain_stitch(windows, floor=None, thr=5.0):
    """windows: list (chronological) of dict date->raw value (each on its own scale).
    floor: if set, raw values are clamped to >= floor in their own window units before scaling.
    Each window is rescaled to the previous (already scaled) one by the median ratio over shared
    dates; in each overlap the series switches from old to new at the overlap midpoint.
    Returns (dict date->value scaled so max=100, list of junction stats)."""
    scaled = []
    stats = []
    prev = None
    for i, w in enumerate(windows):
        wv = {k: (max(v, floor) if floor is not None else v) for k, v in w.items()}
        if prev is None:
            r = 1.0
            stats.append({'window': i, 'ratio': 1.0, 'n': 0, 'thr': None, 'sd_log': None})
        else:
            # ratio from unclamped raw values (clamping would bias the ratio)
            base_raw = {k: scaled_raw[-1][k] for k in scaled_raw[-1]}
            r_raw, n, t, sd = median_ratio(base_raw, w, thr=thr)
            r = r_raw
            stats.append({'window': i, 'ratio': r, 'n': n, 'thr': t, 'sd_log': sd})
        if i == 0:
            scaled_raw = [dict(w)]
        else:
            scaled_raw.append({k: v * r for k, v in w.items()})
        cur = {k: v * r for k, v in wv.items()}
        scaled.append(cur)
        prev = cur
    # assemble: switch at overlap midpoint
    out = {}
    for i, cur in enumerate(scaled):
        ks = sorted(cur)
        if i + 1 < len(scaled):
            nxt = scaled[i + 1]
            shared = sorted(set(cur) & set(nxt))
            cut = shared[len(shared) // 2] if shared else None
        else:
            cut = None
        start_cut = None
        if i > 0:
            shared_prev = sorted(set(scaled[i - 1]) & set(cur))
            start_cut = shared_prev[len(shared_prev) // 2] if shared_prev else None
        for k in ks:
            if start_cut is not None and k < start_cut:
                continue
            if cut is not None and k >= cut:
                continue
            out[k] = cur[k]
    m = max(out.values())
    out = {k: v * 100.0 / m for k, v in out.items()}
    return out, stats
