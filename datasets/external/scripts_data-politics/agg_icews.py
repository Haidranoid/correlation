"""One pass over the ICEWS coded event files -> daily counts by location.

Counts per (event date, location key, kind) where
  kind: 'all' (every event), 'protest' (CAMEO root 14), 'violence' (CAMEO roots 18-20: assault, fight,
        unconventional mass violence)
  location key: event location Country (e.g. 'United States'), and for US events 'US|<Province>'.
Event IDs are de-duplicated across files with a bitmap (the 2023 weekly files can repeat events).
Output: work/icews_agg.json  {"meta": {...}, "counts": {"<kind>|<loc>": {"YYYY-MM-DD": n}}}
"""
import io
import json
import os
import sys
import zipfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, WORK

D = os.path.join(RAW, 'icews')
files = sorted(f for f in os.listdir(D) if f.endswith('.zip') or f.endswith('.tab'))
# yearly files first (sorted by year), then the 2023 weekly files
def sort_key(f):
    low = f.lower()
    if low.startswith('events.'):
        return (0, low.split('.')[1])
    return (1, low)
files.sort(key=sort_key)

seen = np.zeros(60_000_000, dtype=bool)
counts = defaultdict(lambda: defaultdict(int))
meta = {'files': {}, 'dups': 0, 'bad_code': 0, 'bad_rows': 0, 'id_overflow': 0}


def root_of(code):
    c = code.strip().strip('"')
    if not c.isdigit():
        return None
    n = len(c)
    if n == 2:                      # "042" -> "42": root 0X
        return int(c[0])
    if n == 3:
        if c[0] == '0':             # "042"
            return int(c[:2])
        if 10 <= int(c[:2]) <= 20:  # "145", "190", "204"
            return int(c[:2])
        return int(c[0])            # "254" from "0254"
    if n == 4:
        return int(c[:2])           # "0254", "1823", "2041"
    if n == 1:
        return None
    return None


def open_text(fn):
    p = os.path.join(D, fn)
    if fn.endswith('.zip'):
        z = zipfile.ZipFile(p)
        name = [n for n in z.namelist() if n.endswith('.tab')][0]
        return io.TextIOWrapper(z.open(name), encoding='utf-8', errors='replace')
    return open(p, encoding='utf-8', errors='replace')


for fn in files:
    n_rows = n_new = 0
    dmin, dmax = '9999', '0000'
    with open_text(fn) as f:
        header = f.readline().lstrip('﻿').rstrip('\n').split('\t')
        header = [h.strip().strip('"') for h in header]
        ix = {h: i for i, h in enumerate(header)}
        i_id, i_date, i_code = ix['Event ID'], ix['Event Date'], ix['CAMEO Code']
        i_prov, i_ctry = ix['Province'], ix['Country']
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < len(header):
                meta['bad_rows'] += 1
                continue
            n_rows += 1
            try:
                eid = int(parts[i_id].strip('"'))
            except ValueError:
                meta['bad_rows'] += 1
                continue
            if eid >= len(seen):
                meta['id_overflow'] += 1
            else:
                if seen[eid]:
                    meta['dups'] += 1
                    continue
                seen[eid] = True
            n_new += 1
            d = parts[i_date].strip('"')[:10]
            if d < dmin:
                dmin = d
            if d > dmax:
                dmax = d
            ctry = parts[i_ctry].strip('"').strip() or 'NULL'
            r = root_of(parts[i_code])
            if r is None:
                meta['bad_code'] += 1
            kinds = ['all']
            if r == 14:
                kinds.append('protest')
            elif r is not None and 18 <= r <= 20:
                kinds.append('violence')
            for k in kinds:
                counts[f'{k}|{ctry}'][d] += 1
            if ctry == 'United States':
                prov = parts[i_prov].strip('"').strip() or 'NULL'
                for k in kinds:
                    counts[f'{k}|US|{prov}'][d] += 1
    meta['files'][fn] = {'rows': n_rows, 'new': n_new, 'dmin': dmin, 'dmax': dmax}
    print(f'{fn}: rows={n_rows} new={n_new} dates {dmin}..{dmax} dups_total={meta["dups"]}', flush=True)

out = {'meta': meta, 'counts': {k: dict(v) for k, v in counts.items()}}
with open(os.path.join(WORK, 'icews_agg.json'), 'w') as fh:
    json.dump(out, fh)
print('keys', len(out['counts']), 'meta', {k: v for k, v in meta.items() if k != 'files'})
