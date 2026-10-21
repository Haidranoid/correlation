"""Assemble catalog_data-politics.csv from work/catalog_parts/*.jsonl and the series files."""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CAT_COLS, EXT, PARTS, SERIES, read_series

entries = []
for fn in sorted(glob.glob(os.path.join(PARTS, '*.jsonl'))):
    for line in open(fn):
        if line.strip():
            entries.append(json.loads(line))
ids = [e['id'] for e in entries]
dups = sorted({i for i in ids if ids.count(i) > 1})
assert not dups, f'duplicate ids {dups}'
for e in entries:
    rows = read_series(e['id'])
    e['first'], e['last'], e['rows'] = rows[0][0], rows[-1][0], len(rows)
    for k in ('notes', 'name', 'units', 'source'):
        e[k] = ' '.join(str(e[k]).split())
# other agents' catalogs: id collisions
others = {}
for fn in glob.glob(os.path.join(EXT, 'catalog_*.csv')):
    if fn.endswith('catalog_data-politics.csv'):
        continue
    with open(fn, newline='') as f:
        for r in csv.DictReader(f):
            others[r.get('id')] = os.path.basename(fn)
coll = [i for i in ids if i in others]
print('collisions with other catalogs:', [(i, others[i]) for i in coll])
# orphans: series files with my prefixes not in catalog
PREF = ('pu_', 'gpr_c_', 'pci_', 'ccc_', 'icews_', 'oxcgrt_', 'gallup_', 'pol_', 'iem_')
orph = [f[:-4] for f in os.listdir(SERIES) if f.endswith('.csv') and f.startswith(PREF) and f[:-4] not in ids]
print('orphan series files with my prefixes:', orph)
out = os.path.join(EXT, 'catalog_data-politics.csv')
tmp = out + '.tmp'
with open(tmp, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(CAT_COLS)
    for e in sorted(entries, key=lambda e: (e['subcategory'], e['id'])):
        w.writerow([e[c] for c in CAT_COLS])
os.replace(tmp, out)
from collections import Counter
print('wrote', out, len(entries), 'series')
print('by subcategory:', sorted(Counter(e['subcategory'] for e in entries).items()))
print('by frequency:', sorted(Counter(e['frequency'] for e in entries).items()))
print('by transform:', sorted(Counter(e['transform'] for e in entries).items()))
loc = Counter('state' if e['region'].startswith('US-') else ('national' if e['region'] == 'US' else
              ('global' if e['region'] == 'WORLD' else 'country')) for e in entries)
print('by geography:', sorted(loc.items()))
