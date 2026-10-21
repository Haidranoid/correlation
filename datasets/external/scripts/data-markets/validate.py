"""Validate catalog_data-markets.csv against the series files; print counts by category/frequency."""
import csv, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmlib
rows = list(csv.DictReader(open(dmlib.CATALOG)))
hdr = next(csv.reader(open(dmlib.CATALOG)))
assert hdr == dmlib.CAT_COLS, hdr
errs = []
ids = [r['id'] for r in rows]
assert len(ids) == len(set(ids)), 'duplicate ids'
for r in rows:
    sid = r['id']
    if not re.fullmatch(r'[a-z0-9_]+', sid): errs.append(f'bad id {sid}')
    if r['category'] not in dmlib.CATEGORIES: errs.append(f'{sid} category')
    if r['transform'] not in dmlib.TRANSFORMS: errs.append(f'{sid} transform')
    if r['frequency'] not in dmlib.FREQS: errs.append(f'{sid} frequency')
    if not re.fullmatch(r'-?\d+', r['known_after_days']): errs.append(f'{sid} kad')
    for k in ('name', 'source', 'url', 'units', 'notes', 'region', 'subcategory'):
        if not r[k]: errs.append(f'{sid} empty {k}')
    p = os.path.join(dmlib.SERIES, sid + '.csv')
    if not os.path.exists(p): errs.append(f'{sid} missing file'); continue
    L = open(p).read().splitlines()
    if L[0] != 'date,value': errs.append(f'{sid} header')
    d = [x.split(',')[0] for x in L[1:]]
    vals = [x.split(',')[1] for x in L[1:]]
    if d != sorted(set(d)): errs.append(f'{sid} order/dups')
    if any(not re.fullmatch(r'-?\d+(\.\d+)?', v) for v in vals): errs.append(f'{sid} non-plain number')
    if r['transform'] == 'logdiff' and any(float(v) <= 0 for v in vals): errs.append(f'{sid} nonpositive')
    if (d[0], d[-1], str(len(d))) != (r['first'], r['last'], r['rows']): errs.append(f'{sid} first/last/rows mismatch')
print('errors:', errs or 'none')
# files in series/ with my prefixes but not in catalog
mine = ('etf_', 'stooq_', 'fred_', 'eia_', 'bcinfo_', 'altme_')
cat_ids = set(ids)
other_cat = set()
for f in os.listdir(dmlib.EXT):
    if f.startswith('catalog_') and f != os.path.basename(dmlib.CATALOG):
        other_cat |= {x['id'] for x in csv.DictReader(open(os.path.join(dmlib.EXT, f)))}
stray = [f for f in os.listdir(dmlib.SERIES) if f.endswith('.csv') and f.startswith(mine) and f[:-4] not in cat_ids and f[:-4] not in other_cat]
print('stray files with my prefixes:', stray or 'none')
clash = cat_ids & other_cat
print('id clashes with other catalogs:', clash or 'none')
print('total', len(rows))
c = collections.Counter((r['category']) for r in rows)
f = collections.Counter((r['category'], r['frequency']) for r in rows)
for k, v in sorted(c.items(), key=lambda x: -x[1]):
    sub = [r for r in rows if r['category'] == k]
    firsts = sorted(r['first'] for r in sub); lasts = sorted(r['last'] for r in sub)
    fr = ', '.join(f"{fq} {n}" for (cc, fq), n in sorted(f.items()) if cc == k)
    print(f"  {k:<13} {v:>3} ({fr}); first {firsts[0]}..{firsts[-1]}; last {lasts[0]}..{lasts[-1]}")
print('kad distribution:', sorted(collections.Counter(int(r['known_after_days']) for r in rows).items()))
short = [(r['id'], r['first']) for r in rows if r['first'] > '2005-01-03']
print('series starting after 2005-01-03:', len(short))
end_early = [(r['id'], r['last']) for r in rows if r['last'] < '2025-06-27']
print('series ending before 2025-06-27:', end_early)
