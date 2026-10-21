"""Find spike-and-revert patterns (likely bad prints) and gap/flat periods by year in series files."""
import csv, math, os, sys, collections, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmlib
import numpy as np
cat = {r['id']: r for r in csv.DictReader(open(dmlib.CATALOG))}
ids = sys.argv[1:]
for sid in ids:
    r = list(csv.reader(open(os.path.join(dmlib.SERIES, sid + '.csv'))))[1:]
    d = [x[0] for x in r]; v = np.array([float(x[1]) for x in r])
    tr = cat[sid]['transform']
    ch = np.diff(np.log(v)) if tr == 'logdiff' else np.diff(v)
    mad = np.median(np.abs(ch - np.median(ch))) * 1.4826
    sd = mad if mad > 0 else np.std(ch)
    out = []
    for i in range(len(ch) - 1):
        a, b = ch[i], ch[i + 1]
        if abs(a) > 8 * sd and abs(b) > 0.7 * abs(a) and a * b < 0:
            out.append(f"{d[i+1]}({v[i]:g}->{v[i+1]:g}->{v[i+2]:g})")
    ds = [dt.date.fromisoformat(x) for x in d]
    gy = collections.Counter(ds[i].year for i in range(1, len(ds)) if (ds[i] - ds[i-1]).days > 5)
    zy = collections.Counter(ds[i+1].year for i in range(len(ch)) if ch[i] == 0)
    ny = collections.Counter(x.year for x in ds)
    zfrac = {y: round(zy[y] / ny[y], 2) for y in sorted(ny) if zy[y] / ny[y] > 0.1}
    print(f"{sid}: spikes={len(out)} {' '.join(out[:8])}")
    if gy: print(f"   gaps>5d by year: {dict(sorted(gy.items()))}")
    if zfrac: print(f"   years with >10% zero changes: {zfrac}")
