"""Coverage table: grouped stocks, stocks with a valid 10-K, valid groups, and class sizes, averaged by year."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import WORK, np

rows = {}
for p in sorted((WORK / 'classes').glob('*.npz')):
    c = np.load(p)
    y = p.stem[:4]
    rows.setdefault(y, []).append((len(c['rows']), c['has'].sum(), c['valid'].sum(), c['imp'].sum(), c['true'].sum(),
                                   len(c['und_i']), np.nanmedian(c['age'])))
print('year | months | grouped | with 10-K | groups valid | imposters | true members | undiscovered | median months since 10-K available')
for y, v in rows.items():
    a = np.array(v, float).mean(0)
    print(f'{y} | {len(v)} | {a[0]:.0f} | {a[1]:.0f} ({a[1] / a[0]:.0%}) | {a[2]:.0f} | {a[3]:.0f} | {a[4]:.0f} | {a[5]:.0f} | {a[6]:.1f}')
