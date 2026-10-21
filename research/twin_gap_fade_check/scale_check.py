"""D2's check, on daily closes only (no intraday returns): is the store dividend-adjusted relative to Yahoo's
unadjusted closes? Prints the ratio store close / Yahoo close by segment between ex-dates -> out/scale_check.txt.

    python -I scale_check.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # noqa: E402

days, rk, books, y5, y1, yd = A.load()
K = A.scale_table(rk, yd)
kk = K.dropna(subset=['ratio'])
lines = []
seg = kk.groupby(['sym', 'seg']).agg(k=('k', 'first'), n=('ratio', 'size'), spread=('ratio', lambda r: float(np.nanmax(np.abs(r / np.nanmedian(r) - 1)))))
lines.append(f'symbol-days with store and Yahoo closes: {len(kk)}, symbols {kk["sym"].nunique()}')
for s, name in ((0, 'no ex-date after the day (through 2026-10-19)'), (1, 'one ex-date after the day'), (2, 'two or more')):
    x = seg[seg.index.get_level_values('seg') == s] if s < 2 else seg[seg.index.get_level_values('seg') >= 2]
    if len(x):
        lines.append(f'segments with {name}: {len(x)}; k median {x["k"].median():.6f}, 5-95% {x["k"].quantile(.05):.6f}..{x["k"].quantile(.95):.6f}; '
                     f'share |k-1| > 1e-4: {(np.abs(x["k"] - 1) > 1e-4).mean():.1%}; within-segment max deviation median {x["spread"].median():.2e}, '
                     f'share > 0.1%: {(x["spread"] > 1e-3).mean():.1%}')
lines.append(f'symbol-days off their segment k by more than 0.1%: {(~kk["usable"]).sum()} ({(~kk["usable"]).mean():.2%})')
for sym in ('SPY', 'XLK', 'JPM', 'KO', 'T'):
    x = K[(K['sym'] == sym) & K['cs'].notna()]
    ev = K[(K['sym'] == sym) & K['ev']][['date', 'dividends', 'stock_splits']].values.tolist()
    if len(x):
        g = x.groupby('seg')['ratio'].agg(['median', 'min', 'max', 'size'])
        lines.append(f'{sym}: events {ev}; ratio by segment: ' + '; '.join(f'seg {i}: {r["median"]:.6f} [{r["min"]:.6f}, {r["max"]:.6f}] n={int(r["size"])}' for i, r in g.iterrows()))
txt = '\n'.join(lines)
print(txt)
(A.OUT / 'scale_check.txt').write_text(txt + '\n')
