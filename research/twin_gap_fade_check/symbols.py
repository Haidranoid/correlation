"""Symbols to fetch for the 5-minute and daily pass: the ranked set on days from 2026-08-20 (Yahoo's 60-day 5-minute
range starts after that) and their twins, plus SPY. Writes out/symbols_5m.txt.

    python3 -I symbols.py
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = np.load(Path.home() / '.local/share/correlation2/experiments/twin_gap_fade_check/positions.npz')
days = P['days']
t_from = int(np.searchsorted(days, (np.datetime64('2026-08-20') - np.datetime64('1970-01-01')).astype(int)))
m = P['rk_t'] >= t_from
si = np.unique(P['rk_i'][m])
fi = np.unique(P['rk_f'][m])
smap = dict(zip(P['stk_i'].tolist(), P['stk_sym'].tolist()))
fmap = dict(zip(P['fnd_f'].tolist(), P['fnd_sym'].tolist()))
syms = sorted(set(smap[i] for i in si.tolist()) | set(fmap[f] for f in fi.tolist()) | {'SPY'})
(HERE / 'out' / 'symbols_5m.txt').write_text('\n'.join(syms) + '\n')
print(f'days from 2026-08-20: {len(np.unique(P["rk_t"][m]))}; stocks {len(si)}, twins {len(fi)}, symbols {len(syms)}')
