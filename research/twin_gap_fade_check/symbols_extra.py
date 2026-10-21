"""Supplementary symbols: Yahoo's 5-minute range ('60d' = 60 trading days) starts 2026-07-29, before the 2026-08-20
used by symbols.py. The ranked set and twins on days from 2026-07-29 that symbols_5m.txt lacks -> out/symbols_extra.txt.

    python3 -I symbols_extra.py
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = np.load(Path.home() / '.local/share/correlation2/experiments/twin_gap_fade_check/positions.npz')
days = P['days']
t_from = int(np.searchsorted(days, (np.datetime64('2026-07-29') - np.datetime64('1970-01-01')).astype(int)))
m = P['rk_t'] >= t_from
smap = dict(zip(P['stk_i'].tolist(), P['stk_sym'].tolist()))
fmap = dict(zip(P['fnd_f'].tolist(), P['fnd_sym'].tolist()))
syms = set(smap[i] for i in np.unique(P['rk_i'][m]).tolist()) | set(fmap[f] for f in np.unique(P['rk_f'][m]).tolist())
have = set((HERE / 'out' / 'symbols_5m.txt').read_text().split())
extra = sorted(syms - have)
(HERE / 'out' / 'symbols_extra.txt').write_text('\n'.join(extra) + '\n')
print(f'days from 2026-07-29: {len(np.unique(P["rk_t"][m]))}; symbols {len(syms)}, already listed {len(syms & have)}, extra {len(extra)}')
