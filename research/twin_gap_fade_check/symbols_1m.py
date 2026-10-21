"""Symbols for the 1-minute pass: stocks and twins of the all-stocks and 500-largest books' positions on 2026-09-22..
2026-10-19 (Yahoo serves 1-minute bars only from 2026-09-22 on 2026-10-21) -> out/symbols_1m.txt. The variant's extra
positions, known only after the 5-minute pass, are fetched separately (out/symbols_1m_variant.txt, by analyze's helper).

    python3 -I symbols_1m.py
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = np.load(Path.home() / '.local/share/correlation2/experiments/twin_gap_fade_check/positions.npz')
days = P['days']
t_from = int(np.searchsorted(days, (np.datetime64('2026-09-22') - np.datetime64('1970-01-01')).astype(int)))
smap = dict(zip(P['stk_i'].tolist(), P['stk_sym'].tolist()))
fmap = dict(zip(P['fnd_f'].tolist(), P['fnd_sym'].tolist()))
syms = {'SPY'}
for b in ('all', 'b500'):
    m = P[f'{b}_t'] >= t_from
    syms |= set(smap[i] for i in np.unique(P[f'{b}_i'][m]).tolist()) | set(fmap[f] for f in np.unique(P[f'{b}_f'][m]).tolist())
(HERE / 'out' / 'symbols_1m.txt').write_text('\n'.join(sorted(syms)) + '\n')
print(f'days from 2026-09-22: {len(np.unique(P["all_t"][P["all_t"] >= t_from]))}; symbols {len(syms)}')
