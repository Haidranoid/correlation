"""Cross-check the fresh 5m and 1m bars against the twin gap fade check's bars (fetched earlier the same day),
read-only: on every (symbol, timestamp) in both, the share of closes that agree to 1e-6 (relative).

    python -I crosscheck.py      # -> out/crosscheck.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = Path.home() / '.local/share/correlation2/experiments/twin_gap_fade_check/bars'
WORK = Path.home() / '.local/share/correlation2/experiments/intraday_lead_lag'
out = []
for mode, dirs in (('5m', ['5m', '5m_extra']), ('1m', ['1m', '1m_variant'])):
    P = np.load(WORK / f'panel_{mode}.npz')
    syms = [str(s) for s in P['syms']]
    sidx = {s: i for i, s in enumerate(syms)}
    didx = {str(d): i for i, d in enumerate(P['dates'])}
    step = 5 if mode == '5m' else 1
    close = P['close']
    n = agree = 0
    worst = []
    for d in dirs:
        for f in sorted((OLD / d).glob(f'{mode}_*.csv')):
            x = pd.read_csv(f, usecols=['sym', 'ts', 'close'], dtype={'sym': str, 'ts': str})
            x = x[x['sym'].isin(sidx) & x['close'].notna()]
            x = x[x['ts'].str.slice(19) == '-04:00']
            mins = x['ts'].str.slice(11, 13).astype(int) * 60 + x['ts'].str.slice(14, 16).astype(int) - 570
            x = x[(mins >= 0) & (mins < 390)].assign(slot=(mins // step))
            x = x[x['ts'].str.slice(0, 10).isin(didx)]
            if not len(x):
                continue
            si = x['sym'].map(sidx).to_numpy()
            di = x['ts'].str.slice(0, 10).map(didx).to_numpy()
            mine = close[si, di, x['slot'].to_numpy()]
            ok = np.isfinite(mine)
            rel = np.abs(mine[ok] / x['close'].to_numpy()[ok] - 1)
            n += int(ok.sum())
            agree += int((rel <= 1e-6).sum())
            if (rel > 1e-6).any():
                worst.append(float(rel.max()))
    out.append(f'{mode}: bars in both {n:,}; closes agreeing to 1e-6 {agree / max(1, n):.4%}; largest relative gap '
               f'{max(worst) if worst else 0:.2e}')
    print(out[-1], flush=True)
(HERE / 'out' / 'crosscheck.txt').write_text('\n'.join(out) + '\n')
