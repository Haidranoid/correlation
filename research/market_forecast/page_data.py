"""Data for the explainer page (not part of the test): each market's 12-month growth, and the wine forecasts (the
model, own history and "same as last year") against what happened, from the forecasts forecast.py saved.

    python3 -I -B page_data.py     # -> artifact_data.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mf_common as M  # noqa: E402
import numpy as np  # noqa: E402


def pct(v):
    return None if v is None or not np.isfinite(v) else round(float(100 * (np.exp(v) - 1)), 2)


def main():
    z = np.load(M.WORK / 'forecasts.npz')
    months = z['months']
    out = {'growth': {}, 'forecasts': {}}
    for b in M.MARKETS:
        g = z[f'g_{b}_h12']
        out['growth'][b] = [[M.mkey(L), pct(v)] for L, v in zip(months, g) if '2005-01' <= M.mkey(L) <= '2026-08']
        y, fb = z[f'y_{b}_h12'], z[f'fb_{b}_h12']
        fm = z[f'fk_{b}_h12'].mean(1)
        ok = np.isfinite(fm) & np.isfinite(y)
        out['forecasts'][b] = [[M.mkey(L + 12), pct(y[i]), pct(fm[i]), pct(fb[i]), pct(g[i])]
                               for i, L in enumerate(months) if ok[i]]
    (Path(__file__).resolve().parent / 'artifact_data.json').write_text(json.dumps(out, separators=(',', ':')))
    print(len(json.dumps(out)), 'bytes;', {b: (v[0][0], v[-1][0]) for b, v in out['forecasts'].items()})


if __name__ == '__main__':
    main()
