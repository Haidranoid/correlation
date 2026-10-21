"""Overnight peers: peer diagnostics by period from out/build_<tag>.json (overlap as mean Jaccard, mean peer
correlation per type, coverage) and the example peer lists. No outcomes.

    python3 -I report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np

L = []
for tag, periods in (('old', [('2009-2019', '2009-01-01', '2019-12-31'), ('2020-2025-06', '2020-01-01', '2025-06-27')]),
                     ('new', [('unseen', '2025-06-30', '2026-10-19')])):
    B = json.loads((C.OUT / f'build_{tag}.json').read_text())
    cal = B['calendar']
    L.append(f'returns check ({tag}): {B["check"]}')
    for name, a, z in periods:
        if tag == 'new':
            rs = [r for r in cal if r['date'] >= '2025-06-27']   # calendar days from 2025-07-16 (the books' days)
        else:
            rs = [r for r in cal if a <= r['date'] <= z]
        L.append(f'\n## {name}: {len(rs)} calendar days, {rs[0]["date"]} .. {rs[-1]["date"]}')
        L.append(f'universe: mean {np.mean([r["n_uni"] for r in rs]):.0f}')
        L.append('| Type | Share with >= 5 peers | Mean peers kept | Mean peer correlation | Mean best-peer correlation |')
        L.append('|---|---|---|---|---|')
        for ty in C.TYPES:
            L.append(f'| {C.TYPE_NAME[ty]} | {np.mean([r[f"n_{ty}"] / r["n_uni"] for r in rs]):.4f} | '
                     f'{np.mean([r[f"npeer_{ty}"] for r in rs]):.2f} | {np.mean([r[f"corr_{ty}"] for r in rs]):.3f} | '
                     f'{np.mean([r[f"corr1_{ty}"] for r in rs]):.3f} |')
        L.append('| Pair | Mean Jaccard |')
        L.append('|---|---|')
        for p in ('on_whole', 'on_intra', 'intra_whole'):
            L.append(f'| {p} | {np.mean([r[f"jac_{p}"] for r in rs]):.3f} |')
        L.append(f'| two random 20-stock sets (about) | {np.mean([r["jac_random"] for r in rs]):.3f} |')
B = json.loads((C.OUT / 'build_old.json').read_text())
ex = B['examples']
L.append(f'\n## Example peers at {ex["date"]} (top 8, correlation)')
for tk, v in ex.items():
    if tk == 'date':
        continue
    for ty in C.TYPES:
        x = v[ty]
        s = x if isinstance(x, str) else ', '.join(f'{a} {b:.2f}' for a, b in x[:8])
        L.append(f'- {tk} {C.TYPE_NAME[ty]}: {s}')
txt = '\n'.join(L) + '\n'
(C.OUT / 'peers_report.md').write_text(txt)
print(txt)
