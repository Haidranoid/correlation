"""Data checks for the plan (coverage only; no forecast is scored here): each input's first and last month, the
origins where every input exists, and how many forecasts each period gets, per market and horizon.

    python3 -I -B inputs.py     # -> inputs.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mf_common as M  # noqa: E402
import numpy as np  # noqa: E402


def main():
    lines = []

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    X = M.common_inputs()
    for k in ('E1', 'E2', 'E3', 'T1', 'T2', 'T3', 'LG', 'A'):
        ks = sorted(X[k])
        out(f'{k}: {ks[0]}..{ks[-1]} ({len(ks)} months); missing inside: '
            f'{[M.mkey(m) for m in range(M.mindex(ks[0]), M.mindex(ks[-1])) if M.mkey(m) not in X[k]]}')
    A = X['A']
    out('A by January: ' + ', '.join(f'{y}: {100 * A[f"{y}-01"]:+.2f}%' for y in range(2003, 2027) if f'{y}-01' in A))
    L = M.legal_share()
    out('legal share at year end: ' + ', '.join(f'{y} {L[f"{y}-12"]:.3f}' for y in range(2012, 2026)))
    for b, code in M.MARKETS.items():
        S = M.bea(code)
        P = M.price_input(M.bls(M.PRICE[b]), X)
        ks = sorted(P)
        out(f'{b} {M.NAMES[b]} ({code}): {min(S)}..{max(S)}; its price input {ks[0]}..{ks[-1]}')
        for h in (12, 3):
            months, y, g, Xm = M.panel(X, S, M.bls(M.PRICE[b]), h)
            have = np.isfinite(g) & np.isfinite(Xm).all(1)
            o = M.oos(y, g, Xm, months, h, sink=False)
            f = np.isfinite(o['fb'])
            cnt = {p: int((f & M.period_mask(months, h, *r)).sum()) for p, r in
                   (('dev', M.DEV), ('check', M.CHECK), ('unseen', M.UNSEEN))}
            out(f'  h={h}: origins with all inputs {M.mkey(months[have][0])}..{M.mkey(months[have][-1])}; '
                f'first forecast {M.mkey(months[f][0])}; forecasts by target end: {cnt}')
    (Path(__file__).resolve().parent / 'inputs.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
