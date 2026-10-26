"""The unseen months (targets ending 2025-07..2026-08), once, at the end, from the forecasts forecast.py saved: P1,
P1', P2, P2', each market, and the errors in percentage points (C5).

    python3 -I -B unseen.py     # -> unseen.txt, unseen.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mf_common as M  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
KS = {k: i for i, k in enumerate(M.INPUTS)}


def main():
    z = np.load(M.WORK / 'forecasts.npz')
    months = z['months']
    mask = M.period_mask(months, 12, *M.UNSEEN)
    lines, res = [], {}

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    out(f'unseen: targets ending {M.UNSEEN[0]}..{M.UNSEEN[1]} ({int(mask.sum())} origins, '
        f'{M.mkey(months[mask][0])}..{M.mkey(months[mask][-1])})')
    for tag, nm in (('h12', 'P1'), ('h12two', "P1'")):
        ys = [z[f'y_{b}_{tag}'] for b in M.MARKETS]
        fbs = [z[f'fb_{b}_{tag}'] for b in M.MARKETS]
        fks = [z[f'fk_{b}_{tag}'] for b in M.MARKETS]
        e = M.evaluate(ys, fbs, [fk.mean(1) for fk in fks], mask, boot=False)
        res[nm] = {'pooled': e}
        out(f'{nm}: {M.fmt_ev(e)}')
        for th, ks in M.THESES.items():
            et = M.evaluate(ys, fbs, [M.combo(fk, [KS[k] for k in ks]) for fk in fks], mask, boot=False)
            res[nm][th] = et
            out(f'  {th}: R2 {et["r2os"]:+.4f} t {et["cw_t"]:+.2f}')
        for i, b in enumerate(M.MARKETS):
            eb = M.evaluate([ys[i]], [fbs[i]], [fks[i].mean(1)], mask, boot=False)
            res[nm][b] = eb
            out(f'  {M.NAMES[b]}: R2 {eb["r2os"]:+.4f} t {eb["cw_t"]:+.2f}, MAE {100 * eb["mae_b"]:.2f} -> '
                f'{100 * eb["mae_m"]:.2f} pts')
    res['C5'] = {}
    for b in M.MARKETS:
        y, g, f0, fb = (z[f'{k}_{b}_h12'] for k in ('y', 'g', 'f0', 'fb'))
        fm = z[f'fk_{b}_h12'].mean(1)
        ok = mask & np.isfinite(y) & np.isfinite(fm)
        res['C5'][b] = {nm: float(100 * np.mean(np.abs(y[ok] - f[ok])))
                        for nm, f in (('mean', f0), ('no change', g), ('own history', fb), ('model', fm))}
        out(f'C5 {M.NAMES[b]}: ' + '; '.join(f'{k} {v:.2f}' for k, v in res['C5'][b].items()))
    (HERE / 'unseen.txt').write_text('\n'.join(lines) + '\n')
    M.dump(res, HERE / 'unseen.json')


if __name__ == '__main__':
    main()
