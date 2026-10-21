"""Summarise replication.py's per-offset tallies: kept vs chance, and same-sign replication on later data, by window
and offset band. Writes research/v1_pair_finder/replication.txt and replication.json."""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

import v1lib
import c2eval

OUT = c2eval.WORK / 'v1_pair_finder'
BANDS = [('0', 0, 0), ('1-30', 1, 30), ('31-365', 31, 365), ('>365', 366, 10 ** 9)]


def cal_span(w):
    """Median calendar days covered by w trading rows (training period); offsets at least this long are usable."""
    ex = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
    d = ex['days'][c2eval.day_index(ex['days'], '2015-06-29'):c2eval.day_index(ex['days'], '2022-06-28')]
    return int(np.ceil(np.median(d[w:] - d[:-w])))


def agg(t, sel):
    g = lambda f: float(t[f][sel].sum())
    with np.errstate(invalid='ignore', divide='ignore'):
        base_k = np.where(t['all_ok'] > 0, t['all_agree'] / t['all_ok'], np.nan)
        base_sg = np.where(t['all_ok'] > 0, t['all_signed'] / t['all_ok'], np.nan)
        base_rk = np.where(t['all_ok'] > 0, t['all_rekept'] / t['all_ok'], np.nan)
    ko = t['kept_ok'][sel]
    m = ko > 0
    wsum = ko[m].sum()
    nulls = [g(f'null_kept{q}') for q in range(3)]
    out = {
        'offsets': int(sel.sum()), 'tested': g('tested'), 'kept': g('kept'), 'kept_pos_share': g('kept_pos') / max(1, g('kept')),
        'null_kept_mean': float(np.mean(nulls)), 'null_kept_draws': nulls,
        'kept_r50': g('kept_r50'), 'null_r50': g('null_r50'), 'kept_r70': g('kept_r70'), 'null_r70': g('null_r70'),
        'kept_r90': g('kept_r90'), 'null_r90': g('null_r90'),
        'kept_checked': g('kept_ok'),
        'kept_same_sign': g('kept_agree') / max(1, g('kept_ok')),
        'kept_same_sign_kmatched_base': float((ko[m] * base_k[sel][m]).sum() / wsum) if wsum else float('nan'),
        'all_same_sign': g('all_agree') / max(1, g('all_ok')),
        'null_picked_same_sign': g('null_agree') / max(1, g('null_ok')),
        'strong_same_sign': g('strong_agree') / max(1, g('strong_ok')), 'strong_checked': g('strong_ok'),
        'kept_signed_mean': g('kept_signed') / max(1, g('kept_ok')),
        'kept_signed_kmatched_base': float((ko[m] * base_sg[sel][m]).sum() / wsum) if wsum else float('nan'),
        'null_signed_mean': g('null_signed') / max(1, g('null_ok')),
        'kept_rekept': g('kept_rekept') / max(1, g('kept_ok')),
        'kept_rekept_kmatched_base': float((ko[m] * base_rk[sel][m]).sum() / wsum) if wsum else float('nan'),
        'null_rekept': g('null_rekept') / max(1, g('null_ok')),
    }
    return out


def main():
    res, lines = {}, []
    lines.append('v1 search on 500 stocks, 2015-06-29..2022-06-27; later data: B changes wholly in 2022-06-28..2025-06-27.')
    lines.append('kept = |r| >= 0.3, n >= 10 (v1). null = targets sign-flipped per day (3 draws, mean). same sign = later r has the '
                 'training sign; base = all relationships at the same offsets, weighted like the kept ones.')
    tot = {}
    for w in v1lib.WINDOWS:
        p = OUT / f'replication_perk_w{w}.npz'
        if not p.exists():
            continue
        z = np.load(p)
        t = {f: z[f] for f in z.files if z[f].ndim == 1}
        k = np.arange(len(t['kept']))
        span = cal_span(w)
        res[w] = {'cal_span': span}
        lines.append(f'\n== window {w} rows (calendar span ~{span} days), offsets 0..{len(k) - 1}')
        lines.append(f'{"band":>10} {"tested":>10} {"kept":>10} {"null":>10} {"x null":>7} {"pos%":>5} | {"same sign":>9} {"base":>6} {"null-pk":>7} {"strong":>6} | '
                     f'{"sgn*r":>6} {"base":>6} {"null":>6} | {"rekept":>6} {"base":>6} {"null":>6}')
        bands = BANDS + [(f'>={span} (usable)', span, 10 ** 9), ('all', 0, 10 ** 9)]
        for name, lo, hi in bands:
            sel = (k >= lo) & (k <= hi)
            a = agg(t, sel)
            res[w][name] = a
            lines.append(f'{name:>10} {a["tested"]:10.3g} {a["kept"]:10.3g} {a["null_kept_mean"]:10.3g} {a["kept"] / max(1, a["null_kept_mean"]):7.2f} '
                         f'{a["kept_pos_share"]:5.0%} | {a["kept_same_sign"]:9.3f} {a["kept_same_sign_kmatched_base"]:6.3f} {a["null_picked_same_sign"]:7.3f} '
                         f'{a["strong_same_sign"]:6.3f} | {a["kept_signed_mean"]:+6.3f} {a["kept_signed_kmatched_base"]:+6.3f} {a["null_signed_mean"]:+6.3f} | '
                         f'{a["kept_rekept"]:6.3f} {a["kept_rekept_kmatched_base"]:6.3f} {a["null_rekept"]:6.3f}')
        lines.append(f'  tails (all offsets): |r|>=0.5 real {res[w]["all"]["kept_r50"]:.3g} vs null {res[w]["all"]["null_r50"]:.3g}; '
                     f'>=0.7 {res[w]["all"]["kept_r70"]:.3g} vs {res[w]["all"]["null_r70"]:.3g}; >=0.9 {res[w]["all"]["kept_r90"]:.3g} vs {res[w]["all"]["null_r90"]:.3g}')
        for f in t:
            tot.setdefault(f, []).append(t[f])
    # all windows together, by band
    if tot:
        lines.append('\n== all windows together')
        for name, lo, hi in BANDS + [('all', 0, 10 ** 9)]:
            parts = []
            for ti in range(len(tot['kept'])):
                kk = np.arange(len(tot['kept'][ti]))
                parts.append({f: tot[f][ti][(kk >= lo) & (kk <= hi)] for f in tot})
            t = {f: np.concatenate([p[f] for p in parts]) for f in tot}
            a = agg(t, np.ones(len(t['kept']), bool))
            res.setdefault('all_windows', {})[name] = a
            lines.append(f'{name:>10}: kept {a["kept"]:.3g} vs null {a["null_kept_mean"]:.3g} ({a["kept"] / max(1, a["null_kept_mean"]):.2f}x); '
                         f'same sign later {a["kept_same_sign"]:.3f} (base {a["kept_same_sign_kmatched_base"]:.3f}, null-picked {a["null_picked_same_sign"]:.3f}); '
                         f'sign*r {a["kept_signed_mean"]:+.3f} (base {a["kept_signed_kmatched_base"]:+.3f}); re-kept {a["kept_rekept"]:.3f} '
                         f'(base {a["kept_rekept_kmatched_base"]:.3f}, null {a["null_rekept"]:.3f})')
    (HERE / 'replication.txt').write_text('\n'.join(lines) + '\n')
    (HERE / 'replication.json').write_text(json.dumps(res, indent=1, default=float))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
