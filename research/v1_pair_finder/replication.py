"""Chance and replication for v1's pair finder.

v1's search (every ordered pair of 500 liquid stocks, 8 windows, every calendar-day offset 0..v1's maximum, keep
|r| >= 0.3 with n >= 10) on 2015-06-29..2022-06-27, against a sign-flip null on the targets (3 draws), and every
relationship recomputed on later data: B's changes whose whole window lies in 2022-06-28..2025-06-27, each with A's
change k calendar days earlier.

Writes per-offset tallies to ~/.local/share/correlation2/experiments/v1_pair_finder/replication_perk.npz; summarise
with replication_summary.py.

    python3 -I research/v1_pair_finder/replication.py [windows...]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

import v1lib
import c2eval

OUT = c2eval.WORK / 'v1_pair_finder'
TRAIN = ('2015-06-29', '2022-06-28')     # [start, end)
NULL_SEEDS = (101, 102, 103)
THRESH = (0.3, 0.5, 0.7, 0.9)
# per-offset tallies (index k)
FIELDS = ['n_train', 'n_test', 'tested',
          'kept', 'kept_pos', 'kept_r50', 'kept_r70', 'kept_r90',
          'null_kept0', 'null_kept1', 'null_kept2', 'null_r50', 'null_r70', 'null_r90',
          'all_ok', 'all_agree', 'all_rekept', 'all_signed',
          'kept_ok', 'kept_agree', 'kept_rekept', 'kept_signed',
          'strong_ok', 'strong_agree',
          'null_ok', 'null_agree', 'null_rekept', 'null_signed']


def main(windows):
    OUT.mkdir(parents=True, exist_ok=True)
    ex = v1lib.load_panel(c2eval.EXPORT_OLD)
    days = ex['days']
    s, e = c2eval.day_index(days, TRAIN[0]), c2eval.day_index(days, TRAIN[1])
    T = len(days)
    u = v1lib.select_universe(ex, s, e, size=500)
    N = len(u)
    C0 = np.asarray(ex['close'][u, s:T], dtype=np.float64)
    C = v1lib.ffill(C0)      # gaps carried forward (causal)
    cal = days[s:T].astype(np.int64)
    Ltr = e - s
    span = int(cal[Ltr - 1] - cal[0])
    test_complete = (np.isfinite(C0[:, Ltr:]) & (C0[:, Ltr:] > 0)).all(axis=1)
    filled = int((~np.isfinite(C0[:, :Ltr])).sum())
    print(f'universe {N} stocks ({", ".join(ex["tickers"][u[:8]])}, ...); training {c2eval.iso(cal[0])}..{c2eval.iso(cal[Ltr - 1])} '
          f'({Ltr} days, span {span} calendar days); later data to {c2eval.iso(cal[-1])}; '
          f'{test_complete.sum()} of {N} have every later close; {filled} training closes carried forward', flush=True)
    np.save(OUT / 'replication_universe.npy', u)
    off = ~np.eye(N, dtype=bool)
    idx = np.arange(len(cal))
    result = {}
    for w in windows:
        t0 = time.time()
        ch = v1lib.pct_changes(C, w).astype(np.float32)
        chn = [v1lib.pct_changes(v1lib.flip_null_prices(C[:, :Ltr], sd), w).astype(np.float32) for sd in NULL_SEEDS]
        valid_tr = (idx >= w) & (idx < Ltr)
        valid_trn = valid_tr[:Ltr]
        valid_a = idx >= w
        valid_b = idx >= Ltr - 1 + w
        kmax = v1lib.max_offset(span, w)
        tal = {f: np.zeros(kmax + 1) for f in FIELDS}
        for k in range(kmax + 1):
            i, j = v1lib.offset_pairs(cal, valid_tr, valid_tr, k)
            n = len(i)
            tal['n_train'][k] = n
            if n < v1lib.N_MIN:
                continue
            R = v1lib.corr_block(ch[:, i], ch[:, j])
            fin = np.isfinite(R) & off
            tal['tested'][k] = fin.sum()
            kept = v1lib.keep_mask(R, n) & fin
            a = np.abs(R)
            tal['kept'][k] = kept.sum()
            tal['kept_pos'][k] = (kept & (R > 0)).sum()
            for th, f in zip(THRESH[1:], ('kept_r50', 'kept_r70', 'kept_r90')):
                tal[f][k] = (fin & (a >= th)).sum()
            # null: leaders real, targets sign-flipped
            Rn0 = None
            for q, chq in enumerate(chn):
                Rn = v1lib.corr_block(ch[:, i], chq[:, j])
                kn = v1lib.keep_mask(Rn, n) & np.isfinite(Rn) & off
                tal[f'null_kept{q}'][k] = kn.sum()
                if q == 0:
                    Rn0, kn0 = Rn, kn
                    an = np.abs(Rn)
                    for th, f in zip(THRESH[1:], ('null_r50', 'null_r70', 'null_r90')):
                        tal[f][k] = (np.isfinite(Rn) & off & (an >= th)).sum()
            # later data
            it, jt = v1lib.offset_pairs(cal, valid_a, valid_b, k)
            tal['n_test'][k] = len(it)
            if len(it) < v1lib.N_MIN:
                continue
            Rt = v1lib.corr_block(ch[:, it], ch[:, jt])
            ok = fin & np.isfinite(Rt)
            same = np.sign(R) == np.sign(Rt)
            rek = same & (np.abs(Rt) >= v1lib.R_KEEP)
            sg = np.where(ok, np.sign(R) * Rt, 0.0)
            tal['all_ok'][k] = ok.sum()
            tal['all_agree'][k] = (ok & same).sum()
            tal['all_rekept'][k] = (ok & rek).sum()
            tal['all_signed'][k] = sg.sum()
            ko = ok & kept
            tal['kept_ok'][k] = ko.sum()
            tal['kept_agree'][k] = (ko & same).sum()
            tal['kept_rekept'][k] = (ko & rek).sum()
            tal['kept_signed'][k] = sg[ko].sum()
            so = ko & (a >= 0.5)
            tal['strong_ok'][k] = so.sum()
            tal['strong_agree'][k] = (so & same).sum()
            no = kn0 & np.isfinite(Rt)
            nsame = np.sign(Rn0) == np.sign(Rt)
            tal['null_ok'][k] = no.sum()
            tal['null_agree'][k] = (no & nsame).sum()
            tal['null_rekept'][k] = (no & nsame & (np.abs(Rt) >= v1lib.R_KEEP)).sum()
            tal['null_signed'][k] = np.where(no, np.sign(Rn0) * Rt, 0.0).sum()
        result[w] = tal
        kt = tal['kept'].sum()
        nk = np.mean([tal[f'null_kept{q}'].sum() for q in range(3)])
        print(f'w={w}: offsets 0..{kmax}; tested {tal["tested"].sum():.3g}; kept {kt:.3g} ({kt / tal["tested"].sum():.1%}), '
              f'null kept {nk:.3g}; kept same sign later {tal["kept_agree"].sum() / max(1, tal["kept_ok"].sum()):.3f} '
              f'(all {tal["all_agree"].sum() / max(1, tal["all_ok"].sum()):.3f}, null-picked {tal["null_agree"].sum() / max(1, tal["null_ok"].sum()):.3f}); '
              f'{time.time() - t0:.0f} s', flush=True)
        np.savez_compressed(OUT / f'replication_perk_w{w}.npz', **tal, w=w, N=N, span=span)


if __name__ == '__main__':
    ws = [int(x) for x in sys.argv[1:]] or list(v1lib.WINDOWS)
    main(ws)
