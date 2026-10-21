"""Money test for v1's pair finder: relationships re-learned each year by v1's own search, turned into scores, graded
by the shared judge (grade.py).

For each test period [start, end): universe = 500 most-traded complete stocks over the 7 years before it; v1's search
(every ordered pair, every calendar-day offset 0..v1's maximum, keep |r| >= 0.3 with n >= 10) on those 7 years; then on
every decision day d of the period, B's score = sum over kept (A -> B, w, k) of r x A's w-row change ending on calendar
date tgt(d) - k, used only if that date is a trading day on or before d (exact calendar match, as v1 aligns).
  aligned (headline): tgt(d) = calendar date of trading day d + w, i.e. the relationship's forecast of B's change over
                      (d, d + w]; offsets shorter than that calendar span drop out.
  literal:            tgt(d) = d itself (the brief's recipe: A's change ending k days before d).
A stock with no usable relationship on a day gets no score (NaN).

    python3 -I research/v1_pair_finder/walkforward.py --variant aligned --windows 7 14 21 30 40 60 90 180
    python3 -I research/v1_pair_finder/walkforward.py --variant literal --windows 30
    python3 -I research/v1_pair_finder/walkforward.py --variant aligned --null 101 --windows 30
    python3 -I research/v1_pair_finder/walkforward.py --unseen --windows 7 14 21 30 40 60 90 180
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np

import v1lib
import c2eval

OUT = c2eval.WORK / 'v1_pair_finder'
TRAIN_YEARS = 7
UNIVERSE = 500


def extended_calendar(days, extra=600):
    """Export trading days, continued with weekdays after the last one (for target dates past the end of the data)."""
    out = list(days.astype(np.int64))
    d = int(out[-1])
    while len(out) < len(days) + extra:
        d += 1
        if (d + 3) % 7 < 5:          # 1970-01-01 was a Thursday: (d + 3) % 7 is 0 on Mondays
            out.append(d)
    return np.array(out, dtype=np.int64)


def periods_walkforward(days):
    out = []
    for y in range(2006, 2025):
        h0 = c2eval.day_index(days, f'{y}-06-28')
        h1 = min(len(days), c2eval.day_index(days, f'{y + 1}-06-28'))
        s = c2eval.day_index(days, f'{y - TRAIN_YEARS}-06-28')
        out.append((f'{y}-{(y + 1) % 100:02d}', s, h0, h1))
    return out


def run_period(ex, cal_ext, s, h0, h1, windows, variant, null_seed, scores, log):
    """Fill scores[w][u, d] for decision days d in [h0 - 1, h1 - 1) from the 7 years [s, h0)."""
    days = ex['days'].astype(np.int64)
    u = v1lib.select_universe(ex, s, h0, size=UNIVERSE)
    N = len(u)
    wmax = max(windows)
    s_ext = max(0, s - wmax - 10)
    P = v1lib.ffill(np.asarray(ex['close'][u, s_ext:h1], dtype=np.float64))   # gaps carried forward (causal)
    cal_p = days[s_ext:h1]
    tr0, tr1 = s - s_ext, h0 - s_ext
    span = int(days[h0 - 1] - days[s])
    D = np.arange(h0 - 1, h1 - 1)                 # decision days (absolute)
    dcal = days[D]
    alive = np.isfinite(np.asarray(ex['close'][u][:, D]))
    info = {'universe': u.tolist(), 'train': [c2eval.iso(days[s]), c2eval.iso(days[h0 - 1])],
            'decisions': [int(h0 - 1), int(h1 - 1)], 'windows': {}}
    for w in windows:
        t0 = time.time()
        with np.errstate(invalid='ignore', divide='ignore'):
            chA = v1lib.pct_changes(P, w).astype(np.float32)
        trA = chA[:, tr0:tr1]
        if null_seed is None:
            trB = trA
        else:
            trB = v1lib.pct_changes(v1lib.flip_null_prices(P[:, tr0:tr1], null_seed), w).astype(np.float32)
        cal_tr = cal_p[tr0:tr1]
        valid_tr = np.arange(tr1 - tr0) >= w
        tgt = cal_ext[D + w] if variant == 'aligned' else dcal
        S = np.zeros((N, len(D)), np.float32)
        Cn = np.zeros((N, len(D)), np.float32)
        kmax = v1lib.max_offset(span, w)
        n_kept = n_used = 0
        for k in range(kmax + 1):
            src = tgt - k
            okd = src <= dcal
            if not okd.any():
                continue
            pos = np.searchsorted(cal_p, src)
            posc = np.minimum(pos, len(cal_p) - 1)
            hit = okd & (pos < len(cal_p)) & (cal_p[posc] == src) & (posc >= w)
            if not hit.any():
                continue
            i, j = v1lib.offset_pairs(cal_tr, valid_tr, valid_tr, k)
            n = len(i)
            if n < v1lib.N_MIN:
                continue
            R = v1lib.corr_block(trA[:, i], trB[:, j])
            m = v1lib.keep_mask(R, n) & np.isfinite(R)
            nk = int(m.sum())
            if nk == 0:
                continue
            n_kept += nk
            n_used += 1
            Wt = np.where(m, R, 0.0).astype(np.float32)
            G = chA[:, posc[hit]]
            Gv = np.isfinite(G)
            S[:, hit] += Wt.T @ np.where(Gv, G, 0.0).astype(np.float32)
            Cn[:, hit] += m.astype(np.float32).T @ Gv.astype(np.float32)
        out = np.where((Cn > 0) & alive, S, np.nan).astype(np.float32)
        scores[w][np.ix_(u, D)] = out
        cover = float(np.isfinite(out).mean())
        info['windows'][w] = {'kmax': kmax, 'kept_used_offsets': n_kept, 'offsets_with_kept': n_used, 'score_coverage': cover,
                              'mean_terms_per_score': float(Cn[Cn > 0].mean()) if (Cn > 0).any() else 0.0}
        log(f'    w={w}: offsets 0..{kmax}, kept relationships at usable offsets {n_kept:,} on {n_used} offsets; '
            f'scores on {cover:.0%} of stock-days, {info["windows"][w]["mean_terms_per_score"]:,.0f} terms each; {time.time() - t0:.0f} s')
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='aligned', choices=['aligned', 'literal'])
    ap.add_argument('--windows', type=int, nargs='+', default=[30])
    ap.add_argument('--null', type=int, default=None, help='sign-flip null seed for the targets (B) in training')
    ap.add_argument('--unseen', action='store_true', help='the 16 unseen months on the new export (once)')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    export = c2eval.EXPORT_NEW if a.unseen else c2eval.EXPORT_OLD
    ex = v1lib.load_panel(export)
    days = ex['days']
    cal_ext = extended_calendar(days)
    if a.unseen:
        h0 = c2eval.day_index(days, '2025-06-30')
        periods = [('2025-06-30..', c2eval.day_index(days, f'{2025 - TRAIN_YEARS}-06-28'), h0, len(days))]
    else:
        periods = periods_walkforward(days)
    tag = ('unseen_' if a.unseen else '') + a.variant + (f'_null{a.null}' if a.null is not None else '')
    logf = open(HERE / f'walkforward_{tag}.log', 'a')

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log(f'== {time.strftime("%Y-%m-%d %H:%M")} variant={a.variant} null={a.null} windows={a.windows} export={export}')
    N, T = ex['close'].shape
    scores = {w: np.full((N, T), np.nan, np.float32) for w in a.windows}
    infos = {}
    t0 = time.time()
    for name, s, h0, h1 in periods:
        log(f'  {name}: training {c2eval.iso(days[s])}..{c2eval.iso(days[h0 - 1])}, decisions {c2eval.iso(days[h0 - 1])}..{c2eval.iso(days[h1 - 2])}')
        infos[name] = run_period(ex, cal_ext, s, h0, h1, a.windows, a.variant, a.null, scores, log)
    paths = {}
    for w in a.windows:
        paths[w] = str(c2eval.save_scores(scores[w], f'v1_pair_finder/{tag}_w{w}', export=export))
    if len(a.windows) > 1:
        stack = np.stack([scores[w] for w in a.windows])
        comb = np.where(np.isfinite(stack).any(axis=0), np.nansum(stack, axis=0), np.nan).astype(np.float32)
        paths['all'] = str(c2eval.save_scores(comb, f'v1_pair_finder/{tag}_wall', export=export))
    (OUT / f'{tag}_info.json').write_text(json.dumps({'paths': paths, 'periods': infos}, default=str))
    log(f'  done in {time.time() - t0:.0f} s: {paths}')


if __name__ == '__main__':
    main()
