#!/usr/bin/python3
"""Correctness oracle for the Rust census: exact Pearson r for sampled pairs in one window.

Uses the same data handling as lag_spectrum.py for the 2016-06-28 .. 2022-06-27 window (daily,
market-residual returns with window betas, standardized within the window, clipped at +-5 sd,
missing days as 0) and writes, for lags 0 and 1:
  - the universe (tickers, in order),
  - exact r = corrcoef(D[:T-k, i], D[k:, j]) for 2,000 random pairs and the 50 strongest lag-1 pairs.
`c2 census-validate` recomputes the same pairs and reports the largest difference.

    /usr/bin/python3 research/census_oracle.py
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np

import leadlag_holdout as lh

warnings.filterwarnings('ignore', category=RuntimeWarning)
lh.LOAD_FROM = '2016-05-01'
WINDOW = ('2016-06-28', '2022-06-27')
OUT = Path(__file__).with_name('census_oracle.json')


def main():
    kinds = lh.stooq_kinds()
    syms, dates, P, V = lh.load_prices()
    kind = np.array([kinds.get(s, 'unknown') for s in syms])
    kind[[bool(lh.TEST_SYMBOL.match(s)) for s in syms]] = 'test'
    jump, back = np.log(P[1:-1] / P[:-2]), np.log(P[2:] / P[1:-1])
    P[1:-1][(np.abs(jump) > math.log(2)) & (np.abs(jump + back) < 0.1)] = np.nan
    R = np.vstack([np.full(P.shape[1], np.nan), np.log(P[1:] / P[:-1])])

    w = (dates >= np.datetime64(WINDOW[0])) & (dates <= np.datetime64(WINDOW[1]))
    last = np.flatnonzero(w)[-252:]
    coverage = np.mean(~np.isnan(R[w]), 0)
    med_price = np.nanmedian(P[last], 0)
    dollar_vol = np.nanmedian(P[last] * V[last], 0)
    U = np.flatnonzero((kind == 'stock') & (coverage >= lh.MIN_COVERAGE) &
                       (med_price >= lh.MIN_PRICE) & (dollar_vol >= lh.MIN_DOLLAR_VOL))

    Ru = R[:, U]
    mkt = np.nanmean(Ru, 1)
    m = mkt[w] - np.nanmean(mkt[w])
    beta = np.nansum((Ru[w] - np.nanmean(Ru[w], 0)) * m[:, None], 0) / np.nansum(m ** 2)
    E = Ru - beta * mkt[:, None]
    sd = np.nanstd(E[w], 0)
    D = np.nan_to_num(np.clip((E - np.nanmean(E[w], 0)) / np.where(sd > 0, sd, 1), -5, 5))[w]
    T, N = D.shape

    rng = np.random.default_rng(1)
    pairs = []
    for k in (0, 1):
        Zs = (D - D.mean(0)) / D.std(0)
        C = (Zs[:T - k].T @ Zs[k:]) / (T - k)
        np.fill_diagonal(C, 0)
        chosen = [(int(i), int(j)) for i, j in rng.integers(0, N, size=(2000, 2)) if i != j]
        if k == 1:
            flat = np.argpartition(np.abs(C), -50, axis=None)[-50:]
            chosen += [tuple(map(int, np.unravel_index(f, C.shape))) for f in flat]
        for i, j in chosen:
            r = np.corrcoef(D[:T - k, i], D[k:, j])[0, 1]
            pairs.append({'lag': k, 'leader': syms[U[i]], 'target': syms[U[j]], 'r': float(r)})

    OUT.write_text(json.dumps({
        'window': WINDOW, 'days': int(T), 'universe': [syms[u] for u in U],
        'first_day': str(dates[w][0]), 'last_day': str(dates[w][-1]), 'pairs': pairs,
    }))
    print(f'{N} stocks, {T} days, {len(pairs)} pairs -> {OUT}')


if __name__ == '__main__':
    main()
