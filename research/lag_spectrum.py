#!/usr/bin/python3
"""Lag spectrum: at which lags, from 1 day to 1 year, does lead-lag structure between stocks repeat?

leadlag_holdout.py only tried lags of 1-5 days. A stock might lead another by a month or a year,
and several leaders might act at different lags. This searches every pair at every lag, at daily,
weekly and monthly resolution, on 2016-06 .. 2022-06, then asks whether the lag structure found
there shows up again in 2022-06 .. 2025-06, data it never saw.

For each lag k, the all-pairs matrix C_k[i, j] = corr(stock i at t, stock j at t+k) is computed
in both periods (market-residual returns). Per lag:

  similarity  correlation, across all ~4.6M ordered pairs, between the early and later matrices.
              Real, stable lead-lag at lag k makes it positive even if no single pair can be
              proven significant on its own.
  chance      the same similarity after shifting the later period's targets in time, so nothing
              real can line up. z = (similarity - chance mean) / chance sd.
  top pairs   the 1,000 strongest early pairs at lag k: share with the same sign later.

Lag 0 (same-day co-movement) is the check that the method can see structure when it is there.

    /usr/bin/python3 research/lag_spectrum.py | tee research/lag_spectrum.txt
"""
import math
import time
import warnings

import numpy as np

import leadlag_holdout as lh

warnings.filterwarnings('ignore', category=RuntimeWarning)
lh.LOAD_FROM = '2016-05-01'
EARLY = ('2016-06-28', '2022-06-27')
LATER = ('2022-06-28', '2025-06-27')
SCALES = (('daily', 1, 250), ('weekly', 5, 52), ('monthly', 21, 24))   # name, days per step, max lag
SHOW = {'daily': (0, 1, 2, 3, 4, 5, 10, 21, 42, 63, 126, 189, 250),
        'weekly': (0, 1, 2, 3, 4, 8, 13, 26, 39, 52),
        'monthly': (0, 1, 2, 3, 6, 9, 12, 18, 24)}
MIN_OBS = 12
TOP = 1000
rng = np.random.default_rng(2016)


def block_sum(X, step):
    """Non-overlapping sums of daily values over `step` days."""
    n = (len(X) // step) * step
    return X[:n].reshape(-1, step, X.shape[1]).sum(1)


def standardize(X):
    sd = X.std(0)
    return ((X - X.mean(0)) / np.where(sd > 0, sd, 1)).astype(np.float32)


def lag_corr(Z, k, target=None):
    target = Z if target is None else target
    return (Z[:len(Z) - k].T @ target[k:]) / (len(Z) - k)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / math.sqrt((a @ a) * (b @ b)))


def main():
    t0 = time.time()
    kinds = lh.stooq_kinds()
    syms, dates, P, V = lh.load_prices()
    kind = np.array([kinds.get(s, 'unknown') for s in syms])
    kind[[bool(lh.TEST_SYMBOL.match(s)) for s in syms]] = 'test'
    jump, back = np.log(P[1:-1] / P[:-2]), np.log(P[2:] / P[1:-1])
    P[1:-1][(np.abs(jump) > math.log(2)) & (np.abs(jump + back) < 0.1)] = np.nan
    R = np.vstack([np.full(P.shape[1], np.nan), np.log(P[1:] / P[:-1])])

    early = (dates >= np.datetime64(EARLY[0])) & (dates <= np.datetime64(EARLY[1]))
    later = (dates >= np.datetime64(LATER[0])) & (dates <= np.datetime64(LATER[1]))
    last = np.flatnonzero(early)[-252:]
    coverage = np.mean(~np.isnan(R[early]), 0)
    med_price = np.nanmedian(P[last], 0)
    dollar_vol = np.nanmedian(P[last] * V[last], 0)
    U = np.flatnonzero((kind == 'stock') & (coverage >= lh.MIN_COVERAGE) &
                       (med_price >= lh.MIN_PRICE) & (dollar_vol >= lh.MIN_DOLLAR_VOL))
    N = len(U)

    # Market-residual daily returns (betas from the early period), standardized and clipped at
    # +-5 sd using early-period stats, missing days as 0. Weekly/monthly are sums of these.
    Ru = R[:, U]
    mkt = np.nanmean(Ru, 1)
    m = mkt[early] - np.nanmean(mkt[early])
    beta = np.nansum((Ru[early] - np.nanmean(Ru[early], 0)) * m[:, None], 0) / np.nansum(m ** 2)
    E = Ru - beta * mkt[:, None]
    sd = np.nanstd(E[early], 0)
    D = np.nan_to_num(np.clip((E - np.nanmean(E[early], 0)) / np.where(sd > 0, sd, 1), -5, 5))

    print(f'{N:,} liquid US stocks, {N * (N - 1):,} ordered pairs per lag. '
          f'Early {EARLY[0]}..{EARLY[1]} ({early.sum()} days), later {LATER[0]}..{LATER[1]} ({later.sum()} days)')

    for name, step, max_lag in SCALES:
        Ze, Zl = standardize(block_sum(D[early], step)), standardize(block_sum(D[later], step))
        off = ~np.eye(N, dtype=bool)
        rows, chance = [], []
        for k in range(max_lag + 1):
            if len(Zl) - k < MIN_OBS:
                break
            a = lag_corr(Ze, k)[off]
            b = lag_corr(Zl, k)[off]
            nulls = [corr(a, lag_corr(Zl, k, np.roll(Zl, s, axis=0))[off])
                     for s in rng.integers(len(Zl) // 4, 3 * len(Zl) // 4, size=2)]
            chance.extend(nulls)
            top = np.argpartition(np.abs(a), -TOP)[-TOP:]
            rows.append((k, corr(a, b), np.mean(nulls), np.mean(np.sign(a[top]) == np.sign(b[top])),
                         len(Ze) - k, len(Zl) - k))
        c_mean, c_sd = np.mean(chance), np.std(chance)
        unit = {'daily': 'trading days', 'weekly': 'weeks', 'monthly': 'months (21 trading days)'}[name]
        print(f'\n== {name}: lags in {unit}; chance similarity {c_mean:+.4f} +- {c_sd:.4f} ==')
        print(f'{"lag":>5} {"similarity":>11} {"z":>7} {"top 1,000 same sign":>20} {"obs early/later":>16}')
        for k, sim, null, same, n_e, n_l in rows:
            if k in SHOW[name]:
                print(f'{k:>5} {sim:>+11.4f} {(sim - c_mean) / c_sd:>+7.1f} {same:>20.1%} {n_e:>8}/{n_l:<7}')
        lagged = [(k, sim) for k, sim, *_ in rows if k > 0]
        z = np.array([(sim - c_mean) / c_sd for _, sim in lagged])
        best_k, best_sim = max(lagged, key=lambda r: r[1])
        print(f'lags 1..{lagged[-1][0]}: {np.sum(z > 3)} of {len(z)} above z = 3 '
              f'(chance would give about {len(z) * 0.00135:.1f}); strongest at lag {best_k} '
              f'(similarity {best_sim:+.4f}, z = {(best_sim - c_mean) / c_sd:+.1f})')
    # Relationships may last only about a year and then change, which a 6-year window averages away.
    # So also learn from one year and test on the next, for each consecutive pair of years.
    print('\n== rolling: learn from one year, test on the next (daily lags, 9 consecutive year pairs) ==')
    starts = [np.datetime64(f'{y}-06-28') for y in range(2016, 2026)]
    years = [np.flatnonzero((dates >= a) & (dates < b)) for a, b in zip(starts, starts[1:])]
    years = [y for y in years if len(y) > 200]
    Zs = [standardize(D[y]) for y in years]
    off = ~np.eye(N, dtype=bool)
    print(f'{"lag":>5} {"mean similarity":>16} {"z":>7} {"top 1,000 same sign":>20}')
    for k in (0, 1, 2, 3, 4, 5, 10, 21, 42, 63, 126):
        sims, nulls, same = [], [], []
        for Za, Zb in zip(Zs, Zs[1:]):
            a = lag_corr(Za, k)[off]
            b = lag_corr(Zb, k)[off]
            sims.append(corr(a, b))
            nulls.extend(corr(a, lag_corr(Zb, k, np.roll(Zb, s, axis=0))[off])
                         for s in rng.integers(len(Zb) // 4, 3 * len(Zb) // 4, size=2))
            top = np.argpartition(np.abs(a), -TOP)[-TOP:]
            same.append(np.mean(np.sign(a[top]) == np.sign(b[top])))
        z = (np.mean(sims) - np.mean(nulls)) / (np.std(nulls) / math.sqrt(len(sims)))
        print(f'{k:>5} {np.mean(sims):>+16.4f} {z:>+7.1f} {np.mean(same):>20.1%}')
    print(f'\nTotal runtime {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
