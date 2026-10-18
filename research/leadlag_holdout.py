#!/usr/bin/python3
"""Holdout test: can lead-lag correlations learned from the past pick stocks in the next year?

The last year of the restored v1 data (2024-06-28 .. 2025-06-27) is treated as the future.
Lead-lag correlations between stocks are built only from the three years before it, then
used on each holdout day to pick stocks for a $10,000 account. The account is compared with
buying and holding SPY (S&P 500) or QQQ (Nasdaq-100), and with 1,000 accounts that pick the
same number of stocks at random each day.

Prices come from the v1 Postgres container (correlation-db); which symbols are stocks rather
than ETFs comes from Stooq's d_us_txt.zip, because v1's importer dropped that label.
numpy only, so it runs on the system python:

    /usr/bin/python3 research/leadlag_holdout.py | tee research/results.txt
"""
import io
import math
import re
import subprocess
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)   # all-NaN columns for symbols outside the window

ZIP = Path.home() / 'programming/correlation/data/stooq/d_us_txt.zip'
LOAD_FROM = '2021-06-01'
TRAIN = ('2021-06-28', '2024-06-27')
HOLDOUT = ('2024-06-28', '2025-06-27')
LAGS = (1, 2, 3, 4, 5)       # trading days by which the leader moves first
TOP_K = 2000                 # lead-lag pairs kept ("stored") from the training years
PICKS = 20                   # stocks held each holdout day
START_CASH = 10_000.0
COSTS_BPS = (0, 5, 20)       # per side, charged on turnover
RANDOM_RUNS = 1000
MIN_PRICE, MIN_DOLLAR_VOL, MIN_COVERAGE = 5.0, 5e6, 0.95
WINSOR = 5.0                 # clip standardized residual returns at +-5 sd
# Exchange test tickers (ZVZZT etc.) sit in Stooq's stock folders and swing 3,000x in a day.
TEST_SYMBOL = re.compile(r'^(Z[A-Z]ZZT|[A-Z]?TEST[A-Z]?|ZXYZ.*)$')
rng = np.random.default_rng(20240628)


def say(msg=''):
    print(msg, flush=True)


def psql(sql):
    return subprocess.run(['docker', 'exec', 'correlation-db', 'psql', '-U', 'correlation',
                           '-d', 'correlation_development', '-Atc', sql],
                          capture_output=True, text=True, check=True).stdout


def stooq_kinds():
    """Symbol -> 'stock' or 'etf', from Stooq's folder names (e.g. 'nasdaq stocks')."""
    kinds = {}
    with zipfile.ZipFile(ZIP) as z:
        for name in z.namelist():
            m = re.match(r'data/daily/us/([^/]+)/(?:[^/]+/)*([^/]+)\.us\.txt$', name)
            if m:
                folder, sym = m.groups()
                kinds[sym.upper()] = 'etf' if 'etf' in folder else 'stock' if 'stock' in folder else folder
    return kinds


def load_prices():
    syms = psql(f"SELECT symbol FROM (SELECT DISTINCT symbol FROM stock_prices "
                f"WHERE date >= '{LOAD_FROM}') s ORDER BY symbol COLLATE \"C\"").split('\n')[:-1]
    rows = psql(f"COPY (SELECT dense_rank() OVER (ORDER BY symbol COLLATE \"C\") - 1, "
                f"date - DATE '{LOAD_FROM}', close, COALESCE(volume, 0) FROM stock_prices "
                f"WHERE date >= '{LOAD_FROM}' AND close > 0) TO STDOUT WITH CSV")
    a = np.loadtxt(io.StringIO(rows), delimiter=',')
    sym_id, day, close, vol = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3]
    # Trading days = days SPY traded. (Inferring them from how many symbols report drops real
    # days in earlier years, when the survivor-only data has fewer symbols.)
    cal = np.unique(day[sym_id == syms.index('SPY')])
    row = np.full(day.max() + 1, -1)
    row[cal] = np.arange(len(cal))
    keep = row[day] >= 0
    P = np.full((len(cal), len(syms)), np.nan)
    V = np.zeros((len(cal), len(syms)))
    P[row[day[keep]], sym_id[keep]] = close[keep]
    V[row[day[keep]], sym_id[keep]] = vol[keep]
    dates = np.datetime64(LOAD_FROM) + cal
    return syms, dates, P, V


def standardize(X, rows):
    mu, sd = np.nanmean(X[rows], 0), np.nanstd(X[rows], 0)
    Z = np.clip((X - mu) / np.where(sd > 0, sd, 1), -WINSOR, WINSOR)
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def lag_corr(Z, k):
    """C[i, j] = corr(stock i on day t, stock j on day t+k), as one matrix product."""
    return (Z[:-k].T @ Z[k:]) / (len(Z) - k)


def top_pairs(Zlead, Ztarget, k_top):
    """Strongest |r| off-diagonal (leader, target, lag) triples across all lags."""
    found = []
    for k in LAGS:
        C = lag_corr(Zlead, k) if Ztarget is None else (Zlead[:-k].T @ Ztarget[k:]) / (len(Zlead) - k)
        np.fill_diagonal(C, 0)
        flat = np.argpartition(np.abs(C), -k_top, axis=None)[-k_top:]
        i, j = np.unravel_index(flat, C.shape)
        found.append(np.column_stack([np.abs(C[i, j]), np.full(k_top, k), i, j, C[i, j]]))
    found = np.vstack(found)
    return found[np.argsort(-found[:, 0])][:k_top]


def pair_corr(Z, pairs):
    """Correlation of each (lag, leader, target) pair within Z, standardized within Z."""
    Zs = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    out = np.empty(len(pairs))
    for k in LAGS:
        m = pairs[:, 1] == k
        i, j = pairs[m, 2].astype(int), pairs[m, 3].astype(int)
        out[m] = (Zs[:-k, i] * Zs[k:, j]).mean(0)
    return out


def simulate(weights_by_day, simple_ret, cost_bps):
    """Account value path for a list of per-day weight vectors (cash = 1 - sum)."""
    value, prev = START_CASH, np.zeros(simple_ret.shape[1])
    for w, r in zip(weights_by_day, simple_ret):
        turnover = np.abs(w - prev).sum()
        value *= 1 + (w * r).sum() - turnover * cost_bps / 1e4
        prev = w
    return value


def main():
    t0 = time.time()
    kinds = stooq_kinds()
    syms, dates, P, V = load_prices()
    say(f'Loaded {len(syms):,} symbols x {len(dates)} trading days in {time.time() - t0:.0f}s')
    kind = np.array([kinds.get(s, 'unknown') for s in syms])
    kind[[bool(TEST_SYMBOL.match(s)) for s in syms]] = 'test'
    say(f"Stooq labels matched: {(kind != 'unknown').sum():,} of {len(syms):,} "
        f"({(kind == 'stock').sum():,} stocks, {(kind == 'etf').sum():,} ETFs, "
        f"{(kind == 'test').sum()} exchange test tickers dropped)")

    # Stooq prices are already split- and dividend-adjusted. What remains are one-day bad prints:
    # a >2x jump that undoes itself the next day. Blank those closes.
    with np.errstate(all='ignore'):
        jump, back = np.log(P[1:-1] / P[:-2]), np.log(P[2:] / P[1:-1])
    bad = (np.abs(jump) > math.log(2)) & (np.abs(jump + back) < 0.1)
    P[1:-1][bad] = np.nan
    say(f'Bad prints repaired: {bad.sum()}')

    with np.errstate(all='ignore'):
        R = np.vstack([np.full(P.shape[1], np.nan), np.log(P[1:] / P[:-1])])
    tr = (dates >= np.datetime64(TRAIN[0])) & (dates <= np.datetime64(TRAIN[1]))
    ho = (dates >= np.datetime64(HOLDOUT[0])) & (dates <= np.datetime64(HOLDOUT[1]))
    last_year = np.flatnonzero(tr)[-252:]

    # Universe chosen from training data only.
    coverage = np.mean(~np.isnan(R[tr]), 0)
    with np.errstate(all='ignore'):
        med_price = np.nanmedian(P[last_year], 0)
        dollar_vol = np.nanmedian(P[last_year] * V[last_year], 0)
    U = np.flatnonzero((kind == 'stock') & (coverage >= MIN_COVERAGE) &
                       (med_price >= MIN_PRICE) & (dollar_vol >= MIN_DOLLAR_VOL))
    N = len(U)
    say(f'Universe (from training years only): {N:,} liquid US stocks '
        f'(price >= ${MIN_PRICE:.0f}, median dollar volume >= ${MIN_DOLLAR_VOL / 1e6:.0f}M, '
        f'>= {MIN_COVERAGE:.0%} of days traded)')
    say(f'Training: {TRAIN[0]} .. {TRAIN[1]} ({tr.sum()} days)   '
        f'Holdout: {HOLDOUT[0]} .. {HOLDOUT[1]} ({ho.sum()} days)')

    # Market-residual returns: remove each stock's beta to the equal-weight market (betas from training).
    Ru = R[:, U]
    mkt = np.nanmean(Ru, 1)
    m_tr = mkt[tr] - np.nanmean(mkt[tr])
    with np.errstate(all='ignore'):
        beta = np.nansum((Ru[tr] - np.nanmean(Ru[tr], 0)) * m_tr[:, None], 0) / np.nansum(m_tr ** 2)
    E = Ru - beta * mkt[:, None]
    Z = standardize(E, tr)

    say('\n== 1. Lead-lag correlations from the training years ==')
    Zt = Z[tr]
    t1 = time.time()
    stored = top_pairs(Zt, None, TOP_K)
    gemm_s = time.time() - t1
    tests = N * (N - 1) * len(LAGS)
    T = len(Zt)
    say(f'{tests:,} leader/target/lag combinations tested ({len(LAGS)} lags) in {gemm_s:.1f}s')
    say(f'Stored the top {TOP_K:,}: |r| from {stored[-1, 0]:.3f} to {stored[0, 0]:.3f}')

    # Chance level: same search after shifting the targets' timeline, which destroys any real lead-lag.
    null_max, null_kth, null_rev = [], [], []
    for s in rng.integers(126, T - 126, size=3):
        Zs = np.roll(Zt, s, axis=0)
        n = top_pairs(Zt, Zs, TOP_K)
        null_max.append(n[0, 0])
        null_kth.append(n[-1, 0])
        for r_abs, k, i, j, _ in n[:200]:
            k, i, j = int(k), int(i), int(j)
            null_rev.append(abs((Zs[:-k, j] * Zt[k:, i]).sum() / (T - k)) / r_abs)
    say(f'Chance level (timeline-shifted search, 3 runs): best |r| {np.mean(null_max):.3f}, '
        f'{TOP_K:,}th best {np.mean(null_kth):.3f}')
    say(f'Theory: 1 sd of r = {1 / math.sqrt(T):.3f}; expected best of {tests:,} by chance ~ '
        f'{math.sqrt(2 * math.log(tests)) / math.sqrt(T):.3f}')
    beat = (stored[:, 0] > np.mean(null_max)).sum()
    say(f'Stored pairs stronger than the best chance result: {beat:,} of {TOP_K:,}')

    # Direction: true lead-lag should be much stronger forward than backward.
    rev = np.empty(len(stored))
    for k in LAGS:
        m = stored[:, 1] == k
        i, j = stored[m, 2].astype(int), stored[m, 3].astype(int)
        rev[m] = np.abs((Zt[:-k, j] * Zt[k:, i]).sum(0) / (T - k))
    say(f'Direction check: median |reverse r| / |forward r| = {np.median(rev / stored[:, 0]):.2f}, '
        f'chance pairs {np.median(null_rev):.2f} (picking the strongest pairs makes the reverse look weak '
        f'either way, so this only means something if the two differ)')

    say('\n== 2. Do the stored correlations hold up in the holdout year? ==')
    Zh = Z[ho]
    r_ho = pair_corr(Zh, stored)
    persist = np.sign(stored[:, 4]) * r_ho
    rand_pairs = np.column_stack([np.zeros(20000), rng.choice(LAGS, 20000),
                                  rng.integers(0, N, 20000), rng.integers(0, N, 20000), np.zeros(20000)])
    rand_pairs = rand_pairs[rand_pairs[:, 2] != rand_pairs[:, 3]]
    rand_pairs[:, 4] = pair_corr(Zt, rand_pairs)
    rand_persist = np.sign(rand_pairs[:, 4]) * pair_corr(Zh, rand_pairs)
    say(f'{"":28}{"same sign":>10}{"mean signed r":>15}')
    say(f'{"stored pairs":28}{np.mean(persist > 0):>10.1%}{np.mean(persist):>15.4f}')
    say(f'{"random pairs (baseline)":28}{np.mean(rand_persist > 0):>10.1%}{np.mean(rand_persist):>15.4f}')
    say(f'(1 sd of a single holdout r = {1 / math.sqrt(ho.sum()):.3f})')
    tercile = np.digitize(dollar_vol[U], np.quantile(dollar_vol[U], [1 / 3, 2 / 3]))
    for t_i, name in enumerate(['least liquid third', 'middle third', 'most liquid third']):
        m = tercile[stored[:, 3].astype(int)] == t_i
        if m.any():
            say(f'  target in {name:20} {m.sum():5} pairs  same sign {np.mean(persist[m] > 0):.1%}  '
                f'mean signed r {np.mean(persist[m]):+.4f}')

    say('\n== 3. Positive control: do large caps lead small caps by a day? ==')
    broad = np.flatnonzero((kind == 'stock') & (coverage >= 0.9) & (med_price >= 1))
    q = np.quantile(dollar_vol[broad], [1 / 3, 2 / 3])
    big = np.nanmean(R[:, broad[dollar_vol[broad] >= q[1]]], 1)
    small = np.nanmean(R[:, broad[dollar_vol[broad] < q[0]]], 1)
    for name, rows in (('training', tr), ('holdout', ho)):
        b, s_ = big[rows], small[rows]
        say(f'  {name:9} big(t)->small(t+1) r = {np.corrcoef(b[:-1], s_[1:])[0, 1]:+.3f}   '
            f'small(t)->big(t+1) r = {np.corrcoef(s_[:-1], b[1:])[0, 1]:+.3f}')

    say('\n== 4. $10,000 for the holdout year ==')
    W = np.zeros((len(LAGS), N, N), np.float32)
    for r_abs, k, i, j, r in stored:
        W[LAGS.index(int(k)), int(i), int(j)] = r
    targets = np.unique(stored[:, 3].astype(int))
    hidx = np.flatnonzero(ho)
    # Score for day t uses only leader moves on days t-1..t-5, all known at the previous close.
    score = np.zeros((len(hidx), N), np.float32)
    for ki, k in enumerate(LAGS):
        score += Z[hidx - k] @ W[ki]
    simple = np.nan_to_num(np.expm1(Ru[hidx]), nan=0.0)
    resid_next = E[hidx]

    picks_by_day, hits, ics = [], [], []
    for d in range(len(hidx)):
        s = score[d, targets]
        order = targets[np.argsort(-s)]
        chosen = [j for j in order[:PICKS] if score[d, j] > 0]
        w = np.zeros(N)
        w[chosen] = 1.0 / PICKS
        picks_by_day.append(w)
        e = resid_next[d, chosen]
        hits.extend(e[~np.isnan(e)] > 0)
        valid = ~np.isnan(resid_next[d, targets])
        if valid.sum() > 10:
            a = np.argsort(np.argsort(s[valid]))
            b_ = np.argsort(np.argsort(resid_next[d, targets][valid]))
            ics.append(np.corrcoef(a, b_)[0, 1])
    ics = np.array(ics)
    say(f'Picks: top {PICKS} predicted gainers among {len(targets):,} stocks the stored pairs can predict, '
        f'rebalanced every day at the close')
    say(f'Picked stocks beat the market next day: {np.mean(hits):.1%} of {len(hits):,} picks '
        f'(coin flip = 50%)')
    say(f'Daily rank correlation of prediction vs outcome (IC): mean {ics.mean():+.4f}, '
        f't = {ics.mean() / ics.std() * math.sqrt(len(ics)):+.2f}')

    # Random guessers: same number of stocks, same candidates, chosen at random each day.
    rand_final = {c: np.empty(RANDOM_RUNS) for c in COSTS_BPS}
    rv = {c: np.full(RANDOM_RUNS, START_CASH) for c in COSTS_BPS}
    prev_mask = np.zeros((RANDOM_RUNS, len(targets)), bool)
    for d in range(len(hidx)):
        pick = np.argpartition(rng.random((RANDOM_RUNS, len(targets))), PICKS, axis=1)[:, :PICKS]
        mask = np.zeros_like(prev_mask)
        np.put_along_axis(mask, pick, True, axis=1)
        ret = simple[d, targets][pick].mean(1)
        turnover = 2 * (PICKS - (mask & prev_mask).sum(1)) / PICKS if d else np.ones(RANDOM_RUNS)
        for c in COSTS_BPS:
            rv[c] *= 1 + ret - turnover * c / 1e4
        prev_mask = mask
    for c in COSTS_BPS:
        rand_final[c] = rv[c]

    bench = {}
    for name in ('SPY', 'QQQ'):
        col = syms.index(name)
        bench[name] = START_CASH * P[hidx[-1], col] / P[hidx[0] - 1, col]
    avg_stock = START_CASH * np.mean(np.nan_to_num(np.exp(np.nansum(Ru[hidx][:, targets], 0)), nan=1.0))

    say(f'\n{"":34}{"final value":>12}{"return":>9}')
    say(f'{"Buy and hold SPY (S&P 500)":34}{bench["SPY"]:>12,.0f}{bench["SPY"] / START_CASH - 1:>9.1%}')
    say(f'{"Buy and hold QQQ (Nasdaq-100)":34}{bench["QQQ"]:>12,.0f}{bench["QQQ"] / START_CASH - 1:>9.1%}')
    say(f'{"Buy and hold every candidate":34}{avg_stock:>12,.0f}{avg_stock / START_CASH - 1:>9.1%}')
    for c in COSTS_BPS:
        strat = simulate(picks_by_day, simple, c)
        rf = rand_final[c]
        p = np.mean(rf >= strat)
        label = 'no trading costs' if c == 0 else f'{c} bps per trade'
        say(f'{"Correlation picks, " + label:34}{strat:>12,.0f}{strat / START_CASH - 1:>9.1%}'
            f'   random guessers: median {np.median(rf):,.0f}, '
            f'{np.mean(rf >= strat):.1%} did at least as well (p = {p:.3f})')

    say('\n== 5. Speed ==')
    say(f'Training search above: {N:,} stocks, {len(LAGS)} lags, {T} days in {gemm_s:.1f}s on the CPU (numpy)')
    big_n = 11431
    Zb = rng.standard_normal((T, big_n), dtype=np.float32)
    t1 = time.time()
    Cb = Zb[:-1].T @ Zb[1:]
    one = time.time() - t1
    del Cb
    say(f'Full v1 universe, one lag: {big_n:,} x {big_n:,} pairs over {T} days in {one:.2f}s '
        f'({2 * big_n * big_n * T / one / 1e12:.2f} TFLOP/s)')
    v1_days = big_n * (big_n - 1) * len(LAGS) * 0.05 / 24 / 86400
    say(f'v1 at its own 50 ms per correlation on 24 cores, same {len(LAGS)} lags: ~{v1_days:.0f} days')
    say(f'\nTotal runtime {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
