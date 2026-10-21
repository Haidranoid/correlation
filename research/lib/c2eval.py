"""Shared helpers for the experiments in experiments/: read the price store and the judge's exports, write scores
files, grade them with stockai's judge (correlation2's $10,000 harness, bit-identical), and compute "change over
random" the same way for every experiment.

Read-only on the real data. Run with any Python that has numpy (python3 -I keeps the script dir off sys.path).

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib'); import c2eval
"""
import json
import os
import subprocess
from pathlib import Path

import numpy as np

HOME = Path.home()
JUDGE = Path('/home/bhoult/programming/correlation/stockai/judge/target/release/stockai-judge')
# Export of the store the census was built on: prices through 2025-06-27 (8,503 stocks x 5,911 days).
EXPORT_OLD = HOME / '.local/share/correlation2/export/full'
# Export of the fresh store (sandbox copy of stores/stooq-2026-10-19): through 2026-10-19 (9,508 x 6,240).
EXPORT_NEW = HOME / '.local/share/c2-sandbox-2026-10-19/export/full'
STORE_NEW = HOME / '.local/share/correlation2/stores/stooq-2026-10-19'
STORE_OLD = HOME / '.local/share/correlation2/stores/stooq-2025-06-27'
# The judge reads the store as $C2_DATA_DIR/store; these sandboxes hold copies of the matching store.
SANDBOX = {str(EXPORT_OLD): HOME / '.local/share/c2-sandbox-2025-06-27',
           str(EXPORT_NEW): HOME / '.local/share/c2-sandbox-2026-10-19'}
# Large outputs (scores files, caches) go here, not in the repo.
WORK = HOME / '.local/share/correlation2/experiments'


def load_export(path=EXPORT_OLD, arrays=('close', 'dollar_volume', 'v', 'market')):
    """The judge's export: days [T] (days since 1970), tickers [N], ids [N] (store ids), and the requested arrays,
    each [N, T] float32 memory-mapped (market is [T] float64). Row j = stock j, column d = day d."""
    path = Path(path)
    out = {'path': path, 'meta': json.loads((path / 'meta.json').read_text())}
    out['days'] = np.load(path / 'days.npy')
    rows = [l.rstrip('\n').split('\t') for l in open(path / 'symbols.tsv')][1:]
    out['ids'] = np.array([int(r[1]) for r in rows])
    out['tickers'] = np.array([r[2] for r in rows])
    for a in arrays:
        out[a] = np.load(path / f'{a}.npy', mmap_mode='r')
    return out


def day_index(days, iso):
    """Index of the first trading day on or after an ISO date."""
    d = (np.datetime64(iso) - np.datetime64('1970-01-01')).astype(int)
    return int(np.searchsorted(days, d))


def iso(day):
    return str(np.datetime64('1970-01-01') + np.timedelta64(int(day), 'D'))


def load_store(path=STORE_NEW):
    """The raw price store (all symbols, stocks and ETFs, with open/high/low/close/volume).
    Returns dict: n_sym, offs [n_sym+1] row offsets, date/open/high/low/close/volume (flat, memory-mapped),
    and syms: list of dicts (id, ticker, exchange, kind, first, last, rows). Rows of a symbol are offs[i]..offs[i+1],
    sorted by date (days since 1970). Prices are Stooq's split- and dividend-adjusted values, unrepaired."""
    path = Path(path)
    raw = np.memmap(path / 'prices.bin', dtype=np.uint8, mode='r')
    assert bytes(raw[:8]) == b'C2PRICE1'
    n_sym = int(np.frombuffer(raw[12:16], dtype='<u4')[0])
    rows = int(np.frombuffer(raw[16:24], dtype='<u8')[0])
    offs = np.frombuffer(raw[24:24 + 8 * (n_sym + 1)], dtype='<u8').astype(np.int64)
    pos = 24 + 8 * (n_sym + 1)
    cols = {}
    for name, dt in (('date', '<i4'), ('open', '<f4'), ('high', '<f4'), ('low', '<f4'), ('close', '<f4'), ('volume', '<f4')):
        pos += (64 - pos % 64) % 64
        cols[name] = np.frombuffer(raw[pos:pos + 4 * rows], dtype=dt)
        pos += 4 * rows
    syms = []
    for l in list(open(path / 'symbols.tsv'))[1:]:
        f = l.rstrip('\n').split('\t')
        syms.append({'id': int(f[0]), 'ticker': f[1], 'exchange': f[2], 'kind': f[3], 'rows': int(f[6])})
    return {'n_sym': n_sym, 'offs': offs, 'syms': syms, **cols}


def store_panel(store, ids, days, field='close'):
    """[len(ids), len(days)] float32 panel of one store field aligned to the given trading days (NaN where missing)."""
    out = np.full((len(ids), len(days)), np.nan, np.float32)
    for k, i in enumerate(ids):
        a, b = store['offs'][i], store['offs'][i + 1]
        d = store['date'][a:b]
        pos = np.searchsorted(days, d)
        ok = (pos < len(days)) & (days[np.minimum(pos, len(days) - 1)] == d)
        out[k, pos[ok]] = store[field][a:b][ok]
    return out


def save_scores(scores, name, export=EXPORT_OLD):
    """Write a [N, T] scores array (float32, NaN = no score) aligned to an export; returns its path.
    scores[j, d] must use only data through the close of day d."""
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / f'{name}.npy'
    np.save(p, np.ascontiguousarray(scores, dtype=np.float32))
    return p


def judge(scores_path, export=EXPORT_OLD, rebalance=1, walk_forward=True, start='2024-06-28', end=None, delay=0,
          costs='0,5,20', random=1000, random_match='uniform', picks=20, json_out=None, extra=()):
    """Grade a scores file. Defaults: walk-forward over every year 2006-07 .. 2024-25 (19 years, the same years as
    correlation2's own walk-forwards) on the old export. For the unseen months use export=EXPORT_NEW,
    walk_forward=False, start='2025-06-30' (once, after the rules are final). Returns (printed text, parsed JSON)."""
    json_out = Path(json_out or str(scores_path).replace('.npy', f'__r{rebalance}_d{delay}_{random_match}.json'))
    cmd = [str(JUDGE), 'holdout', '--family', 'scores', '--scores', str(scores_path), '--export', str(export),
           '--rebalance', str(rebalance), '--delay', str(delay), '--costs', costs, '--random', str(random),
           '--random-match', random_match, '--picks', str(picks), '--start', start, '--json', str(json_out), '--locked']
    if walk_forward:
        cmd.append('--walk-forward')
    if end:
        cmd += ['--end', end]
    cmd += list(extra)
    env = dict(os.environ, C2_DATA_DIR=str(SANDBOX[str(Path(export))]))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f'judge failed: {r.stderr or r.stdout}')
    return r.stdout, json.loads(json_out.read_text())


def change_over_random(js, cost_index=0):
    """The number in an experiment's file name: the picks' yearly return minus the median random account's, in
    percentage points, at the first cost level (0 bps by default), averaged over the graded years. Also returns the
    per-year differences, how many years were ahead, and the standard error. For a single holdout (not walk-forward)
    it is that one period's difference."""
    if js.get('years'):
        entries = [y['costs'][cost_index] for y in js['years']]
    else:  # a single holdout keeps its cost levels under report.strategy
        entries = [js['report']['strategy'][cost_index]]
    diffs = [c['final_pct'] - c['random_median_pct'] for c in entries]
    d = np.array(diffs)
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float('nan')
    return {'change': float(d.mean()), 'per_year': [round(x, 2) for x in diffs], 'ahead': int((d > 0).sum()),
            'years': len(d), 'se': se}


def summary_line(js):
    """One line with the judge's headline measures (walk-forward JSON)."""
    s = js['summary']
    beat = ', '.join(f"{int(b['cost_bps'])} bps: {b['years']}" for b in s['beat_random_p_lt_0_1'])
    cor = change_over_random(js)
    return (f"years {s['years']}; beat random at p<0.1 ({beat}) of {s['years']}; change over random "
            f"{cor['change']:+.1f} pts/yr (se {cor['se']:.1f}, ahead {cor['ahead']}/{cor['years']}); "
            f"IC actual returns z {s['ic_ret_z']:+.2f}; long-short {s['ls_bps_per_day_mean']:.2f} bps/day z {s['ls_z']:+.2f}, "
            f"turnover {s['ls_turnover_mean']:.0%}, costs {s['ls_cost_bps_per_day_per_bps']:.2f} x c bps/day")


# ---------------------------------------------------------------------------------------------------------------
# Twins: each stock's best-tracking fund (the hedge finder's idea), shared by the experiments that use it.

STORE_FOR = {str(EXPORT_OLD): STORE_OLD, str(EXPORT_NEW): STORE_NEW}


def funds(E):
    """ETFs (store kind 'etf') of the store matching an export, aligned to its days: dict with ids, tickers,
    close [F, T] and dollar volume [F, T] (float32, NaN where missing). Store prices are unrepaired, so daily fund
    returns beyond ±50% (log) are treated as missing wherever this module computes them."""
    store = load_store(STORE_FOR[str(Path(E['path']))])
    by_id = {s['id']: s for s in store['syms']}
    ids = [s['id'] for s in store['syms'] if s['kind'] == 'etf']
    close = store_panel(store, ids, E['days'], 'close')
    vol = store_panel(store, ids, E['days'], 'volume')
    return {'ids': np.array(ids), 'tickers': np.array([by_id[i]['ticker'] for i in ids]), 'close': close, 'dollar_volume': close * vol}


def _log_returns(close, max_abs=None):
    c = np.asarray(close, np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(all='ignore'):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    r[~np.isfinite(r)] = np.nan
    if max_abs is not None:
        r[np.abs(r) > max_abs] = np.nan
    return r


def twins(E, step=21, window=252, min_frac=0.9, min_dv=5e6, max_vol=0.75, max_corr=0.95, cache=True):
    """Each stock's twin: the fund that tracked it best over the trailing `window` days, recomputed every `step`
    trading days on a fixed calendar (export day indices window, window+step, ...). Point in time: the twin chosen
    at rebalance index t0 uses daily log returns of days t0-window+1 .. t0 (closes through t0) and is valid for
    decisions on days t0 .. t0+step-1.

    Eligible funds at t0: store kind 'etf', a return on >= 90% of the window's days, median dollar volume >= $5M,
    annualized volatility <= 75% (the diversifier tool's cap, which drops most leveraged funds), correlation with
    the stock > 0 and < 0.95, and not a single-name fund (one whose best-matching real stock beats its second-best by
    more than 0.12, with the best >= 0.6: option-income and 1x funds on one stock). Choice, as the groups map labels
    its groups: among funds within 0.02 of the best correlation, drop those with variance above 2.25x the calmest,
    then take the most traded. Stocks need returns on >= 90% of
    the window's days. Correlation: cosine of the two return series standardized over their own finite days, with
    missing days counted as 0. beta = r * sd_stock / sd_fund. resid_sd = sd of the daily residual stock - beta * fund
    over the window.

    Also flags fund-like "stocks" (fund_like [K, N], point in time): Stooq files some funds, leveraged single-stock
    funds and duplicate share classes under stocks; a stock moving >= 0.97 with a more-traded stock or fund over the
    window is flagged. Experiments should drop fund-like stocks from their universe on each rebalance.

    Returns dict: rebal [K] (export day indices), fund (ETF store ids), fund_ticker, plus fund_row [K, N] (row in
    funds(E), -1 = no twin), corr, beta, resid_sd, fund_like [K, N]; and 'F' = funds(E). Cached under WORK."""
    WORK.mkdir(parents=True, exist_ok=True)
    key = WORK / f"twins3_{E['meta']['fingerprint']}_{step}_{window}_{min_frac}_{int(min_dv)}_{max_vol}_{max_corr}.npz"
    F = funds(E)
    if cache and key.exists():
        z = np.load(key)
        return {k: z[k] for k in z.files} | {'F': F}
    sr = _log_returns(E['close'])
    fr = _log_returns(F['close'], max_abs=0.5)
    fdv = np.asarray(F['dollar_volume'], np.float64)
    sdv = np.asarray(E['dollar_volume'], np.float64)
    N, T = sr.shape
    rebal = np.arange(window, T, step)
    K = len(rebal)
    fund_row = np.full((K, N), -1, np.int32)
    corr = np.full((K, N), np.nan, np.float32)
    beta = np.full((K, N), np.nan, np.float32)
    resid_sd = np.full((K, N), np.nan, np.float32)
    fund_like = np.zeros((K, N), bool)

    def standardize(X):
        m = np.nanmean(X, 1, keepdims=True)
        s = np.nanstd(X, 1, keepdims=True)
        Z = (X - m) / s
        Z[~np.isfinite(Z)] = 0.0
        Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        return Z, s[:, 0]

    for k, t0 in enumerate(rebal):
        a, b = t0 - window + 1, t0 + 1
        S, Fw = sr[:, a:b], fr[:, a:b]
        s_ok = np.isfinite(S).mean(1) >= min_frac
        with np.errstate(all='ignore'):
            f_ok = ((np.isfinite(Fw).mean(1) >= min_frac) & (np.nanmedian(fdv[:, a:b], 1) >= min_dv)
                    & (np.nanstd(Fw, 1) * np.sqrt(252) <= max_vol))
        si, fi = np.nonzero(s_ok)[0], np.nonzero(f_ok)[0]
        if len(si) == 0 or len(fi) == 0:
            continue
        Zs, sds = standardize(S[si])
        Zf, sdf = standardize(Fw[fi])
        C = Zs @ Zf.T
        with np.errstate(all='ignore'):
            dv_s = np.nanmedian(sdv[si, a:b], 1)
            dv_f = np.nanmedian(fdv[fi, a:b], 1)
        dv_s = np.nan_to_num(dv_s); dv_f = np.nan_to_num(dv_f)
        # Fund-like "stocks": Stooq files some funds, leveraged single-stock funds and duplicate share classes under
        # stocks. A stock moving >= 0.97 with a more-traded stock or fund is a wrapper or duplicate of it.
        Css = (Zs.astype(np.float32) @ Zs.T.astype(np.float32))
        np.fill_diagonal(Css, 0)
        like = ((Css >= 0.97) & (dv_s[None, :] > dv_s[:, None])).any(1)
        like |= ((C >= 0.97) & (dv_f[None, :] > dv_s[:, None])).any(1)
        del Css
        fund_like[k, si[like]] = True
        # Single-name funds (option-income or 1x funds on one stock) are about one stock, not a group: a fund whose
        # best-matching real (not fund-like) stock beats its second-best by more than 0.12, with the best >= 0.6.
        Creal = C[~like]
        if Creal.shape[0] >= 2:
            top2 = np.sort(Creal, 0)[-2:]
            single = (top2[1] >= 0.6) & (top2[1] - top2[0] > 0.12)
            C[:, single] = -np.inf
        C[(C <= 0) | (C >= max_corr)] = -np.inf
        # As the groups map labels groups: among funds within 0.02 of the best correlation, drop those with variance
        # above 2.25x the calmest one, then take the most traded (so a calm 1x sector fund beats its 2x version).
        best0 = C.max(1)
        M = np.isfinite(C) & (C >= best0[:, None] - 0.02)
        var_f = sdf ** 2
        minvar = np.where(M, var_f[None, :], np.inf).min(1)
        M &= var_f[None, :] <= 2.25 * minvar[:, None]
        j = np.argmax(np.where(M, dv_f[None, :], -np.inf), 1)
        best = np.where(M.any(1), C[np.arange(len(si)), j], -np.inf)
        has = np.isfinite(best)
        si, j, best = si[has], j[has], best[has]
        fund_row[k, si] = fi[j]
        corr[k, si] = best
        bet = best * sds[has] / sdf[j]
        beta[k, si] = bet
        res = S[si] - bet[:, None] * Fw[fi[j]]
        resid_sd[k, si] = np.nanstd(res, 1)
    out = {'rebal': rebal, 'fund_row': fund_row, 'corr': corr, 'beta': beta, 'resid_sd': resid_sd, 'fund_like': fund_like,
           'fund': F['ids'], 'fund_ticker': F['tickers']}
    if cache:
        np.savez(key, **out)
    return out | {'F': F}


def twin_residual(E, tw, h, scaled=True, max_missing=1):
    """[N, T] float32: the stock's log return over the h days ending at day d minus beta x its twin fund's, using the
    twin chosen at the latest rebalance at or before d (point in time). NaN where the stock has no twin or more than
    `max_missing` missing daily returns (stock or fund) in the h days. scaled=True divides by resid_sd * sqrt(h), so
    the value is in units of a typical h-day residual move."""
    sr = _log_returns(E['close'])
    fr = _log_returns(tw['F']['close'], max_abs=0.5)
    N, T = sr.shape
    out = np.full((N, T), np.nan, np.float32)
    cs = np.concatenate([np.zeros((N, 1)), np.cumsum(np.nan_to_num(sr), 1)], 1)
    cm = np.concatenate([np.zeros((N, 1)), np.cumsum(~np.isfinite(sr), 1)], 1)
    cf = np.concatenate([np.zeros((fr.shape[0], 1)), np.cumsum(np.nan_to_num(fr), 1)], 1)
    cfm = np.concatenate([np.zeros((fr.shape[0], 1)), np.cumsum(~np.isfinite(fr), 1)], 1)
    rebal = tw['rebal']
    for k, t0 in enumerate(rebal):
        t1 = rebal[k + 1] if k + 1 < len(rebal) else T
        si = np.nonzero(tw['fund_row'][k] >= 0)[0]
        if len(si) == 0:
            continue
        fj = tw['fund_row'][k, si]
        d = np.arange(max(t0, h), t1)
        if len(d) == 0:
            continue
        rs = cs[si][:, d + 1] - cs[si][:, d + 1 - h]
        ms = cm[si][:, d + 1] - cm[si][:, d + 1 - h]
        rf = cf[fj][:, d + 1] - cf[fj][:, d + 1 - h]
        mf = cfm[fj][:, d + 1] - cfm[fj][:, d + 1 - h]
        res = rs - tw['beta'][k, si][:, None] * rf
        if scaled:
            res = res / (tw['resid_sd'][k, si][:, None] * np.sqrt(h))
        res[(ms > max_missing) | (mf > max_missing)] = np.nan
        out[np.ix_(si, d)] = res
    return out


# ---------------------------------------------------------------------------------------------------------------
# One shared exclusion rule for "stock" universes: the fund-like flag plus funds, ETFs, preferreds, warrants and
# units that Stooq files as stocks and the flag misses (built from the betting-against-correlation experiment's list,
# 2026-10-21). Every experiment applies excluded() so they all share one universe definition.

SEC_MAP = WORK / 'imposters/map/sec_tickers.json'  # SEC's company_tickers.json via a public mirror (current names)
HAND_FUNDS = {'ASA', 'TYG', 'SARK', 'JEPQ', 'QTUM', 'ARTY', 'QQQE', 'WGMI', 'FSCO', 'SYSB', 'COLO', 'GARP',
              'PHYS', 'PSLV', 'CEF', 'SPYM', 'SPYU', 'ETHD', 'MSTU', 'MSTZ', 'ETHU'}


def not_common(tickers):
    """[N] bool: tickers that aren't common stock although filed as stocks: '_' in the ticker (preferreds),
    -WS/-U/-R/-RT endings (warrants, units, rights), the hand list, or an SEC company name containing FUND,
    SPROTT PHYSICAL or BLACKROCK ... TRUST."""
    import re
    try:
        names = {r['ticker'].upper(): r['name'].upper() for r in json.loads(Path(SEC_MAP).read_text())}
    except (OSError, ValueError, KeyError):
        names = {}
    out = np.zeros(len(tickers), bool)
    for j, t in enumerate(tickers):
        t = str(t).upper()
        key = t.replace('.', '-')
        name = names.get(key, '')
        if '_' in t or re.search(r'-(WS|U|R|RT)$', t) or key in HAND_FUNDS:
            out[j] = True
        elif re.search(r'\bFUND\b', name) or 'SPROTT PHYSICAL' in name or (name.startswith('BLACKROCK') and 'TRUST' in name):
            out[j] = True
    return out


def rebal_index(tw, t):
    """Index into tw['rebal'] of the latest twin rebalance at or before export day t (-1 if none)."""
    return int(np.searchsorted(tw['rebal'], t, side='right')) - 1


def excluded(E, tw):
    """[K, N] bool on tw['rebal'] days: the fund-like flag OR not_common(). Use excluded(E, tw)[rebal_index(tw, t)]
    for day t. Drop these from every stock universe."""
    return tw['fund_like'] | not_common(E['tickers'])[None, :]
