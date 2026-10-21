"""Fold the downloaded Yahoo bars into dense arrays (one CSV batch file at a time).

    python -I build.py 5m|1m      # -> WORK/panel_<mode>.npz
    python -I build.py 1d         # -> WORK/daily.npz

5m / 1m: syms [S] (order of out/symbols.txt), dates [D] (every date seen, 'YYYY-MM-DD'), close [S, D, B] float64
(NaN = bar absent), volume [S, D, B], with B = 78 or 390 slots from 09:30. Only bars with 09:30 <= time < 16:00
and UTC offset -04:00 are kept; the counts of the others are stored (n_outside, n_other_offset). Duplicate
(symbol, timestamp) rows keep the first. 1d: Yahoo's daily close (split-adjusted, not dividend-adjusted) [S, D1]
and stock_splits [S, D1] on dates1 [D1].
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/intraday_lead_lag'
BARS = WORK / 'bars'


def read(f):
    return pd.read_csv(f, dtype={'sym': str, 'ts': str, 'tz': str}, keep_default_na=False,
                       na_values=['', 'nan', 'NaN'])


def main():
    mode = sys.argv[1]
    syms = [l.strip() for l in (HERE / 'out' / 'symbols.txt').read_text().splitlines() if l.strip()]
    sidx = {s: i for i, s in enumerate(syms)}
    files = sorted((BARS / mode).glob(f'{mode}_*.csv'))
    if mode == '1d':
        parts = []
        for f in files:
            d = read(f)
            d['date'] = d['ts'].str.slice(0, 10)
            parts.append(d[['sym', 'date', 'close', 'stock_splits']])
        d = pd.concat(parts).drop_duplicates(['sym', 'date'])
        dates = np.array(sorted(d['date'].unique()))
        didx = {x: i for i, x in enumerate(dates)}
        close = np.full((len(syms), len(dates)), np.nan)
        split = np.zeros((len(syms), len(dates)))
        si = d['sym'].map(sidx).to_numpy()
        di = d['date'].map(didx).to_numpy()
        close[si, di] = d['close'].to_numpy(float)
        split[si, di] = d['stock_splits'].fillna(0).to_numpy(float)
        np.savez(WORK / 'daily.npz', syms=np.array(syms), dates=dates, close=close, split=split)
        print(f'{len(files)} files; {d["sym"].nunique()} symbols with daily bars; {len(dates)} dates '
              f'{dates[0]}..{dates[-1]}; splits {[(syms[a], dates[b], split[a, b]) for a, b in zip(*np.nonzero(split))]}')
        return
    step = 5 if mode == '5m' else 1
    B = 390 // step
    seen, chunks = set(), []
    n_rows = n_out = n_off = n_nan = n_dup = 0
    for f in files:
        d = read(f)
        n_rows += len(d)
        d['date'] = d['ts'].str.slice(0, 10)
        hh = d['ts'].str.slice(11, 13).astype(int)
        mm = d['ts'].str.slice(14, 16).astype(int)
        off = d['ts'].str.slice(19)
        mins = hh * 60 + mm - (9 * 60 + 30)
        inside = (mins >= 0) & (mins < 390)
        n_out += int((~inside).sum())
        good = inside & (off == '-04:00')
        n_off += int((inside & (off != '-04:00')).sum())
        d = d[good].assign(slot=(mins[good] // step).astype(int))
        assert ((mins[good] % step) == 0).all()
        nan = ~np.isfinite(d['close'].to_numpy(float))
        n_nan += int(nan.sum())
        d = d[~nan]
        before = len(d)
        d = d.drop_duplicates(['sym', 'ts'])
        n_dup += before - len(d)
        chunks.append(d[['sym', 'date', 'slot', 'close', 'volume']])
        seen |= set(d['date'].unique())
    d = pd.concat(chunks)
    before = len(d)
    d = d.drop_duplicates(['sym', 'date', 'slot'])
    n_dup += before - len(d)
    dates = np.array(sorted(seen))
    didx = {x: i for i, x in enumerate(dates)}
    close = np.full((len(syms), len(dates), B), np.nan)
    vol = np.full((len(syms), len(dates), B), np.nan)
    si = d['sym'].map(sidx).to_numpy()
    di = d['date'].map(didx).to_numpy()
    sl = d['slot'].to_numpy()
    close[si, di, sl] = d['close'].to_numpy(float)
    vol[si, di, sl] = d['volume'].to_numpy(float)
    np.savez(WORK / f'panel_{mode}.npz', syms=np.array(syms), dates=dates, close=close, volume=vol,
             n_rows=n_rows, n_outside=n_out, n_other_offset=n_off, n_nan_close=n_nan, n_dup=n_dup)
    print(f'{mode}: {len(files)} files, {n_rows} rows; outside session {n_out}, other UTC offset {n_off}, '
          f'NaN close {n_nan}, duplicates {n_dup}; {d["sym"].nunique()} symbols, {len(dates)} dates '
          f'{dates[0]}..{dates[-1]}, bars kept {int(np.isfinite(close).sum())}')


if __name__ == '__main__':
    main()
