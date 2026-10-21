"""Compact the downloaded Yahoo bars into one row per symbol and day (the analysis reads only these tables).

    python -I extract.py 5m|1m|1d BARS_DIR[,BARS_DIR...] OUT_CSV

5m / 1m: per symbol and date, from bars with 09:30 <= time < 16:00 (America/New_York; the UTC offset of every bar is
kept and checked by the analysis): number of bars, first bar's time and open, the opens of the bars stamped exactly
09:31 (1m only), 09:35, 09:45 and 10:00, the last bar's time and close, whether a 16:00 bar or bars outside the
session came back, and the set of UTC offsets seen.
1d: per symbol and date, Yahoo's open, high, low, close (split-adjusted, not dividend-adjusted), adjusted close,
dividends and splits.
"""
import sys
from pathlib import Path

import pandas as pd


def files(bars, mode):
    return sorted(f for b in bars.split(',') for f in Path(b).glob(f'{mode}_*.csv'))


def read(f):
    return pd.read_csv(f, dtype={'sym': str, 'ts': str, 'tz': str}, keep_default_na=False, na_values=['', 'nan', 'NaN'])


def compact(df, mode):
    df = df.drop_duplicates(['sym', 'ts']).copy()
    df['date'] = df['ts'].str.slice(0, 10)
    if mode == '1d':
        cols = ['sym', 'date', 'tz', 'open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'stock_splits']
        return df[[c for c in cols if c in df.columns]].sort_values(['sym', 'date'])
    df['hhmm'] = df['ts'].str.slice(11, 16)
    df['off'] = df['ts'].str.slice(19)
    df = df[df['open'].notna() & df['close'].notna()]
    inside = (df['hhmm'] >= '09:30') & (df['hhmm'] < '16:00')
    extra = df[~inside].groupby(['sym', 'date']).agg(n_outside=('hhmm', 'size'),
                                                       has_1600=('hhmm', lambda s: bool((s == '16:00').any())))
    d = df[inside].sort_values(['sym', 'ts'])
    g = d.groupby(['sym', 'date'])
    res = g.agg(n_bars=('hhmm', 'size'), first_time=('hhmm', 'first'), first_open=('open', 'first'),
                last_time=('hhmm', 'last'), last_close=('close', 'last'), offs=('off', lambda s: '|'.join(sorted(set(s)))),
                tz=('tz', lambda s: '|'.join(sorted(set(s)))))
    times = ['09:31', '09:35', '09:45', '10:00'] if mode == '1m' else ['09:35', '09:45', '10:00']
    for tm in times:
        x = d[d['hhmm'] == tm].set_index(['sym', 'date'])['open']
        res['open_' + tm.replace(':', '')] = x
    res = res.join(extra, how='left')
    res['n_outside'] = res['n_outside'].fillna(0).astype(int)
    res['has_1600'] = res['has_1600'].fillna(False).astype(bool)
    return res.reset_index()


def main():
    mode, bars, out = sys.argv[1], sys.argv[2], sys.argv[3]
    fs = files(bars, mode)
    parts, nrows = [], 0
    for f in fs:  # one symbol's bars sit in one file (a batch), so each file compacts on its own
        df = read(f)
        nrows += len(df)
        parts.append(compact(df, mode))
    res = pd.concat(parts)
    dup = res.duplicated(['sym', 'date']).sum()
    res = res.drop_duplicates(['sym', 'date'])
    res.to_csv(out, index=False)
    print(f'{len(fs)} files, {nrows} rows, {res["sym"].nunique()} symbols, {len(res)} symbol-days ({dup} duplicates dropped)', flush=True)


if __name__ == '__main__':
    main()
