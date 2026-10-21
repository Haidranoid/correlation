"""Parse the raw downloads (Ken French CSVs, AQR workbooks, Yahoo fund bars) into clean files at the top of the
published_data folder. Needs openpyxl and pandas: run with this folder's venv.

    .venv/bin/python -I -B build_clean.py

Writes (DATA = ~/.local/share/correlation2/experiments/published_data):
  ff3_daily.csv ff5_daily.csv umd_daily.csv ff3_monthly.csv ff5_monthly.csv umd_monthly.csv   (decimal returns)
  aqr_bab_us_monthly.csv aqr_tsmom_monthly.csv                                               (decimal returns)
  etf/<TICKER>.csv   date, close (raw), adj_close (dividend-adjusted), dividends, splits
"""
import calendar
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

DATA = Path.home() / '.local/share/correlation2/experiments/published_data'
RAW = DATA / 'raw'
LAST = '2026-10-20'   # last completed trading day at fetch time
FRENCH = {
    'ff5_daily': 'F-F_Research_Data_5_Factors_2x3_daily.csv',
    'ff3_daily': 'F-F_Research_Data_Factors_daily.csv',
    'umd_daily': 'F-F_Momentum_Factor_daily.csv',
    'ff5_monthly': 'F-F_Research_Data_5_Factors_2x3.csv',
    'ff3_monthly': 'F-F_Research_Data_Factors.csv',
    'umd_monthly': 'F-F_Momentum_Factor.csv',
}


def month_end(y, m):
    return f'{y:04d}-{m:02d}-{calendar.monthrange(y, m)[1]:02d}'


def parse_french(path, daily):
    """First table only: the header line starting with ',' and the contiguous YYYYMMDD / YYYYMM rows under it."""
    lines = [l.rstrip('\r\n') for l in open(path, encoding='latin-1')]
    pat = re.compile(r'^\s*(\d{8})\s*,' if daily else r'^\s*(\d{6})\s*,')
    i0 = next(i for i, l in enumerate(lines) if pat.match(l))
    head = [h.strip() for h in lines[i0 - 1].split(',')][1:]
    assert lines[i0 - 1].startswith(',') and all(head), lines[i0 - 1]
    head = ['UMD' if h.upper() == 'MOM' else h for h in head]
    dates, vals = [], []
    for l in lines[i0:]:
        if not pat.match(l):
            break
        f = [x.strip() for x in l.split(',')]
        d = f[0]
        dates.append(f'{d[:4]}-{d[4:6]}-{d[6:8]}' if daily else month_end(int(d[:4]), int(d[4:6])))
        v = [float(x) for x in f[1:]]
        vals.append([np.nan if x in (-99.99, -999.0) else x / 100.0 for x in v])
    df = pd.DataFrame(vals, columns=head)
    df.insert(0, 'date', dates)
    assert df['date'].is_monotonic_increasing and df['date'].is_unique
    return df


def aqr_sheet(path, sheet):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    return rows


def as_date(x):
    if hasattr(x, 'year'):
        return pd.Timestamp(x)
    return pd.Timestamp(pd.to_datetime(str(x), format='%m/%d/%Y'))


def parse_aqr():
    out = {}
    # BAB: header row starts with 'DATE'; US column is 'USA'.
    rows = aqr_sheet(RAW / 'aqr/Betting-Against-Beta-Equity-Factors-Monthly.xlsx', 'BAB Factors')
    h = next(i for i, r in enumerate(rows) if r and r[0] == 'DATE')
    cols = list(rows[h])
    j = cols.index('USA')
    recs = []
    for r in rows[h + 1:]:
        if r is None or r[0] in (None, ''):
            continue
        v = r[j]
        if v in (None, ''):
            continue
        d = as_date(r[0])
        recs.append((month_end(d.year, d.month), str(d.date()), float(v)))
    out['aqr_bab_us_monthly'] = pd.DataFrame(recs, columns=['date', 'aqr_date', 'BAB_US'])
    # TSMOM: header row holds 'TSMOM', 'TSMOM^CM', ...; the all-asset factor is 'TSMOM' (no superscript).
    rows = aqr_sheet(RAW / 'aqr/Time-Series-Momentum-Factors-Monthly.xlsx', 'TSMOM Factors')
    h = next(i for i, r in enumerate(rows) if r and 'TSMOM' in [str(x).strip() for x in r if x is not None])
    cols = [None if x is None else str(x).strip() for x in rows[h]]
    j = cols.index('TSMOM')
    recs = []
    for r in rows[h + 1:]:
        if r is None or r[0] in (None, '') or r[j] in (None, ''):
            continue
        d = as_date(r[0])
        recs.append((month_end(d.year, d.month), str(d.date()), float(r[j])))
    out['aqr_tsmom_monthly'] = pd.DataFrame(recs, columns=['date', 'aqr_date', 'TSMOM'])
    for k, df in out.items():
        assert df['date'].is_monotonic_increasing and df['date'].is_unique, k
    return out, cols


def build_etf():
    d = pd.concat([pd.read_csv(f) for f in sorted((RAW / 'yahoo_etf').glob('batch_*.csv'))])
    (DATA / 'etf').mkdir(exist_ok=True)
    info = []
    d = d[d['date'] <= LAST]   # drop today's partial bar (fetched during the 2026-10-21 session)
    for s, x in d.groupby('sym'):
        x = x.sort_values('date')[['date', 'close', 'adj_close', 'dividends', 'splits']]
        assert x['date'].is_unique
        x.to_csv(DATA / 'etf' / f'{s}.csv', index=False, float_format='%.10g')
        info.append((s, x['date'].iloc[0], x['date'].iloc[-1], len(x), int((x['dividends'] > 0).sum())))
    return info


def main():
    for name, fn in FRENCH.items():
        df = parse_french(RAW / 'french_csv' / fn, daily=name.endswith('daily'))
        df.to_csv(DATA / f'{name}.csv', index=False, float_format='%.6g')
        print(f'{name}: {df["date"].iloc[0]}..{df["date"].iloc[-1]} rows {len(df)} cols {list(df.columns[1:])} '
              f'NaN {int(df.iloc[:, 1:].isna().sum().sum())}')
    out, tcols = parse_aqr()
    print('TSMOM header row:', tcols)
    for name, df in out.items():
        df.to_csv(DATA / f'{name}.csv', index=False, float_format='%.10g')
        v = df.iloc[:, 2]
        print(f'{name}: {df["date"].iloc[0]}..{df["date"].iloc[-1]} rows {len(df)} mean {v.mean()*12:.4f}/yr '
              f'sd {v.std()*np.sqrt(12):.4f}/yr')
    for r in build_etf():
        print('etf', *r)


if __name__ == '__main__':
    main()
