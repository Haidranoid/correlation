"""Published data for the round-4 replications (correlation2, built 2026-10-21): Ken French's factors, AQR's US BAB
and TSMOM factors, the 3-month T-bill rate, Yahoo's dividend-adjusted fund closes, and a dividend-correction panel for
the exports' stocks. Numpy only; works with system python3:

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data'); import pubdata

Data folder: DATA = ~/.local/share/correlation2/experiments/published_data (clean files at the top level; the raw
downloads, untouched, under raw/). Code that built it: research/published_data/ (tickers.py, fetch_yahoo.py,
build_clean.py, div_panel.py, checks.py); reports in research/published_data/out/.
Files: ff3_daily.csv ff5_daily.csv umd_daily.csv ff3_monthly.csv ff5_monthly.csv umd_monthly.csv
aqr_bab_us_monthly.csv aqr_tsmom_monthly.csv etf/<TICKER>.csv div_fix_old.npy div_fix_new.npy
div_events_old.csv div_events_new.csv READY; raw/ (french/, french_csv/, aqr/, yahoo_etf/, yahoo_stocks/,
yahoo_stocks_retry/). Every loader below reads these files; nothing is fetched at import.

CONVENTIONS
  dates     every loader returns dates as numpy datetime64[D]. Export days (E['days']) are int days since 1970;
            to_days(dates) converts, on_days(dates, x, days) aligns a daily series to export days.
  returns   all factor returns are simple returns in DECIMAL (French publishes percent; divided by 100 here).
  monthly   every monthly series is keyed by the CALENDAR month-end date (YYYY-MM-31/30/28/29) of its month:
            French's YYYYMM -> that month's last calendar day; AQR's dates are month-ends already (BAB's are calendar
            month-ends; TSMOM's sheet gives the last business day, e.g. 1985-03-29, normalized to 1985-03-31 here;
            the original is kept in the CSV's aqr_date column). So the monthly series join on date exactly.

FAMA-FRENCH (Ken French's data library, CRSP 202608 vintage; downloaded 2026-10-21 as CSV zips, no signup)
  ff3_daily()    -> (dates, {'Mkt-RF','SMB','HML','RF'})              1926-07-01 .. 2026-08-31, 26,317 days
  ff5_daily()    -> (dates, {'Mkt-RF','SMB','HML','RMW','CMA','RF'})  1963-07-01 .. 2026-08-31, 15,897 days
  umd_daily()    -> (dates, {'UMD'})   French's 'Mom' (prior return days -250..-21)   1926-11-03 .. 2026-08-31
  ff3_monthly()  -> same columns as ff3_daily                        1926-07 .. 2026-08, 1,202 months
  ff5_monthly()  -> same columns as ff5_daily (2x3 sorts)             1963-07 .. 2026-08, 758 months
  umd_monthly()  -> (dates, {'UMD'})   French's 'Mom' (months -12..-2)                1927-01 .. 2026-08
  RF is French's 1-month T-bill return for the day / month (decimal). Only the first table of each CSV is parsed
  (monthly files also hold annual tables). French's missing codes -99.99 / -999 would become NaN (none occur).
  Daily dates are NYSE trading days; they match the export's days exactly (checked: 0 differences over
  2001-12-31..2026-08-31). French's files end 2026-08-31 (CRSP 202608), so factors are missing after that.

AQR (aqr.com data library, workbooks as downloaded 2026-10-21; AQR rebuilds full history on each update)
  aqr_bab_us_monthly() -> (dates, {'BAB_US'})  sheet 'BAB Factors', column 'USA' (Frazzini-Pedersen 2014 BAB,
                          self-financing excess return)                     1930-12 .. 2026-08, 1,149 months
  aqr_tsmom_monthly()  -> (dates, {'TSMOM'})   sheet 'TSMOM Factors', column 'TSMOM': the all-asset factor (the
                          sheet's notes: "all assets (no superscript)"; the others are TSMOM^CM/EQ/FI/FX), a 12-month
                          time-series momentum strategy with a 1-month holding period (Moskowitz-Ooi-Pedersen 2012),
                          excess return                                     1985-01 .. 2026-05, 497 months
  Both are decimal monthly returns (BAB_US: 7.8%/yr mean, 11.1%/yr sd; TSMOM: 12.3%/yr, 12.5%/yr over full history).
  Date convention: AQR uses month-end dates; see CONVENTIONS above.

RISK-FREE RATE
  rf_daily(days) -> float64 [len(days)]: daily risk-free return per day = FRED DGS3MO (% a year) / 100 / 252, the
                    latest value at or before the day (so it forward-fills holidays and the CSV's end), NaN before the
                    first value. days: int days since 1970 (E['days']) or datetime64[D]. Copied from
                    research/lazy_prices/common.py; the CSV (RF_CSV) covers 1990-01-02 .. 2026-10-15 (values after
                    2026-10-15 repeat 2026-10-15's 4.28%). FRED refuses scripts, so it isn't refetched here.

YAHOO FUNDS (yfinance 1.7.0, auto_adjust=False, actions=True, period='max'; fetched 2026-10-21, last day 2026-10-20)
  ETF_TICKERS: SPY IWM EFA EEM EWJ VGK FXI TLT IEF LQD HYG TIP EMB GLD SLV USO UNG DBC DBA UUP FXE FXY VNQ
  etf_adjclose(tickers) -> (dates, {ticker: adjusted close})  Yahoo's 'Adj Close' (split- and dividend-adjusted;
                           total-return index up to scale). dates = union of the tickers' Yahoo trading days, NaN
                           where a ticker has no bar.
  etf_close(tickers)    -> same, Yahoo's raw 'Close' (split-adjusted only).
  etf_dividends(tickers)-> same, cash dividend per share on its ex-date row (0 elsewhere; split-adjusted).
  First days: SPY 1993-01-29, EWJ 1996-03-18, IWM 2000-05-26, EFA 2001-08-27, TLT/IEF/LQD 2002-07-30,
  EEM 2003-04-14, TIP 2003-12-05, VNQ 2004-09-29, FXI 2004-10-08, GLD 2004-11-18, VGK 2005-03-10, FXE 2005-12-12,
  DBC 2006-02-06, USO 2006-04-10, SLV 2006-04-28, DBA 2007-01-05, FXY 2007-02-13, UUP 2007-03-01, HYG 2007-04-11,
  UNG 2007-04-18, EMB 2007-12-19. All end 2026-10-20.
  Check: SPY adjusted total return 2009-12-31..2019-12-31 = 13.44% a year (raw close: 11.19% a year).
  Files: DATA/etf/<TICKER>.csv (date, close, adj_close, dividends, splits).

DIVIDEND CORRECTION FOR THE EXPORTS' STOCKS (built by div_panel.py; report: research/published_data/out/
div_classification.txt, check: out/checks.txt)
  div_fix(tag) -> float32 [N, T] memmap aligned to the export's close (tag 'old' = c2eval.EXPORT_OLD, 8,503 x 5,911
                  through 2025-06-27; 'new' = c2eval.EXPORT_NEW, 9,508 x 6,240 through 2026-10-19).
  Why: the store's (so the export's) closes are back-adjusted for dividends until 2022-12-19 for most payers and raw
  from 2022-12-20 on (a sharp switch: of the ex-dates classified, ~1-4% raw Jul-Nov 2022, 2 of ~310 from 12-01 to
  12-19, ~92% from 12-20); so from then on a stock's return on an ex-dividend date includes the dividend drop. About
  1-11% of post-switch events a month are still adjusted (177 stocks in the new export whose history the store
  re-adjusted later), so the classification is per event, not a date cutoff.
  Value at [j, d]: log(1 + D / C_d) on ex-dates d (Yahoo) where the export's return was classified as raw, with D =
  Yahoo's cash dividend and C_d = the export's close on d, so that
        log(C_d / C_{d-1}) + div_fix[j, d] = log((C_d + D) / C_{d-1})      (the exact total return, CRSP's definition)
  0 elsewhere. NOTE: the brief asked for log(1 + D / C_{d-1}); that understates the total return by about (D/C)^2 per
  event (MO: -16 bp a year vs Yahoo's adjusted closes), so C_d is used. Where the export's previous close differs from
  Yahoo's raw close by > 2% (split-frame mismatch; 343 corrected events in 'old', 170 in 'new'), Yahoo's own
  D / Close_d is used.
  Use: total daily LOG return = log(C_d / C_{d-1}) + div_fix[j, d] (the panel is 0 before 2022-07-01, and on 'adj'
  events, so it can be added from 2022-07 on, or everywhere).
  Coverage: ex-dates 2022-07-01 .. 2026-10-19 ('old': .. 2025-06-27) for the 3,066 tickers in the judge-like liquid
  set (night_and_day's rule, minus c2eval.excluded) on any twins-calendar day from 2022-01 on in either export
  (tickers list: research/published_data/out/tickers_map.tsv); 2,812 have Yahoo data. The other 254 (Yahoo: "possibly
  delisted") are mostly names that left the market before 2026; in 'old' they are 254 of the 3,008 matched rows, in
  'new' only 9 of 2,807. Rows of other tickers, and those 254, are 0: no correction known.
  Classification of each Yahoo ex-dividend event (stock, d): r_s = the export's log return on d; r_raw / r_adj =
  Yahoo's raw-close / adjusted-close log returns over the same two days; gap g = r_adj - r_raw (~ D / C). The nearer
  of r_raw, r_adj wins ('raw' / 'adj') if g >= 1 bp and the nearer distance <= min(g / 2, 50 bp); otherwise
  'unclassified'. 'missing': ex-date not an export day, a close missing at d or d-1, or calendars disagree.
  'raw_inferred': an unclassified event whose nearest classified event before is raw and after is raw or none.
  'raw_suspect': a raw event with D / C_{d-1} > 10% whose export return doesn't show at least half the drop; not
  corrected (10 events: NFE's seven bogus Yahoo "5" dividends, HR 2022-07-21, LENZ 2024-03-15, WB 2025-04-09).
  How close: for classified events the nearer distance is 0.000 of the gap at the median and 90th percentile, and
  0.03-0.05 at the 99th: the store and Yahoo closes agree to the cent.
  Summary (events 2022-07-01..end):
      'old': 16,873 events: raw 13,603, adj 3,154, raw_inferred 5, raw_suspect 10, unclassified 77, missing 24
      'new': 24,246 events: raw 20,385, adj 3,720, raw_inferred 17, raw_suspect 10, unclassified 89, missing 25
  Corrections larger than 5% (special dividends: AD, AIV, HUYA, NOAH, SITC, VISN, ZIM, ...) are listed in the report.
  div_events(tag) -> dict of arrays from DATA/div_events_<tag>.csv: one row per Yahoo ex-dividend event with
                     ticker, row, date, col (export day index, -1 if none), dividend, yield (D / C_{d-1}), fix (the
                     panel value if corrected), r_store, r_raw, r_adj, dist_raw, dist_adj, class, reason,
                     frame_fallback.
  Check, bp a year, (export log return + div_fix) minus Yahoo adjusted-close log return, 2022-12-30 .. 2025-06-27
  ('old') and .. 2025-12-31 ('new'); without the fix in brackets:
      old: XOM +1.4 (-342)  T +2.1 (-595)  VZ +3.0 (-678)  KO -1.2 (-303)  CVX -1.2  PFE +2.6  MO +8.0 (-822)
           JNJ -0.9  IBM +1.5  PG +1.0
      new: XOM +0.8 (-344)  T +1.9 (-562)  VZ +2.8 (-673)  KO -0.7 (-300)  CVX -0.7  PFE +2.4  MO +7.5 (-796)
           JNJ -0.9  IBM +1.6  PG +0.5
  (The residual is Yahoo's adjustment convention, C_d / (C_{d-1} - D), against the exact total return.)
"""
from pathlib import Path

import numpy as np

DATA = Path.home() / '.local/share/correlation2/experiments/published_data'
RF_CSV = Path.home() / '.local/share/correlation2/external/series/fred_dgs3mo.csv'
ETF_TICKERS = ('SPY', 'IWM', 'EFA', 'EEM', 'EWJ', 'VGK', 'FXI', 'TLT', 'IEF', 'LQD', 'HYG', 'TIP', 'EMB', 'GLD', 'SLV',
               'USO', 'UNG', 'DBC', 'DBA', 'UUP', 'FXE', 'FXY', 'VNQ')
EPOCH = np.datetime64('1970-01-01', 'D')


def _read(path, skip=()):
    """CSV with a 'date' first column -> (datetime64[D] dates, {column: float64 array}); columns in skip dropped."""
    lines = [l for l in Path(path).read_text().splitlines() if l]
    head = lines[0].split(',')
    rows = [l.split(',') for l in lines[1:]]
    dates = np.array([r[0] for r in rows], dtype='datetime64[D]')
    cols = {}
    for i, h in enumerate(head[1:], 1):
        if h in skip:
            continue
        cols[h] = np.array([float(r[i]) if r[i] != '' else np.nan for r in rows], dtype=np.float64)
    return dates, cols


def to_days(dates):
    """datetime64 -> int64 days since 1970 (the exports' E['days'] convention)."""
    return (np.asarray(dates, dtype='datetime64[D]') - EPOCH).astype(np.int64)


def on_days(dates, x, days):
    """Align a daily series (dates datetime64[D], values x) to export days (int days since 1970): float64 [len(days)],
    NaN where the series has no value on that exact day."""
    dd = to_days(dates)
    days = np.asarray(days)
    if np.issubdtype(days.dtype, np.datetime64):
        days = to_days(days)
    pos = np.searchsorted(dd, days)
    ok = (pos < len(dd)) & (dd[np.minimum(pos, len(dd) - 1)] == days)
    out = np.full(len(days), np.nan)
    out[ok] = np.asarray(x, np.float64)[pos[ok]]
    return out


def ff3_daily():
    """French's FF3 daily: Mkt-RF, SMB, HML, RF in decimal."""
    return _read(DATA / 'ff3_daily.csv')


def ff5_daily():
    """French's FF5 (2x3) daily: Mkt-RF, SMB, HML, RMW, CMA, RF in decimal."""
    return _read(DATA / 'ff5_daily.csv')


def umd_daily():
    """French's daily momentum factor ('Mom') as 'UMD', decimal."""
    return _read(DATA / 'umd_daily.csv')


def ff3_monthly():
    """French's FF3 monthly, decimal, dated at calendar month-end."""
    return _read(DATA / 'ff3_monthly.csv')


def ff5_monthly():
    """French's FF5 (2x3) monthly, decimal, dated at calendar month-end."""
    return _read(DATA / 'ff5_monthly.csv')


def umd_monthly():
    """French's monthly momentum factor ('Mom') as 'UMD', decimal, dated at calendar month-end."""
    return _read(DATA / 'umd_monthly.csv')


def aqr_bab_us_monthly():
    """AQR's US BAB factor (sheet 'BAB Factors', column 'USA') as 'BAB_US', decimal, calendar month-end dates."""
    return _read(DATA / 'aqr_bab_us_monthly.csv', skip=('aqr_date',))


def aqr_tsmom_monthly():
    """AQR's all-asset TSMOM factor (sheet 'TSMOM Factors', column 'TSMOM'), decimal, calendar month-end dates."""
    return _read(DATA / 'aqr_tsmom_monthly.csv', skip=('aqr_date',))


def rf_daily(days):
    """Daily risk-free return per day: DGS3MO % a year / 100 / 252, latest value at or before the day (NaN before
    1990-01-02). days: int days since 1970 or datetime64[D]. Copied from research/lazy_prices/common.py."""
    days = np.asarray(days)
    if np.issubdtype(days.dtype, np.datetime64):
        days = to_days(days)
    rows = [l.strip().split(',') for l in open(RF_CSV)][1:]
    d = np.array([(np.datetime64(r[0]) - np.datetime64('1970-01-01')).astype(int) for r in rows if r[1] not in ('', '.')])
    v = np.array([float(r[1]) for r in rows if r[1] not in ('', '.')])
    pos = np.searchsorted(d, days, side='right') - 1
    out = v[np.maximum(pos, 0)] / 100.0 / 252.0
    out[pos < 0] = np.nan
    return out


def _etf(tickers, field):
    if isinstance(tickers, str):
        tickers = [tickers]
    per = {t: _read(DATA / 'etf' / f'{t}.csv') for t in tickers}
    dates = np.unique(np.concatenate([d for d, _ in per.values()]))
    out = {}
    for t, (d, c) in per.items():
        x = np.full(len(dates), np.nan)
        x[np.searchsorted(dates, d)] = c[field]
        out[t] = x
    return dates, out


def etf_adjclose(tickers=ETF_TICKERS):
    """(dates, {ticker: Yahoo Adj Close}) on the union of the tickers' trading days; NaN where no bar."""
    return _etf(tickers, 'adj_close')


def etf_close(tickers=ETF_TICKERS):
    """(dates, {ticker: Yahoo raw Close (split-adjusted only)}) on the union of the tickers' trading days."""
    return _etf(tickers, 'close')


def etf_dividends(tickers=ETF_TICKERS):
    """(dates, {ticker: cash dividend on its ex-date, 0 elsewhere}) on the union of the tickers' trading days."""
    return _etf(tickers, 'dividends')


def div_fix(tag):
    """float32 [N, T] memmap aligned to the export's close ('old' or 'new'): log(1 + D / C_{d-1}) on ex-dates where
    the export's close is raw, else 0. Add to the export's daily log returns."""
    assert tag in ('old', 'new'), tag
    return np.load(DATA / f'div_fix_{tag}.npy', mmap_mode='r')


def div_events(tag):
    """The classified ex-dividend events behind div_fix(tag): dict of column -> array (strings for ticker, date,
    class, reason; ints for row, col, frame_fallback; float64 otherwise)."""
    assert tag in ('old', 'new'), tag
    lines = (DATA / f'div_events_{tag}.csv').read_text().splitlines()
    head = lines[0].split(',')
    rows = [l.split(',') for l in lines[1:] if l]
    out = {}
    for i, h in enumerate(head):
        col = [r[i] for r in rows]
        if h in ('ticker', 'date', 'class', 'reason'):
            out[h] = np.array(col)
        elif h in ('row', 'col', 'frame_fallback'):
            out[h] = np.array([int(x) for x in col], dtype=np.int64)
        else:
            out[h] = np.array([float(x) for x in col], dtype=np.float64)
    return out
