"""Fetch Yahoo daily bars with dividend events through yfinance (own venv: research/published_data/.venv).

    .venv/bin/python -I -B fetch_yahoo.py stocks SYMBOLS_FILE OUT_DIR     # from 2022-01-01, batches of 80
    .venv/bin/python -I -B fetch_yahoo.py etf    SYMBOLS_FILE OUT_DIR     # full history (period='max')

yf.download(..., auto_adjust=False, actions=True): Close is Yahoo's split-adjusted raw close, Adj Close its split-
and dividend-adjusted close, Dividends the cash dividend on its ex-date row (split-adjusted). One CSV per batch, long
format (sym, date, open, high, low, close, adj_close, volume, dividends, splits), into OUT_DIR, which holds downloaded
data only. Batches already written are skipped (resume). Symbols that fail with a rate-limit message (429) are
retried after a back-off; others are listed in out/yahoo_<mode>_failed.tsv. Log: research/published_data/out.
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance import shared as yf_shared

HERE = Path(__file__).resolve().parent
BACKOFF = (60, 120, 240, 480, 900)
PAUSE = 3.0


def log(fh, *a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    fh.write(s + '\n')
    fh.flush()


def is_rate(msg):
    m = str(msg)
    return '429' in m or 'Too Many' in m or 'Rate' in m or 'rate limit' in m.lower()


def get(syms, mode):
    """(long frame, {sym: error})"""
    yf_shared._ERRORS.clear()
    kw = dict(auto_adjust=False, actions=True, group_by='ticker', threads=4, progress=False, interval='1d')
    if mode == 'etf':
        d = yf.download(syms, period='max', **kw)
    else:
        d = yf.download(syms, start='2022-01-01', end='2026-10-21', **kw)
    errs = dict(yf_shared._ERRORS)
    frames = []
    for s in syms:
        if s in errs:
            continue
        try:
            x = d[s] if isinstance(d.columns, pd.MultiIndex) else d
        except KeyError:
            errs[s] = 'missing from frame'
            continue
        x = x.dropna(subset=['Close'])
        if len(x) == 0:
            errs[s] = 'no rows'
            continue
        x = x.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close',
                              'Volume': 'volume', 'Dividends': 'dividends', 'Stock Splits': 'splits'})
        x = x[['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividends', 'splits']].copy()
        x.index = pd.DatetimeIndex(x.index).tz_localize(None) if x.index.tz is not None else x.index
        x.insert(0, 'date', x.index.strftime('%Y-%m-%d'))
        x.insert(0, 'sym', s)
        frames.append(x.reset_index(drop=True))
    return (pd.concat(frames) if frames else None), errs


def main():
    mode, sym_file, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    (HERE / 'out').mkdir(exist_ok=True)
    fh = open(HERE / 'out' / f'fetch_{mode}.log', 'a')
    syms = [l.strip() for l in open(sym_file) if l.strip()]
    bs = 80 if mode == 'stocks' else 25
    batches = [syms[i:i + bs] for i in range(0, len(syms), bs)]
    log(fh, f'{mode}: {len(syms)} symbols in {len(batches)} batches -> {out}')
    failed = {}
    for b, batch in enumerate(batches):
        f = out / f'batch_{b:03d}.csv'
        if f.exists():
            continue
        todo, frames = list(batch), []
        for k, wait in enumerate((0,) + BACKOFF):
            if not todo:
                break
            if wait:
                log(fh, f'batch {b}: rate limited on {len(todo)} symbols; backing off {wait}s')
                time.sleep(wait)
            fr, errs = get(todo, mode)
            if fr is not None:
                frames.append(fr)
            rate = [s for s, e in errs.items() if is_rate(e)]
            for s, e in errs.items():
                if not is_rate(e):
                    failed[s] = str(e).replace('\n', ' ')[:200]
            if k == 0 and b == 0 and fr is None and len(rate) == len(todo):
                log(fh, 'BLOCKED: Yahoo refused the whole first batch')
            todo = rate
        for s in todo:
            failed[s] = 'rate limited after all back-offs'
        if frames:
            pd.concat(frames).to_csv(f, index=False)
        else:
            f.write_text('sym,date,open,high,low,close,adj_close,volume,dividends,splits\n')
        n_ok = sum(len(x['sym'].unique()) for x in frames)
        log(fh, f'batch {b + 1}/{len(batches)}: {n_ok} of {len(batch)} symbols; failed so far {len(failed)}')
        time.sleep(PAUSE)
    with open(HERE / 'out' / f'yahoo_{mode}_failed.tsv', 'w') as g:
        g.write('sym\terror\n')
        for s, e in sorted(failed.items()):
            g.write(f'{s}\t{e}\n')
    log(fh, f'DONE {mode}: failed {len(failed)} of {len(syms)}')


if __name__ == '__main__':
    main()
