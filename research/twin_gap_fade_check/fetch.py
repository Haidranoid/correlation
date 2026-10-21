"""Fetch Yahoo Finance bars with yfinance for the intraday check of the twin gap fade.

    python -I fetch.py 5m  SYMBOLS_FILE OUT_DIR        # 5-minute bars, last 60 days
    python -I fetch.py 1d  SYMBOLS_FILE OUT_DIR        # daily bars 2026-06-01..2026-10-20, unadjusted, with actions
    python -I fetch.py 1m  SYMBOLS_FILE OUT_DIR START  # 1-minute bars, 7-day pieces from START through 2026-10-19

One Ticker.history call per symbol (4 threads), batches of BATCH symbols with a pause between them, exponential
backoff when Yahoo answers 429 (YFRateLimitError), and one retry of the symbols that failed. Writes one CSV per batch
(long format: sym, ts, open, high, low, close, volume[, adj_close, dividends, stock_splits]) into OUT_DIR, a directory
of downloaded data only (no code). Regular session only, auto_adjust=False (Yahoo's prices are then split-adjusted,
not dividend-adjusted). Exits with code 3 and "BLOCKED" in the log if Yahoo refuses the whole first batch.
A log goes to OUT_DIR/../fetch_<mode>.log and failures to OUT_DIR/../failed_<mode>.tsv.
"""
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

yf.config.debug.hide_exceptions = False
BATCH = 40
THREADS = 4
PAUSE = 2.0
BACKOFF = (30, 60, 120, 240, 480, 900)


def log(fh, *a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    fh.write(s + '\n')
    fh.flush()


def one(sym, mode, start, end):
    """(sym, frame or None, error string or None, rate_limited)"""
    try:
        t = yf.Ticker(sym)
        if mode == '5m':
            d = t.history(period='60d', interval='5m', prepost=False, actions=False, auto_adjust=False, repair=False)
        elif mode == '1d':
            d = t.history(start='2026-06-01', end='2026-10-21', interval='1d', prepost=False, actions=True,
                          auto_adjust=False, repair=False)
        else:
            d = t.history(start=start, end=end, interval='1m', prepost=False, actions=False, auto_adjust=False, repair=False)
    except YFRateLimitError as ex:
        return sym, None, str(ex), True
    except Exception as ex:
        msg = f'{type(ex).__name__}: {ex}'
        return sym, None, msg, ('429' in msg or 'Too Many' in msg)
    if d is None or len(d) == 0:
        return sym, None, 'no rows', False
    tz = str(getattr(d.index, 'tz', None))
    d = d.copy()
    d.columns = [str(c).lower().replace(' ', '_') for c in d.columns]
    d.index.name = 'ts'
    d = d.reset_index()
    d['ts'] = d['ts'].astype(str)
    d.insert(0, 'sym', sym)
    d['tz'] = tz
    keep = ['sym', 'ts', 'tz', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'dividends', 'stock_splits']
    return sym, d[[c for c in keep if c in d.columns]], None, False


def run_batch(syms, mode, fh, start, end):
    """Fetch a batch; symbols refused with 429 are retried after backoff. Returns (frames, errors)."""
    frames, errors = [], {}
    todo = list(syms)
    for k, wait in enumerate((0,) + BACKOFF):
        if not todo:
            break
        if wait:
            log(fh, f'rate limited on {len(todo)} symbols; backing off {wait}s')
            time.sleep(wait)
        with ThreadPoolExecutor(THREADS) as ex:
            res = list(ex.map(lambda s: one(s, mode, start, end), todo))
        todo = []
        for sym, d, err, rl in res:
            if d is not None:
                frames.append(d)
            elif rl:
                todo.append(sym)
            else:
                errors[sym] = err
    for s in todo:
        errors[s] = 'rate limited after all backoffs'
    return frames, errors


def main():
    mode, symfile, outdir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    start = sys.argv[4] if len(sys.argv) > 4 else None
    syms = [l.strip() for l in symfile.read_text().splitlines() if l.strip()]
    outdir.mkdir(parents=True, exist_ok=True)
    fh = open(outdir.parent / f'fetch_{mode}.log', 'a')
    pieces = [(None, None)]
    if mode == '1m':
        d0, end_all = dt.date.fromisoformat(start), dt.date(2026, 10, 20)  # the store ends 2026-10-19
        pieces = []
        while d0 < end_all:
            d1 = min(d0 + dt.timedelta(days=7), end_all)
            pieces.append((d0.isoformat(), d1.isoformat()))
            d0 = d1
    log(fh, f'mode {mode}: {len(syms)} symbols, batches of {BATCH}, {THREADS} threads, pieces {pieces}')
    failed_all = {}
    first = True
    for (a, b) in pieces:
        todo = list(syms)
        for attempt in (1, 2):
            failed = {}
            for j in range(0, len(todo), BATCH):
                name = f'{mode}_{a or "all"}_{attempt}_{j // BATCH:04d}.csv'
                if (outdir / name).exists():
                    continue
                batch = todo[j:j + BATCH]
                frames, err = run_batch(batch, mode, fh, a, b)
                failed.update(err)
                if frames:
                    pd.concat(frames).to_csv(outdir / name, index=False)
                tzs = sorted(set(f['tz'].iloc[0] for f in frames))
                log(fh, f'{a or ""} attempt {attempt} batch {j // BATCH}: {len(frames)}/{len(batch)} symbols, '
                        f'{sum(len(f) for f in frames)} rows, tz {tzs}')
                if first and not frames:
                    log(fh, f'BLOCKED: Yahoo returned nothing for the first batch: {list(err.items())[:5]}')
                    sys.exit(3)
                first = False
                time.sleep(PAUSE)
            if not failed:
                break
            log(fh, f'{a or ""} attempt {attempt}: {len(failed)} failed, e.g. {list(failed.items())[:5]}')
            todo = sorted(failed)
            time.sleep(10)
        for s, e in failed.items():
            failed_all[(s, a)] = e
    with open(outdir.parent / f'failed_{mode}.tsv', 'w') as f:
        for (s, a), e in sorted(failed_all.items(), key=lambda x: (x[0][0], str(x[0][1]))):
            f.write(f'{s}\t{a}\t{str(e)[:300]}\n')
    log(fh, f'done: {len(failed_all)} symbol-pieces failed after retry')


if __name__ == '__main__':
    main()
