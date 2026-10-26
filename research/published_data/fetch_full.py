"""Fetch Yahoo's full daily history (period='max') with dividend events for the full-history dividend panel, through
fetch_yahoo.py's own download and back-off code (own venv: research/published_data/.venv). Its own log and failure
list, so the earlier fetches' files are left as they were.

    .venv/bin/python -I -B fetch_full.py SYMBOLS_FILE OUT_DIR NAME     # -> out/fetch_full_NAME.log,
                                                                      #    out/yahoo_full_NAME_failed.tsv
Batches of 50; batches already written are skipped (resume).
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_yahoo import BACKOFF, HERE, PAUSE, get, is_rate, log  # noqa: E402


def main():
    sym_file, out, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    fh = open(HERE / 'out' / f'fetch_full_{name}.log', 'a')
    syms = [l.strip() for l in open(sym_file) if l.strip()]
    batches = [syms[i:i + 50] for i in range(0, len(syms), 50)]
    log(fh, f'full {name}: {len(syms)} symbols in {len(batches)} batches -> {out}')
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
            fr, errs = get(todo, 'etf')          # 'etf' mode = period='max'
            if fr is not None:
                frames.append(fr)
            rate = [s for s, e in errs.items() if is_rate(e)]
            for s, e in errs.items():
                if not is_rate(e):
                    failed[s] = str(e).replace('\n', ' ')[:200]
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
    with open(HERE / 'out' / f'yahoo_full_{name}_failed.tsv', 'w') as g:
        g.write('sym\terror\n')
        for s, e in sorted(failed.items()):
            g.write(f'{s}\t{e}\n')
    log(fh, f'DONE full {name}: failed {len(failed)} of {len(syms)}')


if __name__ == '__main__':
    main()
