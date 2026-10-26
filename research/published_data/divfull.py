"""Full-history dividend corrections (built by div_panel_full.py; report: out/div_full_classification.txt).

NOT YET BUILT (2026-10-25): the Yahoo histories are downloaded (DATA/raw/yahoo_full_stocks, yahoo_full_funds) but
div_panel_full.py hasn't been run, so the panels below don't exist yet. Nothing in the project uses this module.

The store's closes, and so the exports', have no dividend adjustment before about February 2011, are back-adjusted
from then to 2022, and are raw again from about 2022-12-20 (pubdata.div_fix covers only the last part). On ex-dates
where the store's close is raw, a holder's total return is the price return plus the dividend; these panels add it.

    import sys; sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data'); import divfull

  divfull.div_fix_full(tag)    float32 [N, T] memmap aligned to the export's close (tag 'old' = c2eval.EXPORT_OLD,
                               'new' = c2eval.EXPORT_NEW): log(1 + D / C_d) on ex-dates classified raw, else 0, for
                               the 3,211 stocks ever in the judge-like liquid set (out/tickers_full_stocks_map.tsv);
                               other rows are 0 (no correction known).
  divfull.fund_div_fix(tag)    float32 [F, T] aligned to c2eval.funds(E) rows of that tag's export (the store's
                               unrepaired fund closes): the same for the 1,089 funds ever chosen as a twin, plus SPY,
                               QQQ, IWM, DIA and the nine SPDR sector funds (out/tickers_full_funds_map.tsv).
  divfull.div_events_full(tag), divfull.fund_div_events(tag)
                               the classified events behind each panel, as pubdata.div_events returns them.

  Use: total daily log return = log(C_d / C_{d-1}) + panel[j, d]. The panels are 0 wherever the store's close was
  already adjusted, so they can be added on every day.
pubdata.py and div_panel.py are left unchanged: experiments that froze their hashes keep reproducing.
"""
from pathlib import Path

import numpy as np

DATA = Path.home() / '.local/share/correlation2/experiments/published_data'


def _npy(name, tag):
    assert tag in ('old', 'new'), tag
    return np.load(DATA / f'{name}_{tag}.npy', mmap_mode='r')


def _events(name, tag):
    assert tag in ('old', 'new'), tag
    lines = (DATA / f'{name}_{tag}.csv').read_text().splitlines()
    head = lines[0].split(',')
    rows = [l.split(',') for l in lines[1:] if l]
    out = {}
    for i, h in enumerate(head):
        col = [r[i] for r in rows]
        if h in ('ticker', 'date', 'class', 'reason'):
            out[h] = np.array(col)
        elif h in ('row', 'col', 'frame_fallback'):
            out[h] = np.array([int(x) for x in col])
        else:
            out[h] = np.array([float(x) if x not in ('', 'nan') else np.nan for x in col])
    return out


def div_fix_full(tag):
    return _npy('div_fix_full', tag)


def fund_div_fix(tag):
    return _npy('fund_div_fix', tag)


def div_events_full(tag):
    return _events('div_events_full', tag)


def fund_div_events(tag):
    return _events('fund_div_events', tag)
