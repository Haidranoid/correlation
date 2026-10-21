"""Fits, s-scores, state paths and daily books for one export, price source and dividend setting.

    python3 -I -B run.py old|new export|store raw|fix

Writes WORK/fit_<tag>_<source>_<fix>/{beta,m,sigeq,b}.npy (float32 [N, T]) and the daily books to
WORK/books_<tag>_<source>_<fix>.npz: for each set ('all' = the universe, 'big' = the 500 largest) and execution
('same' = at the close the s-score uses, 'next' = the next close), the daily arrays from common.run_book, plus
diagnostics (filter pass shares, holding times by exit type, flips).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

tag, source, fix = sys.argv[1], sys.argv[2], sys.argv[3]
assert tag in ('old', 'new') and source in ('export', 'store') and fix in ('raw', 'fix')
name = f'{tag}_{source}_{fix}'
E, tw, uni, big = C.load(tag)
days = E['days']
N, T = E['close'].shape
C.log(name, 'N', N, 'T', T)
close = C.stock_close(E, tag, source)
has_close = np.isfinite(close)
lr = C.log_returns(close)
dfix = None
if fix == 'fix':
    sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
    import pubdata  # noqa: E402
    dfix = np.asarray(pubdata.div_fix(tag), np.float64)
    assert dfix.shape == (N, T), dfix.shape
    lr = lr + dfix                      # NaN stays NaN
    C.log('div_fix: nonzero stock-days', int((dfix != 0).sum()))
rs = C.signal_returns(lr)
del lr
fclose = np.asarray(tw['F']['close'], np.float64)
rf = C.signal_returns(C.log_returns(fclose))
fdir = C.WORK / f'fit_{name}'
fdir.mkdir(exist_ok=True)
if (fdir / 'b.npy').exists():
    fit = {k: np.load(fdir / f'{k}.npy') for k in ('beta', 'm', 'sigeq', 'b', 'nv')}
    C.log('fits loaded')
else:
    fit = C.fit_panel(rs, rf, tw, uni)
    for k in ('beta', 'm', 'sigeq', 'b', 'nv'):
        np.save(fdir / f'{k}.npy', fit[k])
    C.log('fits done')
del rs, rf
# P&L returns
rs_pnl = C.pnl_returns(close, dfix)
del close, dfix
flr = C.log_returns(C.ffill(fclose))
flr[~np.isfinite(flr) | (np.abs(flr) > C.MAX_LOG)] = 0.0
rf_pnl = np.expm1(flr)
del flr, fclose
fund_row = C.fund_row_by_day(tw, T)
nF = rf_pnl.shape[0]
d_start = int(tw['rebal'][np.nonzero(uni.sum(1))[0][0]])
res = {}
diag = {}
fitted = C.by_day(uni, tw['rebal'], T) & (fit['nv'] >= C.MIN_VALID) & np.isfinite(fit['b'])
for setname, mask_k in (('all', uni), ('big', big)):
    in_set = C.by_day(mask_k, tw['rebal'], T)
    s, ok = C.s_score(fit, in_set)
    if tag == 'old' and source == 'export' and fix == 'raw' and setname == 'all':
        np.save(C.WORK / 'scores_s_old_export_raw_all.npy', s)
    if tag == 'new' and source == 'export' and fix == 'raw' and setname == 'all':
        np.save(C.WORK / 'scores_s_new_export_raw_all.npy', s)
    fs = fitted & in_set
    per_day_fit = fs.sum(0)
    per_day_ok = ok.sum(0)
    p, info = C.state_path(s, has_close, in_set)
    C.log(setname, 'state path done; flips', info['flips'])
    diag[setname] = {'fit_stock_days': int(fs.sum()), 'pass_stock_days': int(ok.sum()),
                     'flips': info['flips'],
                     'hold': {k: [int(len(v)), float(v.mean()) if len(v) else float('nan'),
                                  float(np.median(v)) if len(v) else float('nan')] for k, v in info['hold'].items()}}
    res[f'{setname}_fitn'] = per_day_fit
    res[f'{setname}_okn'] = per_day_ok
    # holding-time arrays by exit type (for period splits later we keep only the overall stats; entries per day:)
    res[f'{setname}_opens'] = ((p != 0) & (np.concatenate([np.zeros((N, 1), np.int8), p[:, :-1]], 1) != p)).sum(0)
    for ex, pp, bs in (('same', p, 0), ('next', C.delay_path(p, has_close), 1)):
        book = C.run_book(pp, fit['beta'], fund_row, rs_pnl, rf_pnl, nF, d_start, beta_shift=bs)
        for k, v in book.items():
            res[f'{setname}_{ex}_{k}'] = v
        C.log(setname, ex, 'book done')
    del s, ok, p, in_set
res['days'] = days
np.savez(C.WORK / f'books_{name}.npz', **res)
(C.OUT / f'diag_{name}.json').write_text(json.dumps(diag, indent=1))
C.log('saved', name)
