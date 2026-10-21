"""Accounting checks of common.run_book on the old export:
  1. the stock legs' summed daily P&L equals the sum over trades of side x (close_exit / close_entry - 1);
  2. the hedge's summed daily P&L equals a per-position recomputation (-beta_d x V_d x r_f(d+1), summed);
  3. a placebo: s replaced by iid N(0, 1) noise on the same stock-days should earn about 0 gross.

    python3 -I -B check_book.py      # writes out/check_book.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    C.log(s)


E, tw, uni, big = C.load('old')
days = E['days']
N, T = E['close'].shape
close = np.asarray(E['close'], np.float64)
has_close = np.isfinite(close)
fdir = C.WORK / 'fit_old_export_raw'
fit = {k: np.load(fdir / f'{k}.npy') for k in ('beta', 'm', 'sigeq', 'b', 'nv')}
in_set = C.by_day(uni, tw['rebal'], T)
s, ok = C.s_score(fit, in_set)
p, info = C.state_path(s, has_close, in_set)
rs_pnl = C.pnl_returns(close)
cff = C.ffill(close)
fclose = np.asarray(tw['F']['close'], np.float64)
flr = C.log_returns(C.ffill(fclose))
flr[~np.isfinite(flr) | (np.abs(flr) > C.MAX_LOG)] = 0.0
rf_pnl = np.expm1(flr)
fund_row = C.fund_row_by_day(tw, T)
d_start = int(tw['rebal'][np.nonzero(uni.sum(1))[0][0]])
book = C.run_book(p, fit['beta'], fund_row, rs_pnl, rf_pnl, rf_pnl.shape[0], d_start)
den = np.maximum(book['n'], 1)
tot_stock = float((book['pnl_stock'] * den).sum())
# 1. trade-level stock P&L
trade_sum = 0.0
ntr = 0
for i in np.nonzero((p != 0).any(1))[0]:
    pi = p[i].astype(int)
    chg = np.nonzero(np.diff(np.concatenate([[0], pi])))[0]   # days where the position changes
    for a_, d in enumerate(chg):
        if pi[d] == 0:
            continue
        x = chg[a_ + 1] if a_ + 1 < len(chg) else T - 1
        trade_sum += pi[d] * (cff[i, x] / cff[i, d] - 1)
        ntr += 1
say(f'1. stock legs: daily P&L summed {tot_stock:.6f}; trades ({ntr}) summed {trade_sum:.6f}; diff {tot_stock - trade_sum:.2e}')
# 2. hedge P&L recomputed per position (beta carried forward where NaN), same-close book
V = np.zeros(N)
bheld = np.full(N, np.nan)
fheld = np.full(N, -1)
hedge = 0.0
prev = np.zeros(N, np.int8)
for d in range(d_start, T):
    if d > d_start:
        held = np.nonzero(V != 0)[0]
        hedge += float((-np.nan_to_num(bheld[held]) * V[held] * rf_pnl[np.maximum(fheld[held], 0), d]
                        * (fheld[held] >= 0)).sum())
        V = V * (1 + rs_pnl[:, d])
    chg = p[:, d] != prev
    V[chg] = p[chg, d]
    bd, fd = fit['beta'][:, d], fund_row[:, d]
    u = np.isfinite(bd) & (fd >= 0)
    bheld[u] = bd[u]
    fheld[u] = fd[u]
    prev = p[:, d].copy()
tot_fund = float((book['pnl_fund'] * den).sum())
say(f'2. hedge legs: daily P&L summed {tot_fund:.6f}; per-position recomputation {hedge:.6f}; diff {tot_fund - hedge:.2e}')
# 3. placebo
rng = np.random.default_rng(11)
sp = np.where(np.isfinite(s), rng.standard_normal(s.shape).astype(np.float32), np.nan)
pp, _ = C.state_path(sp, has_close, in_set)
bp = C.run_book(pp, fit['beta'], fund_row, rs_pnl, rf_pnl, rf_pnl.shape[0], d_start)
pm = C.period_mask(days, '2006-2019')
m, t, n = C.tstat(bp['gross'][pm])
say(f'3. placebo (iid noise s), 2006-2019: gross {1e4 * m:+.2f} bps/day (t {t:+.2f}), positions/day {bp["n"][pm].mean():.0f}')
(C.OUT / 'check_book.txt').write_text('\n'.join(LINES) + '\n')
