"""The daily inputs on the day axis, and their coverage. No outcome (no future return) is computed here.

    python3 -I -B inputs.py old|new|fwd      -> ~/.local/share/correlation2/experiments/credit_spread/inputs_<tag>.npz,
                                                inputs_<tag>.txt here

Axis: Yahoo SPY's trading days, 1993-01-29 .. the tag's export's last day (asserted equal to the export's days from
2001-12-31). Every series is cut at that day. FRED values (BAA10Y, AAA10Y, DGS10, VIXCLS) are carried over at most
one trading day (bond holidays; VIX's two missing days) and never past the file's last date.
"""
import json
import sys
import time

sys.path.insert(0, '/home/bhoult/programming/correlation/research/credit_spread')
import numpy as np  # noqa: E402

import cs_common as K  # noqa: E402

t_start = time.time()
tag = sys.argv[1]
assert tag in K.TAGS
edays = K.export_days(tag)
cut = K.iso(edays[-1])
if tag == 'old':
    assert cut == K.OLD_LAST
if tag == 'new':
    assert cut == K.NEW_LAST
info = {'tag': tag, 'cut': cut}

# ---- The axis: Yahoo SPY's days through the export's last day ---------------------------------------------------
yd, yv = K.read_yahoo('SPY', tag, cut)
ok = np.isfinite(yv) & (yv > 0)
axis = yd[ok]
P = yv[ok]
T = len(axis)
offset = int(np.searchsorted(axis, edays[0]))
assert np.array_equal(axis[offset:], edays), 'Yahoo SPY days differ from the export days'
dates = K.iso_days(axis)
info |= {'axis_first': dates[0], 'axis_last': dates[-1], 'axis_days': T, 'export_first_index': offset,
         'export_days': len(edays)}

# ---- The previous export must be continued exactly (new / fwd) --------------------------------------------------
if tag in ('new', 'fwd'):
    prev = {'new': 'old', 'fwd': 'new'}[tag]
    pdays = K.export_days(prev)
    info['prev_export_overlap_equal'] = bool(np.array_equal(edays[:len(pdays)], pdays))
    assert info['prev_export_overlap_equal'], 'the export does not continue the previous one'

# ---- FRED ---------------------------------------------------------------------------------------------------------
fred = {}
for nm in ('baa10y', 'aaa10y', 'dgs10', 'vixcls'):
    d, v = K.read_fred(nm, tag, cut)
    x, n_car = K.carry(d, v, axis)
    fred[nm] = x
    first = int(np.argmax(np.isfinite(x)))
    miss = [dates[t] for t in range(first, T) if not np.isfinite(x[t])]
    info[f'{nm}_rows'] = int(len(d))
    info[f'{nm}_last_date'] = K.iso(d[-1])
    info[f'{nm}_carried_days'] = n_car
    info[f'{nm}_missing_after_start'] = miss[:12]
    info[f'{nm}_n_missing_after_start'] = len(miss)
S = fred['baa10y']
A = fred['aaa10y']
G = fred['dgs10']
V = fred['vixcls']
D = S - A                                  # the default spread, DBAA - DAAA

# ---- Funds: Yahoo LQD, IEF; the store's SPY (reported repeat) and XLF (C3) ---------------------------------------
lq = K.exact(*K.read_yahoo('LQD', tag, cut), axis)
ie = K.exact(*K.read_yahoo('IEF', tag, cut), axis)
Pst, fill_spy = K.store_close(tag, 'SPY', axis)
X, fill_xlf = K.store_close(tag, 'XLF', axis)
info['store_spy_filled_days'] = fill_spy
info['store_xlf_filled_days'] = fill_xlf
info['lqd_first'] = dates[int(np.argmax(np.isfinite(lq)))]
info['ief_first'] = dates[int(np.argmax(np.isfinite(ie)))]
info['xlf_first'] = dates[int(np.argmax(np.isfinite(X)))]

# data check: Yahoo vs store SPY daily log returns where both exist
ry = np.diff(np.log(P), prepend=np.nan)
rs = np.diff(np.log(Pst), prepend=np.nan)
okr = np.isfinite(ry) & np.isfinite(rs)
dif = ry[okr] - rs[okr]
info['yahoo_vs_store_spy'] = {'days': int(okr.sum()), 'corr': float(np.corrcoef(ry[okr], rs[okr])[0, 1]),
                              'max_abs_diff': float(np.abs(dif).max()), 'days_abs_diff_gt_0.5pct': int((np.abs(dif) > 0.005).sum()),
                              'mean_diff_bps_per_day': float(1e4 * dif.mean())}
if tag in ('new', 'fwd'):
    prev = {'new': 'old', 'fwd': 'new'}[tag]
    zp = np.load(K.WORK / f'inputs_{prev}.npz')
    n0 = len(zp['days'])
    assert np.array_equal(zp['days'], axis[:n0])
    for nm, cur in (('xlf', X), ('spy_store', Pst), ('spy_yahoo', P), ('baa10y', S), ('vix', V)):
        old = zp[{'xlf': 'X', 'spy_store': 'Pst', 'spy_yahoo': 'P', 'baa10y': 'S', 'vix': 'V'}[nm]]
        both = np.isfinite(old) & np.isfinite(cur[:n0])
        rel = np.abs(cur[:n0][both] / old[both] - 1) if nm in ('xlf', 'spy_store', 'spy_yahoo') else np.abs(cur[:n0][both] - old[both])
        info[f'overlap_{nm}'] = {'days': int(both.sum()), 'max_diff': float(rel.max()), 'days_diff': int((rel > 1e-6).sum()),
                                 'missing_now_not_before': int((np.isfinite(old) & ~np.isfinite(cur[:n0])).sum())}

# ---- Derived regressors (all point in time at t) ----------------------------------------------------------------
lp = np.log(P)
r = np.diff(lp, prepend=np.nan)
r2 = np.maximum(r ** 2, K.FLOOR_R2)
t_idx = np.arange(T)


def wsum(x, k):
    """sum of x over the k days ending at t (NaN if any missing or before the start)."""
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x, nan=0.0))])
    cn = np.concatenate([[0], np.cumsum(~np.isfinite(x))])
    out = np.full(len(x), np.nan)
    out[k - 1:] = np.where(cn[k:] - cn[:-k] == 0, cs[k:] - cs[:-k], np.nan)
    return out


def lagdiff(x, lag, k):
    """x(t - lag) - x(t - lag - k)."""
    out = np.full(len(x), np.nan)
    out[lag + k:] = x[k:len(x) - lag] - x[:len(x) - lag - k]
    return out


def shift(x, lag):
    out = np.full(len(x), np.nan)
    out[lag:] = x[:len(x) - lag]
    return out


def har(logp):
    rr = np.diff(logp, prepend=np.nan)
    q = np.maximum(rr ** 2, K.FLOOR_R2)
    q[~np.isfinite(rr)] = np.nan
    return np.log(wsum(q, 1)), np.log(wsum(q, 5)), np.log(wsum(q, 22))


lrv1, lrv5, lrv22 = har(lp)
R = lagdiff(lp, 0, K.R_DAYS)                          # log P(t) - log P(t - 6)
R5 = lagdiff(lp, 0, 5)                                # C1's same-dated 5-day return
lvix = np.log(V)
out = {'days': axis, 'offset': offset, 'P': P, 'Pst': Pst, 'X': X, 'S': S, 'A': A, 'D': D, 'G': G, 'V': V,
       'lqd': lq, 'ief': ie, 'lrv1': lrv1, 'lrv5': lrv5, 'lrv22': lrv22, 'lvix': lvix, 'R': R, 'R5': R5}
# FRED's values have two decimals, so every change is a whole number of basis points. Float subtraction leaves noise
# of about 1e-16, which decided ties with S2's percentile threshold at random (found after the first run); the
# changes are rounded back to FRED's 0.01 grid.
for lag in (0, 1, 2):
    out[f'dS{lag}'] = np.round(lagdiff(S, lag, K.WEEK), 2)   # dS1 = S(t - 1) - S(t - 6), the plan's ΔS
    out[f'dD{lag}'] = np.round(lagdiff(D, lag, K.WEEK), 2)
out['dG1'] = np.round(lagdiff(G, 1, K.WEEK), 2)               # C4: DGS10(t - 1) - DGS10(t - 6)
out['L1'] = shift(S, 1)                               # C6: S(t - 1)
out['cETF'] = -(lagdiff(np.log(lq), 0, 5) - lagdiff(np.log(ie), 0, 5))   # C8: widening = LQD trails IEF
lx = np.log(X)
out['lrv1X'], out['lrv5X'], out['lrv22X'] = har(lx)
out['RX'] = lagdiff(lx, 0, K.R_DAYS)
# store-SPY versions (S2's reported repeat)
lps = np.log(Pst)
out['Rst'] = lagdiff(lps, 0, K.R_DAYS)
np.savez(K.WORK / f'inputs_{tag}.npz', **out)

# ---- Coverage (no outcomes) ---------------------------------------------------------------------------------------
cov = {}
for nm in ('lrv22', 'lvix', 'R', 'dS1', 'dD1', 'dG1', 'L1', 'cETF', 'lrv22X', 'RX'):
    x = out[nm]
    f = np.isfinite(x)
    cov[nm] = {'first': dates[int(np.argmax(f))], 'finite_days': int(f.sum()),
               'missing_after_first': int((~f[int(np.argmax(f)):]).sum())}
info['coverage'] = cov
both = np.isfinite(out['dS1'])
info['dS1_days_exactly_zero_share'] = float(np.mean(out['dS1'][both] == 0))
info['seconds'] = round(time.time() - t_start)
with open(K.HERE / f'inputs_{tag}.json', 'w') as f:
    json.dump(info, f, indent=1, default=float)
L = [f'inputs ({tag}): axis {dates[0]} .. {dates[-1]}, {T} days; export from axis day {offset} ({len(edays)} days, equal to the axis)']
for k, v in info.items():
    if k in ('tag', 'coverage'):
        continue
    L.append(f'  {k}: {v}')
L.append('  coverage (first finite day, finite days, missing after the first):')
for k, v in cov.items():
    L.append(f"    {k}: {v['first']}, {v['finite_days']}, {v['missing_after_first']}")
txt = '\n'.join(L)
print(txt)
(K.HERE / f'inputs_{tag}.txt').write_text(txt + '\n')
