"""Forward confirmation (plan, "Holdout and forward confirmation"): run each later download through the frozen
procedure, keep first-seen values only, and score P1 and T.

    python3 -I -B forward.py DOWNLOAD_DIR [STORE_DIR]
    python3 -I -B forward.py                 (score the log only)

DOWNLOAD_DIR holds a later download: SPY.csv and IWM.csv in pubdata's etf/ format (date, close, adj_close,
dividends, splits), fred_vixcls.csv, fred_vxncls.csv, fred_rvxcls.csv, fred_dgs3mo.csv (FRED's date,value CSVs) and
ff3_daily.csv (pubdata's format). STORE_DIR (optional) is a later Stooq store, for QQQ. The frozen forecast.py runs
with tag `fwd` on it (it checks that the forecasts over the previous run's span are unchanged and records any
difference). Then, for every day after 2026-10-19 not yet in the log, one row is appended to
~/.local/share/correlation2/experiments/variance_risk_premium/forward_log.jsonl, as first seen: SPY's daily log
return (from this download's own adjusted closes, so later rescaling of Yahoo's adjusted series can't shift it), the
risk-free rate, VIX, the VRP, both forecasts, and on calendar dates s2 and c. Later revisions are ignored. A calendar
date counts as a live decision only if it is this download's last or second-to-last day (the trade is at the close
of t + 1), so a missed download leaves a logged gap, never a decision filled in afterwards.
Not yet run on real data: no download after 2026-10-19 exists yet.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/variance_risk_premium')
import numpy as np  # noqa: E402

import vrp_common as V  # noqa: E402

LOG = V.WORK / 'forward_log.jsonl'
REPORT = V.HERE / 'forward_report.txt'


def run_frozen(download, store):
    env = dict(os.environ, VRP_FWD_DIR=str(download))
    if store:
        env['VRP_FWD_STORE'] = str(store)
    for s in ('forecast.py', 'timing.py'):
        r = subprocess.run([sys.executable, '-I', '-B', str(V.HERE / s), 'fwd'], env=env, capture_output=True, text=True)
        (V.WORK / f'{s[:-3]}_fwd.log').write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            if s == 'forecast.py':
                raise RuntimeError(f'{s} fwd failed: {r.stderr[-2000:]}')
            print(f'{s} fwd did not finish (too few forward days yet?); see its log', flush=True)
    return env


def append_log(env):
    os.environ.update({k: v for k, v in env.items() if k.startswith('VRP_FWD')})
    import importlib
    importlib.reload(V)
    D = V.load('fwd', verbose=False)
    S = V.build(D)
    fc = np.load(V.WORK / 'fc_fwd.npz')
    dates, T = D['dates'], D['T']
    cal = set(int(t) for t in D['cal'])
    seen = set()
    if LOG.exists():
        seen = {json.loads(l)['date'] for l in LOG.read_text().splitlines() if l.strip()}
    with open(LOG, 'a') as f:
        for t in range(T):
            if dates[t] < V.FWD[0] or dates[t] in seen:
                continue
            row = {'date': dates[t], 'r_spy': float(S['r'][t]), 'rf': float(D['rf'][t]), 'vix': float(D['vix'][t]),
                   'vrp': float(S['vrp'][t]), 'yhat_vrp': float(fc['yhat_vrp'][t]), 'yhat_mean': float(fc['yhat_mean'][t]),
                   'calendar_day': t in cal, 'live': t >= T - 2, 'first_seen_in': dates[-1]}
            if t in cal:
                prev = np.array([V.sigma2(S['r'], s) for s in range(max(V.RV_WIN, t - V.C_MAX), t)])
                prev = prev[np.isfinite(prev)]
                row['s2'] = V.sigma2(S['r'], t)
                row['c'] = float(np.median(prev)) if len(prev) >= V.C_MIN else float('nan')
            f.write(json.dumps(row) + '\n')


def score():
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []
    if not rows:
        return 'no forward rows yet'
    n = len(rows)
    ex = np.array([r['r_spy'] - np.log1p(r['rf']) for r in rows])
    y = np.full(n, np.nan)
    for t in range(n - V.H - 1):
        w = ex[t + 2:t + 2 + V.H]
        if np.isfinite(w).all():
            y[t] = w.sum()
    fv, fm = np.array([r['yhat_vrp'] for r in rows]), np.array([r['yhat_mean'] for r in rows])
    ok = np.isfinite(y) & np.isfinite(fv) & np.isfinite(fm)
    L = [f'Forward log: {n} days {rows[0]["date"]} .. {rows[-1]["date"]}; {int(ok.sum())} origins with a complete 63-day target']
    if ok.sum() >= 252:
        q = V.evaluate(y, fm, fv, ok)
        L.append(f"P1: R2_oos {100 * q['r2_oos']:+.2f}%, CW t {q['cw_t']:+.2f}, bootstrap p {q.get('cw_boot_p', float('nan')):.4f}")
    L.append('Report every 12 months; judge P1 and T at 60 months (about 20 independent quarters).')
    live = [r for r in rows if r['calendar_day'] and r['live']]
    gaps = [r['date'] for r in rows if r['calendar_day'] and not r['live']]
    L.append(f'T: {len(live)} live decision dates; {len(gaps)} calendar dates missed (gaps): {gaps}')
    if live:
        rets = {'vrp': [], 'vt': []}
        idx = {r['date']: i for i, r in enumerate(rows)}
        for r in live:
            i = idx[r['date']]
            if i + 1 + 21 >= n or not np.isfinite(r.get('c', np.nan)):
                continue
            hold = np.exp(np.sum([rows[j]['r_spy'] for j in range(i + 2, i + 23)])) - 1
            wvt = min(1.0, r['c'] / r['s2'])
            wv = wvt if r['yhat_mean'] <= 0 else min(1.0, max(0.0, (r['c'] / r['s2']) * r['yhat_vrp'] / r['yhat_mean']))
            rets['vt'].append(wvt * hold)
            rets['vrp'].append(wv * hold)
        if rets['vt']:
            L.append(f"T, completed live months {len(rets['vt'])}: mean monthly return VRP rule {100 * np.mean(rets['vrp']):+.2f}%, "
                     f"VT {100 * np.mean(rets['vt']):+.2f}% (before costs)")
    return '\n'.join(L)


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        env = run_frozen(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        append_log(env)
    txt = score()
    print(txt)
    REPORT.write_text(txt + '\n')
