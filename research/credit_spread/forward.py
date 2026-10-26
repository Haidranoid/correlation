"""Forward confirmation (plan, "Holdout and forward confirmation"): run each later download through the frozen
procedure, keep first-seen values only, and score P1 and S2 at 12 and 36 months.

    python3 -I -B forward.py EXPORT_DIR STORE_DIR FRED_DIR YAHOO_DIR [SANDBOX_DIR]

EXPORT_DIR / STORE_DIR: a later Stooq download built and exported the way the 2026-10-19 one was (it must continue
the new export exactly; inputs.py asserts it). FRED_DIR: fred_baa10y.csv, fred_aaa10y.csv, fred_dgs10.csv and
fred_vixcls.csv downloaded by hand (date,value). YAHOO_DIR: SPY.csv, LQD.csv and IEF.csv in pubdata's format
(date,close,adj_close,dividends,splits), e.g. from research/published_data/fetch_yahoo.py. The frozen scripts run with
tag `fwd`. Then, for every axis day after 2026-10-19 not yet in the log, one row is appended to
~/.local/share/correlation2/experiments/credit_spread/forward_log.jsonl: SPY's close, dS, R, both models' forecasts and,
on calendar days, S2's two weights, as first seen. Later revisions are ignored. A calendar decision counts only if it
is this download's last or second-to-last day (the trade is at the close of t + 1), so a missed download leaves a
logged gap, never a decision filled in afterwards. Scores use first-seen SPY closes.
Not yet run on real data: the first download after 2026-10-19 is still to come.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, '/home/bhoult/programming/correlation/research/credit_spread')
import numpy as np  # noqa: E402

import cs_common as K  # noqa: E402

LOG = K.WORK / 'forward_log.jsonl'
REPORT = K.HERE / 'forward_report.txt'
SCRIPTS = ('inputs.py', 'forecast.py', 'timing.py')


def run_frozen(export_dir, store_dir, fred_dir, yahoo_dir, sandbox):
    env = dict(os.environ, C2_FWD_EXPORT=str(export_dir), C2_FWD_STORE=str(store_dir), C2_FWD_FRED_DIR=str(fred_dir),
               C2_FWD_YAHOO_DIR=str(yahoo_dir), C2_FWD_SANDBOX=str(sandbox))
    for s in SCRIPTS:
        r = subprocess.run([sys.executable, '-I', '-B', str(K.HERE / s), 'fwd'], env=env, capture_output=True, text=True)
        (K.WORK / f'{s[:-3]}_fwd.log').write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            if s in ('inputs.py', 'forecast.py'):
                raise RuntimeError(f'{s} fwd failed: {r.stderr[-2000:]}')
            print(f'{s} fwd did not finish (too few forward days yet?); see its log', flush=True)


def append_log():
    z = K.load_inputs('fwd')
    fc = np.load(K.WORK / 'fc_fwd.npz')
    dates = K.iso_days(z['days'])
    T = len(dates)
    off = int(z['offset'])
    cal = set(int(t) + off for t in np.arange(252, T - off, K.H))
    months = {}
    mp = K.HERE / 'months_fwd.tsv'
    if mp.exists():
        rows = [l.split('\t') for l in mp.read_text().splitlines()]
        h = rows[0]
        for r in rows[1:]:
            months[r[0]] = {'w_B': float(r[h.index('w_B')]), 'w_spread': float(r[h.index('w_spread')]), 'warning': int(r[h.index('warning')])}
    seen = set()
    if LOG.exists():
        seen = {json.loads(l)['date'] for l in LOG.read_text().splitlines() if l.strip()}
    with open(LOG, 'a') as f:
        for t in range(T):
            if dates[t] < K.FWD[0] or dates[t] in seen:
                continue
            row = {'date': dates[t], 'spy': float(z['P'][t]), 'dS': float(z['dS1'][t]), 'R': float(z['R'][t]),
                   'yhat_base': float(fc['yhat_base'][t]), 'yhat_spread': float(fc['yhat_spread'][t]),
                   'calendar_day': t in cal, 'live': t >= T - 2, 'first_seen_in': dates[-1]}
            if t in cal and dates[t] in months:
                row |= months[dates[t]]
            f.write(json.dumps(row) + '\n')


def score():
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()] if LOG.exists() else []
    if not rows:
        return 'no forward rows yet'
    lp = np.log(np.array([r['spy'] for r in rows]))
    n = len(rows)
    y = np.full(n, np.nan)
    y[:n - 1 - K.H] = lp[1 + K.H:] - lp[1:n - K.H]
    fb = np.array([r['yhat_base'] for r in rows])
    fs = np.array([r['yhat_spread'] for r in rows])
    ok = np.isfinite(y) & np.isfinite(fb) & np.isfinite(fs)
    L = [f'Forward log: {n} days {rows[0]["date"]} .. {rows[-1]["date"]}; {int(ok.sum())} origins with a complete 21-day target']
    if ok.sum() >= 252:
        eb, es = y[ok] - fb[ok], y[ok] - fs[ok]
        f = eb ** 2 - es ** 2 + (fb[ok] - fs[ok]) ** 2
        dr = K.block_mean_draws(f - f.mean())
        L.append(f'P1: 1 - MSE ratio {100 * (1 - np.mean(es ** 2) / np.mean(eb ** 2)):+.2f}%, CW t {K.nw_t(f, 2 * K.H):+.2f}, '
                 f'bootstrap p {np.mean(dr >= f.mean()):.4f}')
        L.append('Report at 12 months; judge P1 and S2 at 36 (756 scored origins).' if ok.sum() < 756 else 'Judge now: 36 months scored.')
    live = [i for i, r in enumerate(rows) if r['calendar_day'] and r['live'] and 'w_B' in r]
    gaps = sum(1 for r in rows if r['calendar_day'] and not r['live'])
    L.append(f'S2: {len(live)} live decision dates; {gaps} calendar dates missed (gaps)')
    done = [i for i in live if i + 1 + K.H < n]
    if done:
        gs = [rows[i]['w_spread'] * (np.exp(lp[i + 1 + K.H] - lp[i + 1]) - 1) for i in done]
        gb = [rows[i]['w_B'] * (np.exp(lp[i + 1 + K.H] - lp[i + 1]) - 1) for i in done]
        L.append(f'S2, {len(done)} completed live months (before costs): mean monthly return spread rule {100 * np.mean(gs):+.2f}%, '
                 f'rule B {100 * np.mean(gb):+.2f}%; warnings {sum(rows[i]["warning"] for i in done)}')
    return '\n'.join(L)


if __name__ == '__main__':
    if len(sys.argv) >= 5:
        exp, store, fred, yahoo = sys.argv[1:5]
        sandbox = sys.argv[5] if len(sys.argv) > 5 else str(K.Path(exp).parent.parent)
        run_frozen(exp, store, fred, yahoo, sandbox)
        append_log()
    txt = score()
    print(txt)
    REPORT.write_text(txt + '\n')
