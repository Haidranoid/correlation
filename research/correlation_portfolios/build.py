"""Build the five portfolios at every rebalance and their daily returns (the .md, "Rules" and "Decisions").

    python3 -I build.py old      # the judge's export through 2025-06-27: every rebalance from 2004-12-31
    python3 -I build.py new      # the new export: rebalances from 2024-06-01 on (only returns from 2025-06-30 are used)

At the close of each rebalance day t0 (the twins' calendar, export days 252 + 21k):
  - universe: the 500 largest of the judge-like liquid set (night_and_day's rule), minus c2eval.excluded;
  - window: daily log returns of export closes over t0-251..t0; a stock needs >= 227 of the 252; each row is
    demeaned over its finite values and missing values are set to 0;
  - EW, IV (1/sigma), MV-LW (Ledoit-Wolf 2004 constant-correlation covariance, 0 <= w <= 2%), MV-TW (SPY + twin-fund
    factors, 0 <= w <= 2%), HRP (López de Prado 2016 on the Ledoit-Wolf covariance, no cap);
  - held t0+1..t1 (t1 = next rebalance) with weights drifting with forward-filled export closes; turnover
    sum |w_new - w_drifted| charged at 5 bps (per side) on day t0+1.
Writes WORK/series_<tag>.npz and a log line per rebalance.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CAP, COST_BPS, DATA_W, EXPORTS, MIN_RET, PORTS, STORES, WIN, WORK, c2eval, demean_fill,  # noqa: E402
                    ffill_rows, log, np, rf_daily, spy_close, universe, window_logret)
import estimators as est  # noqa: E402


def run(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    T = len(days)
    close = E['close']
    tw = c2eval.twins(E)
    rebal = tw['rebal']
    start = DATA_W if tag == 'old' else c2eval.day_index(days, '2024-06-01')
    ks = [k for k in range(len(rebal)) if rebal[k] >= max(start, DATA_W - 1) and rebal[k] < T - 1]
    log(tag, 'N', close.shape[0], 'T', T, 'rebalances', len(ks), c2eval.iso(days[rebal[ks[0]]]), '..',
        c2eval.iso(days[rebal[ks[-1]]]))
    uni, big = universe(E, tw, set(ks))
    log('universe done; 500 largest sizes', int(big[ks].sum(1).min()), '..', int(big[ks].sum(1).max()))
    spy = spy_close(days, STORES[tag])
    with np.errstate(all='ignore'):
        m_all = np.full(T, np.nan)
        m_all[1:] = np.log(spy[1:] / spy[:-1])
    m_all[~np.isfinite(m_all)] = np.nan
    rf = rf_daily(days)
    Fclose = tw['F']['close']

    P = len(PORTS)
    gross = np.full((P, T), np.nan)
    cost = np.zeros((P, T))
    rec = {k: [] for k in ('t0', 'n', 'n_spy', 'delta', 'rbar', 'n_twin', 'n_funds', 'n_dead', 'pg_iters_lw', 'pg_iters_tw',
                           'method_lw', 'method_tw', 'kkt_lw', 'kkt_tw', 'dropped')}
    turn, names, beta_ea, wmax = [], [], [], []
    flat_idx, flat_w, offs = [], [], [0]
    prev = [dict() for _ in range(P)]
    for k in ks:
        t0 = int(rebal[k])
        t1 = int(min(rebal[k + 1], T - 1)) if k + 1 < len(rebal) else T - 1
        sel = np.nonzero(big[k])[0]
        R = window_logret(np.asarray(close[sel, t0 - WIN:t0 + 1], np.float64))       # returns on t0-251..t0
        okr = np.isfinite(R).sum(1) >= MIN_RET
        dropped = int((~okr).sum())
        sel, R = sel[okr], R[okr]
        n = len(sel)
        X = demean_fill(R)
        Sig, delta, rbar, S = est.ledoit_wolf_cc(X)
        sd = np.sqrt(np.diag(S))
        W = np.zeros((P, n))
        W[0] = 1.0 / n
        W[1] = (1 / sd) / (1 / sd).sum()
        W[2], info_lw = est.minvar_capped(Sig, CAP)
        # MV-TW
        mw = demean_fill(m_all[None, t0 - WIN + 1:t0 + 1])[0]
        frow = tw['fund_row'][k, sel]
        urows = np.unique(frow[frow >= 0])
        fc = np.asarray(Fclose[urows, t0 - WIN:t0 + 1], np.float64)
        G = window_logret(fc)
        G[np.abs(G) > 0.5] = np.nan                    # c2eval's rule for fund returns (unrepaired store prices)
        G = demean_fill(G)
        twin_of = np.where(frow >= 0, np.searchsorted(urows, frow), -1)
        Sig_tw, info_tw = est.twin_cov(X, mw, G, twin_of)
        assert np.isfinite(Sig_tw).all(), (tag, t0)
        n_spy = int(np.isfinite(m_all[t0 - WIN + 1:t0 + 1]).sum())
        W[3], info_tw_qp = est.minvar_capped(Sig_tw, CAP)
        W[4] = est.hrp(Sig)[0]
        assert np.allclose(W.sum(1), 1.0, atol=1e-12) and (W >= 0).all()
        # holding period
        Pw = ffill_rows(np.asarray(close[sel, t0 - 300:t1 + 1], np.float64))[:, 300:]
        assert np.isfinite(Pw[:, 0]).all(), (tag, t0)
        V = Pw / Pw[:, :1]
        assert np.isfinite(V).all(), (tag, t0)
        tr = np.zeros(P)
        for p in range(P):
            val = W[p] @ V
            gross[p, t0 + 1:t1 + 1] = val[1:] / val[:-1] - 1
            new = dict(zip(sel.tolist(), W[p].tolist()))
            tr[p] = sum(abs(new.get(j, 0.0) - prev[p].get(j, 0.0)) for j in set(new) | set(prev[p]))
            cost[p, t0 + 1] = COST_BPS / 1e4 * tr[p]
            end = W[p] * V[:, -1] / val[-1]
            prev[p] = dict(zip(sel.tolist(), end.tolist()))
        turn.append(tr)
        names.append((W > 1e-3).sum(1))
        beta_ea.append(W @ info_tw['b_m'])
        wmax.append(W.max(1))
        flat_idx.append(sel)
        flat_w.append(W.T)
        offs.append(offs[-1] + n)
        for key, val in (('t0', t0), ('n', n), ('n_spy', n_spy), ('delta', delta), ('rbar', rbar), ('n_twin', info_tw['n_twin']),
                         ('n_funds', info_tw['n_funds']), ('n_dead', info_tw['n_dead']),
                         ('pg_iters_lw', info_lw['pg_iters']), ('pg_iters_tw', info_tw_qp['pg_iters']),
                         ('method_lw', info_lw['method']), ('method_tw', info_tw_qp['method']),
                         ('kkt_lw', max(info_lw['kkt_primal'], info_lw['kkt_dual'])),
                         ('kkt_tw', max(info_tw_qp['kkt_primal'], info_tw_qp['kkt_dual'])), ('dropped', dropped)):
            rec[key].append(val)
        log(f'{c2eval.iso(days[t0])} n={n} drop={dropped} spy={n_spy} delta={delta:.3f} rbar={rbar:.3f} twins={info_tw["n_twin"]} '
            f'funds={info_tw["n_funds"]} | names>0.1%: ' + ' '.join(f'{PORTS[p]}={names[-1][p]}' for p in range(P))
            + f' | turnover ' + ' '.join(f'{x:.2f}' for x in tr)
            + f' | qp {info_lw["method"]}/{info_lw["pg_iters"]} kkt {rec["kkt_lw"][-1]:.0e}, '
              f'{info_tw_qp["method"]}/{info_tw_qp["pg_iters"]} kkt {rec["kkt_tw"][-1]:.0e}')
    net = gross - np.where(np.isfinite(gross), cost, 0.0)
    out = {'days': days, 'rf': rf, 'spy': spy, 'gross': gross, 'cost': cost, 'net': net, 'ports': np.array(PORTS),
           'turn': np.array(turn), 'names': np.array(names), 'beta_ea': np.array(beta_ea), 'wmax': np.array(wmax),
           'w_idx': np.concatenate(flat_idx), 'w': np.concatenate(flat_w), 'w_offs': np.array(offs),
           'tickers': E['tickers']}
    for key, v in rec.items():
        out[key] = np.array(v)
    np.savez(WORK / f'series_{tag}.npz', **out)
    log('saved', WORK / f'series_{tag}.npz')


if __name__ == '__main__':
    run(sys.argv[1])
