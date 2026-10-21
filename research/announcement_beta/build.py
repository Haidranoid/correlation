"""Build the beta panel, SPY returns and the universe for one export.

    python3 -I -B build.py old|new     -> WORK/beta_{tag}.npy, WORK/meta_{tag}.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, EXPORTS, c2eval, log, log_returns, np, rolling_beta, spy_close, universe, L  # noqa: E402


def build(tag):
    E = c2eval.load_export(EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    log(tag, 'N', N, 'T', T, c2eval.iso(days[0]), '..', c2eval.iso(days[-1]))
    sc = spy_close(tag, days)
    m = log_returns(sc[None, :])[0]
    first = int(np.nonzero(np.isfinite(sc))[0][0])
    log('SPY first store close', c2eval.iso(days[first]), '; days with a return', int(np.isfinite(m).sum()))
    beta = rolling_beta(E['close'], m)
    np.save(WORK / f'beta_{tag}.npy', beta)
    fin = np.isfinite(beta).sum(0)
    d0 = int(np.nonzero(fin > 0)[0][0])
    log('beta: first day with any beta', c2eval.iso(days[d0]), '; finite stock-days', int(fin.sum()))
    # self-check: one stock, one day, against a direct regression
    rng = np.random.default_rng(0)
    ok = np.argwhere(np.isfinite(beta[:, -300:]))
    for j, dd in ok[rng.choice(len(ok), 5, replace=False)]:
        d = T - 300 + dd
        r = log_returns(np.asarray(E['close'][j:j + 1], np.float64))[0][d - L:d]
        x = m[d - L:d]
        k = np.isfinite(r) & np.isfinite(x)
        b = np.polyfit(x[k], r[k], 1)[0]
        log(f'  check stock {j} day {c2eval.iso(days[d])}: rolling {beta[j, d]:.6f} direct {b:.6f} (n {k.sum()})')
    del beta
    tw = c2eval.twins(E)
    uni = universe(E, tw)
    sz = uni.sum(1)
    log('universe sizes: first nonempty rebalance', c2eval.iso(days[tw['rebal'][np.nonzero(sz)[0][0]]]),
        'median', int(np.median(sz[sz > 0])), 'min', int(sz[sz > 0].min()), 'max', int(sz.max()))
    np.savez(WORK / f'meta_{tag}.npz', days=days, m=m, spy_close=sc, rebal=tw['rebal'], uni=uni)
    log('saved')


if __name__ == '__main__':
    build(sys.argv[1])
