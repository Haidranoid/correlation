"""Check the vectorized simulation against a plain scalar loop, pair by pair, and each trade's summed daily P&L against
its buy-and-hold leg returns from prices (the two must agree). Also re-checks the SSD top 20 against brute force at
a few formations on a 150-stock subset.

    python3 -I -B check_sim.py old export nodiv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np, log  # noqa: E402

tag, source, div = sys.argv[1:4]
U = np.load(C.WORK / f'universe_{tag}.npz')
R = np.load(C.WORK / f'run_{tag}_{source}_{div}_all.npz')
E = c2eval.load_export(C.EXPORTS[tag], arrays=('close',))
days = E['days']
N, T = E['close'].shape
c = C.gapfill(C.source_close(tag, source, E), list(U['gap_days']))
fin = np.isfinite(c)
divfix = None
if div == 'div':
    sys.path.insert(0, '/home/bhoult/programming/correlation/research/published_data')
    import pubdata  # noqa: E402
    divfix = pubdata.div_fix(tag)
L = C.log_tr(c, days, divfix)
rebal = R['rebal']
bps = C.SIDE_BPS * 1e-4


def scalar(i, j, t0, sigma):
    a = t0 - C.FORM + 1
    hmax = min(C.H, T - 1 - t0)
    A = [np.exp(L[i, t0 + h] - L[i, a]) for h in range(hmax + 1)]
    B = [np.exp(L[j, t0 + h] - L[j, a]) for h in range(hmax + 1)]
    oa = [bool(fin[i, t0 + h]) for h in range(hmax + 1)]
    ob = [bool(fin[j, t0 + h]) for h in range(hmax + 1)]
    la = max(h for h in range(hmax + 1) if oa[h])
    lb = max(h for h in range(hmax + 1) if ob[h])
    endp = min(la, lb, hmax)
    pnl = np.zeros(C.H + 1)
    cost = np.zeros(C.H + 1)
    state, pending, vA, vB = 0, None, 0.0, 0.0
    trades = []   # (pos, open h, close h)
    for h in range(1, endp + 1):
        if state:
            ra, rb = A[h] / A[h - 1] - 1, B[h] / B[h - 1] - 1
            pnl[h] = state * (vA * ra - vB * rb)
            vA *= 1 + ra
            vB *= 1 + rb
        if pending == 'close':
            cost[h] += bps * (vA + vB)
            trades.append((state, oh, h))
            state = 0
        elif pending in (1, -1):
            state, vA, vB, oh = pending, 1.0, 1.0, h
            cost[h] += 2 * bps
        pending = None
        if h == endp and state:
            cost[h] += bps * (vA + vB)
            trades.append((state, oh, h))
            state = 0
        if h < endp and oa[h] and ob[h]:
            D = A[h] - B[h]
            if state and ((state > 0 and D >= 0) or (state < 0 and D <= 0)):
                pending = 'close'
            elif state == 0 and abs(D) > 2 * sigma and h + 1 < endp:
                pending = -1 if D > 0 else 1
    # trade P&L from prices: long leg A/A_o - 1, short leg B/B_o - 1
    tp = sum(s * ((A[ch] / A[oh] - 1) - (B[ch] / B[oh] - 1)) for s, oh, ch in trades)
    return pnl, cost, tp, len(trades)


ks = np.nonzero(R['valid'])[0]
sel = ks[np.linspace(0, len(ks) - 1, 12).astype(int)]
md_pnl = md_cost = md_trade = 0.0
nt_diff = 0
for k in sel:
    t0 = rebal[k]
    for p in range(C.NPAIRS):
        i, j = R['pairs'][k, p]
        pnl, cost, tp, nt = scalar(i, j, t0, R['sigma'][k, p])
        mp = C.month_sums(pnl[None, :])[0]
        mc = C.month_sums(cost[None, :])[0]
        md_pnl = max(md_pnl, float(np.abs(mp - R['pnl'][k, p]).max()))
        md_cost = max(md_cost, float(np.abs(mc - R['cost'][k, p]).max()))
        md_trade = max(md_trade, abs(pnl.sum() - tp))
        nt_diff += int(nt != R['n_open'][k, p])
    # sigma recomputed from the formation paths
    a = t0 - C.FORM + 1
    i, j = R['pairs'][k, 0]
    Pi = np.exp(L[i, a:t0 + 1] - L[i, a])
    Pj = np.exp(L[j, a:t0 + 1] - L[j, a])
    assert abs((Pi - Pj).std() - R['sigma'][k, 0]) < 1e-9
    assert abs(((Pi - Pj) ** 2).sum() - R['ssd'][k, 0]) < 1e-8
log(f'{len(sel)} formations x 20 pairs: max |month P&L diff| {md_pnl:.2e}, max |month cost diff| {md_cost:.2e}, '
    f'max |daily-sum vs trade-from-prices| {md_trade:.2e}, pairs with a different trade count {nt_diff}')
# SSD brute force at several formations (150-stock subsets of the universe)
for k in sel[::3]:
    t0 = rebal[k]
    a = t0 - C.FORM + 1
    idx = np.nonzero(U['uni'][k] & U['complete'][k])[0]
    rng = np.random.default_rng(int(k))
    sub = np.sort(rng.choice(idx, min(150, len(idx)), replace=False))
    P = np.exp(L[sub, a:t0 + 1] - L[sub, a:a + 1])
    bi, bj, bs = C.brute_ssd(P)
    gi, gj, gs = C.top_pairs(P)
    log(f'k {k} {c2eval.iso(days[t0])}: brute force vs Gram top 20:',
        'same' if np.array_equal(bi, gi) and np.array_equal(bj, gj) else 'DIFFERENT', f'max SSD diff {np.abs(bs - gs).max():.1e}')
