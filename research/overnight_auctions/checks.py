"""Overnight auctions: plumbing checks (no outcomes).

1. Timing: for sampled signal days t, every stock and fund close, high and low from day t on and every open from
   day t+1 on are replaced with garbage; the legs and S(:, t) are recomputed and must be bit-identical (same NaN
   pattern). The universe and the SPY beta for day t are recomputed from the export truncated before day t's close
   and must be identical. (The twin and its beta come from c2eval.twins at rebal[k] <= t-1, whose inputs end at that
   rebalance's close, per c2eval's docstring; the code asserts rebal[k] <= t-1.)
2. S against a direct computation from the store's prices for random (stock, day) pairs.

    python3 -I checks.py old      # -> out/checks_old.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import build as B  # noqa: E402
np = C.np
c2eval = C.c2eval

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
L = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    L.append(s)
    print(s, flush=True)


E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
days = E['days']
N, T = E['close'].shape
tw = c2eval.twins(E)
tw.pop('F')
rebal = tw['rebal']
P = B.load(tag)
c0, first_day = P['c0'], P['first_day']
S_store = c2eval.load_store(C.STORES[tag])
Ps = B.store_panels(S_store, E['ids'], days, c0)
Pf = B.store_panels(S_store, tw['fund'], days, c0)
spy_full = c2eval.store_panel(S_store, [int(tw['fund'][P['spy_row']])], days, 'close')[0]
rng = np.random.default_rng(11)

# 1. timing
cands = [t for t in range(first_day + 40, T - 2, 97)]
sample = sorted(rng.choice(cands, 12, replace=False).tolist())
all_ok = True
for t in sample:
    k = C.kidx(rebal, t - 1)
    assert rebal[k] <= t - 1
    j = t - c0

    def garble(Pp):
        Q = {f: Pp[f].copy() for f in Pp}
        for f in ('close', 'high', 'low'):
            Q[f][:, j:] *= rng.uniform(0.5, 1.5, Q[f][:, j:].shape).astype(np.float32)
        Q['open'][:, j + 1:] *= rng.uniform(0.5, 1.5, Q['open'][:, j + 1:].shape).astype(np.float32)
        return Q
    on_s, _, _, _ = B.legs(garble(Ps), first_day - c0)
    on_f, _, _, _ = B.legs(garble(Pf), first_day - c0)
    S2, Sr2 = B.signal(on_s, on_f, tw, P['uni'], c0, T, [k])
    a, b = P['S'][:, t], S2[:, t]
    same = np.array_equal(np.isfinite(a), np.isfinite(b)) and np.array_equal(a[np.isfinite(a)], b[np.isfinite(b)])
    ar, br = P['Sraw'][:, t], Sr2[:, t]
    same_r = np.array_equal(np.isfinite(ar), np.isfinite(br)) and np.array_equal(ar[np.isfinite(ar)], br[np.isfinite(br)])
    # universe and SPY beta from the export truncated before day t's close
    Ecut = {'close': E['close'][:, :t], 'dollar_volume': E['dollar_volume'][:, :t], 'tickers': E['tickers']}
    twc = {'rebal': rebal[:k + 1], 'fund_like': tw['fund_like'][:k + 1]}
    ucut, bcut = C.universe(Ecut, twc)
    same_u = np.array_equal(ucut[k], P['uni'][k]) and np.array_equal(bcut[k], P['big'][k])
    bs = B.spy_beta(E['close'][:, :t], spy_full[:t], rebal, [k])
    x, y = bs[k], P['bspy'][k]
    same_b = np.array_equal(np.isfinite(x), np.isfinite(y)) and np.array_equal(x[np.isfinite(x)], y[np.isfinite(y)])
    ok = same and same_r and same_u and same_b
    all_ok &= ok
    say(f'1. t = {c2eval.iso(days[t])} (rebalance {c2eval.iso(days[rebal[k]])}): finite S {int(np.isfinite(a).sum())}; '
        f'S identical {same}, S_raw identical {same_r}, universe identical {same_u}, SPY beta identical {same_b}')
say(f'1. timing check over {len(sample)} days: {"PASS" if all_ok else "FAIL"}')
# control: garbling day t's open (which S may use) must change S(:, t), so the check has power
t = sample[len(sample) // 2]
k = C.kidx(rebal, t - 1)
Q = {f: Ps[f].copy() for f in Ps}
Q['open'][:, t - c0] *= rng.uniform(0.9, 1.1, N).astype(np.float32)
on_s, _, _, _ = B.legs(Q, first_day - c0)
on_f, _, _, _ = B.legs(Pf, first_day - c0)
S2, _ = B.signal(on_s, on_f, tw, P['uni'], c0, T, [k])
f = np.isfinite(P['S'][:, t]) & np.isfinite(S2[:, t])
say(f'1. control: garbling day t\'s open ({c2eval.iso(days[t])}) changes S for {int((P["S"][f, t] != S2[f, t]).sum())} of {int(f.sum())} stocks')

# 2. direct computation of S for random stock-days
o, c = Ps['open'].astype(np.float64), Ps['close'].astype(np.float64)
fo, fc = Pf['open'].astype(np.float64), Pf['close'].astype(np.float64)
errs, nn, n_nan_agree = [], 0, 0
fin = np.argwhere(np.isfinite(P['S'][:, first_day + 30:T]))
pick = fin[rng.choice(len(fin), 300, replace=False)]
for i, tt in pick:
    t = int(tt) + first_day + 30
    k = C.kidx(rebal, t - 1)
    fj = tw['fund_row'][k, i]
    bt = float(tw['beta'][k, i])
    xs = []
    for d in range(t - 20, t + 1):
        jj = d - c0
        if jj < 1 or d < first_day:
            continue
        ps, pc = o[i, jj], c[i, jj - 1]
        qs, qc = fo[fj, jj], fc[fj, jj - 1]
        if not (np.isfinite(ps) and np.isfinite(pc) and ps > 0 and pc > 0 and np.isfinite(qs) and np.isfinite(qc) and qs > 0 and qc > 0):
            continue
        if Ps['open'][i, jj] == Ps['close'][i, jj - 1] or Pf['open'][fj, jj] == Pf['close'][fj, jj - 1]:
            continue
        a1, a2 = np.log(ps / pc), np.log(qs / qc)
        if abs(a1) > 0.5 or abs(a2) > 0.5:
            continue
        xs.append(a1 - bt * a2)
    direct = np.mean(xs) if len(xs) >= 17 else np.nan
    errs.append(abs(direct - float(P['S'][i, t])))
say(f'2. S vs direct computation on 300 random stock-days: max abs diff {np.nanmax(errs):.2e} (float32 storage)')
(C.OUT / f'checks_{tag}.txt').write_text('\n'.join(L) + '\n')
