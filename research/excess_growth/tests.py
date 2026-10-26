"""Engine checks, run before any result (writes tests_out.txt). Synthetic data only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg_common as C  # noqa: E402
np = C.np
OUT = []


def say(s):
    OUT.append(s)
    print(s, flush=True)


rng = np.random.default_rng(1)

# 1. The identity: delay 0, daily, no costs, everything tradeable -> log V - mean g = sum_d [log mean e^l - mean l].
worst = 0.0
for case in range(20):
    K, n = int(rng.integers(2, 60)), int(rng.integers(5, 300))
    L = rng.normal(0.0003, 0.02, (K, n + 1)) + rng.normal(0, 0.01, (1, n + 1))
    R = np.expm1(L)
    H = {'R': R, 'trade': np.ones_like(R, bool), 'ends': np.full(K, n + 1), 'n': n}
    g, t, w0, we = C.run_period(H, 'd', 0)
    lhs = np.log1p(g[0]).sum() - L[:, 1:].sum(1).mean()
    rhs = sum(np.log(np.exp(L[:, d]).mean()) - L[:, d].mean() for d in range(1, n + 1))
    worst = max(worst, abs(lhs - rhs))
say(f'1. EG identity (delay 0, daily), 20 random cases: worst |engine - formula| = {worst:.2e}')
assert worst < 1e-11

# 2. Two stocks, known answer: A doubles then halves, B flat. Rebalanced at offset 1: 1.5 -> 1.125; held: 1.0.
R = np.array([[0.0, 1.0, -0.5], [0.0, 0.0, 0.0]])
H = {'R': R, 'trade': np.ones_like(R, bool), 'ends': np.full(2, 3), 'n': 2}
gR, tR, _, _ = C.run_period(H, 'd', 0)
gB, tB, _, _ = C.run_period(H, 'B', 0)
vR, vB = np.prod(1 + gR[0]), np.prod(1 + gB[0])
say(f'2. Two-stock example: rebalanced {vR:.6f} (expected 1.125), held {vB:.6f} (expected 1.0), '
    f'rebalancing turnover {tR[0].sum():.6f} (expected 1/3: sell 0.25 of A and buy 0.25 of B, sum |trade| 0.5 '
    f'over the value 1.5)')
assert abs(vR - 1.125) < 1e-12 and abs(vB - 1.0) < 1e-12 and abs(tR[0].sum() - 0.5 / 1.5) < 1e-12

# 3. Buy-and-hold never trades except forced sales; 4. turnover and holdings against a plain loop (delay 1, missing
# days, untradeable days, a forced sale).
K, n = 7, 60
L = rng.normal(0, 0.02, (K, n + 1))
R = np.expm1(L)
trade = rng.random((K, n + 1)) > 0.1
R[~trade & (rng.random((K, n + 1)) > 0.5)] = 0.0          # some untradeable days are missing closes (return 0)
ends = np.full(K, n + 1)
ends[3] = 25
for j in range(K):
    R[j, ends[j] + 1:] = 0.0 if ends[j] <= n else R[j, ends[j] + 1:]
H = {'R': R, 'trade': trade, 'ends': ends, 'n': n}
gB, tB, _, _ = C.run_period(H, 'B', 1)
nz = np.nonzero(tB[0])[0]
say(f'3. Buy-and-hold turnover nonzero only on offsets {[int(x) for x in nz + 1]} (the forced sale is at offset 25)')
assert list(nz + 1) == [25]


def loop(H, freq, delay):
    R, trade, ends, n = H['R'], H['trade'], H['ends'], H['n']
    K = R.shape[0]
    w = (1 + R[:, 0]) / (1 + R[:, 0]).sum() if delay == 1 else np.full(K, 1.0 / K)
    h = list(w)
    alive = [True] * K
    fills = set(C.fills_for(freq, n, delay))
    out_g, out_t = [], []
    V = sum(h)
    for d in range(1, n + 1):
        for j in range(K):
            h[j] *= 1 + R[j, d]
        Vn = sum(h)
        out_g.append(Vn / V - 1)
        tt = 0.0
        for j in range(K):
            if alive[j] and ends[j] == d:
                p = h[j]
                h[j] = 0.0
                rest = Vn - p
                for i in range(K):
                    h[i] *= Vn / rest
                tt += 2 * p / Vn
                alive[j] = False
        if d in fills:
            trd = [alive[j] and trade[j, d] for j in range(K)]
            keep = sum(h[j] for j in range(K) if alive[j] and not trd[j])
            tau = [(1 + R[j, d]) if delay == 1 else 1.0 for j in range(K)]
            ts = sum(tau[j] for j in range(K) if trd[j])
            for j in range(K):
                if trd[j]:
                    new = tau[j] / ts * (Vn - keep)
                    tt += abs(new - h[j]) / Vn
                    h[j] = new
        out_t.append(tt)
        V = Vn
    return np.array(out_g), np.array(out_t)


worst = 0.0
for freq in ('d', 'm', 'B'):
    for delay in (0, 1):
        g, t, _, _ = C.run_period(H, freq, delay)
        lg, lt = loop(H, None if freq == 'B' else freq, delay)
        worst = max(worst, np.abs(g[0] - lg).max(), np.abs(t[0] - lt).max())
say(f'4. Engine against a plain loop (daily, monthly, hold; delay 0 and 1; missing and untradeable days; a forced '
    f'sale): worst difference {worst:.2e}')
assert worst < 1e-12

# 5. Independent returns with equal means: mean end wealth of rebalanced and held agree (Stage 1, item 4).
B, K, n = 40000, 5, 30
mu, sd = 0.001, 0.03
common = rng.normal(0, 0.015, (B, 1, n + 1))
Rb = mu + sd * rng.standard_normal((B, K, n + 1)) + common
w0 = np.full((B, K), 1.0 / K)
gR, _, _ = C.simulate(n, lambda d: Rb[:, :, d], w0, C.fills_for('d', n, 0), 0)
gB, _, _ = C.simulate(n, lambda d: Rb[:, :, d], w0, (), 0)
vR, vB = np.prod(1 + gR, 1), np.prod(1 + gB, 1)
diff = vR - vB
z = diff.mean() / (diff.std(ddof=1) / np.sqrt(B))
say(f'5. iid returns, equal means ({B} draws, 5 stocks, 30 days): mean wealth rebalanced {vR.mean():.5f}, held '
    f'{vB.mean():.5f}, theory {(1 + mu) ** n:.5f}; difference z = {z:+.2f}. Mean log gap {np.log(vR / vB).mean():+.2e}')
assert abs(z) < 4
# and with unequal means, held is ahead in mean wealth
mus = np.linspace(-0.002, 0.004, K)[None, :, None]
Rb2 = mus + sd * rng.standard_normal((B, K, n + 1)) + common
gR, _, _ = C.simulate(n, lambda d: Rb2[:, :, d], w0, C.fills_for('d', n, 0), 0)
gB, _, _ = C.simulate(n, lambda d: Rb2[:, :, d], w0, (), 0)
vR, vB = np.prod(1 + gR, 1), np.prod(1 + gB, 1)
theory = (1 + mus[0, :, 0]).__pow__(n).mean() - (1 + mus.mean()) ** n
say(f'   unequal means: mean wealth held - rebalanced {vB.mean() - vR.mean():+.5f} (theory {theory:+.5f})')

# 6. Chain: the reconstitution turnover is sum |w_new - w_old| over the union; the first purchase is 1.0.
per = [{'rows': np.array([1, 2]), 'gross': np.zeros(3), 'turn': np.zeros(3), 'w0': np.array([.5, .5]),
        'wend': np.array([.6, .4]), 'days': np.arange(3)},
       {'rows': np.array([2, 3]), 'gross': np.zeros(2), 'turn': np.zeros(2), 'w0': np.array([.5, .5]),
        'wend': np.array([.5, .5]), 'days': np.arange(3, 5)}]
_, _, t, _ = C.chain(per)
say(f'6. Chain turnover: {t.tolist()} (expected [1, 0, 1.2, 0, 0]: 0.6 + |0.5 - 0.4| + 0.5)')
assert np.allclose(t, [1, 0, 1.2, 0, 0])

# 7. The second-order lag decomposition: -sum_l A_l = 1/2 (sum x^2 - (sum x)^2) per stock, via FFT.
x = rng.normal(0, 0.02, (30, 200))
lhs = 0.5 * ((x ** 2).sum(1) - x.sum(1) ** 2)
F = np.fft.rfft(x, 400, axis=1)
ac = np.fft.irfft(F * np.conj(F), 400, axis=1)[:, :200]
rhs = -ac[:, 1:].sum(1)
say(f'7. Lag decomposition identity: worst |difference| {np.abs(lhs - rhs).max():.2e}')
assert np.abs(lhs - rhs).max() < 1e-12

# 8. Statistics: NW t and the bootstrap on noise and on a shifted series.
z = rng.normal(0, 1, 3000)
p0, _, _ = C.boot_p(z)
p1, _, _ = C.boot_p(z + 0.1)
say(f'8. Noise: NW t {C.nw_t(z):+.2f} (plain t {z.mean() / z.std(ddof=1) * np.sqrt(len(z)):+.2f}), bootstrap p '
    f'{p0:.3f}; noise + 0.1 sd: NW t {C.nw_t(z + 0.1):+.2f}, bootstrap p {p1:.4f}, normal p {C.normal_p(C.nw_t(z + 0.1)):.4f}')
print('Holm check:', C.holm([0.01, 0.02]), C.holm([0.02, 0.01]), C.holm([0.012, 0.001]), C.holm([0.001, 0.03]))
assert C.holm([0.01, 0.02]) == [True, True] and C.holm([0.012, 0.001]) == [True, True] and C.holm([0.001, 0.03]) == [True, False]
assert C.holm([0.02, 0.02]) == [False, False]
say('All checks passed.')
(C.HERE / 'tests_out.txt').write_text('\n'.join(OUT) + '\n')
