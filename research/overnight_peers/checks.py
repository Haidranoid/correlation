"""Plumbing checks (no outcomes): the float32 pairwise-complete correlations against a direct float64 computation;
the top-20 peer choice against a brute-force pick; point in time (the residual signal at d0 is unchanged when every
return after d0 is removed); the overnight + intraday = whole-day check is in build.py.

    python3 -I checks.py old
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
np = C.np
c2eval = C.c2eval
iso = c2eval.iso

tag = sys.argv[1] if len(sys.argv) > 1 else 'old'
M = C.meta(tag)
days, rebal, uni = M['days'], M['rebal'], M['uni']
T = len(days)
R = C.raw_returns(tag)
A = {ty: C.market_adjust(R[ty], rebal, uni)[0] for ty in C.TYPES}
PE = C.load_peers(tag)
rng = np.random.default_rng(11)
ks = C.k_range(tag, M)
lines = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


# 1. correlations against direct float64 pairwise-complete Pearson
for k in (ks[0], ks[len(ks) // 2], ks[-1]):
    t = int(rebal[k])
    rows = np.nonzero(uni[k])[0]
    for ty in C.TYPES:
        Xw = A[ty][rows, t - C.WIN + 1:t + 1]
        Cm, n = C.corr_matrix(Xw)
        errs, nerr = [], 0
        for _ in range(300):
            i, j = rng.integers(len(rows), size=2)
            if i == j:
                continue
            a, b = Xw[i].astype(np.float64), Xw[j].astype(np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            nerr += int(ok.sum() != n[i, j])
            if ok.sum() < 3:
                continue
            r = np.corrcoef(a[ok], b[ok])[0, 1]
            if np.isfinite(r):
                errs.append(abs(r - Cm[i, j]))
        say(f'1. {iso(days[t])} {ty}: max |float32 pairwise - direct float64| over {len(errs)} pairs {max(errs):.2e}; count mismatches {nerr}')

# 2. top-20 choice against brute force
k = ks[len(ks) // 3]
t = int(rebal[k])
rows = np.nonzero(uni[k])[0]
bad = 0
for ty in C.TYPES:
    Xw = A[ty][rows, t - C.WIN + 1:t + 1].astype(np.float64)
    prow, ppeer, pcorr = PE[ty][k]
    assert np.array_equal(prow, rows)
    for i in rng.choice(len(rows), 20, replace=False):
        cands = []
        for j in range(len(rows)):
            if j == i:
                continue
            ok = np.isfinite(Xw[i]) & np.isfinite(Xw[j])
            if ok.sum() < C.MIN_COMMON:
                continue
            r = np.corrcoef(Xw[i, ok], Xw[j, ok])[0, 1]
            if r >= C.MIN_CORR:
                cands.append((r, rows[j]))
        cands.sort(reverse=True)
        want = [c[1] for c in cands[:C.NPEER]] if len(cands) >= C.MIN_PEERS else []
        got = [p for p in ppeer[i] if p >= 0]
        bad += int(set(want) != set(got))
say(f'2. {iso(days[t])}: brute-force top-20 peer sets differing from the saved ones: {bad} of 60')

# 3. point in time: remove every return after d0, rebuild adjusted panels, S21, peers and the residual at d0
for d0s in ('2012-06-29', '2018-03-15', '2024-11-20'):
    d0 = c2eval.day_index(days, d0s)
    k = C.kidx(rebal, d0)
    Rt = {ty: R[ty].copy() for ty in C.TYPES}
    for ty in C.TYPES:
        Rt[ty][:, d0 + 1:] = np.nan
    At = {ty: C.market_adjust(Rt[ty], rebal, uni)[0] for ty in C.TYPES}
    S21t = C.roll_sum(At['whole'])
    rows = np.nonzero(uni[k])[0]
    for ty in C.TYPES:
        peer, corr = C.peers_at(At[ty], rows, int(rebal[k]))
        has = peer[:, 0] >= 0
        r, _, _ = C.residual_block(S21t, rows[has], peer[has], np.array([d0]))
        saved = np.load(C.WORK / f'resid_{ty}_{tag}.npy', mmap_mode='r')[rows[has], d0]
        same_peers = np.array_equal(peer, PE[ty][k][1])
        dif = np.nanmax(np.abs(r[:, 0] - saved))
        same_nan = np.array_equal(np.isfinite(r[:, 0]), np.isfinite(saved))
        say(f'3. {iso(days[d0])} {ty}: peers identical {same_peers}; residual max diff {dif:.2e}; same NaN pattern {same_nan}')
    del Rt, At, S21t

(C.OUT / f'checks_{tag}.txt').write_text('\n'.join(lines) + '\n')
