"""Decoupling: independent recomputation of the panel values for random stock-days and events (straight loops over one
stock at a time, no sliding windows or cumulative sums), the twin rule, and how many candidates the extra
"twin at t-21 = twin at t" requirement removes. Reads no outcome statistics; it only checks that fwd matches.

    python3 -I -B checks.py old|new      # writes out/checks_{tag}.txt
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
    print(s, flush=True)


def direct(E, tw, j, t):
    k = C.kidx(tw['rebal'], t)
    f = int(tw['fund_row'][k, j])
    if f < 0 or tw['fund_row'][k - 3, j] != f:
        return None
    b = float(tw['beta'][k, j])
    c = np.asarray(E['close'][j], np.float64)
    fc = np.asarray(tw['F']['close'][f], np.float64)
    with np.errstate(all='ignore'):
        r = np.r_[np.nan, np.log(c[1:] / c[:-1])]
        q = np.r_[np.nan, np.log(fc[1:] / fc[:-1])]
    q[np.abs(q) > 0.5] = np.nan
    # now
    ri, qi = r[t - 62:t + 1], q[t - 62:t + 1]
    e = ri - b * qi
    fin = np.isfinite(e)
    idx = np.nonzero(fin)[0]
    big = idx[np.argsort(-np.abs(e[idx]))[:3]]
    keep = fin.copy()
    keep[big] = False
    rho_n = np.corrcoef(ri[keep], qi[keep])[0, 1] if keep.sum() >= 50 else np.nan
    S3 = e[keep].sum() if keep.sum() >= 50 else np.nan
    Sall = e[fin].sum() if keep.sum() >= 50 else np.nan
    # before
    rb, qb = r[t - 314:t - 62], q[t - 314:t - 62]
    m = np.isfinite(rb) & np.isfinite(qb)
    rho_b = np.corrcoef(rb[m], qb[m])[0, 1] if m.sum() >= 200 else np.nan
    # forward
    if t + 63 <= len(r) - 1:
        rf, qf = r[t + 1:t + 64], q[t + 1:t + 64]
        if (~np.isfinite(rf)).sum() > 3 or (~np.isfinite(qf)).sum() > 3:
            fwd = np.nan
        else:
            fwd = np.nansum(rf) - b * np.nansum(qf)
    else:
        fwd = np.nan
    return dict(rho_b=rho_b, rho_n=rho_n, S3=S3, Sall=Sall, fwd=fwd)


def main(tag):
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
    tw = c2eval.twins(E)
    P = {p: np.load(C.WORK / f'{p}_{tag}.npy', mmap_mode='r') for p in C.PANELS}
    ev = dict(np.load(C.WORK / f'events_{tag}.npz'))
    U = np.load(C.WORK / f'univ_{tag}.npz')
    rng = np.random.default_rng(1)
    # random valid stock-days in the universe
    pairs = []
    while len(pairs) < 300:
        t = int(rng.integers(800, E['close'].shape[1] - 1))
        k = C.kidx(tw['rebal'], t)
        js = np.nonzero(U['uni'][k])[0]
        if len(js):
            pairs.append((int(rng.choice(js)), t))
    pairs += list(zip(ev['j'].tolist(), ev['t'].tolist()))
    worst = {p: 0.0 for p in C.PANELS}
    nan_mismatch = {p: 0 for p in C.PANELS}
    n_ok = 0
    for j, t in pairs:
        d = direct(E, tw, j, t)
        if d is None:
            for p in C.PANELS:
                nan_mismatch[p] += int(np.isfinite(P[p][j, t]))
            continue
        n_ok += 1
        for p in C.PANELS:
            a, b = float(P[p][j, t]), float(d[p])
            if np.isfinite(a) != np.isfinite(b):
                nan_mismatch[p] += 1
            elif np.isfinite(a):
                worst[p] = max(worst[p], abs(a - b))
    say(f'# Checks, {tag} export')
    say(f'{len(pairs)} stock-days ({len(ev["t"])} events + 300 random universe stock-days); {n_ok} pass the twin rule')
    say('largest |panel - direct| by panel: ' + ', '.join(f'{p} {worst[p]:.2e}' for p in C.PANELS))
    say('NaN-pattern mismatches by panel: ' + ', '.join(f'{p} {nan_mismatch[p]}' for p in C.PANELS))
    # events: twin same at k-3, k-1, k; universe at k; rule thresholds; quiet period
    k = ev['k']
    fr = tw['fund_row']
    say(f'events: twin k-3 == k: {np.mean(fr[k - 3, ev["j"]] == fr[k, ev["j"]]):.0%}; k-1 == k: '
        f'{np.mean(fr[k - 1, ev["j"]] == fr[k, ev["j"]]):.0%}; in universe at k: {np.mean(U["uni"][k, ev["j"]]):.0%}; '
        f'rho_b >= 0.5: {np.mean(ev["rho_b"] >= 0.5):.0%}; drop >= 0.3: {np.mean(ev["drop"] >= 0.3):.0%}; '
        f'drop21 >= 0.3: {np.mean(ev["drop21"] >= 0.3):.0%}')
    gaps = []
    for j in set(ev['j'].tolist()):
        ts = np.sort(ev['t'][ev['j'] == j])
        gaps += np.diff(ts).tolist()
    say(f'smallest gap between events of one stock: {min(gaps) if gaps else "none"} trading days')
    # how many candidates the extra k-1 requirement removes (no outcomes)
    drop = np.asarray(P['rho_b']) - np.asarray(P['rho_n'])
    drop21 = np.full_like(drop, np.nan)
    drop21[:, 21:] = drop[:, :-21]
    T = drop.shape[1]
    with_k1 = without_k1 = 0
    for kk in range(1, len(tw['rebal'])):
        t0 = int(tw['rebal'][kk])
        t1 = int(tw['rebal'][kk + 1]) if kk + 1 < len(tw['rebal']) else T
        rows = np.nonzero(U['uni'][kk])[0]
        with np.errstate(invalid='ignore'):
            c = ((np.asarray(P['rho_b'][rows, t0:t1]) >= 0.5) & (drop[rows, t0:t1] >= 0.3)
                 & (drop21[rows, t0:t1] >= 0.3) & np.isfinite(np.asarray(P['S3'][rows, t0:t1])))
        same = (fr[kk - 1, rows] == fr[kk, rows])
        without_k1 += int(c.sum())
        with_k1 += int((c & same[:, None]).sum())
    say(f'candidate stock-days: {without_k1} without the twin-at-t-21 requirement, {with_k1} with it')
    (C.OUT / f'checks_{tag}.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
