"""Gravity, step 1: the universe, the open-high-low-close panel, bad prints, sigma, the market-residual move z.

    python -I -B data.py old|new|fwd

Universe: at each twin refit t0 (c2eval.twins' fixed 21-day calendar), the 500 stocks with the highest 63-day median
export dollar volume (>= 57 of 63 days finite, finite export close at t0), minus c2eval.excluded(E, tw); membership holds
for t0 .. next refit - 1, from the refit in force on U_START[tag].

Prices (rows = every stock ever in the universe):
- C: the export's close before 2008 (history only: correlations, beta, spreads' windows, 2007's moves), the store's
  close on valid days from 2008 (the export's close has a repair that looks one day ahead);
- O, H, L: the store's, valid from 2008 only; a day failing the bad-print rule is missing (no force, no return).
sigma[t]: Garman-Klass daily volatility over the 63 days t-62..t (>= 57 valid OHLC days), else the sd of daily log
  close-to-close returns over the same days (>= 57). The force on day t uses sigma[t - 1] ("through t - 1").
z[t] = (r[t] - beta_M[t] r_M[t]) / sigma[t - 1], clipped at +-5; beta_M from the 252 days t-252..t-1 (>= 227).
D0 (the first force day): the first day t >= 2008 on which GK sigma[t - 1] exists for >= 90% of the universe.

Writes BIG/panel_<tag>/ and out/data_<tag>.txt / .json.
"""
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as K  # noqa: E402
np = K.np
c2eval = K.c2eval

U_START = {'old': '2008-01-01', 'new': '2025-01-01', 'fwd': os.environ.get('C2_FWD_USTART', '2025-10-01')}
GK_C = 2 * np.log(2) - 1


def universe(dv, close, tw, ex, u0, T):
    """Per refit in force from u0: (t0, export rows of the universe). dv, close: export arrays [N, >= T] (sliced to
    the data the build may see)."""
    k0 = max(c2eval.rebal_index(tw, u0), 0)
    out = []
    for k in range(k0, len(tw['rebal'])):
        t0 = int(tw['rebal'][k])
        if t0 >= T:
            break
        w = np.asarray(dv[:, t0 - K.TOP_W + 1:t0 + 1], np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            md = np.nanmedian(w, 1)
        ok = (~ex[k]) & np.isfinite(np.asarray(close[:, t0])) & (np.isfinite(w).sum(1) >= K.TOP_MIN) & (np.nan_to_num(md) > 0)
        idx = np.nonzero(ok)[0]
        idx = idx[np.argsort(-md[idx], kind='stable')[:K.TOP]]
        out.append((t0, np.sort(idx)))
    return out


def roll(x, m, W):
    """Rolling sums over the W days through each day: (sum of x where m, count of m); [R, T] float64; NaN before W-1."""
    R, T = x.shape
    cs = np.concatenate([np.zeros((R, 1)), np.cumsum(np.where(m, x, 0.0), 1)], 1)
    cm = np.concatenate([np.zeros((R, 1)), np.cumsum(m, 1)], 1)
    s = np.full((R, T), np.nan)
    n = np.zeros((R, T))
    s[:, W - 1:] = cs[:, W:] - cs[:, :-W]
    n[:, W - 1:] = cm[:, W:] - cm[:, :-W]
    return s, n


def derive(raw, mkt, f0):
    """Everything computed from the raw arrays (sliced to what the build may see). raw: O, H, L, Cs (store, float32),
    ec (export close). Returns dict of panels."""
    O, H, L, Cs, ec = (np.asarray(raw[k], np.float64) for k in ('O', 'H', 'L', 'Cs', 'ec'))
    R, T = O.shape
    bad, parts = K.bad_print(O, H, L, Cs)
    vO = ~bad
    vO[:, :f0] = False
    C = np.full((R, T), np.nan)
    C[:, :f0] = ec[:, :f0]
    C[:, f0:] = np.where(vO[:, f0:], Cs[:, f0:], np.nan)
    C[~(C > 0)] = np.nan
    vC = np.isfinite(C)
    with np.errstate(all='ignore'):
        lnC = np.log(C)
        lnO = np.where(vO, np.log(O), np.nan)
        lnH = np.where(vO, np.log(H), np.nan)
        lnL = np.where(vO, np.log(L), np.nan)
    r = np.full((R, T), np.nan)
    r[:, 1:] = lnC[:, 1:] - lnC[:, :-1]
    vr = np.isfinite(r)
    # sigma through t
    gkv = 0.5 * (lnH - lnL) ** 2 - GK_C * (lnC - lnO) ** 2
    vg = np.isfinite(gkv)
    s, n = roll(np.nan_to_num(gkv), vg, K.SIG_W)
    with np.errstate(all='ignore'):
        sig_gk = np.sqrt(s / n)
    gk_ok = n >= K.SIG_MIN
    s1, n1 = roll(np.nan_to_num(r), vr, K.SIG_W)
    s2, _ = roll(np.nan_to_num(r) ** 2, vr, K.SIG_W)
    with np.errstate(all='ignore'):
        sd_cc = np.sqrt(np.maximum(s2 / n1 - (s1 / n1) ** 2, 0) * n1 / (n1 - 1))
    cc_ok = n1 >= K.SIG_MIN
    sig = np.where(gk_ok, sig_gk, np.where(cc_ok, sd_cc, np.nan))
    # beta to the market over t-252..t-1, z
    rM = np.asarray(mkt, np.float64)[None, :].repeat(R, 0)
    vb = vr & np.isfinite(rM)
    x, y = np.where(vb, rM, 0.0), np.where(vb, r, 0.0)
    W = K.BETA_M_W
    sx, nb = roll(x, vb, W)
    sy, _ = roll(y, vb, W)
    sxx, _ = roll(x * x, vb, W)
    sxy, _ = roll(x * y, vb, W)
    with np.errstate(all='ignore'):
        bm_thru = (nb * sxy - sx * sy) / (nb * sxx - sx * sx)
    bm_thru[nb < K.BETA_M_MIN] = np.nan
    beta_m = np.full((R, T), np.nan)
    beta_m[:, 1:] = bm_thru[:, :-1]          # through t - 1
    sig_prev = np.full((R, T), np.nan)
    sig_prev[:, 1:] = sig[:, :-1]
    with np.errstate(all='ignore'):
        z = (r - beta_m * rM) / sig_prev
    z = np.clip(z, -K.Z_CLIP, K.Z_CLIP)
    z[~np.isfinite(z)] = np.nan
    return {'C': C, 'vC': vC, 'vO': vO, 'lnC': lnC, 'lnO': lnO, 'lnH': lnH, 'lnL': lnL, 'r': r, 'sig': sig, 'gk_ok': gk_ok,
            'z': z, 'beta_m': beta_m, 'parts': parts, 'bad': bad}


def build(tag, cut=None, rows=None):
    """Build the panel from the raw data, optionally truncated after export day index `cut` (the look-ahead check).
    rows: export rows to carry (default: every stock in the universe of any refit from U_START). Returns dict."""
    exp, sto = K.SRC[tag]
    E = c2eval.load_export(exp, arrays=('close', 'dollar_volume', 'market'))
    days = np.asarray(E['days'])
    T = len(days) if cut is None else cut + 1
    u0 = K.day_of(days, U_START[tag])
    f0 = K.day_of(days, K.FIRST)
    tw = c2eval.twins(E)
    ex = c2eval.excluded(E, tw)
    U = universe(E['dollar_volume'][:, :T], E['close'][:, :T], tw, ex, u0, T)
    if rows is None:
        rows = np.unique(np.concatenate([u for _, u in U]))
    pos = {int(e): i for i, e in enumerate(rows)}
    S = c2eval.load_store(sto)
    ids = E['ids'][rows]
    raw = {}
    for f, name in (('open', 'O'), ('high', 'H'), ('low', 'L'), ('close', 'Cs')):
        raw[name] = c2eval.store_panel(S, ids, days, f)[:, :T]
    raw['ec'] = np.asarray(E['close'][rows][:, :T], np.float32)
    P = derive(raw, np.asarray(E['market'][:T]), f0)
    R = len(rows)
    refit = np.array([t0 for t0, _ in U], np.int64)
    uni = np.full((len(U), K.TOP), -1, np.int32)
    inU = np.zeros((R, T), bool)
    for k, (t0, u) in enumerate(U):
        pr = np.array([pos[int(e)] for e in u], np.int32)
        uni[k, :len(pr)] = pr
        t1 = int(refit[k + 1]) if k + 1 < len(refit) else T
        inU[pr, max(t0, u0):t1] = True
    # D0: the first day >= 2008 with GK sigma[t-1] for >= 90% of that day's universe
    share = np.zeros(T)
    for t in range(max(f0, 1), T):
        n = inU[:, t].sum()
        share[t] = (P['gk_ok'][inU[:, t], t - 1].sum() / n) if n else 0.0
    ok = np.nonzero((share >= K.D0_SHARE) & (np.arange(T) >= f0))[0]
    d0 = int(ok[0]) if len(ok) else T
    P.update({'rows': np.asarray(rows), 'days': days[:T], 'inU': inU, 'refit': refit, 'uni': uni, 'd0': d0, 'f0': f0, 'u0': u0,
              'tickers': E['tickers'][rows], 'ids': ids, 'fingerprint': E['meta']['fingerprint'], 'raw': raw, 'T': T,
              'export': str(exp), 'store': str(sto)})
    return P


SAVE = ('rows', 'days', 'C', 'vC', 'vO', 'lnC', 'lnO', 'lnH', 'lnL', 'r', 'sig', 'gk_ok', 'z', 'inU', 'refit', 'uni')


def load_panel(tag, mmap=True):
    p = K.BIG / f'panel_{tag}'
    out = {k: np.load(p / f'{k}.npy', mmap_mode='r' if mmap else None) for k in SAVE}
    out['meta'] = json.loads((p / 'meta.json').read_text())
    out['d0'] = out['meta']['d0']
    out['f0'] = out['meta']['f0']
    out['u0'] = out['meta']['u0']
    out['T'] = len(out['days'])
    out['tickers'] = np.array(out['meta']['tickers'])
    return out


def main(tag):
    K.OUT.mkdir(exist_ok=True)
    fh = open(K.OUT / f'data_{tag}.txt', 'w')
    P = build(tag)
    days, T, R = P['days'], P['T'], len(P['rows'])
    sz = (P['uni'] >= 0).sum(1)
    K.log(f'{tag}: export {P["export"]} ({T} days, {K.iso(days, 0)}..{K.iso(days, T - 1)}); store {P["store"]}', fh)
    K.log(f'universe: {len(P["refit"])} refits from {K.iso(days, int(P["refit"][0]))}, size min {sz.min()} median '
          f'{int(np.median(sz))} max {sz.max()}; {R} distinct stocks; OHLC from {K.iso(days, P["f0"])}', fh)
    K.log(f'D0 (first force day, GK sigma for >= 90% of the universe): {K.iso(days, P["d0"])} (index {P["d0"]})', fh)
    inU = P['inU']
    has = inU & ~P['parts']['no row']
    n = int(has.sum())
    rep = {'tag': tag, 'T': T, 'rows': R, 'refits': len(P['refit']), 'size_min': int(sz.min()), 'size_max': int(sz.max()),
           'd0': P['d0'], 'd0_date': K.iso(days, P['d0']), 'universe_stock_days': int(inU.sum()), 'with_store_row': n,
           'bad_prints': {}}
    K.log(f'universe stock-days {int(inU.sum()):,}; with a store row {n:,}', fh)
    for k, v in P['parts'].items():
        if k == 'no row':
            continue
        c = int((v & has).sum())
        rep['bad_prints'][k] = c
        K.log(f'  bad print {k:>12}: {c:>9,} ({100 * c / max(n, 1):.4f}%)', fh)
    c = int((P['bad'] & has).sum())
    rep['bad_prints']['any'] = c
    K.log(f'  bad print {"any":>12}: {c:>9,} ({100 * c / max(n, 1):.4f}%)', fh)
    ec = np.asarray(P['raw']['ec'], np.float64)
    with np.errstate(all='ignore'):
        rel = np.abs(P['C'] / ec - 1)
    m = P['vO'] & np.isfinite(rel)
    rep['store_vs_export_close'] = {'days': int(m.sum()), 'max_rel': float(rel[m].max()), 'share_exact': float(np.mean(rel[m] == 0))}
    K.log(f'store close vs export close on valid days from 2008: {int(m.sum()):,} stock-days, max relative difference '
          f'{rel[m].max():.3g}, exact {100 * np.mean(rel[m] == 0):.3f}%', fh)
    d0 = P['d0']
    uu = inU[:, d0:]
    rep['sigma_gk_share_universe'] = float(P['gk_ok'][:, d0 - 1:T - 1][uu].mean())
    rep['sigma_missing_universe'] = int((~np.isfinite(P['sig'][:, d0 - 1:T - 1]) & uu).sum())
    rep['z_missing_universe'] = int((~np.isfinite(P['z'][:, d0:]) & uu).sum())
    K.log(f'from D0: sigma is Garman-Klass on {100 * rep["sigma_gk_share_universe"]:.2f}% of universe stock-days; sigma '
          f'missing on {rep["sigma_missing_universe"]:,}; z missing on {rep["z_missing_universe"]:,}', fh)
    sg = P['sig'][:, d0 - 1:T - 1][uu]
    rep['sigma_median'] = float(np.nanmedian(sg))
    K.log(f'median sigma over universe stock-days from D0: {rep["sigma_median"]:.4f}', fh)
    out = K.BIG / f'panel_{tag}'
    out.mkdir(parents=True, exist_ok=True)
    for k in SAVE:
        np.save(out / f'{k}.npy', P[k])
    meta = {'tag': tag, 'export': P['export'], 'store': P['store'], 'fingerprint': P['fingerprint'], 'tickers': P['tickers'].tolist(),
            'ids': P['ids'].tolist(), 'd0': P['d0'], 'f0': P['f0'], 'u0': P['u0']}
    (out / 'meta.json').write_text(json.dumps(meta))
    K.jdump(rep, K.OUT / f'data_{tag}.json')
    fh.close()


if __name__ == '__main__':
    main(sys.argv[1])
