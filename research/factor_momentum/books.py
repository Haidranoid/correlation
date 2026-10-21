"""Runs the plan's book, the 1-month diagnostic book and the two baselines over consecutive formations of one
export, with per-component contributions, eigenvalue shares, stability and betas.

    import books; R = books.run(D, ks)      # D = common.load(tag)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import np  # noqa: E402


def exante_beta(D, F):
    """sum_i W_i beta_i, beta_i = cov(r_i, m) / var(m) over the window's days where both have a simple return
    (SPY from store closes). NaN if SPY has returns on < 90% of the window."""
    t = F['t']
    sc = D['spy_close'][t - C.WIN:t + 1]
    with np.errstate(all='ignore'):
        m = sc[1:] / sc[:-1] - 1
    if np.isfinite(m).mean() < C.MIN_FRAC:
        return float('nan'), None
    R = np.expm1(F['r'])
    M = np.isfinite(R) & np.isfinite(m)[None, :]
    x = np.where(M, m[None, :], 0.0)
    y = np.where(M, R, 0.0)
    cnt = M.sum(1)
    mx, my = x.sum(1) / cnt, y.sum(1) / cnt
    cov = (x * y).sum(1) / cnt - mx * my
    var = (x * x).sum(1) / cnt - mx ** 2
    beta = cov / var
    return float(F['W'] @ beta), beta


def stability(prev, cur):
    """Best |cosine| of each of cur's components 2-21 among prev's top components (aligned on the union of stocks,
    zeros where absent), the 1-based index matched, |cos| with the same index, and the retention in prev's 2-21
    subspace (norm of the projection)."""
    union = np.union1d(prev['rows'], cur['rows'])
    A = np.zeros((len(union), C.NK))
    A[np.searchsorted(union, cur['rows'])] = cur['V'][:, C.FIRST - 1:C.LAST]
    B = np.zeros((len(union), prev['V'].shape[1]))
    B[np.searchsorted(union, prev['rows'])] = prev['V']
    cs = np.abs(A.T @ B)                                   # [20, 50]
    best = cs.max(1)
    arg = cs.argmax(1) + 1
    same = cs[np.arange(C.NK), np.arange(C.FIRST - 1, C.LAST)]
    ret = np.sqrt((cs[:, C.FIRST - 1:C.LAST] ** 2).sum(1))
    # component 1 too, for reference
    a1 = np.zeros(len(union))
    a1[np.searchsorted(union, cur['rows'])] = cur['V'][:, 0]
    c1 = float(np.abs(a1 @ B[:, 0]))
    return dict(best=best, arg=arg, same=same, retention=ret, c1=c1)


def run(D, ks, verbose=True):
    tw = D['tw']
    book, book1, ba, bb = C.Book(D), C.Book(D), C.Book(D), C.Book(D)
    forms = []           # one dict per formation (all ks, including those without a complete hold)
    saved = dict(k=[], t=[], offs=[0], rows=[], W=[])
    prev = None
    for i, k in enumerate(ks):
        F = C.formation(D, k)
        t, rows = F['t'], F['rows']
        beta_ex, _ = exante_beta(D, F)
        rec = dict(k=k, t=t, date=C.iso(D, t), n=F['n'], share=(F['lam'][:C.LAST] / F['n']).tolist(),
                   past12=F['past12'].tolist(), past1=F['past1'].tolist(), vol=F['vol'].tolist(),
                   sign=F['sign'].tolist(), scale=F['scale'].tolist(), net_exp=float(F['W'].sum()),
                   long_exp=float(F['W'][F['W'] > 0].sum()), short_exp=float(-F['W'][F['W'] < 0].sum()),
                   n_long=int((F['W'] > 0).sum()), beta_exante=beta_ex,
                   comp_gross=(np.abs(F['B']).sum(0)).tolist())
        if prev is not None:
            st = stability(prev, F)
            rec.update(stab_best=st['best'].tolist(), stab_arg=st['arg'].tolist(), stab_same=st['same'].tolist(),
                       stab_ret=st['retention'].tolist(), stab_c1=st['c1'], prev_date=prev['date'])
        saved['k'].append(k)
        saved['t'].append(t)
        saved['rows'].append(rows)
        saved['W'].append(F['W'])
        saved['offs'].append(saved['offs'][-1] + len(rows))
        if t + C.HOLD <= D['T'] - 1:
            g = C.hold_growth(D, rows, t)
            contrib = F['B'].T @ g                              # each component's part of the book's gross return
            fac = F['w'].T @ g                                  # each gross-1 eigenportfolio's buy-and-hold return
            m = book.add(k, rows, F['W'])
            assert abs(contrib.sum() - m['gross']) < 1e-10
            m1 = book1.add(k, rows, F['W1'])
            big = np.nonzero(D['big'][k])[0]
            sa = C.mom12_1(D, big, t)
            ma = ba.add(k, big, C.decile_weights(sa))
            fr = tw['fund_row'][k, big]
            has = fr >= 0
            sb = np.full(len(big), np.nan)
            sb[has] = C.fund_mom12_1(D, fr[has], t)
            b_ok = has.mean() >= C.B_MIN_FRAC and len(set(fr[has].tolist())) >= C.B_MIN_DISTINCT
            mb = bb.add(k, big, C.decile_weights(sb)) if b_ok else None
            rec.update(hold=True, gross=m['gross'], net=m['net'], cost=m['cost'], turnover=m['turnover'],
                       long=m['long'], short=m['short'], spy=m['spy'],
                       contrib=contrib.tolist(), fac=fac.tolist(), signed_fac=(F['sign'] * fac).tolist(),
                       m1_gross=m1['gross'], m1_net=m1['net'], m1_turnover=m1['turnover'],
                       a_gross=ma['gross'], a_net=ma['net'], a_turnover=ma['turnover'], a_long=ma['long'],
                       a_short=ma['short'],
                       b_gross=mb['gross'] if mb else float('nan'), b_net=mb['net'] if mb else float('nan'),
                       b_turnover=mb['turnover'] if mb else float('nan'), b_ok=bool(b_ok))
        else:
            rec.update(hold=False)
        forms.append(rec)
        prev = dict(rows=rows, V=F['V'], date=rec['date'])
        if verbose and i % 25 == 0:
            C.log(f'formation {rec["date"]} n {F["n"]} gross {rec.get("gross", float("nan")):+.4f}')
    np.savez(C.WORK / f'book_{D["tag"]}.npz', k=np.array(saved['k']), t=np.array(saved['t']),
             offs=np.array(saved['offs']), rows=np.concatenate(saved['rows']), W=np.concatenate(saved['W']))
    return forms
