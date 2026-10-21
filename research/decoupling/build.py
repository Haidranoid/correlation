"""Decoupling: build the break-measure panels, the universe and the event table for one export, and print the event
counts (no forward returns are read here).

    python3 -I -B build.py old|new      # writes WORK/{panel}_{tag}.npy, univ_{tag}.npz, events_{tag}.npz, out/events_{tag}.txt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np, log  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def find_events(P, uni, big, tw, T):
    """Event days, sequential per stock (no event of the same stock on t-126..t-1)."""
    rebal = tw['rebal']
    N = uni.shape[1]
    drop = P['rho_b'] - P['rho_n']
    drop21 = np.full_like(drop, np.nan)
    drop21[:, C.SUSTAIN:] = drop[:, :-C.SUSTAIN]
    base = np.zeros((N, T), bool)          # universe at k(t), twin at k-1 == twin at k (Drop(t-21) uses k-1's twin)
    for k in range(1, len(rebal)):
        t0 = int(rebal[k])
        t1 = int(rebal[k + 1]) if k + 1 < len(rebal) else T
        fk = tw['fund_row'][k]
        ok = uni[k] & (fk >= 0) & (tw['fund_row'][k - 1] == fk)
        base[ok, t0:t1] = True
    with np.errstate(invalid='ignore'):
        cand = (base & (P['rho_b'] >= C.RHO_BEFORE_MIN) & (drop >= C.DROP_MIN) & (drop21 >= C.DROP_MIN)
                & np.isfinite(P['S3']))
    js, ts = np.nonzero(cand)              # row-major: sorted by stock, then day
    ev_j, ev_t = [], []
    last_j, last_t = -1, -10 ** 9
    for j, t in zip(js.tolist(), ts.tolist()):
        if j != last_j:
            last_j, last_t = j, -10 ** 9
        if t - last_t > C.QUIET:
            ev_j.append(j)
            ev_t.append(t)
            last_t = t
    ev_j, ev_t = np.array(ev_j), np.array(ev_t)
    o = np.lexsort((ev_j, ev_t))
    ev_j, ev_t = ev_j[o], ev_t[o]
    ks = np.searchsorted(rebal, ev_t, side='right') - 1
    ev = dict(j=ev_j, t=ev_t, k=ks, rho_b=P['rho_b'][ev_j, ev_t], rho_n=P['rho_n'][ev_j, ev_t],
              drop=drop[ev_j, ev_t], drop21=drop21[ev_j, ev_t], S3=P['S3'][ev_j, ev_t], Sall=P['Sall'][ev_j, ev_t],
              fwd=P['fwd'][ev_j, ev_t], fund_row=tw['fund_row'][ks, ev_j], beta=tw['beta'][ks, ev_j],
              twin_corr=tw['corr'][ks, ev_j], big=big[ks, ev_j])
    ev['D'] = np.sign(ev['S3'])
    ev['Dall'] = np.sign(ev['Sall'])
    return ev, int(cand.sum()), base


def main(tag):
    E = c2eval.load_export(C.EXPORTS[tag], arrays=('close', 'dollar_volume'))
    days = E['days']
    N, T = E['close'].shape
    log(tag, 'N', N, 'T', T, E['meta']['first_day'], '..', E['meta']['last_day'])
    tw = c2eval.twins(E)
    log('twins: K', len(tw['rebal']), 'funds', len(tw['fund_ticker']))
    uni, big = C.universe(E, tw)
    log('universe: median size', int(np.median(uni.sum(1)[uni.sum(1) > 0])), 'max', int(uni.sum(1).max()))
    P = C.build_panels(E, tw)
    log('panels built')
    for p in C.PANELS:
        np.save(C.WORK / f'{p}_{tag}.npy', P[p])
    np.savez(C.WORK / f'univ_{tag}.npz', uni=uni, big=big, rebal=tw['rebal'], days=days)
    ev, n_cand, base = find_events(P, uni, big, tw, T)
    ev['ticker'] = E['tickers'][ev['j']]
    ev['twin'] = tw['fund_ticker'][ev['fund_row']]
    ev['day'] = days[ev['t']]
    np.savez(C.WORK / f'events_{tag}.npz', **ev)
    log('events', len(ev['t']), 'candidate stock-days', n_cand)

    # ---- counts and descriptions (no forward returns)
    say(f'# Decoupling events, {tag} export ({E["meta"]["first_day"]} .. {E["meta"]["last_day"]})\n')
    # coverage of the measure among universe stock-days
    blk = np.searchsorted(tw['rebal'], np.arange(T), side='right') - 1
    yrs = C.year_of(days)
    say('Universe stock-days by year: with a twin; with a valid Drop (twin at t-63 = twin at t, enough data); '
        'rho_before >= 0.5 among valid')
    say(f'{"year":>6} {"univ":>9} {"valid":>9} {"share":>6} {"rb>=.5":>7} {"Drop>=.3":>9} {"Drop<.1":>8}')
    drop = P['rho_b'] - P['rho_n']
    cov = {}
    for y in range(int(yrs.min()), int(yrs.max()) + 1):
        dd = np.nonzero((yrs == y) & (blk >= 0))[0]
        if len(dd) == 0:
            continue
        U = uni[blk[dd]].T                        # [N, nd]
        nu = int(U.sum())
        if nu == 0:
            continue
        dv = drop[:, dd]
        with np.errstate(invalid='ignore'):
            V = U & np.isfinite(dv) & np.isfinite(P['S3'][:, dd])
            rb = V & (P['rho_b'][:, dd] >= C.RHO_BEFORE_MIN)
            br = rb & (dv >= C.DROP_MIN)
            nb = V & (dv < C.NONBREAK_MAX)
        cov[y] = (nu, int(V.sum()), int(rb.sum()), int(br.sum()), int(nb.sum()))
        say(f'{y:>6} {nu:>9,} {int(V.sum()):>9,} {V.sum() / nu:6.1%} {rb.sum() / max(1, V.sum()):7.1%} '
            f'{br.sum() / max(1, V.sum()):9.2%} {nb.sum() / max(1, V.sum()):8.1%}')

    ey = C.year_of(ev['day'])
    say('\nEvents per year (t = event day), share with D > 0, the 500 largest, medians')
    say(f'{"year":>6} {"events":>7} {"D>0":>6} {"Dall>0":>7} {"D!=Dall":>8} {"top500":>7} {"rho_b":>6} {"rho_n":>6} '
        f'{"drop":>6} {"|S3|%":>6} {"|Sall|%":>7} {"has fwd":>8}')
    for y in sorted(set(ey.tolist())):
        m = ey == y
        say(f'{y:>6} {int(m.sum()):>7} {np.mean(ev["D"][m] > 0):6.1%} {np.mean(ev["Dall"][m] > 0):7.1%} '
            f'{np.mean(ev["D"][m] != ev["Dall"][m]):8.1%} {int(ev["big"][m].sum()):>7} '
            f'{np.median(ev["rho_b"][m]):6.2f} {np.median(ev["rho_n"][m]):6.2f} {np.median(ev["drop"][m]):6.2f} '
            f'{100 * np.median(np.abs(ev["S3"][m])):6.1f} {100 * np.median(np.abs(ev["Sall"][m])):7.1f} '
            f'{int(np.isfinite(ev["fwd"][m]).sum()):>8}')
    m = np.ones(len(ey), bool)
    say(f'{"all":>6} {int(m.sum()):>7} {np.mean(ev["D"] > 0):6.1%} {np.mean(ev["Dall"] > 0):7.1%} '
        f'{np.mean(ev["D"] != ev["Dall"]):8.1%} {int(ev["big"].sum()):>7} {np.median(ev["rho_b"]):6.2f} '
        f'{np.median(ev["rho_n"]):6.2f} {np.median(ev["drop"]):6.2f} {100 * np.median(np.abs(ev["S3"])):6.1f} '
        f'{100 * np.median(np.abs(ev["Sall"])):7.1f} {int(np.isfinite(ev["fwd"]).sum()):>8}')

    # most common twins
    tw_u, tw_c = np.unique(ev['twin'], return_counts=True)
    o = np.argsort(-tw_c)[:15]
    say('\nMost common twins among events: ' + ', '.join(f'{tw_u[i]} {tw_c[i]}' for i in o))
    say(f'Distinct stocks with an event: {len(set(ev["j"].tolist()))}; events per such stock: '
        f'{len(ev["j"]) / max(1, len(set(ev["j"].tolist()))):.2f}')

    # examples (seeded), no outcomes
    rng = np.random.default_rng(C.SEED)
    pick = np.sort(rng.choice(len(ev['t']), size=min(12, len(ev['t'])), replace=False))
    say('\nExample events (seeded draw): ticker, date, twin, beta, twin corr at k, rho_before, rho_now, Drop at t, '
        'Drop at t-21, S3, Sall, D')
    for i in pick:
        say(f'  {ev["ticker"][i]:<8} {c2eval.iso(ev["day"][i])} {ev["twin"][i]:<6} b={ev["beta"][i]:+.2f} '
            f'corr_k={ev["twin_corr"][i]:.2f} rho_b={ev["rho_b"][i]:+.2f} rho_n={ev["rho_n"][i]:+.2f} '
            f'drop={ev["drop"][i]:+.2f} drop21={ev["drop21"][i]:+.2f} S3={100 * ev["S3"][i]:+.1f}% '
            f'Sall={100 * ev["Sall"][i]:+.1f}% D={int(ev["D"][i]):+d}')
    (C.OUT / f'events_{tag}.txt').write_text('\n'.join(LINES) + '\n')
    log('done')


if __name__ == '__main__':
    main(sys.argv[1])
