"""The 16 unseen months (2025-06-30..2026-10-19), run once after the rules were final: the fund long-short and its
baselines on the new store, a check that the new store reproduces the old weekly returns, and the story check.
The judge's single holdout is `money.py new`.

    python3 -I unseen.py        # writes out/unseen.txt and out/unseen.json
"""
import json
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402
from tests import story  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def main():
    Po = C.build('old')
    Pn = C.build('new')
    days = Pn['days']
    T = len(days)
    for n in Pn['notes']:
        say('data (new store):', n)
    hn = Pn['idx'][:, 0] >= 0
    ho = Po['idx'][:, 0] >= 0
    f0 = C.first_formation(hn, T - 1)
    forms = np.array([f for f in range(f0, T) if f % C.STEP == 0 and f + C.H <= T - 1])
    assert hn[forms].all()
    unseen = forms >= C.LAST_OLD
    say(f'new store: first forecast {c2eval.iso(days[np.argmax(hn)])}, book from {c2eval.iso(days[forms[0]])}; '
        f'unseen weeks {unseen.sum()}, formed {c2eval.iso(days[forms[unseen][0]])} .. {c2eval.iso(days[forms[unseen][-1]])}, '
        f'returns through {c2eval.iso(days[forms[unseen][-1] + C.H])}')
    mom, rev = C.signals(Pn)
    runs = {'analog': C.weekly(Pn, forms, Pn['F']), 'momentum': C.weekly(Pn, forms, mom), 'reversal': C.weekly(Pn, forms, rev)}

    # does the new store reproduce the old one before 2025-06-27?
    fo = np.array([f for f in forms if f + C.H <= C.LAST_OLD])
    old_an = C.weekly(Po, fo, Po['F'])
    new_an = {k: v[:len(fo)] for k, v in runs['analog'].items()}
    L = min(C.LAST_OLD + 1, len(Po['days']))
    same_idx = float((Po['idx'][:L][ho[:L]] == Pn['idx'][:L][ho[:L]]).all(1).mean())
    dc = np.nanmax(np.abs(np.log(Pn['c'][:, :L] / Po['c'][:, :L])[:, 1:] - np.log(Pn['c'][:, :L] / Po['c'][:, :L])[:, :-1]))
    dn = np.abs(new_an['net'] - old_an['net'])
    say(f'new vs old store before 2025-06-27: largest gap in a daily log return {dc:.2e}; analog sets identical on '
        f'{100 * same_idx:.2f}% of days; weekly analog net returns differ by at most {1e4 * dn.max():.3f} bps '
        f'(mean {1e4 * dn.mean():.4f}); old-store weekly net mean {1e4 * old_an["net"].mean():+.3f} vs new {1e4 * new_an["net"].mean():+.3f} bps')

    res = {'weeks': int(unseen.sum()), 'forms': [c2eval.iso(days[f]) for f in forms[unseen]], 'strategies': {}}
    say(f'\n--- Test 4, the unseen weeks ---')
    say(f'{"":<40} {"bps/wk":>8} {"t":>6} {"n":>4} {"sum %":>7}')
    for k, r in runs.items():
        for kind in ('gross', 'net'):
            x = r[kind][unseen]
            mu, t, n = C.tstat(x)
            say(f'{k + " " + kind:<40} {1e4 * mu:+8.2f} {t:+6.2f} {n:4d} {100 * x.sum():+7.2f}')
            res['strategies'][f'{k}|{kind}'] = dict(mean=mu, t=t, n=n, sum=float(x.sum()), weekly=x.tolist())
        say(f'{"":<40} turnover {r["turnover"][unseen].mean():.2f}/wk')
    for b in ('momentum', 'reversal'):
        for kind in ('net', 'gross'):
            d = runs['analog'][kind][unseen] - runs[b][kind][unseen]
            mu, t, n = C.tstat(d)
            say(f'{"analog - " + b + " (" + kind + ", paired)":<40} {1e4 * mu:+8.2f} {t:+6.2f} {n:4d}')
            res['strategies'][f'diff|{b}|{kind}'] = dict(mean=mu, t=t, n=n)
    G, N = C.random_weekly(Pn, forms)
    gm = G[:, unseen].mean(1) * 1e4
    nm = N[:, unseen].mean(1) * 1e4
    an = runs['analog']['net'][unseen].mean() * 1e4
    ag = runs['analog']['gross'][unseen].mean() * 1e4
    res['random'] = dict(net_median=float(np.median(nm)), p_net=float((nm >= an).mean()), p_gross=float((gm >= ag).mean()))
    say(f'random (10,000): net median {np.median(nm):+.2f} bps/wk (5-95%: {np.percentile(nm, 5):+.2f}..{np.percentile(nm, 95):+.2f}); '
        f'analog p net {res["random"]["p_net"]:.3f}, gross {res["random"]["p_gross"]:.3f}')
    m = runs['analog']['net'][unseen].mean()
    say(f'\nTest 4: analog net {1e4 * m:+.2f} bps/wk -> {"PASS" if m > 0 else "FAIL"} (needs positive)')
    res['test4'] = bool(m > 0)

    say('\nby week (analog gross, net; momentum net; reversal net), bps; longs / shorts:')
    for i in np.nonzero(unseen)[0]:
        w = runs['analog']['w'][i]
        lo = '+'.join(C.SECT[j] for j in np.nonzero(w > 0)[0])
        sh = '+'.join(C.SECT[j] for j in np.nonzero(w < 0)[0])
        say(f'  {c2eval.iso(days[forms[i]])}  {1e4 * runs["analog"]["gross"][i]:+7.1f} {1e4 * runs["analog"]["net"][i]:+7.1f}  '
            f'{1e4 * runs["momentum"]["net"][i]:+7.1f}  {1e4 * runs["reversal"]["net"][i]:+7.1f}   long {lo} / short {sh}')

    say('\n--- Story check, unseen weeks ---')
    S = story(Pn, forms[unseen])
    da, dr, aa, ar = S['d_an'], S['d_rn'], S['a_an'], S['a_rn']
    say(f'dispersion analogs {100 * da.mean():.3f}% vs random {100 * dr.mean():.3f}% (ratio {da.mean() / dr.mean():.3f}; tighter in {100 * (da < dr).mean():.0f}% of weeks)')
    say(f'agreement analogs {aa.mean():+.4f} vs random {ar.mean():+.4f} (higher in {100 * (aa > ar).mean():.0f}% of weeks); '
        f'share of analogs from the 63 days before t {100 * S["recent"].mean():.1f}% (random {100 * S["recent_rn"].mean():.1f}%)')
    res['story'] = dict(disp_an=float(da.mean()), disp_rn=float(dr.mean()), agree_an=float(aa.mean()), agree_rn=float(ar.mean()),
                        recent=float(S['recent'].mean()))
    (C.OUT / 'unseen.txt').write_text('\n'.join(LINES) + '\n')
    (C.OUT / 'unseen.json').write_text(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
