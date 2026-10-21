"""The unseen months (2025-06-30..2026-10-19), run once after the walk-forward results were final: the plan's book
and the baselines on the new export (the book runs continuously from the first formation, so the first unseen hold
pays real turnover), holds formed on or after 2025-06-27 that end by 2026-10-19. Writes book_new.npz for
money.py new.

    python3 -I -B unseen.py      # writes out/unseen.txt, out/unseen.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from common import c2eval, np  # noqa: E402
import books  # noqa: E402

LINES = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    LINES.append(s)
    print(s, flush=True)


def pct(x):
    return f'{100 * x:+.3f}'


def main():
    old_days = np.load(Path(c2eval.EXPORT_OLD) / 'days.npy')
    D = C.load('new')
    assert np.array_equal(D['days'][:len(old_days)], old_days), 'exports disagree on the shared days'
    say(f'new export: N {D["N"]} T {D["T"]}, through {C.iso(D, D["T"] - 1)}; day indices match the old export on '
        f'its {len(old_days)} days; gap days filled {D["gap_days"]}')
    k0 = C.first_k(D)
    forms = books.run(D, list(range(k0, D['K'])), verbose=False)
    first = c2eval.day_index(D['days'], '2025-06-27')
    H = [f for f in forms if f['hold'] and f['t'] >= first]
    say(f'unseen holds: {len(H)}, formed {H[0]["date"]} .. {H[-1]["date"]}, returns through '
        f'{C.iso(D, H[-1]["t"] + C.HOLD)}; stocks per formation {sorted(set(h["n"] for h in H))}')
    res = dict(holds=len(H), first=H[0]['date'], last=H[-1]['date'], last_return_day=C.iso(D, H[-1]['t'] + C.HOLD),
               series={})
    say(f'\n{"":<34} {"mean %":>8} {"t":>6} {"n":>3} {"sum %":>7} {"sd %":>6} {"SR":>6}')
    for nm, key in (('book', ''), ('diag: 1-month sign', 'm1_'), ('(a) plain 12-1', 'a_'), ('(b) twin-group', 'b_')):
        for kind in ('gross', 'net'):
            x = np.array([h[key + kind] for h in H])
            mu, t, n = C.tstat(x)
            sr = C.sharpe(x)
            res['series'][f'{nm}|{kind}'] = dict(mean=mu, t=t, n=n, sum=float(x.sum()), sd=float(x.std(ddof=1)),
                                                 sharpe=sr, holds=[float(v) for v in x])
            say(f'{nm + " " + kind:<34} {pct(mu):>8} {t:+6.2f} {n:3d} {100 * x.sum():+7.2f} {100 * x.std(ddof=1):6.2f} {sr:+6.2f}')
    g = np.array([h['gross'] for h in H])
    s = np.array([h['spy'] for h in H])
    b = C.ols(g, [s])
    r2 = C.ols([h['net'] for h in H], [[h['a_net'] for h in H], [h['b_net'] for h in H]])
    res['beta'] = dict(b=b['b'][1], t=b['t'][1])
    res['alpha_on_baselines'] = r2
    res['turnover'] = float(np.mean([h['turnover'] for h in H]))
    res['exposure'] = {f: float(np.mean([h[f] for h in H])) for f in ('long_exp', 'short_exp', 'net_exp', 'beta_exante')}
    say(f'\nbook: beta to SPY {b["b"][1]:+.3f} (t {b["t"][1]:+.2f}); turnover {res["turnover"]:.2f} per formation; '
        f'net exposure {res["exposure"]["net_exp"]:+.3f}; ex-ante beta {res["exposure"]["beta_exante"]:+.3f}')
    say(f'book net on (a) and (b) net: alpha {pct(r2["b"][0])} (t {r2["t"][0]:+.2f}), n {r2["n"]} (diagnostic only)')
    Cn = np.array([h['contrib'] for h in H])
    gsum = {gname: C.tstat(Cn[:, sl].sum(1)) for gname, sl in {'2-5': slice(0, 4), '6-11': slice(4, 10), '12-21': slice(10, 20)}.items()}
    res['groups'] = gsum
    res['per_component_mean'] = Cn.mean(0).tolist()
    say('contribution by component group: ' + '; '.join(f'{k} {pct(v[0])} (t {v[1]:+.2f})' for k, v in gsum.items()))
    say('\nby hold, % (book gross [net] | 1-month | (a) | (b) | SPY):')
    res['by_hold'] = []
    for h in H:
        say(f'  {h["date"]}  {100 * h["gross"]:+6.2f} [{100 * h["net"]:+6.2f}] | {100 * h["m1_gross"]:+6.2f} | '
            f'{100 * h["a_gross"]:+6.2f} | {100 * h["b_gross"]:+6.2f} | {100 * h["spy"]:+5.1f}')
        res['by_hold'].append((h['date'], h['gross'], h['net'], h['m1_gross'], h['a_gross'], h['b_gross'], h['spy']))
    res['test3_unseen_pass'] = bool(res['series']['book|net']['mean'] > 0)
    say(f'\nTest 3, unseen part (book net mean positive): {pct(res["series"]["book|net"]["mean"])} -> '
        f'{"PASS" if res["test3_unseen_pass"] else "FAIL"}')
    (C.OUT / 'unseen.json').write_text(json.dumps(res, indent=1, default=float))
    (C.OUT / 'unseen.txt').write_text('\n'.join(LINES) + '\n')


if __name__ == '__main__':
    main()
