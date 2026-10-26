"""C5: cannabis legalization events. ALC - SPY (and the cannabis basket - SPY, events from 2018-06) log returns
over days +1..+2 and +1..+21 after the event day's close, and over the 20 days up to it (anticipation). Reported
addition (plan, Decisions): the event day's own close-to-close return, where news that broke during trading lands.

    python3 -I -B events.py old     # -> events_old.txt, events_old.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ad_common as K  # noqa: E402
import numpy as np  # noqa: E402

# Fixed in the plan (C5): the day each decision became public.
EVENTS = [
    ('2012-11-06', 'ballot', 'Colorado, Washington'),
    ('2014-11-04', 'ballot', 'Alaska, Oregon, DC'),
    ('2016-11-08', 'ballot', 'California, Nevada, Massachusetts, Maine'),
    ('2018-01-22', 'law', 'Vermont'),
    ('2018-06-19', 'law', "Canada's Senate passes C-45"),
    ('2018-11-06', 'ballot', 'Michigan'),
    ('2019-06-25', 'law', 'Illinois'),
    ('2020-11-03', 'ballot', "Arizona, Montana, New Jersey, South Dakota; Oregon's psilocybin measure"),
    ('2021-03-31', 'law', 'New York'),
    ('2021-04-07', 'law', 'Virginia'),
    ('2021-04-12', 'law', 'New Mexico'),
    ('2021-06-22', 'law', 'Connecticut'),
    ('2022-05-25', 'law', 'Rhode Island'),
    ('2022-11-08', 'ballot', "Maryland, Missouri; Colorado's psychedelics measure"),
    ('2023-04-24', 'law', 'Delaware'),
    ('2023-05-30', 'law', 'Minnesota'),
    ('2023-08-30', 'federal', 'HHS recommends Schedule III'),
    ('2023-11-07', 'ballot', 'Ohio'),
    ('2024-04-30', 'federal', 'reports that the DEA will reschedule'),
]


def summary(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return {'n': len(v)}
    return {'n': len(v), 'mean': float(v.mean()), 't': float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v)))),
            'share_negative': float(np.mean(v < 0))}


def main(tag):
    z = np.load(K.WORK / f'inputs_{tag}.npz')
    axis = z['axis']
    P = {t: z[f'P_{t}'] for t in K.TICKERS}
    dates = K.iso_days(axis)
    rows, lines = [], []

    def out(s=''):
        print(s, flush=True)
        lines.append(s)

    out('C5: log returns against SPY around cannabis legalization events (event day = e; windows by closes).')
    out('date        kind     ALC: pre(-20..e) day e   +1..+2   +1..+21    cannabis: day e  +1..+2   +1..+21  names  event')
    for d, kind, what in EVENTS:
        e = int(np.searchsorted(dates, d))
        assert dates[e] == d, (d, dates[e])
        if e + 21 >= len(axis):
            continue

        def rel(names, a, b):
            v, n = K.basket_log(P, names, a, b)
            s, _ = K.basket_log(P, ('SPY',), a, b)
            return v - s, n
        pre, _ = rel(K.ALC, e - 20, e)
        a0, _ = rel(K.ALC, e - 1, e)           # reported addition (Decisions): the event day's own close-to-close
        a2, _ = rel(K.ALC, e, e + 2)
        a21, _ = rel(K.ALC, e, e + 21)
        if d >= '2018-06-01':
            c0, _ = rel(K.CANNABIS, e - 1, e)
            c2, nc = rel(K.CANNABIS, e, e + 2)
            c21, _ = rel(K.CANNABIS, e, e + 21)
        else:
            c0 = c2 = c21 = np.nan
            nc = 0
        rows.append({'date': d, 'kind': kind, 'what': what, 'alc_pre': pre, 'alc_day': a0, 'alc_2': a2,
                     'alc_21': a21, 'can_day': c0, 'can_2': c2, 'can_21': c21, 'can_names': nc})
        out(f'{d}  {kind:7s}  {pre:+.4f}  {a0:+.4f}  {a2:+.4f}  {a21:+.4f}    {c0:+.4f}  {c2:+.4f}  {c21:+.4f}  '
            f'{nc}  {what}')
    res = {'events': rows}
    for nm, m in (('all', lambda r: True), ('2008-2019', lambda r: r['date'] <= '2019-12-31'),
                  ('2020-2025-06', lambda r: r['date'] >= '2020-01-01')):
        sub = [r for r in rows if m(r)]
        res[nm] = {k: summary([r[k] for r in sub]) for k in ('alc_pre', 'alc_day', 'alc_2', 'alc_21', 'can_day', 'can_2', 'can_21')}
        out(f'{nm}: ' + '; '.join(f'{k} n {v["n"]} mean {v.get("mean", np.nan):+.4f} t {v.get("t", np.nan):+.2f} '
                                  f'negative {v.get("share_negative", np.nan):.2f}' for k, v in res[nm].items()))
    (K.HERE / f'events_{tag}.txt').write_text('\n'.join(lines) + '\n')
    K.dump(res, K.HERE / f'events_{tag}.json')


if __name__ == '__main__':
    main(sys.argv[1])
