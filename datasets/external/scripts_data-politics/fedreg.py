"""Federal Register daily document counts via the federalregister.gov API (no key), 1994 onward.

Downloads one 'daily' facet per year per filter into raw/federal_register/, then writes series on publication
days only (days with at least one document of any type), so weekends/holidays are absent rather than zero.
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, catalog, qc, reset_group, write_series

OUT = os.path.join(RAW, 'federal_register')
os.makedirs(OUT, exist_ok=True)
UA = 'Mozilla/5.0 (research data download; contact bhoult)'
FILTERS = {
    'all': {},
    'rule': {'conditions[type][]': 'RULE'},
    'prorule': {'conditions[type][]': 'PRORULE'},
    'presdocu': {'conditions[type][]': 'PRESDOCU'},
    'sigrule': {'conditions[type][]': 'RULE', 'conditions[significant]': '1'},
}
YEARS = range(1994, 2027)
for key, cond in FILTERS.items():
    for y in YEARS:
        fn = os.path.join(OUT, f'daily_{key}_{y}.json')
        if os.path.exists(fn) and os.path.getsize(fn) > 100:
            continue
        q = dict(cond)
        q['conditions[publication_date][year]'] = str(y)
        url = 'https://www.federalregister.gov/api/v1/documents/facets/daily?' + urllib.parse.urlencode(q)
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': UA})
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                json.loads(body)
                open(fn, 'wb').write(body)
                break
            except Exception as e:
                print('retry', key, y, e)
                time.sleep(10 * (attempt + 1))
        time.sleep(1.0)
    print('downloaded', key)


def load(key):
    out = {}
    for y in YEARS:
        fn = os.path.join(OUT, f'daily_{key}_{y}.json')
        if not os.path.exists(fn):
            continue
        for d, v in json.load(open(fn)).items():
            out[dt.date.fromisoformat(d)] = int(v['count'])
    return out


G = 'fedreg'
reset_group(G)
allc = load('all')
today = dt.date(2026, 10, 17)
pubdays = sorted(d for d, c in allc.items() if c > 0 and d <= today)
print('publication days', len(pubdays), pubdays[0], pubdays[-1])
START = {'rule': dt.date(1995, 1, 1), 'prorule': dt.date(1995, 1, 1), 'presdocu': dt.date(1994, 1, 1),
         'sigrule': dt.date(1996, 1, 1)}   # API coverage of 1994 rules is partial; significance flag from 1996
SPEC = [('rule', 'fr_final_rules', 'Federal Register: final rules published per day', 'final rules (type RULE)'),
        ('prorule', 'fr_proposed_rules', 'Federal Register: proposed rules published per day', 'proposed rules (type PRORULE)'),
        ('presdocu', 'fr_presidential_docs', 'Federal Register: presidential documents published per day',
         'presidential documents (executive orders, proclamations, memoranda, determinations)'),
        ('sigrule', 'fr_significant_rules', 'Federal Register: significant final rules (EO 12866) published per day',
         'final rules flagged significant under EO 12866')]
for key, sid, name, what in SPEC:
    c = load(key)
    rows = [(d, c.get(d, 0)) for d in pubdays if d >= START[key]]
    write_series(sid, rows)
    rep = qc(sid, 'daily', lo=0, hi=5000, show=2, quiet=False)
    catalog(G, sid, name=name, subcategory='federal_policy_activity', region='US',
            source='Office of the Federal Register via federalregister.gov API (public domain)',
            url='https://www.federalregister.gov/developers/documentation/api/v1', frequency='daily',
            units=f'documents per publication day: {what}', transform='level', known_after_days=0,
            notes=('Publication days only (weekends/federal holidays absent, not zero). The daily issue is public by '
                   '~6 am ET (public inspection the previous afternoon), so 0. API rule coverage is partial in 1994 (series start '
                   f'{START[key]}); counts are '
                   'by publication date and include corrections only as filed by type. '
                   + ('The significance flag is sparse/inconsistent before the late 1990s. ' if key == 'sigrule' else '')
                   + 'Strong end-of-administration surges (midnight regulations) and weekday patterns.'))
