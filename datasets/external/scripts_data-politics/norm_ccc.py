"""Crowd Counting Consortium (Harvard Dataverse, CC0) -> daily US protest series."""
import csv
import datetime as dt
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, catalog, qc, reset_group, write_series

G = 'ccc'
reset_group(G)
csv.field_size_limit(10 ** 9)
D = os.path.join(RAW, 'ccc')
FILES = [('ccc_compiled_20172020.tab', '\t', 'doi:10.7910/DVN/6OPP7H'),
         ('ccc_compiled_20212024.tab', '\t', 'doi:10.7910/DVN/9MMYDI'),
         ('ccc-phase3-public.csv', ',', 'doi:10.7910/DVN/RI9JFU')]

ev = Counter()
left = Counter()
right = Counter()
disrupt = Counter()
size = defaultdict(float)
by_state = defaultdict(Counter)
skipped = Counter()
dmin, dmax = '9999', '0000'


def flag(row, *names):
    for n in names:
        v = (row.get(n) or '').strip()
        if v not in ('', '0', 'NA', 'NULL'):
            try:
                if float(v) > 0:
                    return True
            except ValueError:
                return True
    return False


for fn, delim, doi in FILES:
    with open(os.path.join(D, fn), encoding='utf-8', errors='replace', newline='') as f:
        for row in csv.DictReader(f, delimiter=delim):
            d = (row.get('date') or '').strip()[:10]
            try:
                dt.date.fromisoformat(d)
            except ValueError:
                skipped['bad_date'] += 1
                continue
            if (row.get('online') or '').strip() == '1':
                skipped['online'] += 1
                continue
            dmin, dmax = min(dmin, d), max(dmax, d)
            ev[d] += 1
            v = (row.get('valence') or '').strip()
            if v == '1':
                left[d] += 1
            elif v == '2':
                right[d] += 1
            if flag(row, 'arrests_any', 'property_damage_any', 'injuries_crowd_any', 'injuries_police_any',
                    'participant_casualties_any', 'police_casualties_any'):
                disrupt[d] += 1
            try:
                s = float(row.get('size_mean') or 'nan')
                if s == s and s > 0:
                    size[d] += s
            except ValueError:
                pass
            st = (row.get('state') or '').strip().upper()
            by_state[st][d] += 1

print('skipped', dict(skipped), 'range', dmin, dmax)
start, end = dt.date.fromisoformat(dmin), dt.date.fromisoformat(dmax)
days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
tot_state = sorted(((sum(c.values()), s) for s, c in by_state.items() if len(s) == 2), reverse=True)
print('top states', tot_state[:14])

COMMON = ('Crowd Counting Consortium (Harvard Kennedy School / UConn) compiles events from news, social media and '
          'organizer reports, with retroactive additions. Phase breaks: 2017-2020, 2021-2024 (adds many strike/picket '
          'and other events; roughly 2x the yearly count), 2025- (new coding guidelines), so levels shift at '
          '2021-01-01 and 2025-01-01. Online-only events excluded. Every calendar day written (0 = no recorded event). '
          'Events were public on the day; the dataset is compiled later, so known_after_days=1 is information timing.')
URL = 'https://dataverse.harvard.edu/dataverse/crowdcountingconsortium'
SRC = 'Crowd Counting Consortium US protest event data (Dataverse 6OPP7H, 9MMYDI, RI9JFU; CC0)'


def emit(sid, counter, name, region, units, notes, lo=0, hi=1e9, fn=None):
    rows = [(d, (fn(counter.get(d.isoformat(), 0)) if fn else counter.get(d.isoformat(), 0))) for d in days]
    write_series(sid, rows)
    rep = qc(sid, 'daily', lo=lo, hi=hi, show=0, quiet=True)
    catalog(G, sid, name=name, subcategory='protest', region=region, source=SRC, url=URL, frequency='daily',
            units=units, transform='level', known_after_days=1, notes=notes + ' ' + COMMON)
    print(f"{sid}: {rep['first']}..{rep['last']} n={rep['n']} min={rep['min']} max={rep['max']} zeros={rep['zeros']}")


emit('ccc_us_events', ev, 'US protest events per day (all in-person events, CCC)', 'US', 'events per day',
     'All in-person protests, demonstrations, rallies, marches, vigils, strikes/pickets, walkouts.')
emit('ccc_us_events_left', left, 'US left-valence protest events per day (CCC valence=1, e.g. anti-Trump)', 'US',
     'events per day', 'CCC valence 1 = left-wing / anti-Trump.')
emit('ccc_us_events_right', right, 'US right-valence protest events per day (CCC valence=2, e.g. pro-Trump)', 'US',
     'events per day', 'CCC valence 2 = right-wing / pro-Trump.')
emit('ccc_us_events_disruptive', disrupt,
     'US protest events with arrests, injuries or property damage per day (CCC)', 'US', 'events per day',
     'Events with any arrests, crowd/police injuries (casualties in phase 3) or property damage.')
emit('ccc_us_participants_log', size, 'US protest participants per day, log scale (CCC size_mean, summed)', 'US',
     'ln(1 + sum of event size_mean estimates)',
     'Only about half of events carry a size estimate; days dominated by a few mass events (Women\'s March '
     '2017-01-21, June 2020, "No Kings" 2025). Log taken because raw totals span 0 to millions.',
     fn=lambda x: math.log1p(x))
STATES = {'CA': 'California', 'NY': 'New York', 'DC': 'District of Columbia', 'MA': 'Massachusetts',
          'IL': 'Illinois', 'PA': 'Pennsylvania', 'WA': 'Washington', 'TX': 'Texas', 'FL': 'Florida', 'OR': 'Oregon'}
for st, nm in STATES.items():
    emit(f'ccc_{st.lower()}_events', by_state[st], f'Protest events per day in {nm} (CCC)', f'US-{st}',
         'events per day', f'In-person events with state={st}.')
