"""State-level partisan control changes: governors (Wikipedia yearly results) + legislatures (Ballotpedia).

Scores: R = +1, D = -1, independent/other/split = 0.  A change is (new - old) / 2, so D->R = +1, R->D = -1.
Per-state series value on a date = governor change + (upper change + lower change) / 2   (range -2..+2).
"""
import datetime as dt
import html
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, WORK, catalog, qc, reset_group, write_series

G = 'states'
reset_group(G)
D = dt.date.fromisoformat
START, END = dt.date(1995, 1, 1), dt.date(2026, 9, 30)
DAYS = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
ABBR = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO',
        'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
        'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
        'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
        'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
        'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
        'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
        'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX',
        'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
        'Wisconsin': 'WI', 'Wyoming': 'WY'}
SCORE = {'R': 1, 'D': -1, 'O': 0, 'I': 0, 'S': 0}
ODD_YEAR_LEG = {'LA', 'MS', 'NJ', 'VA'}
MN_SENATE = {1992, 1996, 2000, 2002, 2006, 2010, 2012, 2016, 2020, 2022}
NJ_SENATE = {1993, 1997, 2001, 2003, 2007, 2011, 2013, 2017, 2021, 2023}


def leg_election_year(ab, chamber, y):
    """True if chamber `chamber` of state `ab` held a regular general election in year y."""
    upper = chamber == 'Senate'
    if ab in ('LA', 'MS'):
        return y % 4 == 3                      # 1995, 1999, ..., 2023
    if ab == 'VA':
        return (y % 4 == 3) if upper else (y % 2 == 1)
    if ab == 'NJ':
        return (y in NJ_SENATE) if upper else (y % 2 == 1)
    if ab in ('AL', 'MD') or (ab == 'MI' and upper):
        return y % 4 == 2                      # 1994, 1998, ..., 2022
    if ab == 'MN' and upper:
        return y in MN_SENATE
    return y % 2 == 0


def election_day(y):
    d = dt.date(y, 11, 1)
    while d.weekday() != 0:
        d += dt.timedelta(days=1)
    return d + dt.timedelta(days=1)


# Louisiana gubernatorial decisive dates (jungle primary or runoff) and legislative general/runoff dates
LA_GOV = {1999: '1999-10-23', 2003: '2003-11-15', 2007: '2007-10-20', 2011: '2011-10-22', 2015: '2015-11-21',
          2019: '2019-11-16', 2023: '2023-10-14'}
LA_LEG = {1999: '1999-11-20', 2003: '2003-11-15', 2007: '2007-11-17', 2011: '2011-11-19', 2015: '2015-11-21',
          2019: '2019-11-16', 2023: '2023-11-18'}
SPECIAL_DATES = {(2003, 'California'): '2003-10-07', (2012, 'Wisconsin'): '2012-06-05',
                 (2021, 'California'): '2021-09-14'}
# Close flips (winner lead < ~2.2 pts): first day by whose US close the outcome was public (researched; see notes)
CALLED = json.load(open(os.path.join(WORK, 'close_race_calls.json')))  # {"YYYY|State": ["YYYY-MM-DD", "why"]}
# Governor party changes outside elections: (known date, state, old, new, note)
MIDTERM = [('2008-12-01', 'Arizona', 'D', 'R', 'Napolitano nominated DHS Secretary (announced Dec 1, 2008); '
            'Secretary of State Brewer (R) succeeded on 2009-01-21'),
           ('2010-04-30', 'Florida', 'R', 'O', 'Crist left the GOP to run as an independent (announced evening Apr 29)'),
           ('2013-05-31', 'Rhode Island', 'O', 'D', 'Chafee (I) joined the Democratic Party (announced May 30)'),
           ('2017-08-04', 'West Virginia', 'D', 'R', 'Justice switched to the GOP (announced evening Aug 3)')]

races = json.load(open(os.path.join(WORK, 'governor_races.json')))
races += json.load(open(os.path.join(WORK, 'governor_races_manual.json')))
races = [r for r in races if r['year'] >= 1996]
races.sort(key=lambda r: (r['year'], SPECIAL_DATES.get((r['year'], r['state']), '9999'), r['state']))

gov_events = []      # (date, state, change, note)
race_rows = []       # (date, state, margin_rd, prev_margin_rd, lead)
prev_margin = {}
close_holds = []
for r in races:
    st, y = r['state'], r['year']
    if (y, st) in SPECIAL_DATES:
        eday = D(SPECIAL_DATES[(y, st)])
    elif st == 'Louisiana' and y in LA_GOV:
        eday = D(LA_GOV[y])
    else:
        eday = election_day(y)
    lead = (r['winner_pct'] - r['runnerup_pct']) if (r['winner_pct'] is not None and r['runnerup_pct'] is not None) else None
    key = f'{y}|{st}'
    if key in CALLED:
        known = D(CALLED[key][0])
    elif lead is not None and lead < 2.0:
        known = eday + dt.timedelta(days=14)      # conservative for close races without a researched call date
        close_holds.append(key)
    else:
        known = eday + dt.timedelta(days=1)
    inc, win = r['incumbent_party'], r['winner_party']
    if inc and win and inc != win:
        gov_events.append((known, st, (SCORE[win] - SCORE[inc]) / 2.0, f'{y} {st} {inc}->{win}'))
    m = r['margin_rd']
    race_rows.append((known, st, m, prev_margin.get(st), lead, y))
    if m is not None:
        prev_margin[st] = m
for d, st, old, new, note in MIDTERM:
    gov_events.append((D(d), st, (SCORE[new] - SCORE[old]) / 2.0, note))

# ---------------------------------------------------------------- legislatures (Ballotpedia annual control)
t = open(os.path.join(RAW, 'ballotpedia', 'Historical_and_potential_changes_in_trifectas.html'), encoding='utf-8',
         errors='replace').read()
pos = [m.start() for m in re.finditer(r'<table', t)]
tabs = re.findall(r'<table.*?</table>', t, flags=re.S)
leg = {}
for i, tb in enumerate(tabs):
    rows = [[html.unescape(re.sub(r'<[^>]+>', '', c)).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, flags=re.S)]
            for r in re.findall(r'<tr[^>]*>(.*?)</tr>', tb, flags=re.S)]
    if not rows or rows[0][:2] != ['Year', '92']:
        continue
    pre = t[pos[i - 1]:pos[i]]
    m = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)?) Party Control: 1992', html.unescape(re.sub(r'<[^>]+>', ' ', pre)))
    st = m[-1]
    years = [1900 + int(x) if int(x) > 50 else 2000 + int(x) for x in rows[0][1:]]
    ch = {}
    for row in rows[1:]:
        ch[row[0]] = dict(zip(years, row[1:]))
    leg[st] = ch
assert len(leg) == 50, len(leg)
leg_events = []
skipped_nonelection = []
for st, ch in leg.items():
    ab = ABBR[st]
    for cname, byyear in ch.items():
        if cname == 'Governor':
            continue
        for y in range(1993, 2027):
            a, b = byyear.get(y - 1), byyear.get(y)
            if a in (None, '-', '') or b in (None, '-', ''):
                continue
            if a == b:
                continue
            ey = y - 1
            regular = leg_election_year(ab, cname, ey)
            if not regular:
                skipped_nonelection.append(f'{st} {cname} {ey}->{y} {a}->{b}')
                continue
            if ab == 'LA' and ey in LA_LEG:
                eday = D(LA_LEG[ey])
            else:
                eday = election_day(ey)
            leg_events.append((eday + dt.timedelta(days=2), st, cname, (SCORE.get(b, 0) - SCORE.get(a, 0)) / 2.0,
                               f'{st} {cname} {ey} {a}->{b}'))
print('governor changes', len(gov_events), '; legislative chamber changes', len(leg_events),
      '; non-election-year chamber changes skipped', len(skipped_nonelection))
print('close races without researched call (dated +14 d):', close_holds)

# cross-check governor flips vs Ballotpedia governor rows (year after the election)
mism = []
for d, st, v, note in gov_events:
    y = int(note[:4]) if note[:4].isdigit() else None
    if y is None:
        continue
    g = leg[st]['Governor']
    nxt = g.get(y + 1)
    win = note.split('->')[-1]
    if nxt and SCORE.get(nxt, 0) != SCORE.get(win, 0):
        mism.append((note, 'ballotpedia next-year governor', nxt))
print('governor flip vs Ballotpedia mismatches:', mism)


def daily_sum(events):
    acc = defaultdict(float)
    for e in events:
        acc[e[0]] += e[-2] if len(e) == 5 else e[2]
    return [(d, acc.get(d, 0.0)) for d in DAYS]


BIG15 = ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania', 'Illinois', 'Ohio', 'Georgia',
         'North Carolina', 'Michigan', 'New Jersey', 'Virginia', 'Washington', 'Arizona', 'Massachusetts']
GOVSRC = 'Wikipedia "YYYY United States gubernatorial elections" summary tables (raw/wikipedia_elections)'
LEGSRC = 'Ballotpedia "Historical and potential changes in trifectas" state tables 1992-2026 (GFDL)'
for st in BIG15:
    acc = defaultdict(float)
    notes = []
    for d, s, v, note in gov_events:
        if s == st and d >= START:
            acc[d] += v
            notes.append(f'{d} gov {v:+g} ({note.split(" ", 1)[-1] if note[:4].isdigit() else note[:40]})')
    for d, s, cname, v, note in leg_events:
        if s == st and d >= START:
            acc[d] += v / 2.0
            notes.append(f'{d} {cname} {v / 2:+g}')
    rows = [(d, acc.get(d, 0.0)) for d in DAYS]
    sid = f'pol_ctl_{ABBR[st].lower()}'
    write_series(sid, rows)
    rep = qc(sid, 'daily', show=0, quiet=True)
    print(f"{sid}: nonzero={sum(1 for _, v in rows if v)} {notes}")
    catalog(G, sid, name=f'{st}: change in partisan control of state government (governor +-1, each chamber +-0.5; + toward R)',
            subcategory='state_partisan_control', region=f'US-{ABBR[st]}', source=GOVSRC + '; ' + LEGSRC,
            url='https://ballotpedia.org/Historical_and_potential_changes_in_trifectas', frequency='daily',
            units='change in control score (governor R=+1/D=-1 counts 1, each legislative chamber counts 0.5)',
            transform='level', known_after_days=0,
            notes=('SPARSE event series (0 on all other days). Governor changes dated the day after the election '
                   '(close races: researched call/concession date, else +14 d); legislative chamber changes dated '
                   'election day + 2 (chamber control from annual Ballotpedia tables; only changes following a '
                   'regular legislative election are kept). Events: ' + ('; '.join(notes) if notes else 'none')))

# ---------------------------------------------------------------- national aggregates of state-level changes
rows = daily_sum([(d, s, v, n) for d, s, v, n in gov_events])
write_series('pol_gov_net_flips', rows)
catalog(G, 'pol_gov_net_flips', name='Net US governorships changing party toward Republicans (all 50 states)',
        subcategory='state_partisan_control', region='US', source=GOVSRC + '; mid-term switches hand-coded',
        url='https://en.wikipedia.org/wiki/2018_United_States_gubernatorial_elections', frequency='daily',
        units='sum of governor changes (D->R +1, R->D -1, to/from independent +-0.5)', transform='level',
        known_after_days=0,
        notes='Aggregate of state events; dated as in pol_ctl_*. Mid-term party changes: '
              + '; '.join(f'{d} {s} {o}->{n}' for d, s, o, n, _ in MIDTERM) + '. 1996-2025.')
rows = daily_sum([(d, s, v, n) for d, s, c, v, n in leg_events])
write_series('pol_leg_net_flips', rows)
catalog(G, 'pol_leg_net_flips', name='Net US state legislative chambers changing control toward Republicans',
        subcategory='state_partisan_control', region='US', source=LEGSRC,
        url='https://ballotpedia.org/Historical_and_potential_changes_in_trifectas', frequency='daily',
        units='sum of chamber changes (D->R +1, R->D -1, to/from split +-0.5)', transform='level', known_after_days=0,
        notes=('Annual control per chamber from Ballotpedia, changes following regular elections only (LA, MS, NJ, VA '
               'odd years), dated election day + 2 as a conservative bound. Mid-session changes (party switches, '
               f'coalitions, special elections; {len(skipped_nonelection)} chamber-year transitions) are not included. '
               'Nebraska (nonpartisan) excluded.'))
# swing and closeness
sw = defaultdict(list)
cl = defaultdict(int)
for known, st, m, pm, lead, y in race_rows:
    if m is not None and pm is not None:
        sw[known].append(m - pm)
    if lead is not None and lead < 3.0:
        cl[known] += 1
rows = [(d, sum(sw[d]) / len(sw[d]) if d in sw else 0.0) for d in DAYS]
write_series('pol_gov_swing', rows)
catalog(G, 'pol_gov_swing', name='Mean swing toward Republicans in governor races decided that day (vs the previous race for the seat)',
        subcategory='election_closeness', region='US', source=GOVSRC,
        url='https://en.wikipedia.org/wiki/2018_United_States_gubernatorial_elections', frequency='daily',
        units='percentage points of (R share - D share), change vs previous race in the same state; 0 = no race',
        transform='level', known_after_days=0,
        notes=('Uses final certified shares (election-night shares differed slightly: mild magnitude look-ahead; '
               'the sign and size were broadly public). Races without both an R and a D candidate omitted. '
               '0 on days without decided races. Covers 1996-2025 races; each state\'s first race in the sample only '
               'provides the baseline (no swing value).'))
rows = [(d, float(cl.get(d, 0))) for d in DAYS]
write_series('pol_gov_close_races', rows)
catalog(G, 'pol_gov_close_races', name='Number of governor races decided that day with a winning lead under 3 points',
        subcategory='election_closeness', region='US', source=GOVSRC,
        url='https://en.wikipedia.org/wiki/2018_United_States_gubernatorial_elections', frequency='daily',
        units='count of races', transform='level', known_after_days=0,
        notes='Lead = winner share - runner-up share (final results). Close races resolved late are dated at the '
              'researched call date, else election day + 14 d.')
for sid in ['pol_gov_net_flips', 'pol_leg_net_flips', 'pol_gov_swing', 'pol_gov_close_races']:
    rep = qc(sid, 'daily', show=0, quiet=True)
    rr = [v for d, v in [(x, y) for x, y in []]]
    print(sid, rep['first'], rep['last'], 'n', rep['n'], 'min', rep['min'], 'max', rep['max'])
