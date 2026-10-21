"""US federal political event series (daily, hand-coded from cited sources).

Sign convention for control series: +1 = a branch moves to Republican control, -1 = to Democratic control.
Event dates are the first calendar day by whose 4 pm New York close the outcome was public (known_after_days=0).
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import catalog, qc, reset_group, write_series

G = 'federal'
reset_group(G)
START, END = dt.date(1990, 1, 1), dt.date(2026, 9, 30)
DAYS = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
D = dt.date.fromisoformat


def election_day(y):
    """First Tuesday after the first Monday in November."""
    d = dt.date(y, 11, 1)
    while d.weekday() != 0:      # Monday
        d += dt.timedelta(days=1)
    return d + dt.timedelta(days=1)


def daily(events, default=0.0):
    out = {}
    for d, v in events:
        out[d] = out.get(d, 0.0) + v
    return [(d, out.get(d, default)) for d in DAYS]


def emit(sid, rows, **cat):
    write_series(sid, rows)
    rep = qc(sid, 'daily', show=0, quiet=True)
    nz = sum(1 for _, v in rows if v != 0)
    print(f"{sid}: {rep['first']}..{rep['last']} n={rep['n']} nonzero={nz} min={rep['min']} max={rep['max']}")
    catalog(G, sid, frequency='daily', **cat)


SEN = 'senate.gov party divisions (raw/politics_events/senate_partydiv.html)'
WIKI_CONG = 'Wikipedia "Party divisions of United States Congresses"'

# ---------------------------------------------------------------- election days (scheduled)
ed = [(election_day(y), 1.0) for y in range(1990, 2026, 2)]
emit('pol_us_election_day', daily(ed), name='US federal general election day (0/1)', subcategory='election_calendar',
     region='US', source='Statutory date (first Tuesday after the first Monday in November, even years)',
     url='https://www.law.cornell.edu/uscode/text/2/7', units='indicator (1 on election day)', transform='level',
     known_after_days=0, notes='Scheduled years in advance; carries no surprise. 1990-2024.')

# ---------------------------------------------------------------- presidency flips, dated when the winner was public
PRES = [  # (result-known date, R-D popular-vote margin in pct points, flip value, note)
    ('1992-11-04', None, -1, 'Clinton projected ~11 pm ET Nov 3'),
    ('1996-11-06', None, 0, 'Clinton re-election projected ~9 pm ET Nov 5'),
    ('2000-12-13', None, +1, 'Bush v. Gore decided ~10 pm ET Dec 12; Gore conceded Dec 13'),
    ('2004-11-03', None, 0, 'Kerry conceded ~11 am ET Nov 3'),
    ('2008-11-05', None, -1, 'Obama projected 11 pm ET Nov 4'),
    ('2012-11-07', None, 0, 'Obama re-election projected ~11:15 pm ET Nov 6'),
    ('2016-11-09', None, +1, 'Trump projected ~2:30 am ET Nov 9'),
    ('2020-11-07', None, -1, 'Biden projected ~11:25 am ET Saturday Nov 7'),
    ('2024-11-06', None, +1, 'Trump projected ~5:30 am ET Nov 6'),
]
emit('pol_us_pres_flip', daily([(D(d), float(f)) for d, _, f, _ in PRES if f]),
     name='US presidency changes party, dated when the winner became public (+1 to R, -1 to D)',
     subcategory='federal_control_change', region='US', source='Network/AP projection and concession times',
     url='https://en.wikipedia.org/wiki/List_of_United_States_presidential_elections_by_popular_vote_margin',
     units='+1 presidency flips to Republican, -1 to Democratic, 0 otherwise', transform='level', known_after_days=0,
     notes='Result-known dates: ' + '; '.join(f'{d} {n}' for d, _, _, n in PRES) +
           '. Holds (1996, 2004, 2012) are 0. Dated so that the value is public by that day\'s 4 pm ET close.')


# ---------------------------------------------------------------- presidential popular-vote margin (closeness)
import json, re
from common import RAW
_t = json.load(open(os.path.join(RAW, 'wikipedia_elections', 'List_of_United_States_presidential_elections_by_popular_vote_margin.json')))['parse']['wikitext']
_t = re.sub(r'<ref.*?(/>|</ref>)', '', _t, flags=re.S)
MARGIN = {}
for blk in _t.split('\n|-'):
    m = re.search(r'\[\[(\d{4}) United States presidential election\|\d{4}\]\]', blk)
    if not m:
        continue
    y = int(m.group(1))
    if not 1992 <= y <= 2024 or y in MARGIN:
        continue
    pm_ = re.search(r'\|\s*(Rep|Dem)\.', blk)
    pcts = re.findall(r'\n\|\s*([\u2212\-]?\d+\.\d+)%', blk)
    if not pm_ or len(pcts) < 3:
        continue
    party = pm_.group(1)
    win_margin = float(pcts[2].replace('\u2212', '-'))     # EV%, PV%, PV margin%
    MARGIN[y] = win_margin if party == 'Rep' else -win_margin
assert sorted(MARGIN) == list(range(1992, 2025, 4)), MARGIN
print('presidential R-D margins:', MARGIN)
pm = [(D(d), MARGIN[int(d[:4]) if d[5:7] != '12' else int(d[:4])]) for d, _, _, _ in PRES]
emit('pol_us_pres_margin', daily(pm),
     name='US presidential election: Republican minus Democratic national popular-vote margin, dated when the winner became public',
     subcategory='election_closeness', region='US',
     source='Wikipedia "List of United States presidential elections by popular vote margin" (FEC counts)',
     url='https://en.wikipedia.org/wiki/List_of_United_States_presidential_elections_by_popular_vote_margin',
     units='percentage points of the national popular vote (R minus D); 0 on non-event days', transform='level',
     known_after_days=0,
     notes='Values ' + ', '.join(f'{y}: {v:+.2f}' for y, v in sorted(MARGIN.items())) + '. Final certified margins: '
           'the margin reported on the result day differed by up to ~2 points (late-counted ballots), so the '
           'magnitude is mildly look-ahead; sign and rough size were public. Dated as pol_us_pres_flip.')

# ---------------------------------------------------------------- Congress control flips (result known)
CONG = [  # (date, value, note)
    ('1994-11-09', +2, 'House and Senate to R, called on election night Nov 8'),
    ('2001-05-24', -1, 'Senate to D: Jeffords announced leaving the GOP (effective Jun 6)'),
    ('2002-11-06', +1, 'Senate to R, called on election night Nov 5'),
    ('2006-11-08', -1, 'House to D, called on election night Nov 7'),
    ('2006-11-09', -1, 'Senate to D: Virginia (Webb) settled, Allen conceded Nov 9'),
    ('2010-11-03', +1, 'House to R, called on election night Nov 2'),
    ('2014-11-05', +1, 'Senate to R, called ~11:30 pm ET Nov 4'),
    ('2018-11-07', -1, 'House to D, called ~11 pm ET Nov 6'),
    ('2021-01-07', -1, 'Senate to D: Georgia runoffs Jan 5; Ossoff called late Jan 6 (dated Jan 7)'),
    ('2022-11-17', +1, 'House to R: AP called the majority ~7 pm ET Nov 16'),
    ('2024-11-06', +1, 'Senate to R, called on election night Nov 5'),
]
emit('pol_us_congress_flip', daily([(D(d), float(v)) for d, v, _ in CONG]),
     name='US House/Senate majority changes party, dated when it became public (+1 per chamber to R, -1 to D)',
     subcategory='federal_control_change', region='US', source=SEN + '; ' + WIKI_CONG + '; AP race calls',
     url='https://www.senate.gov/history/partydiv.htm', units='net chambers moving to Republican control',
     transform='level', known_after_days=0,
     notes='Events: ' + '; '.join(f'{d} {v:+d} {n}' for d, v, n in CONG) +
           '. 2000 Senate (50-50, R control via VP after Jan 20) treated as a hold. Conservative dating for late calls.')

# ---------------------------------------------------------------- power changes hands (scheduled)
POWER = [  # (effective date, value, note)
    ('1993-01-20', -1, 'presidency to D (Clinton)'),
    ('1995-01-04', +2, '104th Congress: House and Senate to R'),
    ('2001-01-03', -1, '107th Congress convenes 50-50, VP Gore gives D the Senate'),
    ('2001-01-20', +2, 'presidency to R (Bush); VP Cheney gives R the Senate'),
    ('2001-06-06', -1, 'Senate to D (Jeffords switch effective)'),
    ('2002-11-12', +1, 'Senate to R in lame duck (Talent sworn in)'),
    ('2007-01-04', -2, '110th Congress: House and Senate to D'),
    ('2009-01-20', -1, 'presidency to D (Obama)'),
    ('2011-01-05', +1, '112th Congress: House to R'),
    ('2015-01-06', +1, '114th Congress: Senate to R'),
    ('2017-01-20', +1, 'presidency to R (Trump)'),
    ('2019-01-03', -1, '116th Congress: House to D'),
    ('2021-01-20', -2, 'presidency to D (Biden); Senate to D (VP Harris, Georgia senators sworn in)'),
    ('2023-01-03', +1, '118th Congress: House to R'),
    ('2025-01-03', +1, '119th Congress: Senate to R'),
    ('2025-01-20', +1, 'presidency to R (Trump)'),
]
emit('pol_us_power_change', daily([(D(d), float(v)) for d, v, _ in POWER]),
     name='US federal power changes hands (inauguration / new majority takes over; +1 per branch to R, -1 to D)',
     subcategory='federal_control_change', region='US', source=SEN + '; ' + WIKI_CONG,
     url='https://www.senate.gov/history/partydiv.htm', units='net branches (presidency, House, Senate) moving to R',
     transform='level', known_after_days=0,
     notes='Effective dates, scheduled once the result is known, so little surprise: '
           + '; '.join(f'{d} {v:+d} {n}' for d, v, n in POWER) + '.')

# ---------------------------------------------------------------- shutdowns (lapse in appropriations with furloughs)
SHUT = [('1995-11-14', '1995-11-19'), ('1995-12-16', '1996-01-05'), ('2013-10-01', '2013-10-16'),
        ('2018-01-20', '2018-01-22'), ('2018-12-22', '2019-01-25'), ('2025-10-01', '2025-11-12'),
        ('2026-01-31', '2026-02-03'), ('2026-02-14', '2026-04-30')]
ev = []
for a, b in SHUT:
    d = D(a)
    while d <= D(b):
        ev.append((d, 1.0))
        d += dt.timedelta(days=1)
emit('pol_us_shutdown', daily(ev), name='US federal government shutdown in progress (0/1)',
     subcategory='fiscal_standoff', region='US',
     source='Wikipedia shutdown articles (1995-96, 2013, Jan 2018, 2018-19, 2025, 2026) and overview table',
     url='https://en.wikipedia.org/wiki/Government_shutdowns_in_the_United_States',
     units='indicator (1 = appropriations lapsed with furloughs that day)', transform='level', known_after_days=0,
     notes='Periods: ' + '; '.join(f'{a}..{b}' for a, b in SHUT) + '. The few-hour lapse of 2018-02-09 is not '
           'counted. 2026-02-14..04-30 was DHS-only (partial). A lapse is known in advance (midnight deadline), so 0. '
           'Series ends 2026-09-30.')

# ---------------------------------------------------------------- debt limit binding (Treasury extraordinary measures)
DISP = [('2011-05-16', '2011-08-02'), ('2012-12-31', '2013-02-04'), ('2013-05-20', '2013-10-17'),
        ('2014-02-10', '2014-02-15'), ('2015-03-16', '2015-11-02'), ('2017-03-16', '2017-09-08'),
        ('2017-12-11', '2018-02-09'), ('2019-03-04', '2019-08-02'), ('2021-08-02', '2021-12-16'),
        ('2023-01-19', '2023-06-03'), ('2025-01-21', '2025-07-04')]
ev = []
for a, b in DISP:
    d = D(a)
    while d <= D(b):
        ev.append((d, 1.0))
        d += dt.timedelta(days=1)
rows = [(d, v) for d, v in daily(ev) if d >= dt.date(2011, 1, 1)]
write_series('pol_us_debtlimit_binding', rows)
rep = qc('pol_us_debtlimit_binding', 'daily', show=0, quiet=True)
print(f"pol_us_debtlimit_binding: {rep['first']}..{rep['last']} n={rep['n']} days_on={sum(v for _, v in rows):.0f}")
catalog(G, 'pol_us_debtlimit_binding', name='US debt limit binding: Treasury in a debt issuance suspension period (0/1)',
        subcategory='fiscal_standoff', region='US',
        source='CRS R43389 "The Debt Limit Since 2011" (Table 2, text); 2023 from Wikipedia; 2025 from Treasury letters (1/17/25, OCR) and Treasury debt-limit page',
        url='https://www.everycrsreport.com/reports/R43389.html', frequency='daily',
        units='indicator (1 from the day extraordinary measures were invoked through enactment of the fix)',
        transform='level', known_after_days=0,
        notes='Periods: ' + '; '.join(f'{a}..{b}' for a, b in DISP) + '. Starts 2011-01-01: earlier episodes '
              '(2002-2007) need GAO DISP tables that could not be downloaded (gao.gov returned 403). 2021 DISP was '
              'extended Oct 18 and Nov 16, 2021 and ended with the Dec 16, 2021 increase. 2025: Yellen letter of Jan 17, 2025 (scanned PDF, OCR) sets the DISP start on Tuesday Jan 21, 2025; Treasury lists a DISP ending July 4, 2025 (OBBBA raised the limit by $5T). A DISP is announced in advance, so known_after_days=0. Ends 2026-09-30.')
