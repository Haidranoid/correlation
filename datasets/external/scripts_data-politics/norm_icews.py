"""ICEWS daily counts (from work/icews_agg.json) -> series."""
import datetime as dt
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, catalog, qc, reset_group, write_series

G = 'icews'
reset_group(G)
A = json.load(open(os.path.join(WORK, 'icews_agg.json')))['counts']
START, END = dt.date(1995, 1, 1), dt.date(2023, 4, 9)   # 2023-04-10 (last file day) is incomplete
days = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
SRC = 'ICEWS Coded Event Data (Lockheed Martin / Harvard Dataverse doi:10.7910/DVN/28075)'
URL = 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075'
COMMON = ('Machine-coded (CAMEO) events from international news wires, located by event location (Country field). '
          'Ends 2023-04-09 (ICEWS discontinued; last file day 2023-04-10 incomplete and dropped). Coverage changes: '
          'yearly volumes shift with the source set (notably the 2014 production change and a dip in 2020); event IDs '
          'de-duplicated across files. Every calendar day written (0 = no coded event). Events were reported in the news '
          'on/after the day; the dataset itself was published with a lag (before v23 in 2018 the latest year was '
          'withheld), so known_after_days=1 is information timing, not vintage availability. Terms on Dataverse: '
          '"made available with limited information on how it can be used".')


def series(kind, countries):
    out = defaultdict(int)
    for c in countries:
        for d, n in A.get(f'{kind}|{c}', {}).items():
            out[d] += n
    return out


def world(kind):
    out = defaultdict(int)
    for k, v in A.items():
        if k.startswith(kind + '|') and not k.startswith(kind + '|US|'):
            for d, n in v.items():
                out[d] += n
    return out


def emit(sid, counts, name, sub, region, units, notes, share_of=None):
    if share_of is None:
        rows = [(d, counts.get(d.isoformat(), 0)) for d in days]
    else:
        rows = []
        for d in days:
            den = share_of.get(d.isoformat(), 0)
            if den >= 200:   # require a reasonable daily base; thin days left out
                rows.append((d, 100.0 * counts.get(d.isoformat(), 0) / den))
    write_series(sid, rows)
    rep = qc(sid, 'daily', lo=0, hi=1e7, show=0, quiet=True)
    catalog(G, sid, name=name, subcategory=sub, region=region, source=SRC, url=URL, frequency='daily', units=units,
            transform='level', known_after_days=1, notes=(notes + ' ' + COMMON).strip())
    print(f"{sid}: {rep['first']}..{rep['last']} n={rep['n']} min={rep['min']:.4g} max={rep['max']:.4g} "
          f"zeros={rep['zeros']} gaps={rep['gaps']}/{rep['gap_units']}")


wp, wa, wv = world('protest'), world('all'), world('violence')
emit('icews_protest_world', wp, 'Protest events worldwide per day (ICEWS, CAMEO root 14)', 'protest', 'WORLD',
     'events per day', 'All event locations.')
emit('icews_protest_share_world', wp, 'Protest share of all coded events worldwide (ICEWS, CAMEO root 14)', 'protest',
     'WORLD', 'percent of all coded events that day',
     'Stationary variant: protests / all events located anywhere (days with < 200 events left out).', share_of=wa)
emit('icews_violence_world', wv, 'Violent conflict events worldwide per day (ICEWS, CAMEO roots 18-20)',
     'violent_conflict', 'WORLD', 'events per day',
     'CAMEO 18 assault, 19 fight, 20 unconventional mass violence. Includes crime/policing coded as assault.')

PROTEST = [('China', 'cn', 'CN', 'China (excl. Hong Kong)'), ('Hong Kong', 'hk', 'HK', 'Hong Kong'),
           ('India', 'in', 'IN', 'India'), ('Russian Federation', 'ru', 'RU', 'Russia'),
           ('United Kingdom', 'gb', 'GB', 'United Kingdom'), ('France', 'fr', 'FR', 'France'),
           ('Germany', 'de', 'DE', 'Germany'), ('Brazil', 'br', 'BR', 'Brazil'), ('Turkey', 'tr', 'TR', 'Turkey'),
           ('Iran', 'ir', 'IR', 'Iran'), ('Egypt', 'eg', 'EG', 'Egypt'), ('Ukraine', 'ua', 'UA', 'Ukraine')]
for c, slug, iso2, label in PROTEST:
    emit(f'icews_protest_{slug}', series('protest', [c]), f'Protest events per day in {label} (ICEWS, CAMEO root 14)',
         'protest', iso2, 'events per day', f'Event location {c}.')

VIOL = [(['Ukraine'], 'ua', 'UA', 'Ukraine'),
        (['Israel', 'Occupied Palestinian Territory'], 'il_ps', 'IL;PS', 'Israel and the Palestinian territories'),
        (['Iraq'], 'iq', 'IQ', 'Iraq'), (['Syria'], 'sy', 'SY', 'Syria')]
for cs, slug, reg, label in VIOL:
    emit(f'icews_violence_{slug}', series('violence', cs),
         f'Violent conflict events per day in {label} (ICEWS, CAMEO roots 18-20)', 'violent_conflict', reg,
         'events per day', 'Event location ' + ' + '.join(cs) + '.')
