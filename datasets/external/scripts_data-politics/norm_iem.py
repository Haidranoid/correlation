"""Iowa Electronic Markets monthly price-history tables -> daily election-market series.

Per market and day we use only contracts that actually traded that day (Units > 0), so no stale prices:
  p_day = mean over traded contracts of (price if contract is the 'D' side else 1 - price)   [2-contract markets]
For the combined Congress markets (DH_DS, DH_RS, RH_DS, RH_RS) the House-D probability is
  (DH_DS + DH_RS) / (sum of the four last prices), using the source's last prices, on days with >= 1 trade.
Cycles are chained: the stored value is the cumulative sum of within-cycle day-to-day changes (cross-cycle
jumps removed), so `diff` gives exactly the within-cycle change in the market-implied probability.
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

G = 'iem'
reset_group(G)
P = os.path.join(RAW, 'iem', 'prices')


def parse_market(name):
    out = defaultdict(dict)   # date -> contract -> (units, last)
    for fn in sorted(os.listdir(P)):
        if not fn.startswith(name + '_'):
            continue
        t = open(os.path.join(P, fn), encoding='utf-8', errors='replace').read()
        for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', t, flags=re.S | re.I):
            # <td> cells are not always closed in the source HTML: split on each opening tag instead
            cells = [html.unescape(re.sub(r'<[^>]+>', '', c)).strip() for c in re.split(r'<td[^>]*>', tr, flags=re.I)[1:]]
            cells = [c for c in cells if c != '']
            if len(cells) != 7 or not re.fullmatch(r'\d{2}/\d{2}/\d{2}', cells[0]):
                continue
            d = dt.datetime.strptime(cells[0], '%m/%d/%y').date()
            units = float(cells[2].replace(',', ''))
            last = float(cells[6])
            out[d][cells[1]] = (units, last)
    return out


MARKETS = sorted(set(re.sub(r'_\d{4}_\d{2}\.html$', '', f) for f in os.listdir(P)))
contracts = {}
data = {}
for m in MARKETS:
    data[m] = parse_market(m)
    contracts[m] = sorted({c for d in data[m].values() for c in d})
json.dump({m: [contracts[m], min(data[m]).isoformat() if data[m] else None,
               max(data[m]).isoformat() if data[m] else None] for m in MARKETS},
          open(os.path.join(WORK, 'iem_markets.json'), 'w'), indent=1)
for m in MARKETS:
    print(m, contracts[m], (min(data[m]), max(data[m])) if data[m] else 'EMPTY')


def traded_pair(m, dname, rname):
    """D-side probability from the exact D/R contracts that traded that day (no stale prices)."""
    rows = []
    for d in sorted(data[m]):
        day = data[m][d]
        vals = []
        if dname in day and day[dname][0] > 0:
            vals.append(day[dname][1])
        if rname in day and day[rname][0] > 0:
            vals.append(1 - day[rname][1])
        if vals:
            rows.append((d, sum(vals) / len(vals)))
    return rows


def ratio_pair(m, dname, rname):
    """D / (D + R) from last prices, on days when either traded (3-party 2000 vote-share market)."""
    rows = []
    for d in sorted(data[m]):
        day = data[m][d]
        if not any(day.get(c, (0, 0))[0] > 0 for c in (dname, rname)):
            continue
        dl, rl = day.get(dname, (0, 0))[1], day.get(rname, (0, 0))[1]
        if dl + rl > 0.5:
            rows.append((d, dl / (dl + rl)))
    return rows


def congress(m, want):
    """P(Democratic/non-Republican control) of 'house' or 'senate' from the combined Congress market:
    sum of last prices of outcomes with that chamber D (or N = non-Republican) over the sum of the four named
    House x Senate outcomes (OTHER excluded); days with >= 1 trade in a named outcome and sum >= 0.5."""
    rows = []
    ever = set()                                       # contracts that have traded at least once so far
    for d in sorted(data[m]):
        day = {c: v for c, v in data[m][d].items() if not c.upper().startswith('OTHER')}
        ever |= {c for c, (u, _) in day.items() if u > 0}
        if not day or not any(u > 0 for u, _ in day.values()):
            continue
        tot = sum(last for _, last in day.values())
        if not (0.85 <= tot <= 1.2) or not all(c in ever for c in day):   # coherent, no never-traded outcome
            continue
        num = 0.0
        for c, (_, last) in day.items():
            h, s_ = re.sub(r'\d+$', '', c).split('_')
            side = h if want == 'house' else s_
            if side in ('DH', 'NH', 'NRH', 'DS', 'NS', 'NRS'):
                num += last
        rows.append((d, num / tot))
    return rows


def chain(cycles):
    """cycles: list of (label, rows). Returns chained cumulative series and boundary notes."""
    out = []
    level = 0.0
    notes = []
    last_date = None
    for label, rows in cycles:
        rows = [(d, v) for d, v in rows if last_date is None or d > last_date]
        if not rows:
            continue
        notes.append(f'{label} {rows[0][0]}..{rows[-1][0]} (p {rows[0][1]:.2f}->{rows[-1][1]:.2f})')
        prev = None
        for d, v in rows:
            if prev is not None:
                level += (v - prev) * 100.0
            out.append((d, round(level, 6)))
            prev = v
        last_date = rows[-1][0]
    return out, notes


def is_dem(c):
    return c.upper().startswith(('DEM', 'UDEM', 'D_', 'DH', 'DS')) or re.match(r'^(DEM|D)\d', c.upper()) is not None


def is_rep(c):
    return c.upper().startswith(('REP', 'UREP', 'R_', 'RH', 'RS')) or re.match(r'^(REP|R)\d', c.upper()) is not None


def emit(sid, cycles, name, units, notes):
    rows, bnotes = chain(cycles)
    write_series(sid, rows)
    rep = qc(sid, 'daily', show=0, quiet=True)
    print(f"{sid}: {rep['first']}..{rep['last']} n={rep['n']} min={rep['min']:.1f} max={rep['max']:.1f} gaps={rep['gaps']}")
    catalog(G, sid, name=name, subcategory='election_market', region='US',
            source='Iowa Electronic Markets (University of Iowa Tippie College of Business) price histories',
            url='https://iemweb.biz.uiowa.edu/markets/historical-data/', frequency='daily', units=units,
            transform='diff', known_after_days=1,
            notes=(notes + ' Cycles chained (cross-cycle jumps removed): ' + '; '.join(bnotes) + '. Days with no '
                   'trade in the relevant contracts are left out (thin, small-stakes market; many gaps). Last price = '
                   'last trade before midnight Central, hence known_after_days=1.'))


WTA = [('2000', 'pres00_wta', 'Dem', 'Rep'), ('2004', 'pres04_wta', 'DEM04', 'REP04'),
       ('2008', 'pres08_wta', 'DEM08_WTA', 'REP08_WTA'), ('2012', 'pres12_wta', 'DEM12_WTA', 'REP12_WTA'),
       ('2016', 'pres16_wta', 'DEM16_WTA', 'REP16_WTA'), ('2020', 'pres20_wta', 'DEM20_WTA', 'REP20_WTA'),
       ('2024', '2024-u-s-presidential-winner-takes-all-market', 'DEM24_WTA', 'REP24_WTA')]
wta = [(y, traded_pair(m, dn, rn)) for y, m, dn, rn in WTA]
VS = [('2008', 'pres08_vs', 'UDEM08_VS', 'UREP08_VS'), ('2012', 'pres12_vs', 'UDEM12_VS', 'UREP12_VS'),
      ('2016', 'pres16_vs', 'UDEM16_VS', 'UREP16_VS'), ('2020', 'pres20_vs', 'UDEM20_VS', 'UREP20_VS'),
      ('2024', 'pres24_vs', 'UDEM24_VS', 'UREP24_VS')]
vs = [('2000', ratio_pair('pres00_vs', 'DemVS', 'RepVS'))] + [(y, traded_pair(m, dn, rn)) for y, m, dn, rn in VS]
vs = [(y, [(d, v) for d, v in rows if 0.35 <= v <= 0.65]) for y, rows in vs]   # drop implausible thin-market prints
emit('iem_pres_wta_dem', wta, 'IEM presidential winner-take-all market: Democratic popular-vote win probability (chained)',
     'cumulative change in implied probability, percentage points (chained across cycles)',
     'Contracts DEM/REP (2000: Dem/Rep; 2004: DEM04/REP04 - the G52/L52 vote-band contracts are ignored). WTA pays on '
     'the POPULAR-VOTE plurality winner, not the Electoral College (2000 and 2016 popular-vote winners lost). '
     'The 2004 DEM04/REP04 contracts stop trading on 2004-09-21, so election night 2004 is not covered. '
     'Daily value = mean of DEM last price and 1 - REP last price over the contracts that traded that day.')
emit('iem_pres_vs_dem', vs, 'IEM presidential vote-share market: expected Democratic share of the two-party vote (chained)',
     'cumulative change in expected Democratic two-party vote share, percentage points (chained across cycles)',
     'VS contracts pay the nominee share of the two-party vote (2000: three-party with Reform, converted to '
     'Dem/(Dem+Rep)). 2004 omitted: that market traded nominee-specific contracts (KERR, DEAN, BU|KERR, ...). '
     'Prints outside 0.35-0.65 dropped as implausible thin-market trades (e.g. a 1.00 first print in 2019).')
CONG = [('2002', 'cong02'), ('2004', 'congress04'), ('2006', 'congress06'), ('2008', 'congress08'),
        ('2010', 'congress10'), ('2012', 'congress12'), ('2016', 'congress16'), ('2018', 'congress18'),
        ('2020', 'congress20'), ('2022', 'congress22'), ('2024', 'congress24'), ('2026', 'congress26')]
house = [(y, congress(m, 'house')) for y, m in CONG]
senate = [(y, congress(m, 'senate')) for y, m in CONG]
COMMON_C = ('From the combined Congressional Control markets (outcomes House x Senate: DH/RH x DS/RS; through 2006 '
            'and late Oct 2010 coded N = non-Republican, i.e. Democrats plus independents). Value = sum of last prices '
            'of outcomes with that chamber Democratic over the sum of the four named outcomes; the 2012+ OTHER '
            'contract is excluded (it traded erratically, 0.2-0.4 in 2023-24). Days kept only if every named '
            'outcome has traded at least once and their last prices sum to 0.85-1.2 (drops early days with untraded '
            'outcomes and incoherent thin-market days; election-night settlement moves are kept). The 2020 market\'s last trade '
            '(2021-01-04) predates the Georgia runoffs. Uses the source last prices on days '
            'with at least one trade, so some components can be stale. No 2014 market history is published. The '
            'separate House/Senate markets are gain/hold/lose ladders and are not used.')
emit('iem_house_dem', house, 'IEM congressional control market: probability Democrats control the US House (chained)',
     'cumulative change in implied probability, percentage points (chained across cycles)', COMMON_C)
emit('iem_senate_dem', senate, 'IEM congressional control market: probability Democrats control the US Senate (chained)',
     'cumulative change in implied probability, percentage points (chained across cycles)', COMMON_C)
