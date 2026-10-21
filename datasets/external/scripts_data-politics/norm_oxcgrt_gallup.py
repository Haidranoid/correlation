"""OxCGRT US state stringency (daily 2020-2022) and Gallup presidential approval (UCSB APP)."""
import csv
import datetime as dt
import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, catalog, qc, reset_group, write_series

# ------------------------------------------------------------------ OxCGRT
G = 'oxcgrt'
reset_group(G)
WANT = {'': ('us', 'US', 'United States federal government'), 'US_CA': ('ca', 'US-CA', 'California'),
        'US_TX': ('tx', 'US-TX', 'Texas'), 'US_FL': ('fl', 'US-FL', 'Florida'), 'US_NY': ('ny', 'US-NY', 'New York'),
        'US_PA': ('pa', 'US-PA', 'Pennsylvania'), 'US_IL': ('il', 'US-IL', 'Illinois'), 'US_OH': ('oh', 'US-OH', 'Ohio'),
        'US_GA': ('ga', 'US-GA', 'Georgia'), 'US_NC': ('nc', 'US-NC', 'North Carolina'),
        'US_MI': ('mi', 'US-MI', 'Michigan'), 'US_NJ': ('nj', 'US-NJ', 'New Jersey'), 'US_WA': ('wa', 'US-WA', 'Washington')}
data = {k: [] for k in WANT}
with open(os.path.join(RAW, 'oxcgrt_us', 'OxCGRT_US_latest.csv'), newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        code = row['RegionCode']
        if code not in WANT:
            continue
        if code == '' and row['Jurisdiction'] != 'NAT_GOV':
            continue
        v = row['StringencyIndex'].strip()
        if v == '':
            continue
        d = dt.datetime.strptime(row['Date'], '%Y%m%d').date()
        data[code].append((d, float(v)))
for code, (slug, region, label) in WANT.items():
    sid = f'oxcgrt_stringency_{slug}'
    write_series(sid, data[code])
    rep = qc(sid, 'daily', lo=0, hi=100, show=0, quiet=True)
    print(f"{sid}: {rep['first']}..{rep['last']} n={rep['n']} min={rep['min']} max={rep['max']} gaps={rep['gaps']}/{rep['gap_units']}")
    catalog(G, sid, name=f'COVID-19 government response stringency index, {label} (OxCGRT)',
            subcategory='covid_policy_stringency', region=region,
            source='Oxford COVID-19 Government Response Tracker, US subnational data (CC BY 4.0)',
            url='https://github.com/OxCGRT/USA-covid-policy', frequency='daily', units='index 0-100',
            transform='diff', known_after_days=1,
            notes=('SHORT: 2020-01-01 to 2022-12-31 only. Stringency of closures/restrictions on day d (policies '
                   'are announced in advance or on the day; coded by OxCGRT retrospectively and revised). '
                   + ('NAT_GOV row = federal policies only. ' if code == '' else
                      'State-wide policies (OxCGRT state rows; general vs targeted flags as in the source index). ')
                   + 'Moves are policy changes; the 2020-03 jump is common to all states.'))

# ------------------------------------------------------------------ Gallup via UCSB APP
G = 'approval'
reset_group(G)
D = os.path.join(RAW, 'ucsb_app_approval')
PAGES = [('william-j-clinton-public-approval.html', 'Clinton'), ('george-w-bush-public-approval.html', 'G.W. Bush'),
         ('barack-obama-public-approval.html', 'Obama'), ('donald-j-trump-public-approval.html', 'Trump (1st)'),
         ('joseph-r-biden-public-approval.html', 'Biden'), ('donald-j-trump-2nd-term-public-approval.html', 'Trump (2nd)')]
obs = {}
fixes = []
skipped_nongallup = []
for fn, who in PAGES:
    t = open(os.path.join(D, fn), encoding='utf-8').read()
    i = t.find('<table')
    j = t.find('</table>', i)
    rows = []
    for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', t[i:j], flags=re.S):
        cells = [html.unescape(re.sub(r'<[^>]+>', '', c)).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr, flags=re.S)]
        cells = [c for c in cells if c != '']
        if len(cells) >= 4 and re.match(r'\d{1,2}/\d{1,2}/\d{2,4}$', cells[0]):
            rows.append(cells)
    for c in rows:
        if who == 'Trump (2nd)':
            # from 2026 the APP table mixes pollsters; last cell is the source name -> keep Gallup rows only
            if 'gallup' not in c[-1].lower():
                skipped_nongallup.append((c[1], c[-1]))
                continue
        s, e = c[0], c[1]
        m = re.fullmatch(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', e)
        y = m.group(3)
        if len(y) != 4:      # e.g. Biden row '2/1/2024' -> '2/20/224': take the start-date year
            ys = re.fullmatch(r'\d{1,2}/\d{1,2}/(\d{4})', s).group(1)
            fixes.append((who, e, ys))
            y = ys
        end = dt.date(int(y), int(m.group(1)), int(m.group(2)))
        appr = float(c[2])
        if end in obs:
            raise ValueError(f'duplicate end date {end}')
        obs[end] = (appr, who)
print('approval date fixes:', fixes)
print('non-Gallup rows skipped (Trump 2nd-term table):', len(skipped_nongallup), skipped_nongallup[:3])
ser = sorted((d, v[0]) for d, v in obs.items())
write_series('gallup_pres_approval', ser)
rep = qc('gallup_pres_approval', 'weekly', lo=10, hi=95, show=2, quiet=False)
trans = []
prev = None
for d in sorted(obs):
    if prev and obs[d][1] != obs[prev][1]:
        trans.append(f'{prev}->{d} ({obs[prev][1]} {obs[prev][0]:.0f} -> {obs[d][1]} {obs[d][0]:.0f})')
    prev = d
catalog(G, 'gallup_pres_approval', name='US presidential job approval, Gallup (% approve)',
        subcategory='presidential_approval', region='US',
        source='Gallup via UCSB American Presidency Project', url='https://www.presidency.ucsb.edu/statistics/data/presidential-job-approval-all-data',
        frequency='weekly', units='percent approving', transform='diff', known_after_days=7,
        notes=('Irregular poll dates (weekly in 2008-2017 Gallup daily-tracking weeks, ~10-day to monthly otherwise; '
               'Biden era monthly). Dated at fieldwork end; Gallup usually publishes 1-5 days later, 7 d bound. '
               'Diffs across presidencies are a change of president, not of opinion: ' + '; '.join(trans)
               + '. One source typo fixed (Biden poll ending "2/20/224" -> 2024-02-20). APP switched to a multi-pollster '
               f'table after January 2026; only rows with source Gallup kept ({len(skipped_nongallup)} non-Gallup rows dropped), '
               'so the series ends with the last Gallup poll.'))
