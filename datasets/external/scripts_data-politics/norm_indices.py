"""Normalize the policy-uncertainty / geopolitical-risk / partisan-conflict index families.

Sources (all read from raw/, never modified):
  raw/policyuncertainty_politics/  State EPU, All_Country EPU, Global EPU, Categorical EPU,
                                   US monthly EPU, Twitter EPU, UK daily EPU
  raw/gpr_country/data_gpr_export.dta   (read with the shared venv's pandas Stata reader -> json)
  raw/philfed_pci/                       Philly Fed xlsx (to 2025-06) + Richmond Fed csv (continuation)
"""
import csv
import datetime as dt
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (RAW, catalog, excel_date, month_end, qc, read_xlsx, reset_group,
                    write_series, xlsx_highlighted, WORK)

G = 'indices'
reset_group(G)
PU = os.path.join(RAW, 'policyuncertainty_politics')
PU_URL = 'https://www.policyuncertainty.com/'
CUT = dt.date(2000, 1, 1)
report = []


def choose_transform(rows):
    """rows sorted [(date, v)]. Rule: any v<=0 on/after 2000-01-01 -> diff (full history);
    zeros only before 2000 -> logdiff, series starts the period after the last zero; else logdiff."""
    zeros_post = [d for d, v in rows if v <= 0 and d >= CUT]
    zeros_pre = [d for d, v in rows if v <= 0 and d < CUT]
    if zeros_post:
        return 'diff', rows, f'{len(zeros_post)} zero values since 2000 (so diff, not logdiff)'
    if zeros_pre:
        last0 = max(zeros_pre)
        return 'logdiff', [(d, v) for d, v in rows if d > last0], f'starts after last zero ({last0}) so logdiff is defined'
    return 'logdiff', rows, ''


def drop_zeros(rows):
    """Treat v<=0 as missing (archive/collection gaps) -> (rows_without, note)."""
    z = [d for d, v in rows if v <= 0]
    if not z:
        return rows, ''
    return [(d, v) for d, v in rows if v > 0], (f'{len(z)} zero values ({min(z):%Y-%m-%d}..{max(z):%Y-%m-%d}) '
                                                 'dropped as missing (archive/collection gaps, not zero uncertainty)')


def emit(sid, rows, freq, lo, hi, **cat):
    first, last, n = write_series(sid, rows)
    rep = qc(sid, freq, lo=lo, hi=hi, show=2, quiet=True)
    report.append(rep)
    catalog(G, sid, frequency=freq, **cat)
    return rep


# ---------------------------------------------------------------- State EPU (Baker, Davis & Levy)
STATE_NAMES = {
    'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California', 'CO': 'Colorado',
    'CT': 'Connecticut', 'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine',
    'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington',
    'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'}
rows = read_xlsx(os.path.join(PU, 'State_Policy_Uncertainty.xlsx'))['Sheet1']
hdr = rows[0]
data = [r for r in rows[1:] if r and isinstance(r[0], float)]
for st in sorted(STATE_NAMES):
    j = hdr.index('EPU_State' + st)
    ser = [(month_end(int(r[0]), int(r[1])), r[j]) for r in data if j < len(r) and isinstance(r[j], float)]
    ser, zwhy = drop_zeros(ser)
    tr, ser, why = choose_transform(ser)
    why = '; '.join(x for x in (zwhy, why) if x)
    small = ' Small newspaper sample: zeros and flat/step stretches occur (e.g. SD 2006-07 Jan-Jul zeros).' \
        if st in ('DC', 'DE', 'SD', 'WY', 'ND', 'MT', 'NH', 'KY', 'MS', 'AK', 'VT', 'RI', 'HI', 'ID') else ''
    if st == 'DC':
        small += ' DC index moves in steps of ~9 points (one article); very noisy.'
    emit(f'pu_state_epu_{st.lower()}', ser, 'monthly', 0, 5000,
         name=f'State-level economic policy uncertainty (state & local policy terms), {STATE_NAMES[st]}',
         subcategory='policy_uncertainty_state', region=f'US-{st}',
         source='Baker, Davis & Levy (2022) State-Level EPU via policyuncertainty.com',
         url=PU_URL + 'state_epu.html',
         units='index (share of local-newspaper articles with economy+uncertainty+state/local-policy terms; normalized to state EPU_National mean pre-2018 = 100)',
         transform=tr, known_after_days=100,
         notes=('EPU_State column of State_Policy_Uncertainty.xlsx (EPU_National/Composite variants not used). '
                'Dated at month end. Retrospective research index: first published 2022-04 (data to 2021-06); '
                'Wayback vintages show irregular updates (2022-10 file to 2022-06, lag 97 d; 2026-01 file to 2025-12, lag 20 d; '
                'no update captured in between). 100 d is a steady-state bound, not real-time vintage availability. '
                + (why + '.' if why else '') + small).strip())

# ---------------------------------------------------------------- Country EPU (All_Country_Data)
COUNTRIES = [('Australia', 'aus', 'AU'), ('Brazil', 'bra', 'BR'), ('Canada', 'can', 'CA'), ('Chile', 'chl', 'CL'),
             ('China', 'chn', 'CN'), ('France', 'fra', 'FR'), ('Germany', 'deu', 'DE'), ('Greece', 'grc', 'GR'),
             ('India', 'ind', 'IN'), ('Ireland', 'irl', 'IE'), ('Italy', 'ita', 'IT'), ('Japan', 'jpn', 'JP'),
             ('Korea', 'kor', 'KR'), ('Mexico', 'mex', 'MX'), ('Pakistan', 'pak', 'PK'), ('Russia', 'rus', 'RU'),
             ('Spain', 'esp', 'ES'), ('Singapore', 'sgp', 'SG'), ('UK', 'gbr', 'GB')]
path = os.path.join(PU, 'All_Country_Data.xlsx')
rows = read_xlsx(path)['EPU']
hl = xlsx_highlighted(path)['EPU']
hdr = rows[0]
for cname, iso3, iso2 in COUNTRIES:
    j = hdr.index(cname)
    ser = []
    imputed = []
    for i, r in enumerate(rows):
        if i == 0 or not r or len(r) <= j or not isinstance(r[j], float):
            continue
        try:
            y, m = int(float(r[0])), int(float(r[1]))
        except (TypeError, ValueError):
            continue
        d = month_end(y, m)
        if (i, j) in hl:
            imputed.append(d)
            continue
        ser.append((d, r[j]))
    tr, ser, why = choose_transform(ser)
    imp = ''
    if imputed:
        imp = (f' Dropped {len(imputed)} source-imputed (yellow-highlighted) months '
               f'({min(imputed):%Y-%m}..{max(imputed):%Y-%m}); left as gaps.')
    extra = ''
    if cname == 'China':
        extra = ' Hybrid China index (SCMP spliced with Mainland-paper index, as in the source file).'
    emit(f'pu_epu_{iso3}', ser, 'monthly', 0, 5000,
         name=f'Economic policy uncertainty index, {cname if cname != "UK" else "United Kingdom"}',
         subcategory='policy_uncertainty_country', region=iso2,
         source='Baker, Bloom & Davis and co-authors, All_Country_Data.xlsx via policyuncertainty.com',
         url=PU_URL + 'all_country_data.html',
         units='index (newspaper-article share, country-specific normalization, ~100 = sample mean)',
         transform=tr, known_after_days=200,
         notes=('Dated at month end. The compiled All_Country file lagged 89-193 days behind month end at Wayback '
                'captures (2022-11, 2023-03, 2024-09) and 79 d on 2026-10-18; 200 d is a conservative bound '
                '(single-country source files may be fresher). Indices are rescaled/revised across vintages.'
                + extra + imp + (' ' + why + '.' if why else '')).strip())

# Global EPU (GDP-weighted, current-price GDP)
rows = read_xlsx(os.path.join(PU, 'Global_Policy_Uncertainty_Data.xlsx'))['Sheet1']
ser = [(month_end(int(r[0]), int(r[1])), r[2]) for r in rows[1:] if r and isinstance(r[0], float) and len(r) > 2 and isinstance(r[2], float)]
tr, ser, why = choose_transform(ser)
emit('pu_gepu_current', ser, 'monthly', 0, 5000,
     name='Global economic policy uncertainty index (GDP-weighted, current-price GDP)',
     subcategory='policy_uncertainty_global', region='WORLD',
     source='Davis (2016) Global EPU via policyuncertainty.com', url=PU_URL + 'global_monthly.html',
     units='index (GDP-weighted average of national EPU indices, 1997-2015 mean = 100)', transform=tr,
     known_after_days=90,
     notes=('GEPU_current column. Dated at month end. Wayback captures show 24-31 d lag (2022); file lagged 79 d on '
            '2026-10-18, so 90 d bound. Built from national EPUs, some imputed for recent months by the source.'))

# ---------------------------------------------------------------- US categorical EPU
rows = read_xlsx(os.path.join(PU, 'Categorical_EPU_Data.xlsx'))['Indices']
hdr = rows[0]
CATS = [('2. Monetary policy', 'monetary', 'monetary policy'),
        ('Fiscal Policy (Taxes OR Spending)', 'fiscal', 'fiscal policy (taxes or government spending)'),
        ('5. Health care', 'healthcare', 'health care policy'),
        ('6. National security', 'natsecurity', 'national security policy'),
        ('7. Entitlement programs', 'entitlement', 'entitlement programs'),
        ('8. Regulation', 'regulation', 'regulation'),
        ('Financial Regulation', 'finreg', 'financial regulation'),
        ('9. Trade policy', 'trade', 'trade policy'),
        ('10. Sovereign debt, currency crises', 'sovdebt', 'sovereign debt and currency crises')]
for col, slug, label in CATS:
    j = hdr.index(col)
    ser = []
    for r in rows[1:]:
        if not r or len(r) <= j or not isinstance(r[j], float):
            continue
        try:
            y, m = int(float(r[0])), int(float(r[1]))
        except (TypeError, ValueError):
            continue
        ser.append((month_end(y, m), r[j]))
    tr, ser, why = choose_transform(ser)
    emit(f'pu_cat_{slug}', ser, 'monthly', 0, 20000,
         name=f'US categorical economic policy uncertainty: {label}',
         subcategory='policy_uncertainty_us_category', region='US',
         source='Baker, Bloom & Davis categorical EPU (NewsBank US newspapers) via policyuncertainty.com',
         url=PU_URL + 'categorical_epu.html',
         units='index (1985-2010 mean of overall EPU = 100)', transform=tr, known_after_days=21,
         notes=('Dated at month end. Updated monthly: Wayback vintages 2019-11, 2024-09, 2025-07, 2025-11 were 9-18 d '
                'behind month end; 21 d bound. Revised/rescaled across vintages.' + (' ' + why + '.' if why else '')))

# ---------------------------------------------------------------- US monthly EPU (news-based, 3-component)
wb = read_xlsx(os.path.join(PU, 'US_Policy_Uncertainty_Data.xlsx'))
rows = wb['Main News Index']
ser = []
for r in rows[1:]:
    try:
        y, m = int(float(r[0])), int(float(r[1]))
    except (TypeError, ValueError, IndexError):
        continue
    if y >= 1985 and len(r) > 2 and isinstance(r[2], float):
        ser.append((month_end(y, m), r[2]))
tr, ser, why = choose_transform(sorted(ser))
emit('pu_us_epu_news_m', ser, 'monthly', 0, 2000,
     name='US economic policy uncertainty, news-based monthly index (10 major newspapers)',
     subcategory='policy_uncertainty_us', region='US',
     source='Baker, Bloom & Davis (2016) via policyuncertainty.com', url=PU_URL + 'us_monthly.html',
     units='index (1985-2009 mean = 100)', transform=tr, known_after_days=21,
     notes=('News_Based_Policy_Uncert_Index, kept from 1985 (sheet goes back to 1900). Dated at month end. '
            'Monthly 10-paper index; distinct from the daily NewsBank-based USEPUINDXD that data-markets holds. '
            'Same release cycle as the categorical file (<=18 d observed); 21 d bound.'))
rows = wb['Legacy Three Component EPU']
ser = []
for r in rows[1:]:
    try:
        y, m = int(float(r[0])), int(float(r[1]))
    except (TypeError, ValueError, IndexError):
        continue
    if len(r) > 2 and isinstance(r[2], float):
        ser.append((month_end(y, m), r[2]))
tr, ser, why = choose_transform(sorted(ser))
emit('pu_us_epu_3comp_m', ser, 'monthly', 0, 2000,
     name='US economic policy uncertainty, three-component index (news, tax expirations, forecaster disagreement)',
     subcategory='policy_uncertainty_us', region='US',
     source='Baker, Bloom & Davis (2016) via policyuncertainty.com', url=PU_URL + 'us_monthly.html',
     units='index (1985-2009 mean = 100)', transform=tr, known_after_days=21,
     notes=('Three_Component_Index ("Legacy" sheet). Dated at month end. Components: news (1/2), CBO tax-code '
            'expirations, SPF CPI and government-purchases forecaster disagreement; the SPF parts are quarterly, so '
            'the latest months can be revised. 21 d bound as for the news index.'))

# ---------------------------------------------------------------- Twitter EPU (daily, 2011-06 .. 2023-04)
rows = read_xlsx(os.path.join(PU, 'Twitter_Economic_Uncertainty.xlsx'))['Sheet1']
hdr = rows[0]
for col, slug, label in [('TEU-USA', 'teu_usa', 'US-located English tweets (geo classifier)'),
                         ('TEU-SCA', 'teu_sca', 'all English tweets scaled by tweets containing "have"')]:
    j = hdr.index(col)
    ser = [(excel_date(r[0]), r[j]) for r in rows[1:] if r and isinstance(r[0], float) and len(r) > j and isinstance(r[j], float)]
    ser, zwhy = drop_zeros(ser)
    tr, ser, why = choose_transform(ser)
    why = '; '.join(x for x in (zwhy, why) if x)
    emit(f'pu_twitter_{slug}', ser, 'daily', 0, 100000,
         name=f'Twitter-based economic uncertainty (TEU), {label}',
         subcategory='policy_uncertainty_twitter', region='US' if col == 'TEU-USA' else 'WORLD',
         source='Baker, Bloom, Davis & Renault Twitter Economic Uncertainty via policyuncertainty.com',
         url=PU_URL + 'twitter_uncert.html',
         units='index (daily count of tweets with economy+uncertainty terms; TEU-SCA scaled by "have" tweets; normalized)',
         transform=tr, known_after_days=1,
         notes=('SHORT: 2011-06-01 to 2023-04-21 (ends when Twitter/X API access ended). Tweets were public on the day; '
                'the compiled index was published later and is rescaled across vintages. TEU is economic (not '
                'policy-specific) uncertainty; the site files it with the EPU family. ' + (why + '.' if why else '')).strip())

# ---------------------------------------------------------------- UK daily EPU
ser = []
with open(os.path.join(PU, 'UK_Daily_Policy_Data.csv')) as f:
    for r in csv.DictReader(f):
        ser.append((dt.date(int(r['year']), int(r['month']), int(r['day'])), float(r['daily_policy_index'])))
ser, zwhy = drop_zeros(sorted(ser))
tr, ser, why = choose_transform(ser)
why = '; '.join(x for x in (zwhy, why) if x) + ' (mostly 25 Dec and scattered 2001-2006 days)'

emit('pu_epu_gbr_daily', ser, 'daily', 0, 100000,
     name='Economic policy uncertainty index, United Kingdom, daily (news-based)',
     subcategory='policy_uncertainty_country', region='GB',
     source='Baker, Bloom & Davis UK daily EPU via policyuncertainty.com', url=PU_URL + 'uk_daily.html',
     units='index (daily share of UK newspaper articles with economy+policy+uncertainty terms, normalized)',
     transform=tr, known_after_days=2,
     notes=('Daily from 2001-01-01 (calendar days incl. weekends). Updated near-daily: Wayback vintages '
            '2025-11-24 and 2026-03-14 ended 1-2 days before capture; 2 d bound. Day-to-day values are very noisy. '
            + (why + '.' if why else '')).strip())

# ---------------------------------------------------------------- GPR country indices (monthly)
js = os.path.join(WORK, 'gpr_country.json')
venv_py = os.path.expanduser('~/programming/c2-wt/venv-data/bin/python')
code = r'''
import json, sys
import pandas as pd
df = pd.read_stata(sys.argv[1])
cols = [c for c in df.columns if c.startswith("GPRC_")]
out = {"months": [d.strftime("%Y-%m-%d") for d in df["month"]]}
for c in cols:
    out[c] = [None if pd.isna(v) else float(v) for v in df[c]]
labels = {}
try:
    from pandas.io.stata import StataReader
    with StataReader(sys.argv[1]) as rdr:
        labels = rdr.variable_labels()
except Exception as e:
    labels = {"_err": str(e)}
out["_labels"] = {c: labels.get(c, "") for c in cols}
json.dump(out, open(sys.argv[2], "w"))
'''
subprocess.run([venv_py, '-I', '-c', code, os.path.join(RAW, 'gpr_country', 'data_gpr_export.dta'), js], check=True)
gp = json.load(open(js))
ISO = {'ARG': ('AR', 'Argentina'), 'AUS': ('AU', 'Australia'), 'BEL': ('BE', 'Belgium'), 'BRA': ('BR', 'Brazil'),
       'CAN': ('CA', 'Canada'), 'CHE': ('CH', 'Switzerland'), 'CHL': ('CL', 'Chile'), 'CHN': ('CN', 'China'),
       'COL': ('CO', 'Colombia'), 'DEU': ('DE', 'Germany'), 'DNK': ('DK', 'Denmark'), 'EGY': ('EG', 'Egypt'),
       'ESP': ('ES', 'Spain'), 'FIN': ('FI', 'Finland'), 'FRA': ('FR', 'France'), 'GBR': ('GB', 'United Kingdom'),
       'HKG': ('HK', 'Hong Kong'), 'HUN': ('HU', 'Hungary'), 'IDN': ('ID', 'Indonesia'), 'IND': ('IN', 'India'),
       'ISR': ('IL', 'Israel'), 'ITA': ('IT', 'Italy'), 'JPN': ('JP', 'Japan'), 'KOR': ('KR', 'South Korea'),
       'MEX': ('MX', 'Mexico'), 'MYS': ('MY', 'Malaysia'), 'NLD': ('NL', 'Netherlands'), 'NOR': ('NO', 'Norway'),
       'PER': ('PE', 'Peru'), 'PHL': ('PH', 'Philippines'), 'POL': ('PL', 'Poland'), 'PRT': ('PT', 'Portugal'),
       'RUS': ('RU', 'Russia'), 'SAU': ('SA', 'Saudi Arabia'), 'SWE': ('SE', 'Sweden'), 'THA': ('TH', 'Thailand'),
       'TUN': ('TN', 'Tunisia'), 'TUR': ('TR', 'Turkey'), 'TWN': ('TW', 'Taiwan'), 'UKR': ('UA', 'Ukraine'),
       'USA': ('US', 'United States'), 'VEN': ('VE', 'Venezuela'), 'VNM': ('VN', 'Vietnam'), 'ZAF': ('ZA', 'South Africa')}
months = [dt.date.fromisoformat(m) for m in gp['months']]
for c in sorted(k for k in gp if k.startswith('GPRC_')):
    iso3 = c[5:]
    iso2, cname = ISO[iso3]
    ser = [(month_end(d.year, d.month), v) for d, v in zip(months, gp[c]) if v is not None and d >= dt.date(1985, 1, 1)]
    tr, ser, why = choose_transform(ser)
    emit(f'gpr_c_{iso3.lower()}', ser, 'monthly', 0, 100,
         name=f'Country-specific geopolitical risk index (Caldara-Iacoviello), {cname}',
         subcategory='geopolitical_risk_country', region=iso2,
         source='Caldara & Iacoviello (2022, AER) GPR, data_gpr_export (GPRC_ columns)',
         url='https://www.matteoiacoviello.com/gpr.htm',
         units='percent of articles in 10 US/UK/CA newspapers that mention geopolitical risk and the country',
         transform=tr, known_after_days=7,
         notes=(f'{c}, 1985 onward (the GPRHC 1900+ historical variant not used). Dated at month end. Source updates '
                'the monthly file at the beginning of each month (last update 2026-10-01, data to 2026-09); 7 d bound. '
                'Measured from US/UK/Canadian newspapers, i.e. how these papers cover risk in/involving the country. '
                'Retrospective construction (published 2021-22). ' + (why + '.' if why else '')).strip())

# ---------------------------------------------------------------- Partisan Conflict Index
rows = read_xlsx(os.path.join(RAW, 'philfed_pci', 'partisan-conflict.xlsx'))
sh = list(rows)[0]
rows = rows[sh]
MONTHS = {m: i for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                       'september', 'october', 'november', 'december'], 1)}
philly = {}
for r in rows[1:]:
    if len(r) >= 3 and isinstance(r[0], float) and isinstance(r[2], float) and isinstance(r[1], str):
        m = MONTHS.get(r[1].strip().lower())
        if m:
            philly[(int(r[0]), m)] = r[2]
rich = {}
fixes = []
prev = None
with open(os.path.join(RAW, 'philfed_pci', 'richmond_partisan_conflict_index.csv'), encoding='utf-8-sig') as f:
    for r in csv.reader(f):
        if r and r[0].strip().isdigit():
            key = (int(r[0]), int(r[1]))
            if prev is not None:
                exp = (prev[0] + 1, 1) if prev[1] == 12 else (prev[0], prev[1] + 1)
                if key != exp and key[1] == exp[1]:
                    fixes.append((key, exp))   # e.g. row labelled 2024,1 that sits between 2024-12 and 2025-02
                    key = exp
            assert key not in rich, key
            rich[key] = float(r[2])
            prev = key
print('Richmond PCI label fixes (stated -> by sequence):', fixes)
overlap = sorted(set(philly) & set(rich))
maxdiff = max(abs(round(philly[k]) - rich[k]) for k in overlap)
print(f'PCI: philly {min(philly)}..{max(philly)} n={len(philly)}; richmond {min(rich)}..{max(rich)} n={len(rich)}; '
      f'overlap {len(overlap)} max |round(philly)-richmond| = {maxdiff}')
assert maxdiff <= 1, 'Richmond continuation does not match the Philly Fed archive'
ser = [(month_end(*k), v) for k, v in philly.items()] + [(month_end(*k), v) for k, v in rich.items() if k not in philly]
tr, ser, why = choose_transform(sorted(ser))
emit('pci_partisan_conflict', ser, 'monthly', 0, 1000,
     name='Partisan Conflict Index (US federal political disagreement in newspapers)',
     subcategory='partisan_conflict', region='US',
     source='Azzimonti; Philadelphia Fed (archive to 2025-06), Richmond Fed continuation',
     url='https://www.richmondfed.org/research/national_economy/partisan_conflict_index',
     units='index (1990 = 100)', transform=tr, known_after_days=30,
     notes=(f'Philly Fed values (decimals) through 2025-06; Richmond Fed values (rounded to integers) from 2025-07. '
            f'Overlap {len(overlap)} months agree to within {maxdiff:g} after rounding (Richmond csv row labelled '
            f'2024,1 between 2024-12 and 2025-02 read as 2025-01). Philly Fed released mid-following '
            'month (June 2025 published 2025-07-09); Philly sunset July 2025. 30 d bound. Dated at month end.'))

# ---------------------------------------------------------------- report
bad = [r for r in report if r['bad']]
print(f'{len(report)} series written; {len(bad)} flagged')
for r in report:
    lg = r['largest_gap']
    print(f"  {r['id']:24s} {r['first']}..{r['last']} n={r['n']:5d} min={r['min']:.4g} max={r['max']:.4g} "
          f"zeros={r['zeros']} gaps={r['gaps']}/{r['gap_units']}" + (f" largest={lg[0]}..{lg[1]}({lg[2]})" if lg else '')
          + (' BAD ' + ';'.join(r['bad']) if r['bad'] else ''))
json.dump(report, open(os.path.join(WORK, 'qc_indices.json'), 'w'), default=str, indent=0)
