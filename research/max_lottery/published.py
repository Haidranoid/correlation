"""Published claim vs this data: Chen & Zimmermann's Open Source Asset Pricing long-short returns for MaxRet (Bali,
Cakici & Whitelaw 2011, VW deciles, CRSP) and IdioVol3F (Ang et al. 2006), by period -> published_osap.txt.
Source file (downloaded 2026-10-21 from openassetpricing.com's Google Drive link, no signup, data through 2024-12):
WORK/downloads/osap/osap_ls_wide.csv; signal definitions in WORK/downloads/osap/SignalDoc.csv.
Returns are % a month, signed so the predicted direction is positive (MaxRet: low MAX minus high MAX)."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import np, WORK, HERE, mean_t  # noqa: E402

SRC = WORK / 'downloads/osap/osap_ls_wide.csv'
rows = list(csv.reader(open(SRC)))
head = rows[0]
dates = np.array([r[0][:7] for r in rows[1:]])
L = [f'Open Source Asset Pricing long-short returns, % a month (t NW 4 lags; plain t), from {SRC.name}; last month {dates[-1]}',
     '| Signal | Period | Months | Mean | t NW (plain) |', '|---|---|---|---|---|']
for sig in ('MaxRet', 'IdioVol3F'):
    j = head.index(sig)
    x = np.array([float(r[j]) if r[j] not in ('', 'NA') else np.nan for r in rows[1:]])
    for a, b, lab in (('1962-07', '2005-12', "Bali et al.'s sample"), ('2006-01', '2019-12', 'test 1 years'),
                      ('2020-01', '2025-06', 'test 3 years (OSAP ends 2024-12)'), ('2012-01', '2024-12', 'since publication'),
                      ('2006-01', '2009-12', ''), ('2010-01', '2019-12', '')):
        m = (dates >= a) & (dates <= b)
        mu, t, tnw, n = mean_t(x[m])
        L.append(f'| {sig} | {a}..{b} {lab} | {n} | {mu:+.3f} | {tnw:+.2f} ({t:+.2f}) |')
doc = {r['Acronym']: r for r in csv.DictReader(open(WORK / 'downloads/osap/SignalDoc.csv'))}
d = doc['MaxRet']
L.append(f"SignalDoc MaxRet: {d['Authors']} {d['Year']} {d['Journal']}; sample {d['SampleStartYear']}-{d['SampleEndYear']}; "
         f"OP return {d['Return']} t {d['T-Stat']} ({d['Stock Weight']}, quantile {d['LS Quantile']}); notes: {d['Notes']}")
(HERE / 'published_osap.txt').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
