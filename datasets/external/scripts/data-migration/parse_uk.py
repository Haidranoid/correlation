"""UK Home Office: migrants detected crossing the English Channel in small boats, daily from 2018,
plus (if downloaded) quarterly asylum applications from the Immigration system statistics."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from xlsx_read import ods_rows

F = os.path.join(RAW, "uk_homeoffice", "16_October_2026_Small_boats_-_time_series.ods")
rows = []
for r in ods_rows(F, "SB_01"):
    if not r or not r[0] or not str(r[0])[:2] == "20":
        continue
    d = dt.date.fromisoformat(str(r[0])[:10])
    v = r[1]
    if not isinstance(v, float):
        raise SystemExit(f"non-numeric {r}")
    rows.append((d, v))
# completeness: one row per calendar day
days = (rows[-1][0] - rows[0][0]).days + 1
assert days == len(rows), (days, len(rows))
zeros = sum(1 for _, v in rows if v == 0)
write_series(
    "ukho_small_boats_daily", rows, name="Migrants arriving in the UK by small boat across the English Channel (daily)",
    category="migration", subcategory="irregular_arrivals", region="GB", source="UK Home Office",
    url="https://www.gov.uk/government/publications/migrants-detected-crossing-the-english-channel-in-small-boats",
    frequency="daily", units="persons per day", transform="level", known_after_days=7,
    notes=(f"Home Office time series (SB_01) from the file dated 16 October 2026; every calendar day "
           f"2018-01-01 to 2026-10-15, explicit zeros on {zeros} of {len(rows)} days (no crossings, usually bad "
           "weather). Provisional operational data, revised (9 days flagged 'Revised'); the quarterly "
           "'Illegal entry routes' release is authoritative. known_after_days=7: the full series is refreshed "
           "weekly on Fridays through the previous day; since 2022 a 'last 7 days' page is updated daily "
           "(next-day figures), but 2018-2021 values were compiled later, so 7 is the conservative bound. "
           "Transform level (daily event counts) rather than anomaly_log1p: zero-heavy, weather-driven and "
           "seasonal (summer-autumn peaks), but the series starts in 2018, so the 3-year anomaly warm-up "
           "would leave it out of every test period. Only the 2020-2025 period can include it."),
    value_range=(0, 5000))

# ------------------------------------------------------------------ quarterly asylum claims 2001-
from collections import defaultdict
from seasonality import seasonal_r2
from xlsx_read import xlsx_rows

A = os.path.join(RAW, "uk_homeoffice", "asylum-claims-datasets-jun-2026.xlsx")
tab = xlsx_rows(A, "Data_Asy_D01")
hdr = tab[1]
assert hdr[:2] == ["Year", "Quarter"] and hdr[-1] == "Claims", hdr
ia = hdr.index("Applicant type")
q = defaultdict(float)
types = defaultdict(float)
for r in tab[2:]:
    if not r or not r[1] or r[-1] is None:
        continue
    y, qq = str(r[1]).split(" Q")
    d = dt.date(int(y), 3 * (int(qq) - 1) + 1, 1)
    q[d] += float(r[-1])
    types[r[ia]] += float(r[-1])
print("UK asylum applicant types:", dict(types))
rows_q = sorted(q.items())
r2, n = seasonal_r2(rows_q, period="quarter")
print(f"UK asylum quarterly seasonal R2 {r2:.3f} (n={n})")
write_series(
    "ukho_asylum_claims_q", rows_q, name="People claiming asylum in the UK, main applicants and dependants (quarterly)",
    category="migration", subcategory="asylum", region="GB", source="UK Home Office",
    url="https://www.gov.uk/government/statistical-data-sets/immigration-system-statistics-data-tables",
    frequency="quarterly", units="persons per quarter", transform="anomaly_log1p" if r2 >= 0.2 else "logdiff",
    known_after_days=155,
    notes=("Immigration system statistics, Asylum claims detailed dataset Asy_D01 (year ending June 2026 "
           "edition), sum of Claims over nationality, age, sex, UASC and location for main applicants and "
           "dependants; dated the first day of the quarter. known_after_days=155 from the quarter start: the "
           "quarterly release comes about 8 weeks after quarter end (Q2 2026 published 2026-08-27, 149 days "
           f"after 1 April). Calendar quarters explain {100 * r2:.0f}% of quarterly log changes and the history "
           "is long (2001-), hence anomaly_log1p (moves start 2004 after the 3-year warm-up). "
           "Complements eurostat_asy_uk (monthly, 2009-2020). Revised in later editions."),
    value_range=(1, 1e6))
