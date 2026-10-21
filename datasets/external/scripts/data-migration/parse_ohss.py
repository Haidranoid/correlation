"""DHS Office of Homeland Security Statistics (OHSS):
- Immigration Enforcement and Legal Processes Monthly Tables (last edition 2025-01-16, data Oct 2013 -
  Nov 2024; discontinued): DHS removals, ICE ERO administrative arrests, credible-fear referrals.
- Legal Immigration and Adjustment of Status (LIAS) report, FY2016-FY2025 year-end editions:
  quarterly lawful permanent residents (total, new arrivals) and naturalizations."""
import datetime as dt
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from xlsx_read import xlsx_rows

D = os.path.join(RAW, "ohss")
MON = ["October", "November", "December", "January", "February", "March", "April", "May", "June", "July",
       "August", "September"]


def monthly_total(sheet):
    rows = xlsx_rows(os.path.join(D, "ohss_monthly_tables_2024-11.xlsx"), sheet)
    hdr = next(i for i, r in enumerate(rows) if r and r[0] == "Fiscal Year")
    assert rows[hdr][1] == "Month" and rows[hdr][2] == "Total", rows[hdr]
    out, fy, months, fy_total = {}, None, [], None

    def close():
        if fy is not None and fy_total is not None:
            s = sum(v for _, v in months)
            # cells are rounded to the nearest ten, so allow rounding slack
            if abs(s - fy_total) > 10 * len(months):
                raise SystemExit(f"{sheet} FY{fy}: months {s} vs total {fy_total}")
        for d, v in months:
            out[d] = v

    for r in rows[hdr + 1:]:
        if not r or r[1] is None:
            continue
        if r[1] == "Total":
            close()
            fy = int(re.match(r"\d{4}", str(r[0]).replace(".0", "")).group(0))
            months, fy_total = [], float(r[2])
        elif r[1] in MON:
            k = MON.index(r[1])
            d = dt.date(fy - 1, 10 + k, 1) if k < 3 else dt.date(fy, k - 2, 1)
            months.append((d, float(r[2])))
    close()
    return sorted(out.items())


URL_M = "https://ohss.dhs.gov/topics/immigration/immigration-enforcement/monthly-tables"
m_notes = ("OHSS 'Immigration Enforcement and Legal Processes Monthly Tables', final edition (2025-01-16, data "
           "through November 2024; the series was discontinued afterwards, so it ends there). Dated the 1st of "
           "the calendar month (fiscal months converted). Cells rounded to the nearest ten; fiscal-year totals "
           "checked against the months. known_after_days=130 from the 1st: editions came 96-130 days after the "
           "first of the reference month (e.g. July 2024 data on 2024-11-08). The tables were first published in "
           "January 2024, so months before late 2023 were not public on this schedule (compiled later; earlier "
           "timing is notional). Starts Oct 2013, too short for the 3-year anomaly warm-up to keep the 2015-2020 "
           "period, so logdiff.")
for sid, sheet, name, sub, extra in [
        ("ohss_dhs_removals", "DHS Removals by Citizenship", "DHS removals, all citizenships (monthly)", "enforcement",
         " Removals by ICE/ERO and CBP; excludes administrative and enforcement returns."),
        ("ohss_ice_arrests", "ERO Arrests by Citizenship", "ICE ERO administrative arrests (monthly)", "enforcement",
         " Interior enforcement arrests."),
        ("ohss_cf_referrals", "Nationwide CF Filed by Citp", "Credible-fear screenings referred to USCIS, nationwide (monthly)",
         "asylum", " Unique applicants by USCIS case-created date; a leading indicator of asylum caseload.")]:
    rows = monthly_total(sheet)
    write_series(sid, rows, name=name, category="migration", subcategory=sub, region="US",
                 source="DHS Office of Homeland Security Statistics", url=URL_M, frequency="monthly",
                 units="persons per month", transform="logdiff", known_after_days=130,
                 notes=m_notes + extra, value_range=(1, 1e6))

# ------------------------------------------------------------------ LIAS quarterly
lpr, newarr, nat = {}, {}, {}
for f in sorted(glob.glob(os.path.join(D, "lias_*.xlsx"))):
    t1 = xlsx_rows(f, "Table 1A")
    title = " ".join(str(c) for r in t1[:4] for c in r if c)
    fy = int(re.search(r"FISCAL YEAR (\d{4})", title.upper()).group(1))
    tot = next(r for r in t1 if r and isinstance(r[0], str) and r[0].strip() == "Total")
    vals = [float(x) for x in tot[1:16]]
    T, Q, A, AQ, N, NQ = vals[0], vals[1:5], vals[5], vals[6:10], vals[10], vals[11:15]
    assert abs(sum(Q) - T) <= 40 and abs(sum(AQ) - A) <= 40 and abs(sum(NQ) - N) <= 40, (fy, vals)
    assert all(abs(Q[i] - AQ[i] - NQ[i]) <= 30 for i in range(4)), (fy, vals)
    t3 = xlsx_rows(f, "Table 3")
    tn = next(r for r in t3 if r and isinstance(r[0], str) and r[0].strip() == "Total")
    nv = [float(x) for x in tn[1:6]]
    assert abs(sum(nv[1:5]) - nv[0]) <= 40, (fy, nv)
    for i in range(4):
        d = dt.date(fy - 1, 10, 1) if i == 0 else dt.date(fy, 3 * i - 2, 1)
        assert d not in lpr, (fy, d)
        lpr[d], newarr[d], nat[d] = Q[i], NQ[i], nv[1 + i]
    print(f"LIAS FY{fy}: LPR {T:.0f} (new arrivals {N:.0f}), naturalized {nv[0]:.0f}  [{os.path.basename(f)[:50]}]")

URL_L = "https://ohss.dhs.gov/topics/immigration/legal-immigration-and-adjustment-status-report"
l_notes = ("OHSS Legal Immigration and Adjustment of Status report, year-end editions FY2016-FY2025 (one file "
           "per fiscal year, quarterly columns; quarter and component sums checked). Dated the first day of the "
           "fiscal quarter (FY Q1 = October). Recent editions rounded to the nearest ten. known_after_days=640 "
           "from the quarter start is a WORST-CASE bound: only year-end editions survive on the site; the FY2025 "
           "Q1 value first appears in the edition posted 2026-06-04 (611 days after 2024-10-01). Where quarterly "
           "editions did exist (FY2024 Q1 was posted 2024-09-16, 351 days after its quarter start) the data were "
           "public earlier. So this series is aligned late on purpose and only tests leads at long horizons. "
           "Quarterly, 40 observations (FY2016-FY2025).")
for sid, src, name, extra in [
        ("ohss_lpr_total", lpr, "Persons obtaining lawful permanent resident status (green cards), US (quarterly)",
         " New arrivals plus adjustments of status; adjustments depend on USCIS processing capacity."),
        ("ohss_lpr_new_arrivals", newarr, "New-arrival lawful permanent residents (immigrant visas used at entry), US (quarterly)",
         " Immigrants arriving from abroad with immigrant visas (State Department issuance driven); the COVID "
         "visa halt shows in 2020."),
        ("ohss_naturalizations", nat, "Persons naturalized as US citizens (quarterly)", "")]:
    write_series(sid, sorted(src.items()), name=name, category="migration", subcategory="legal_immigration",
                 region="US", source="DHS Office of Homeland Security Statistics", url=URL_L,
                 frequency="quarterly", units="persons per quarter", transform="logdiff", known_after_days=640,
                 notes=l_notes + extra, value_range=(1, 2e6))
