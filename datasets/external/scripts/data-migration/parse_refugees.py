"""US refugee admissions by month, FY2001-FY2026, from the Refugee Processing Center's Refugee
Admissions Report (one sheet per fiscal year, 'Grand Totals' row, OCT..SEP)."""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from seasonality import seasonal_r2
from xlsx_read import xlsx_rows, xlsx_sheets

F = os.path.join(RAW, "wrapsnet", "current.xlsx")
out = []
for s in xlsx_sheets(F):
    if not s.isdigit():
        continue
    fy = int(s)
    rows = xlsx_rows(F, s)
    hdr = next(r for r in rows if r and "OCT" in [str(c).strip() for c in r])
    io = [str(c).strip() for c in hdr].index("OCT")
    assert [str(c).strip() for c in hdr[io:io + 12]] == ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY",
                                                         "JUN", "JUL", "AUG", "SEP"], hdr
    gt = [r for r in rows if r and any(isinstance(c, str) and c.strip().lower().startswith("grand total") for c in r)]
    assert len(gt) == 1, (fy, len(gt))
    g = gt[0]
    months = [float(g[io + k] or 0) for k in range(12)]
    tot = float(g[io - 1])
    if abs(sum(months) - tot) > 0.5:
        raise SystemExit(f"FY{fy}: months {sum(months)} != total {tot}")
    for k, v in enumerate(months):
        d = dt.date(fy - 1, 10 + k, 1) if k < 3 else dt.date(fy, k - 2, 1)
        out.append((d, v))
out.sort()
r2, n = seasonal_r2(out)
zeros = sum(1 for _, v in out if v == 0)
write_series(
    "wraps_refugee_arrivals", out, name="US refugee admissions (arrivals) per month", category="migration",
    subcategory="refugees", region="US", source="US State Department, Refugee Processing Center (WRAPS)",
    url="https://www.wrapsnet.org/admissions-and-arrivals/", frequency="monthly", units="persons per month",
    transform="anomaly_log1p", known_after_days=60,
    notes=("Refugee Admissions Report as of 30 September 2026 (one sheet per fiscal year FY2001-FY2026), "
           "'Grand Totals' row by month; every fiscal year's months sum to its FY total, and FY2012-2017 and "
           "FY2019-2020 match the archived 'Arrivals by Region by Month' files exactly (check_refugee_archives.py). Dated the 1st of the calendar "
           "month. Excludes SIV recipients. Policy-driven: post-9/11 halt (Oct-Nov 2001: 4 a month), FY2018-21 "
           "ceilings cut, COVID pause (Apr 2020: 27), the January 2025 suspension (Feb-Sep 2025 under 200 a "
           f"month), FY2026 almost all South Africans. {zeros} zero months. Seasonal (end-of-fiscal-year push in "
           f"August-September; calendar months explain {100 * r2:.0f}% of monthly log changes), hence "
           "anomaly_log1p (zero-safe). known_after_days=60 from the 1st: the report is normally updated early "
           "the next month (the 30 Sep 2026 edition was posted 2026-10-05); 60 allows for delays such as the "
           "Oct-Nov 2025 shutdown (the FY2025 year-end archive file is dated 2025-11-21)."),
    value_range=(0, 1e5))
