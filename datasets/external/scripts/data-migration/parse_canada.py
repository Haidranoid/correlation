"""Canada: IRCC monthly permanent-resident admissions (2015-01 onward) and Statistics Canada
quarterly components of international migration (table 17-10-0040-01, 1946 onward)."""
import csv
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from seasonality import seasonal_r2
from xlsx_read import xlsx_rows, xlsx_sheets

# ------------------------------------------------------------------ IRCC monthly
X = os.path.join(RAW, "ircc", "EN_ODP-PR-ProvImmCat.xlsx")
rows = xlsx_rows(X, xlsx_sheets(X)[0])
yr_row, mon_row = rows[2], rows[4]
MON = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
colmap = {}
year = None
for c in range(len(mon_row)):
    y = yr_row[c] if c < len(yr_row) else None
    if y not in (None, ""):
        try:
            year = int(float(y))
        except ValueError:
            pass
    m = mon_row[c]
    if isinstance(m, str) and m.strip() in MON and year:
        colmap[c] = dt.date(year, MON[m.strip()], 1)
print("IRCC month columns:", len(colmap), min(colmap.values()), max(colmap.values()))


def val(x):
    if x in (None, "", "--"):
        return None
    return float(str(x).replace(",", ""))


tot = {}
for r in rows:
    if r and isinstance(r[0], str) and r[0].strip().endswith("Total") or (r and r[0] == "Total"):
        tot[r[0].strip()] = {d: val(r[c]) for c, d in colmap.items() if c < len(r) and val(r[c]) is not None}
# identity check: sum of province totals vs Canada total (rounding: each cell rounded to 5)
provs = [k for k in tot if k.endswith(" - Total")]
worst = max(abs(sum(tot[p].get(d, 0) for p in provs) - tot["Total"][d]) for d in tot["Total"])
print(f"IRCC provinces vs Canada total, worst monthly abs diff: {worst:.0f} (rounding to 5 per cell)")

K_IRCC = 100
ircc_notes = ("IRCC open data 'Permanent Residents - Monthly IRCC Updates' (EN_ODP-PR-ProvImmCat.xlsx), "
              "admissions by province of intended destination; dated the 1st of the month. Counts are "
              "rounded to the nearest 5 and preliminary (revised in later monthly updates). "
              "known_after_days=100 from the 1st: the file posted 2026-09-15 runs to July 2026 (76 days); "
              "updates are roughly monthly but irregular and sometimes run about two months behind. Starts 2015, too short for the 3-year anomaly "
              "warm-up, so logdiff; mildly seasonal (calendar months explain {r2:.0f}% of monthly log "
              "changes); COVID collapse April-June 2020.")
URL_IRCC = "https://open.canada.ca/data/en/dataset/f7e5498e-0ad8-4417-85c9-9b8aff9b9eda"
for key, slug, label, region in [("Total", "total", "Canada", "CA"),
                                 ("Ontario - Total", "on", "Ontario", "CA-ON"),
                                 ("Quebec - Total", "qc", "Quebec", "CA-QC"),
                                 ("British Columbia - Total", "bc", "British Columbia", "CA-BC"),
                                 ("Alberta - Total", "ab", "Alberta", "CA-AB")]:
    s = sorted(tot[key].items())
    r2, _ = seasonal_r2(s)
    write_series(f"ircc_pr_{slug}", s, name=f"New permanent residents admitted, {label} (monthly)",
                 category="migration", subcategory="permanent_residents", region=region, source="IRCC (Canada)",
                 url=URL_IRCC, frequency="monthly", units="persons per month", transform="logdiff",
                 known_after_days=K_IRCC, notes=ircc_notes.format(r2=100 * r2), value_range=(1, 1e6))

# ------------------------------------------------------------------ StatCan quarterly
S = os.path.join(RAW, "statcan_migration", "unz", "17100040.csv")
data = {}
with open(S, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r["GEO"] != "Canada" or r["VALUE"] in ("", None):
            continue
        y, m = r["REF_DATE"].split("-")
        data.setdefault(r["Components of population growth"], {})[dt.date(int(y), int(m), 1)] = float(r["VALUE"])
for k, v in data.items():
    print("StatCan", k, min(v), max(v), len(v))
K_SC = 185
URL_SC = "https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710004001"
sc_base = ("Statistics Canada table 17-10-0040-01 (quarterly demographic estimates), Canada; dated the "
           "first day of the quarter (the flow covers that quarter). known_after_days=185 from the quarter "
           "start: Q2 2026 was released 2026-09-23 (85 days after quarter end, 176 after its start); "
           "quarterly estimates normally appear about 80-90 days after the quarter. Preliminary values are "
           "revised for several years (stored values are the latest revision).")
r2i, _ = seasonal_r2(sorted(data["Immigrants"].items()), period="quarter")
write_series("statcan_immigrants", sorted((d, v) for d, v in data["Immigrants"].items() if d.year >= 1971),
             name="Immigrants to Canada (quarterly)", category="migration", subcategory="immigration",
             region="CA", source="Statistics Canada", url=URL_SC, frequency="quarterly",
             units="persons per quarter", transform="anomaly_log1p", known_after_days=K_SC,
             notes=sc_base + f" Strongly seasonal (calendar quarters explain {100 * r2i:.0f}% of quarterly "
                             "log changes), long history, hence anomaly_log1p. Kept from 1971 (series starts 1946).",
             value_range=(1, 5e5))
r2e, _ = seasonal_r2(sorted(data["Emigrants"].items()), period="quarter")
write_series("statcan_emigrants", sorted((d, v) for d, v in data["Emigrants"].items() if d.year >= 1971),
             name="Emigrants from Canada (quarterly)", category="migration", subcategory="emigration",
             region="CA", source="Statistics Canada", url=URL_SC, frequency="quarterly",
             units="persons per quarter", transform="anomaly_log1p", known_after_days=K_SC,
             notes=sc_base + f" Emigration is largely modelled (not counted); strongly seasonal (calendar "
                             f"quarters explain {100 * r2e:.0f}% of quarterly log changes), hence anomaly_log1p.",
             value_range=(1, 2e5))
write_series("statcan_net_npr", sorted((d, v) for d, v in data["Net non-permanent residents"].items() if d.year >= 1971),
             name="Net non-permanent residents, Canada (quarterly)", category="migration",
             subcategory="temporary_residents", region="CA", source="Statistics Canada", url=URL_SC,
             frequency="quarterly", units="persons per quarter (net inflow; can be negative)", transform="diff",
             known_after_days=K_SC,
             notes=sc_base + " Net change in temporary residents (students, workers, asylum claimants): the "
                             "2022-2024 surge and the 2025-26 reversal under the new caps dominate. Can be "
                             "negative, hence diff. Method revised in 2023 (new NPR estimation), so the level "
                             "before and after is not fully comparable.",
             value_range=(-1e6, 1e6))
