"""CBP monthly border series.

USBP (Border Patrol) apprehensions/encounters by sector: FY2000-FY2020 from the CBP PDF table
"U.S. Border Patrol Monthly Encounters (FY 2000 - FY 2020)" (pdftotext -layout), FY2021 onward from
the Nationwide Encounters CSVs (component U.S. Border Patrol, Title 8 apprehensions + Title 42
expulsions). FY2020 is in both and is used as the splice check. Total CBP encounters (USBP + OFO)
exist only from FY2020 (CSV). Observations are dated the 1st of the calendar month.
"""
import csv
import datetime as dt
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from seasonality import seasonal_r2

D = os.path.join(RAW, "cbp")
MONTHS = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP"]


def cal(fy, mon):
    i = MONTHS.index(mon)
    return dt.date(fy - 1, 10 + i, 1) if i < 3 else dt.date(fy, i - 2, 1)


# ------------------------------------------------------------------ PDF FY2000-FY2020
txt = subprocess.run(["pdftotext", "-layout", os.path.join(D, "usbp_monthly_encounters_fy2000-fy2020.pdf"), "-"],
                     capture_output=True, text=True, check=True).stdout
pdf = defaultdict(dict)  # (sector) -> {date: value}
fy = None
prev_label = ""
NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
for line in txt.splitlines():
    m = re.search(r"by month - FY (\d{4})", line, re.I)
    if m:
        fy = int(m.group(1))
        continue
    if fy is None:
        continue
    toks = re.findall(NUM + r"|N/A", line)
    label = re.sub(r"\s*(" + NUM + r"|N/A)(\s+|$)", " ", line).strip().rstrip("*").strip()
    nums = [t for t in re.findall(r"\S+", line) if re.fullmatch(NUM + "|N/A", t)]
    if len(nums) == 13:
        name = label if label and not label.startswith("(") else prev_label
        name = re.sub(r"\s+", " ", name)
        if "N/A" in nums:
            continue
        vals = [int(x.replace(",", "")) for x in nums]
        if sum(vals[:12]) != vals[12]:
            raise SystemExit(f"FY{fy} {name}: months sum {sum(vals[:12])} != total {vals[12]}")
        for mon, v in zip(MONTHS, vals[:12]):
            d = cal(fy, mon)
            if d in pdf[name]:
                raise SystemExit(f"duplicate {name} {d}")
            pdf[name][d] = v
    elif line.strip() and not re.search(r"\d", line):
        prev_label = line.strip()

SW = ["Big Bend", "Del Rio", "El Centro", "El Paso", "Laredo", "Rio Grande Valley", "San Diego", "Tucson", "Yuma"]
NORTH = ["Blaine", "Buffalo", "Detroit", "Grand Forks", "Havre", "Houlton", "Spokane", "Swanton"]
print("PDF rows:", sorted(pdf))
# accounting identities in the PDF
for d in sorted(pdf["Monthly Total"]):
    sw = sum(pdf[s].get(d, 0) for s in SW)
    if sw != pdf["Southwest Border"][d]:
        raise SystemExit(f"SW sum mismatch {d}: {sw} vs {pdf['Southwest Border'][d]}")
    tot = pdf["Southwest Border"][d] + pdf["Northern Border"][d] + pdf["Coastal Border"][d]
    if tot != pdf["Monthly Total"][d]:
        raise SystemExit(f"total mismatch {d}: {tot} vs {pdf['Monthly Total'][d]}")
print("PDF identities OK:", len(pdf["Monthly Total"]), "months",
      min(pdf["Monthly Total"]), "..", max(pdf["Monthly Total"]))

# ------------------------------------------------------------------ CSVs FY2020+
SRC = [("nationwide-encounters-fy20-fy23-aor.csv", {2020}),
       ("nationwide-encounters-fy21-fy24-aor.csv", {2021}),
       ("nationwide-encounters-fy22-fy25-aor.csv", {2022}),
       ("nationwide-encounters-fy23-fy26-aug-aor.csv", {2023, 2024, 2025, 2026})]
csvd = defaultdict(lambda: defaultdict(int))  # key -> {date: count}
for fname, fys in SRC:
    with open(os.path.join(D, fname), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            fyr = int(r["Fiscal Year"][:4])
            if fyr not in fys:
                continue
            d = cal(fyr, r["Month (abbv)"])
            n = int(r["Encounter Count"])
            comp = "USBP" if r["Component"] == "U.S. Border Patrol" else "OFO"
            reg = r["Land Border Region"]
            aor = r["Area of Responsibility"].replace(" Sector", "")
            csvd[("ALL",)][d] += n
            csvd[(comp, "ALL")][d] += n
            csvd[(comp, reg)][d] += n
            csvd[("ALL", reg)][d] += n
            if comp == "USBP":
                csvd[("USBP", "AOR", aor)][d] += n
# splice check on FY2020 (Oct 2019 - Sep 2020)
fy20 = [cal(2020, m) for m in MONTHS]
checks = [("Monthly Total", ("USBP", "ALL")), ("Southwest Border", ("USBP", "Southwest Land Border")),
          ("Northern Border", ("USBP", "Northern Land Border")), ("Rio Grande Valley", ("USBP", "AOR", "Rio Grande Valley")),
          ("Tucson", ("USBP", "AOR", "Tucson")), ("San Diego", ("USBP", "AOR", "San Diego"))]
for pname, key in checks:
    diffs = [(d, pdf[pname][d], csvd[key][d]) for d in fy20 if pdf[pname][d] != csvd[key][d]]
    print(f"FY2020 splice {pname}: {'identical' if not diffs else diffs}")

K = 65
common_notes = (
    "Dated the 1st of the calendar month (fiscal months converted: Oct FY2020 = 2019-10). Source: "
    "FY2000-FY2020 from the CBP PDF 'U.S. Border Patrol Monthly Encounters (FY 2000 - FY 2020)' (sums "
    "verified: months = yearly totals, sectors = Southwest Border, regions = Monthly Total); FY2021 on "
    "from the Nationwide Encounters CSVs, the latest file covering each fiscal year (FY2020 identical "
    "in both, so the splice is seamless). Definition change: from March 2020 USBP counts include Title "
    "42 expulsions (encounters, not only Title 8 apprehensions); Title 42 ended 2023-05-11 and repeat "
    "crossers inflated 2020-2023 counts. known_after_days=65 from the 1st: CBP's monthly update typically "
    "came in the second half of the following month in 2014-2024 (the December 2023 update was reported for "
    "late January 2024), and 5-10 days after month end in 2025-26 (file Last-Modified dates, e.g. Aug 2026 data "
    "posted 2026-09-10); 65 covers releases up to about a month after month end. Later files revise earlier "
    "months slightly. Monthly numbers before about FY2016 "
    "were not published monthly in real time (this table was compiled in 2020-21), so the timing of "
    "early years is notional.")


def series_from(pname, key):
    out = dict(pdf[pname]) if pname else {}
    for d, v in csvd[key].items():
        if d.year > 2020 or (d.year == 2020 and d.month >= 10):
            out[d] = v
    return sorted(out.items())


USBP = [
    ("cbp_usbp_total", "US Border Patrol apprehensions/encounters, nationwide (monthly)", "Monthly Total", ("USBP", "ALL"), "US", "national"),
    ("cbp_usbp_swb", "US Border Patrol apprehensions/encounters, Southwest border (monthly)", "Southwest Border", ("USBP", "Southwest Land Border"), "US-SWB", "southwest_border"),
    ("cbp_usbp_northern", "US Border Patrol apprehensions/encounters, Northern border (monthly)", "Northern Border", ("USBP", "Northern Land Border"), "US-NB", "northern_border"),
    ("cbp_usbp_rgv", "US Border Patrol apprehensions/encounters, Rio Grande Valley sector TX (monthly)", "Rio Grande Valley", ("USBP", "AOR", "Rio Grande Valley"), "US-TX", "sector"),
    ("cbp_usbp_tucson", "US Border Patrol apprehensions/encounters, Tucson sector AZ (monthly)", "Tucson", ("USBP", "AOR", "Tucson"), "US-AZ", "sector"),
    ("cbp_usbp_elpaso", "US Border Patrol apprehensions/encounters, El Paso sector TX/NM (monthly)", "El Paso", ("USBP", "AOR", "El Paso"), "US-TX", "sector"),
    ("cbp_usbp_sandiego", "US Border Patrol apprehensions/encounters, San Diego sector CA (monthly)", "San Diego", ("USBP", "AOR", "San Diego"), "US-CA", "sector"),
    ("cbp_usbp_delrio", "US Border Patrol apprehensions/encounters, Del Rio sector TX (monthly)", "Del Rio", ("USBP", "AOR", "Del Rio"), "US-TX", "sector"),
]
for sid, name, pname, key, region, sub in USBP:
    rows = series_from(pname, key)
    extra = ""
    if sid == "cbp_usbp_rgv":
        extra = " Called McAllen sector before FY2012 (same sector)."
    if sid == "cbp_usbp_northern":
        extra = " Small counts (tens to a few thousand); the 2023-24 rise is mostly Swanton sector (NY/VT)."
    r2, _ = seasonal_r2(rows)
    tf = "anomaly_log1p" if r2 >= 0.2 else "logdiff"
    season = (f" Calendar months explain {100 * r2:.0f}% of monthly log changes (spring peak); "
              + ("at least 20% with history from 1999, hence anomaly_log1p (the test needs 3 prior years, so moves "
                 "start Oct 2002)." if tf == "anomaly_log1p" else
                 "under 20% (large non-seasonal surges dominate), hence logdiff."))
    write_series(sid, rows, name=name, category="migration", subcategory=f"border_{sub}", region=region,
                 source="US Customs and Border Protection", url="https://www.cbp.gov/document/stats/nationwide-encounters",
                 frequency="monthly", units="persons encountered per month", transform=tf,
                 known_after_days=K, notes=common_notes + season + extra, value_range=(1, 1e6))

# FY2020+ only (USBP + OFO); short history, so logdiff (anomaly would leave only 2022 on)
short_notes = (
    "Nationwide Encounters CSVs only (from FY2020 = Oct 2019), latest file per fiscal year; dated the 1st "
    "of the calendar month. Encounters = USBP Title 8 apprehensions + OFO Title 8 inadmissibles + Title 42 "
    "expulsions (Mar 2020 - May 2023). OFO counts include CBP One appointments at ports (2023-2024). "
    "known_after_days=65 as for the USBP series. Seasonal, but history is too short for the 3-year "
    "anomaly warm-up, so logdiff (the seasonal swing stays in the moves).")
for sid, name, key, region, sub in [
        ("cbp_enc_total", "CBP total encounters nationwide, USBP + OFO (monthly)", ("ALL",), "US", "national"),
        ("cbp_enc_swb", "CBP total encounters, Southwest land border, USBP + OFO (monthly)", ("ALL", "Southwest Land Border"), "US-SWB", "southwest_border"),
        ("cbp_ofo_total", "CBP Office of Field Operations encounters (ports of entry, airports) nationwide (monthly)", ("OFO", "ALL"), "US", "ports_of_entry")]:
    rows = sorted(csvd[key].items())
    write_series(sid, rows, name=name, category="migration", subcategory=f"border_{sub}", region=region,
                 source="US Customs and Border Protection", url="https://www.cbp.gov/document/stats/nationwide-encounters",
                 frequency="monthly", units="persons encountered per month", transform="logdiff",
                 known_after_days=K, notes=short_notes, value_range=(1, 1e6))
