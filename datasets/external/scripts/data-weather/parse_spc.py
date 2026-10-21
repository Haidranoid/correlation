"""Daily US tornado counts from the NOAA SPC severe weather database (final, 1950-2025).

Usage: python3 -I parse_spc.py
"""
import collections
import csv
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, drop_entries, qa_line  # noqa: E402

F = os.path.join(RAW, "spc", "1950-2025_actual_tornadoes.csv")
URL = "https://www.spc.noaa.gov/wcm/data/1950-2025_actual_tornadoes.csv"
START, END = dt.date(1980, 1, 1), dt.date(2025, 12, 31)


def main():
    drop_entries("spc_")
    allc, sigc = collections.Counter(), collections.Counter()
    stats = collections.Counter()
    with open(F, newline="", encoding="latin-1") as f:
        for r in csv.DictReader(f):
            d = dt.date.fromisoformat(r["date"])
            if not (START <= d <= END):
                continue
            stats["records"] += 1
            if r["sg"] != "1":
                stats["non_sg1"] += 1
                continue
            if r["st"] in ("PR", "VI"):
                stats["pr_vi"] += 1
                continue
            if r["tz"] != "3":
                stats["tz_not_cst"] += 1
            allc[d] += 1
            mag = int(r["mag"])
            if mag == -9:
                stats["mag_unknown"] += 1
            if mag >= 2:
                sigc[d] += 1
    print(dict(stats))
    days = [START + dt.timedelta(i) for i in range((END - START).days + 1)]
    common = ("NOAA SPC severe weather database (final; the 2025 database was released 2026-04-24), one record per tornado "
              "(sg=1); dates are CST days (all but %d records since 1980 are CST; the rest are pre-1995 MST/unknown and "
              "left as given); every day 1980-01-01..2025-12-31 written, zero = no tornado; Puerto Rico/USVI excluded; "
              "the final database revises the real-time local storm reports (duplicates removed, damage surveys), so it is "
              "cleaner than what was known on the day; counts are not stationary: weak-tornado reports rose with NEXRAD "
              "Doppler radar (1990s) and spotter networks; F-scale became the EF-scale on 2007-02-01" % stats["tz_not_cst"])
    out = [
        ("spc_tornadoes_us", "US tornadoes per day (all ratings)", [(d, float(allc.get(d, 0))) for d in days], 3,
         common + "; known_after_days=3: preliminary storm reports are public the same day (the CST day ends 01:00 ET), "
         "but the final count also includes tornadoes found by NWS damage surveys in the following days"),
        ("spc_sig_tornadoes_us", "US significant tornadoes per day ((E)F2 or stronger)",
         [(d, float(sigc.get(d, 0))) for d in days], 7,
         common + "; (E)F2+ by the final rating; ratings come from NWS damage surveys, usually within 1-3 days but up to "
         "about a week after big outbreaks, so known_after_days=7; (E)F2+ counts are much less affected by the "
         "reporting trend than all-tornado counts"),
    ]
    for sid, name, rows, k, notes in out:
        m = dict(name=name, category="weather", subcategory="tornado", region="US (50 states)",
                 source="NOAA Storm Prediction Center", url=URL, frequency="daily", units="count",
                 transform="level", known_after_days=k, notes=notes)
        qa = write_series(sid, rows, m, 0, 400, nd=0)
        print(qa_line(qa))
    top = sorted(allc.items(), key=lambda x: -x[1])[:5]
    print("top days:", [(d.isoformat(), n, sigc.get(d, 0)) for d, n in top])
    yearly = collections.Counter()
    for d, n in allc.items():
        yearly[d.year] += n
    print("yearly counts:", " ".join(f"{y}:{yearly[y]}" for y in sorted(yearly) if y >= 2000))


if __name__ == "__main__":
    main()
