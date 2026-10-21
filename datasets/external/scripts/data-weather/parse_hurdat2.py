"""Daily Atlantic tropical-cyclone event series from NOAA NHC HURDAT2 best tracks.

Usage: python3 -I parse_hurdat2.py
Series (UTC days, every day 1980-01-01 .. 2025-12-31, zeros are real values):
  hurdat_atl_hurricanes_active   distinct storms with status HU in any record that day
  hurdat_atl_ace                 daily accumulated cyclone energy (synoptic 6-hourly TS/SS/HU >= 35 kt, v^2/1e4)
  hurdat_us_landfall_maxwind_kt  max wind (kt) of any CONUS landfall record that day (any status, >= 34 kt), else 0
  hurdat_us_hurricane_landfall   1 if a CONUS landfall record with wind >= 64 kt that day (any status), else 0
"""
import collections
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, drop_entries, qa_line  # noqa: E402

F = os.path.join(RAW, "hurdat2", "hurdat2-1851-2025-092326.txt")
URL = "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt"
START = dt.date(1980, 1, 1)
END = dt.date(2025, 12, 31)


def latlon(s_lat, s_lon):
    lat = float(s_lat[:-1]) * (1 if s_lat[-1] == "N" else -1)
    lon = float(s_lon[:-1]) * (-1 if s_lon[-1] == "W" else 1)
    return lat, lon


def is_conus(lat, lon):
    """Heuristic box for the US Gulf/Atlantic coast (Texas to Maine)."""
    if not (24.4 <= lat <= 47.5 and -98.0 <= lon <= -66.9):
        return False
    if lat < 25.95 and lon < -96.5:     # Mexico (Tamaulipas) south of the Rio Grande mouth
        return False
    if lat < 27.6 and lon > -79.6:      # Bahamas (Bimini/Grand Bahama are east of -79.3)
        return False
    if lat > 43.0 and lon > -66.98:     # Canada (Nova Scotia / New Brunswick)
        return False
    return True


def parse():
    storms = []
    with open(F, encoding="latin-1") as f:
        cur = None
        for line in f:
            p = [x.strip() for x in line.rstrip("\n").split(",")]
            if len(p) >= 3 and p[0][:2] in ("AL",) and len(p[0]) == 8:
                cur = {"id": p[0], "name": p[1], "recs": []}
                storms.append(cur)
                continue
            if len(p) < 8 or cur is None:
                continue
            d = dt.date(int(p[0][:4]), int(p[0][4:6]), int(p[0][6:8]))
            lat, lon = latlon(p[4], p[5])
            cur["recs"].append(dict(date=d, hhmm=p[1], rid=p[2], status=p[3], lat=lat, lon=lon,
                                    wind=int(p[6]), pres=int(p[7])))
    return storms


def main():
    drop_entries("hurdat_")
    storms = parse()
    hur = collections.defaultdict(set)
    ace = collections.defaultdict(float)
    lf_wind = collections.defaultdict(int)
    landfalls = []
    neg_wind = 0
    for s in storms:
        for r in s["recs"]:
            d = r["date"]
            if not (START <= d <= END):
                continue
            if r["wind"] < 0:
                neg_wind += 1
                continue
            if r["status"] == "HU":
                hur[d].add(s["id"])
            if r["hhmm"] in ("0000", "0600", "1200", "1800") and r["status"] in ("TS", "SS", "HU") and r["wind"] >= 35:
                ace[d] += r["wind"] ** 2 / 1e4
            if r["rid"] == "L" and r["wind"] >= 34 and is_conus(r["lat"], r["lon"]):
                lf_wind[d] = max(lf_wind[d], r["wind"])
                landfalls.append((d, r["hhmm"], s["id"], s["name"], r["status"], r["lat"], r["lon"], r["wind"]))
    days = [START + dt.timedelta(i) for i in range((END - START).days + 1)]
    notes_common = ("NHC HURDAT2 Atlantic best track (file revised 2026-09-23, seasons through 2025); UTC days; every day "
                    "1980-01-01..2025-12-31 is written and zero means no activity; best track is a post-season reanalysis "
                    "(intensities/status sometimes revised from the real-time advisories, e.g. storms upgraded or added "
                    "later), so it is slightly better than what was known in real time; known_after_days=1 because the "
                    "operational advisories (every 6 h) made the day's activity public by the next day")
    lf_note = ("CONUS landfall = HURDAT2 record identifier L at a point inside a heuristic Texas-to-Maine coastal box "
               "(excludes Mexico south of 25.95N/-96.5, the Bahamas east of -79.6 below 27.6N, Cuba below 24.4N, "
               "Canada east of -66.98 above 43N; Puerto Rico/USVI excluded); any status counts (e.g. post-tropical Sandy "
               "2012-10-29), so this is not the official US hurricane landfall list")
    out = [
        ("hurdat_atl_hurricanes_active", "Atlantic hurricanes active (count of distinct storms at hurricane status)",
         "count", [(d, float(len(hur.get(d, ())))) for d in days], 0, 8, 0, notes_common),
        ("hurdat_atl_ace", "Atlantic daily accumulated cyclone energy (ACE)", "1e4 kt^2",
         [(d, ace.get(d, 0.0)) for d in days], 0, 30, 4,
         notes_common + "; ACE = sum over 00/06/12/18 UTC fixes of TS/SS/HU systems with wind >= 35 kt of wind^2/1e4"),
        ("hurdat_us_landfall_maxwind_kt", "US (CONUS) tropical-cyclone landfall: max sustained wind at landfall",
         "kt", [(d, float(lf_wind.get(d, 0))) for d in days], 0, 170, 0,
         notes_common + "; max wind (kt) among CONUS landfall records with wind >= 34 kt, else 0; " + lf_note),
        ("hurdat_us_hurricane_landfall", "US (CONUS) hurricane-force landfall indicator (>=64 kt at landfall)",
         "0/1", [(d, 1.0 if lf_wind.get(d, 0) >= 64 else 0.0) for d in days], 0, 1, 0,
         notes_common + "; 1 if a CONUS landfall record that day has wind >= 64 kt, else 0; " + lf_note),
    ]
    for sid, name, units, rows, lo, hi, nd, notes in out:
        m = dict(name=name, category="weather", subcategory="tropical_cyclone",
                 region="North Atlantic basin" if "atl" in sid else "US Gulf/Atlantic coast (CONUS)",
                 source="NOAA National Hurricane Center HURDAT2", url=URL, frequency="daily", units=units,
                 transform="level", known_after_days=1, notes=notes)
        qa = write_series(sid, rows, m, lo, hi, nd=nd)
        print(qa_line(qa))
    print("records with negative wind (skipped):", neg_wind)
    print("CONUS landfalls since 1980:", len(landfalls), " hurricane-force days:",
          sum(1 for d in days if lf_wind.get(d, 0) >= 64))
    for chk in ("2005-08-25", "2005-08-29", "2012-10-29", "2017-08-26", "2017-09-10", "2022-09-28", "2024-09-27",
                "2024-10-10", "2021-08-29", "2018-10-10", "2020-08-27", "2024-07-08"):
        d = dt.date.fromisoformat(chk)
        hits = [l for l in landfalls if l[0] == d]
        print(chk, "maxwind", lf_wind.get(d, 0), [(l[1], l[3], l[4], l[5], l[6], l[7]) for l in hits])
    # landfalls near the box edges, for eyeballing
    print("edge cases:")
    for l in landfalls:
        lat, lon = l[5], l[6]
        if lat < 26.3 or lon > -76 or lon < -96.8:
            print("  ", l)
    # L records just outside the box (could be missed US landfalls)
    print("L records >=34 kt just outside the box:")
    for s in storms:
        for r in s["recs"]:
            if r["rid"] == "L" and START <= r["date"] <= END and r["wind"] >= 34 and not is_conus(r["lat"], r["lon"]):
                if 24.0 <= r["lat"] <= 48 and -98.5 <= r["lon"] <= -66.0:
                    print("  ", r["date"], r["hhmm"], s["name"], r["status"], r["lat"], r["lon"], r["wind"])


if __name__ == "__main__":
    main()
