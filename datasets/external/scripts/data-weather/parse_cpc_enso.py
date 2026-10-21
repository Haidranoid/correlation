"""Normalize CPC ENSO indices: ONI (monthly, 3-month running mean) and weekly OISST Nino SST anomalies.

Usage: python3 -I parse_cpc_enso.py
"""
import calendar
import datetime as dt
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, drop_entries, qa_line  # noqa: E402

D = os.path.join(RAW, "cpc", "enso")
SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
MON = {m.upper(): i for i, m in enumerate(calendar.month_abbr) if m}


def month_end(y, m):
    return dt.date(y, m, calendar.monthrange(y, m)[1])


def parse_oni():
    rows = []
    with open(os.path.join(D, "oni.ascii.txt"), encoding="latin-1") as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) != 4:
                continue
            seas, yr, _total, anom = p[0], int(p[1]), float(p[2]), float(p[3])
            i = SEAS.index(seas)  # middle month index = i (Jan=0 for DJF)
            # Season year = year of its middle month; the season ends one month after the middle month.
            mid_m = i + 1          # 1..12
            end_m = mid_m + 1
            end_y = yr
            if end_m == 13:
                end_m, end_y = 1, yr + 1
            rows.append((month_end(end_y, end_m), anom))
    return rows


def parse_weekly():
    out = {"nino12": [], "nino3": [], "nino34": [], "nino4": []}
    keys = ["nino12", "nino3", "nino34", "nino4"]
    with open(os.path.join(D, "wksst9120.for"), encoding="latin-1") as f:
        for line in f:
            m = re.match(r"\s(\d{2})([A-Z]{3})(\d{4})", line)
            if not m:
                continue
            d = dt.date(int(m.group(3)), MON[m.group(2)], int(m.group(1)))
            nums = re.findall(r"-?\d+\.\d", line[10:])
            assert len(nums) == 8, line
            for i, k in enumerate(keys):
                sst, ssta = float(nums[2 * i]), float(nums[2 * i + 1])
                out[k].append((d, ssta))
    return out


def main():
    drop_entries("cpc_enso_")
    oni = parse_oni()
    m = dict(name="CPC Oceanic Nino Index (ONI), 3-month running mean Nino-3.4 SST anomaly",
             category="climate_index", subcategory="enso", region="Equatorial Pacific Nino-3.4 (5N-5S, 170W-120W)",
             source="NOAA Climate Prediction Center (ERSSTv5)",
             url="https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", frequency="monthly",
             units="degC anomaly", transform="level", known_after_days=15,
             notes=("3-month running mean of ERSSTv5 Nino-3.4 anomalies; each row is dated at the LAST day of its 3-month "
                    "season (e.g. DJF 2000 -> 2000-02-29), so consecutive rows overlap by 2 months and changes are smoothed; "
                    "known_after_days is from that end date: the file was last updated 2026-10-03 with JAS 2026 (3 days), but "
                    "historically CPC updated the ONI with the monthly ENSO discussion (second Thursday, up to day 14), so 15 "
                    "is a conservative upper bound; values are revised: ERSST versions changed (v3b/v4/v5) and CPC uses "
                    "centered 30-year base periods updated every 5 years, so today's history differs from what was "
                    "published in real time (typically by 0.1-0.2 degC); already an anomaly -> level"))
    qa = write_series("cpc_enso_oni", oni, m, -4.0, 4.0, nd=2)
    print(qa_line(qa))

    wk = parse_weekly()
    for key, label, region in (("nino34", "Nino-3.4", "Equatorial Pacific Nino-3.4 (5N-5S, 170W-120W)"),
                               ("nino12", "Nino-1+2", "Far-eastern Pacific off Peru/Ecuador (0-10S, 90W-80W)")):
        m = dict(name=f"CPC weekly {label} SST anomaly (OISST)", category="climate_index", subcategory="enso",
                 region=region, source="NOAA Climate Prediction Center (OISST v2/v2.1)",
                 url="https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for", frequency="weekly",
                 units="degC anomaly", transform="level", known_after_days=6,
                 notes=("weekly mean OISST anomaly vs the 1991-2020 climatology; dated at the week's centre Wednesday "
                        "(the week runs Sunday-Saturday); CPC releases it the following Monday (5 days after the "
                        "centre) with the weekly ENSO update, +1 day for Monday holidays/late posting; the 1991-2020 "
                        "base was applied retroactively to the whole history (in real time older bases were used); "
                        "OISST moved from v2 to v2.1 in 2020; already an anomaly -> level"))
        qa = write_series(f"cpc_enso_{key}_weekly", wk[key], m, -5.0, 6.0, nd=1)
        print(qa_line(qa))


if __name__ == "__main__":
    main()
