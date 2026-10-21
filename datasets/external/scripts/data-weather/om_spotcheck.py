"""Spot checks for Open-Meteo points against known events, plus raw null counts.

Usage: python3 -I om_spotcheck.py
"""
import collections
import datetime as dt
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, SERIES_DIR  # noqa: E402


def load(sid):
    p = os.path.join(SERIES_DIR, sid + ".csv")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as f:
        next(f)
        for ln in f:
            d, v = ln.strip().split(",")
            out[dt.date.fromisoformat(d)] = float(v)
    return out


def main():
    print("== raw null counts")
    for p in sorted(glob.glob(os.path.join(RAW, "openmeteo", "*.json"))):
        js = json.load(open(p))
        nulls = {k: sum(1 for x in v if x is None) for k, v in js["daily"].items() if k != "time"}
        bad = {k: n for k, n in nulls.items() if n}
        print(f"  {os.path.basename(p):42} days={len(js['daily']['time'])} nulls={bad or 0}")

    s = load("om_br_coffee_tmin")
    if s:
        print("== Brazil coffee Tmin around the 2021-07-20 frost (lowest days of July 2021, and the all-period minimum)")
        jul = sorted(((v, d) for d, v in s.items() if d.year == 2021 and d.month == 7))[:4]
        print("  ", [(d.isoformat(), v) for v, d in jul])
        print("   all-period min:", min((v, d.isoformat()) for d, v in s.items()))
    s = load("om_ar_pampas_soilm")
    if s:
        print("== Pampas 7-28 cm soil moisture: yearly mean and min (2022-23 drought should be lowest)")
        by = collections.defaultdict(list)
        for d, v in s.items():
            by[d.year].append(v)
        print("  ", " ".join(f"{y}:{sum(v)/len(v):.3f}/{min(v):.3f}" for y, v in sorted(by.items()) if y >= 2015))
        low = min((v, d.isoformat()) for d, v in s.items())
        print("   all-period min:", low)
    s = load("om_br_soy_prcp")
    if s:
        print("== Mato Grosso precipitation: mean mm/day by month (dry Jun-Aug, wet Dec-Feb)")
        by = collections.defaultdict(list)
        for d, v in s.items():
            by[d.month].append(v)
        print("  ", " ".join(f"{m}:{sum(v)/len(v):.1f}" for m, v in sorted(by.items())))
    s = load("om_ks_wheat_soilm")
    if s:
        print("== Kansas soil moisture: lowest years (2011-12 and 2022-23 droughts expected)")
        by = collections.defaultdict(list)
        for d, v in s.items():
            by[d.year].append(v)
        print("  ", sorted(((round(sum(v)/len(v), 3), y) for y, v in by.items()))[:5])
    s = load("om_il_corn_soilm")
    if s:
        print("== Illinois soil moisture, July means (2012 drought expected lowest)")
        by = collections.defaultdict(list)
        for d, v in s.items():
            if d.month == 7:
                by[d.year].append(v)
        print("  ", sorted(((round(sum(v)/len(v), 3), y) for y, v in by.items()))[:4])
    s = load("om_tokyo_tmax")
    if s:
        print("== Tokyo Tmax: hottest days (2018-07-23 record ~41C nearby; ERA5 grid cells run cooler)")
        print("  ", sorted(((v, d.isoformat()) for d, v in s.items()), reverse=True)[:3])


if __name__ == "__main__":
    main()
