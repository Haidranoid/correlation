"""Download USGS ComCat earthquakes by year (raw CSV, unmodified).

world: M>=4.5, all regions.  ca: M>=3.0 in the California box (lat 32-42, lon -125 to -114).
eventtype=earthquake excludes explosions, quarry blasts, nuclear tests, ice quakes.
Each query is checked with the count endpoint first (API cap is 20,000 events per query).
"""
import os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW, UA
import time

BASE = "https://earthquake.usgs.gov/fdsnws/event/1/"
OUT = os.path.join(RAW, "usgs")
LAST_YEAR = 2026
END_2026 = "2026-10-18"  # exclusive bound: last complete UTC day is 2026-10-17

SETS = {
    "world_m45": dict(minmagnitude="4.5"),
    "ca_m30": dict(minmagnitude="3.0", minlatitude="32", maxlatitude="42",
                   minlongitude="-125", maxlongitude="-114"),
}

def q(params):
    return "&".join(f"{k}={v}" for k, v in params.items())

for name, extra in SETS.items():
    for y in range(1995, LAST_YEAR + 1):
        start = f"{y}-01-01"
        end = f"{y+1}-01-01" if y < LAST_YEAR else END_2026
        p = dict(starttime=start, endtime=end, eventtype="earthquake", **extra)
        dest = os.path.join(OUT, f"{name}_{y}.csv")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        req = urllib.request.Request(BASE + "count?" + q(p), headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r:
            n = int(r.read().decode().strip())
        time.sleep(1)
        if n >= 20000:
            print(f"{name} {y}: {n} events >= 20000 cap, needs splitting", flush=True)
            sys.exit(1)
        p.update(format="csv", orderby="time-asc")
        fetch(BASE + "query?" + q(p), dest, sleep=2, timeout=300)
        print(f"{name} {y}: count={n} -> {os.path.getsize(dest)} bytes", flush=True)
print("done", flush=True)
