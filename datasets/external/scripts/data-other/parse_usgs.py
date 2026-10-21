"""USGS ComCat -> daily (UTC date) earthquake series with explicit zeros on quiet days."""
import csv, datetime as dt, glob, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, daterange, HERE

D = os.path.join(RAW, "usgs")
START, END = dt.date(1995, 1, 1), dt.date(2026, 10, 17)
URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
logcounts = {}
for line in open(os.path.join(HERE, "logs", "usgs.log")):
    m = re.match(r"(\w+) (\d{4}): count=(\d+)", line)
    if m:
        logcounts[(m.group(1), int(m.group(2)))] = int(m.group(3))

def load(prefix, minmag):
    ev = {}
    per_year = defaultdict(int)
    for p in sorted(glob.glob(os.path.join(D, f"{prefix}_*.csv"))):
        y = int(p[-8:-4])
        n = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            n += 1
            assert r["type"] == "earthquake", r["type"]
            if r["mag"] == "" or float(r["mag"]) < minmag:
                if r["mag"] != "":
                    print(f"  dropped sub-threshold event {r['id']} {r['time']} M{r['mag']} {r['magType']} "
                          "(server matched a non-preferred magnitude)")
                continue
            t = dt.datetime.fromisoformat(r["time"].replace("Z", "+00:00"))
            ev[r["id"]] = (t.date(), float(r["mag"]), r["magType"], r["status"])
        per_year[y] = n
        assert n == logcounts[(prefix, y)], (prefix, y, n, logcounts[(prefix, y)])
    return ev, per_year

world, wy = load("world_m45", 4.5)
ca, cy = load("ca_m30", 3.0)
print(f"world M4.5+ events: {len(world)} (rows per year match the count endpoint)")
print(f"CA-box M3+ events: {len(ca)}")
status = defaultdict(int)
for d, m, mt, st in world.values():
    status[st] += 1
print("world status:", dict(status))
assert min(m for _, m, _, _ in world.values()) >= 4.5

cnt5, cnt6, mx = defaultdict(int), defaultdict(int), {}
for d, m, mt, st in world.values():
    if m >= 5.0:
        cnt5[d] += 1
    if m >= 6.0:
        cnt6[d] += 1
    mx[d] = max(mx.get(d, 0), m)
cac = defaultdict(int)
for d, m, mt, st in ca.values():
    cac[d] += 1
days = list(daterange(START, END))
no_m45 = [d for d in days if d not in mx]
print(f"days without any M4.5+ event: {len(no_m45)} {no_m45[:10]}")

common = ("USGS ComCat, eventtype=earthquake (explosions and quarry blasts excluded), UTC calendar days; "
          "every day present, with 0 on days with no qualifying event. Catalog is revised after the fact "
          "(magnitudes reviewed, events added/removed, magnitude types change), so counts near a threshold "
          "differ slightly from what was known in real time; large events are reported within minutes. "
          "The UTC day ends at 7-8 pm New York, so the daily total is known by the next US close.")
write_series("usgs_world_m5_count", [(d, cnt5.get(d, 0)) for d in days],
             name="Worldwide earthquakes of magnitude 5+ per day", category="geophysical",
             subcategory="earthquake", region="world", source="USGS ComCat", url=URL, frequency="daily",
             units="events per UTC day", transform="level", known_after_days=1, value_range=(0, 500),
             notes="Count of M>=5.0 events. Aftershock sequences (e.g. 2004 Sumatra, 2011 Tohoku) create "
                   "multi-week bursts. " + common)
write_series("usgs_world_m6_count", [(d, cnt6.get(d, 0)) for d in days],
             name="Worldwide earthquakes of magnitude 6+ per day", category="geophysical",
             subcategory="earthquake", region="world", source="USGS ComCat", url=URL, frequency="daily",
             units="events per UTC day", transform="level", known_after_days=1, value_range=(0, 100),
             notes="Count of M>=6.0 events (about 130-150 a year, the newsworthy ones). Mostly 0. " + common)
write_series("usgs_world_max_mag", [(d, mx[d]) for d in days if d in mx],
             name="Largest earthquake magnitude worldwide per day", category="geophysical",
             subcategory="earthquake", region="world", source="USGS ComCat", url=URL, frequency="daily",
             units="magnitude (mixed types, mostly Mw/mb)", transform="level", known_after_days=1,
             value_range=(4.5, 9.5),
             notes="Max magnitude over M>=4.5 events of the UTC day; days with no M4.5+ event are left out "
                   f"(missing, not zero; {len(no_m45)} such days). " + common)
write_series("usgs_ca_m3_count", [(d, cac.get(d, 0)) for d in days],
             name="California-region earthquakes of magnitude 3+ per day", category="geophysical",
             subcategory="earthquake", region="US-CA (box 32-42N, 125-114W)", source="USGS ComCat",
             url=URL, frequency="daily", units="events per UTC day", transform="level", known_after_days=1,
             value_range=(0, 2000),
             notes="Box 32-42N, 125-114W (California plus western Nevada and northern Baja). M>=3 is well above "
                   "the regional completeness magnitude throughout. Dominated by aftershock sequences (e.g. "
                   "2010 El Mayor-Cucapah, 2019 Ridgecrest). " + common)
# biggest days, sanity
big = sorted(((m, d) for d, m in mx.items()), reverse=True)[:6]
print("largest days:", [(str(d), m) for m, d in big])
print("busiest CA days:", sorted(((v, str(d)) for d, v in cac.items()), reverse=True)[:4])
