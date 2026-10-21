"""Look up GHCN-Daily stations by id or name pattern and show element coverage.

Usage: python3 -I find_stations.py <ghcnd_dir> <query> [<query> ...]
A query is either an 11-char station id or 'CC:pattern' (country/state prefix of the
id, e.g. 'USW:CHICAGO', 'UK:HEATHROW') matched case-insensitively against the name.
"""
import sys, os, re

ghcnd = sys.argv[1]
queries = sys.argv[2:]
ELEMS = ("TMAX", "TMIN", "PRCP", "SNOW", "AWND")

stations = {}
with open(os.path.join(ghcnd, "ghcnd-stations.txt"), encoding="latin-1") as f:
    for line in f:
        sid = line[0:11]
        stations[sid] = dict(lat=float(line[12:20]), lon=float(line[21:30]),
                             elev=line[31:37].strip(), state=line[38:40].strip(),
                             name=line[41:71].strip(), flags=line[72:85].strip())
inv = {}
with open(os.path.join(ghcnd, "ghcnd-inventory.txt"), encoding="latin-1") as f:
    for line in f:
        sid, el = line[0:11], line[31:35]
        if el in ELEMS:
            inv.setdefault(sid, {})[el] = (int(line[36:40]), int(line[41:45]))

def show(sid):
    s = stations[sid]
    cov = " ".join(f"{e}:{inv.get(sid,{}).get(e,('-','-'))[0]}-{inv.get(sid,{}).get(e,('-','-'))[1]}" for e in ELEMS)
    print(f"{sid} {s['state']:2} {s['name'][:30]:30} {s['lat']:8.3f} {s['lon']:9.3f}  {cov}")

for q in queries:
    if re.fullmatch(r"[A-Z0-9]{11}", q):
        if q in stations:
            show(q)
        else:
            print(q, "NOT FOUND")
        continue
    prefix, pat = q.split(":", 1)
    rx = re.compile(pat, re.I)
    hits = [sid for sid, s in stations.items() if sid.startswith(prefix) and rx.search(s["name"])]
    # only show stations with TMAX coverage into 2025+
    hits = [h for h in hits if inv.get(h, {}).get("TMAX", (0, 0))[1] >= 2025 or inv.get(h, {}).get("PRCP", (0, 0))[1] >= 2025]
    print(f"--- {q}: {len(hits)} hits")
    for h in sorted(hits, key=lambda h: inv.get(h, {}).get("TMAX", (9999, 0))[0])[:12]:
        show(h)
