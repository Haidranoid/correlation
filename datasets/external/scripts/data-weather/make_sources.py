"""Write SOURCES_data-weather.md (prose + generated tables).

Usage: python3 -I make_sources.py
"""
import collections
import io
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENTRIES, EXT, QA_DIR, SCR  # noqa: E402
from stations import STATIONS  # noqa: E402

OUT = os.path.join(EXT, "SOURCES_data-weather.md")


def entries():
    with open(ENTRIES) as f:
        return sorted((json.loads(l) for l in f if l.strip()), key=lambda e: e["id"])


def qa(sid):
    return json.load(open(os.path.join(QA_DIR, sid + ".json")))


def station_table(es):
    ids = {e["id"] for e in es}
    meta = {}
    with open(os.path.join(EXT, "raw", "ghcnd", "ghcnd-stations.txt"), encoding="latin-1") as f:
        for line in f:
            meta[line[0:11]] = (float(line[12:20]), float(line[21:30]))
    out = io.StringIO()
    out.write("| key | GHCN id | station | role | lat, lon | series | TMAX first..last | TMAX missing days since 2000 |\n")
    out.write("|---|---|---|---|---|---|---|---|\n")
    for key, sid, disp, role, *_ in STATIONS:
        lat, lon = meta[sid]
        if sid == "UKE00107650":
            lon = -abs(lon)
        vars_ = [v for v in ("tmax", "tmin", "prcp", "snow", "awnd", "hdd", "cdd") if f"wx_{key}_{v}" in ids]
        q = qa(f"wx_{key}_tmax")
        # missing since 2000
        path = os.path.join(EXT, "series", f"wx_{key}_tmax.csv")
        with open(path) as f:
            next(f)
            n2000 = sum(1 for ln in f if ln[:4] >= "2000" and ln[:10] <= q["last"])
        import datetime as dt
        span = (dt.date.fromisoformat(q["last"]) - dt.date(2000, 1, 1)).days + 1
        out.write(f"| {key} | {sid} | {disp} | {role} | {lat:.3f}, {lon:.3f} | {', '.join(vars_)} | "
                  f"{q['first']}..{q['last']} | {span - n2000} |\n")
    return out.getvalue()


def timing_table(es):
    groups = collections.OrderedDict()
    for e in es:
        p = e["id"].split("_")[0]
        key = (p if p != "cpc" else "cpc_" + e["id"].split("_")[1], e["frequency"], e["known_after_days"])
        groups.setdefault(key, []).append(e["id"])
    out = io.StringIO()
    out.write("| group | frequency | known_after_days | n | ids (examples) |\n|---|---|---|---|---|\n")
    for (g, fr, k), ids in groups.items():
        out.write(f"| {g} | {fr} | {k} | {len(ids)} | {', '.join(ids[:3])}{', ...' if len(ids) > 3 else ''} |\n")
    return out.getvalue()


def counts(es):
    out = io.StringIO()
    by = collections.Counter((e["category"], e["subcategory"]) for e in es)
    out.write("| category | subcategory | series |\n|---|---|---|\n")
    for (c, s), n in sorted(by.items()):
        out.write(f"| {c} | {s} | {n} |\n")
    out.write(f"| **total** | | **{len(es)}** |\n")
    return out.getvalue()


def incomplete_years(es):
    """Daily series with a full calendar year since 2000 that is under 95% complete."""
    import calendar
    out = io.StringIO()
    out.write("| series | incomplete years since 2000 (days present / days in year) |\n|---|---|\n")
    n_rows = 0
    for e in es:
        if e["frequency"] != "daily":
            continue
        cnt = collections.Counter()
        last = None
        with open(os.path.join(EXT, "series", e["id"] + ".csv")) as f:
            next(f)
            for ln in f:
                cnt[int(ln[:4])] += 1
                last = ln[:10]
        bad = []
        for y in range(max(2000, int(e["first"][:4])), int(last[:4])):
            n = 366 if calendar.isleap(y) else 365
            if cnt[y] < 0.95 * n:
                bad.append(f"{y}: {cnt[y]}/{n}")
        if bad:
            out.write(f"| {e['id']} | {', '.join(bad)} |\n")
            n_rows += 1
    if not n_rows:
        out.write("| (none) | |\n")
    return out.getvalue()


def main():
    es = entries()
    cross = subprocess.run([sys.executable, "-I", os.path.join(SCR, "crosscheck.py")], capture_output=True,
                           text=True).stdout
    qa_md = subprocess.run([sys.executable, "-I", os.path.join(SCR, "qa_table.py")], capture_output=True,
                           text=True).stdout
    om_log = open(os.path.join(SCR, "openmeteo_download.log")).read() if os.path.exists(
        os.path.join(SCR, "openmeteo_download.log")) else "(no log)"
    om_ids = [e["id"] for e in es if e["id"].startswith("om_")]
    om_points = sorted({"_".join(i.split("_")[1:-1]) for i in om_ids})
    text = TEMPLATE.format(
        n=len(es), counts=counts(es), stations=station_table(es), timing=timing_table(es), cross=cross.strip(),
        incomplete=incomplete_years(es),
        qa=qa_md.strip(), om_log=om_log.strip(), om_points=", ".join(om_points), n_om=len(om_ids),
        om_used=f"{__import__('dl_openmeteo').used_so_far():,.0f}")
    with open(OUT + ".tmp", "w") as f:
        f.write(text)
    os.replace(OUT + ".tmp", OUT)
    print("wrote", OUT, len(text), "chars")


TEMPLATE = open(os.path.join(SCR, "sources_template.md")).read()

if __name__ == "__main__":
    main()
