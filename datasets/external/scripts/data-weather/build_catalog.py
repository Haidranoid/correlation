"""Build catalog_data-weather.csv from catalog_entries.jsonl, re-validating every series file.

Usage: python3 -I build_catalog.py [--print-rows]
"""
import csv
import datetime as dt
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CATALOG_COLS, ENTRIES, EXT, SERIES_DIR, VALID_CATEGORIES, VALID_FREQ, VALID_TRANSFORM  # noqa: E402

OUT = os.path.join(EXT, "catalog_data-weather.csv")


def check_file(sid):
    path = os.path.join(SERIES_DIR, sid + ".csv")
    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0] == "date,value", (sid, lines[0])
    prev = None
    vals = []
    for ln in lines[1:]:
        d_s, v_s = ln.split(",")
        d = dt.date.fromisoformat(d_s)
        assert len(d_s) == 10, (sid, ln)
        v = float(v_s)
        assert not math.isnan(v) and not math.isinf(v), (sid, ln)
        assert v > -9990 or "-999" not in v_s, (sid, ln)
        if prev is not None:
            assert d > prev, (sid, "not strictly ascending / duplicate", prev, d)
        prev = d
        vals.append(v)
    first = lines[1].split(",")[0]
    last = lines[-1].split(",")[0]
    return first, last, len(lines) - 1, lines[1:4], lines[-3:], min(vals), max(vals)


def main():
    with open(ENTRIES) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate ids"
    entries.sort(key=lambda e: e["id"])
    rows = []
    for e in entries:
        assert e["category"] in VALID_CATEGORIES and e["frequency"] in VALID_FREQ and e["transform"] in VALID_TRANSFORM, e["id"]
        first, last, n, head, tail, vmin, vmax = check_file(e["id"])
        assert (first, last, n) == (e["first"], e["last"], e["rows"]), (e["id"], first, last, n, e["first"], e["last"], e["rows"])
        e = dict(e, first=first, last=last, rows=n)
        rows.append(e)
        if "--print-rows" in sys.argv:
            print(f"{e['id']:32} {e['frequency']:7} {e['transform']:13} k={e['known_after_days']:<3} {first}..{last} n={n:6} "
                  f"[{vmin:g},{vmax:g}] head={head[0]} tail={tail[-1]}")
    with open(OUT + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_COLS, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for e in rows:
            w.writerow({k: e.get(k, "") for k in CATALOG_COLS})
    os.replace(OUT + ".tmp", OUT)
    # summary
    from collections import Counter
    print(f"wrote {OUT}: {len(rows)} series")
    print("by category:", dict(Counter(e["category"] for e in rows)))
    print("by prefix:", dict(Counter(e["id"].split("_")[0] for e in rows)))
    print("by frequency:", dict(Counter(e["frequency"] for e in rows)))
    short = [e["id"] for e in rows if e["last"] < "2025-06-27" or e["first"] > "2005-01-03"]
    print("ending before 2025-06-27 or starting after 2005-01-03:", short)


if __name__ == "__main__":
    main()
