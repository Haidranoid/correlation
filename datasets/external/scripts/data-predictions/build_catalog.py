"""Assemble catalog_data-predictions.csv from meta/*.json and the written series files.

Re-checks every series (header, ascending unique dates, finite values, value range for
probabilities) and fills first/last/rows from the file itself.
"""
import csv
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXT, META, SERIES  # noqa: E402

COLS = ["id", "name", "category", "subcategory", "region", "source", "url", "frequency", "units",
        "transform", "known_after_days", "first", "last", "rows", "notes"]


def main():
    out = []
    problems = []
    for f in sorted(glob.glob(os.path.join(META, "*.json"))):
        m = json.load(open(f))
        sid = m["id"]
        path = os.path.join(SERIES, f"{sid}.csv")
        if not os.path.exists(path):
            problems.append(f"{sid}: series file missing")
            continue
        with open(path) as fh:
            r = csv.reader(fh)
            head = next(r)
            rows = [(a, float(b)) for a, b in r]
        if head != ["date", "value"]:
            problems.append(f"{sid}: bad header {head}")
        ds = [a for a, _ in rows]
        if ds != sorted(set(ds)):
            problems.append(f"{sid}: dates not strictly ascending/unique")
        if any(not math.isfinite(v) for _, v in rows):
            problems.append(f"{sid}: non-finite values")
        if m["units"].startswith("probability") and any(v < 0 or v > 1 for _, v in rows):
            problems.append(f"{sid}: probability outside [0,1]")
        rec = {k: m.get(k, "") for k in COLS}
        rec["first"], rec["last"], rec["rows"] = ds[0], ds[-1], len(rows)
        out.append(rec)
    dest = os.path.join(EXT, "catalog_data-predictions.csv")
    tmp = dest + ".part"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        for rec in sorted(out, key=lambda x: x["id"]):
            w.writerow(rec)
    os.replace(tmp, dest)
    print(f"wrote {dest}: {len(out)} series")
    for p in problems:
        print("PROBLEM", p)


if __name__ == "__main__":
    main()
