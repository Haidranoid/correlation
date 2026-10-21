"""Write catalog_data-news.csv from meta/<id>.json plus the normalized series files."""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXT, META, SERIES, ID_PREFIXES

COLS = ["id", "name", "category", "subcategory", "region", "source", "url", "frequency", "units", "transform",
        "known_after_days", "first", "last", "rows", "notes"]


def main():
    out = []
    for fp in sorted(glob.glob(os.path.join(META, "*.json"))):
        m = json.load(open(fp))
        assert m["id"].startswith(ID_PREFIXES), m["id"]
        with open(os.path.join(SERIES, m["id"] + ".csv")) as f:
            rows = list(csv.reader(f))[1:]
        m["first"], m["last"], m["rows"] = rows[0][0], rows[-1][0], len(rows)
        out.append(m)
    dest = os.path.join(EXT, "catalog_data-news.csv")
    with open(dest + ".part", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for m in sorted(out, key=lambda x: x["id"]):
            w.writerow(m)
    os.replace(dest + ".part", dest)
    print(f"wrote {dest}: {len(out)} series")


if __name__ == "__main__":
    main()
