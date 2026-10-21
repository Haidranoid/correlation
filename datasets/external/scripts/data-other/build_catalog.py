"""Assemble catalog_data-other.csv from meta/*.json + the series files, and run final QA."""
import csv, datetime as dt, glob, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXT, SERIES, META, SPACING

COLS = ["id", "name", "category", "subcategory", "region", "source", "url", "frequency", "units",
        "transform", "known_after_days", "first", "last", "rows", "notes"]
OUT = os.path.join(EXT, "catalog_data-other.csv")

metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(META, "*.json")))]
ids = [m["id"] for m in metas]
assert len(ids) == len(set(ids))
# collisions with other agents' catalogs / series files they own
others = {}
for p in glob.glob(os.path.join(EXT, "catalog_*.csv")):
    if os.path.basename(p) == os.path.basename(OUT):
        continue
    for r in csv.DictReader(open(p)):
        others[r["id"]] = os.path.basename(p)
clash = sorted(set(ids) & set(others))
assert not clash, f"id clash with other catalogs: {[(c, others[c]) for c in clash]}"

rows_out, qa = [], []
for m in sorted(metas, key=lambda m: (m["category"], m["id"])):
    path = os.path.join(SERIES, f"{m['id']}.csv")
    with open(path) as f:
        r = csv.reader(f)
        assert next(r) == ["date", "value"]
        data = [(dt.date.fromisoformat(a), float(b)) for a, b in r]
    dates = [d for d, _ in data]
    vals = [v for _, v in data]
    assert all(dates[i] < dates[i + 1] for i in range(len(dates) - 1)), m["id"]
    assert all(math.isfinite(v) for v in vals), m["id"]
    if m["transform"] == "logdiff":
        assert min(vals) > 0, m["id"]
    sp = SPACING[m["frequency"]]
    lim = sp * 1.5 + (2 if m["frequency"] in ("monthly", "quarterly") else 0)
    gaps2000 = [(a, b, (b - a).days) for a, b in zip(dates, dates[1:])
                if (b - a).days > lim and b >= dt.date(2000, 1, 1)]
    in_window = sum(1 for d in dates if dt.date(2005, 1, 3) <= d <= dt.date(2025, 6, 27))
    row = dict(m, first=dates[0].isoformat(), last=dates[-1].isoformat(), rows=len(data))
    rows_out.append(row)
    qa.append((m["id"], m["category"], m["frequency"], m["transform"], m["known_after_days"],
               dates[0], dates[-1], len(data), in_window, len(gaps2000),
               max((g for g in gaps2000), key=lambda x: x[2]) if gaps2000 else None, min(vals), max(vals)))
with open(OUT + ".part", "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(COLS)
    for row in rows_out:
        w.writerow([row[c] for c in COLS])
os.replace(OUT + ".part", OUT)
# orphan check: series files I wrote that are not in the catalog (by my id prefixes)
mine = set(ids)
prefixes = ("gfz_", "silso_", "usgs_", "wiki_", "delphi_", "nyt_", "astro_", "census_", "wb_", "fred_pop",
            "fred_cnp", "emv_", "gpr_", "eia_gasoline", "eia930_", "tsa_", "sffed_")
orph = [os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SERIES, "*.csv"))
        if os.path.basename(p).startswith(prefixes) and os.path.basename(p)[:-4] not in mine]
print(f"catalog: {OUT}  ({len(rows_out)} series); orphans with my prefixes: {orph}")
print(f"{'id':30s} {'cat':13s} {'freq':7s} {'trans':8s} kad first      last       rows   in_census_window gaps2000 largest_gap_since_2000")
for q in qa:
    sid, cat, fr, tr, kad, a, b, n, nw, ng, big, lo, hi = q
    print(f"{sid:30s} {cat:13s} {fr:7s} {tr:8s} {kad:3d} {a} {b} {n:6d} {nw:6d} {ng:4d} "
          f"{'' if not big else f'{big[0]}->{big[1]} ({big[2]}d)'}")
from collections import Counter
print("by category:", dict(Counter(m["category"] for m in metas)))
print("by frequency:", dict(Counter(m["frequency"] for m in metas)))
