"""Coverage and seasonality report for every data-news series (for SOURCES_data-news.md).
Per series: first/last/rows, days missing inside the span, per-year coverage, and two
seasonality measures on daily series after removing a centred 365-day mean (trend):
  doy_r2  share of variance explained by month-of-year means (annual cycle)
  dow_r2  share of variance explained by day-of-week means
Usage: python3 -I coverage.py [prefix ...]  (prints a markdown table)
"""
import csv
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import META, SERIES


def load(sid):
    with open(os.path.join(SERIES, sid + ".csv")) as f:
        r = csv.reader(f)
        next(r)
        return [(dt.date.fromisoformat(a), float(b)) for a, b in r]


def seasonality(rows):
    d0 = rows[0][0]
    n = (rows[-1][0] - d0).days + 1
    x = np.full(n, np.nan)
    for d, v in rows:
        x[(d - d0).days] = v
    ok = np.isfinite(x)
    # centred 365-day mean via cumulative sums over finite values
    xv = np.where(ok, x, 0.0)
    c = np.concatenate([[0], np.cumsum(xv)])
    k = np.concatenate([[0], np.cumsum(ok)])
    idx = np.arange(n)
    lo, hi = np.clip(idx - 182, 0, n), np.clip(idx + 183, 0, n)
    cnt = k[hi] - k[lo]
    trend = np.where(cnt > 100, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)
    y = x - trend
    good = np.isfinite(y)
    if good.sum() < 730:
        return None, None
    y = y[good]
    days = [d0 + dt.timedelta(int(i)) for i in idx[good]]
    tot = np.var(y)
    if tot == 0:
        return 0.0, 0.0
    out = []
    for key in (lambda d: d.month, lambda d: d.weekday()):
        g = np.array([key(d) for d in days])
        means = {v: y[g == v].mean() for v in set(g.tolist())}
        fit = np.array([means[v] for v in g])
        out.append(float(np.var(fit) / tot))
    return out[0], out[1]


def main(prefixes):
    metas = [json.load(open(fp)) for fp in sorted(glob.glob(os.path.join(META, "*.json")))]
    if prefixes:
        metas = [m for m in metas if m["id"].startswith(tuple(prefixes))]
    print("| id | first | last | rows | missing days | years >= 90% covered | doy_r2 | dow_r2 |")
    print("|---|---|---|---|---|---|---|---|")
    for m in metas:
        rows = load(m["id"])
        first, last = rows[0][0], rows[-1][0]
        if m["frequency"] == "daily":
            span = (last - first).days + 1
            miss = span - len(rows)
            years = {}
            for d, _ in rows:
                years[d.year] = years.get(d.year, 0) + 1
            full = [y for y in range(first.year, last.year + 1)
                    if years.get(y, 0) >= 0.9 * ((min(last, dt.date(y, 12, 31)) - max(first, dt.date(y, 1, 1))).days + 1)]
            yrs = f"{len(full)}/{last.year - first.year + 1}"
            doy, dow = seasonality(rows)
        else:
            miss, yrs, doy, dow = "", "", None, None
        f = lambda v: "" if v is None else f"{v:.3f}"
        print(f"| {m['id']} | {first} | {last} | {len(rows)} | {miss} | {yrs} | {f(doy)} | {f(dow)} |")


def years_matrix(ids, y0=2000, y1=2026):
    """Per-year share of calendar days with a value (weekly series: share of weeks)."""
    print("| series | " + " | ".join(str(y)[2:] for y in range(y0, y1 + 1)) + " |")
    print("|---|" + "---|" * (y1 - y0 + 1))
    for sid in ids:
        rows = load(sid)
        weekly = len(rows) > 2 and (rows[1][0] - rows[0][0]).days == 7
        cnt = {}
        for d, _ in rows:
            cnt[d.year] = cnt.get(d.year, 0) + 1
        cells = []
        for y in range(y0, y1 + 1):
            n = cnt.get(y, 0)
            denom = 52.2 if weekly else (366 if y % 4 == 0 else 365)
            cells.append("" if n == 0 else f"{min(1.0, n / denom):.2f}".lstrip("0"))
        print(f"| {sid} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--years"]:
        years_matrix(sys.argv[2:])
    else:
        main(sys.argv[1:])
