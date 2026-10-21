"""Shared helpers for the data-markets agent: normalize, validate, write series, keep catalog parts.

Every series goes through write_series(), which:
  - sorts by date, rejects conflicting duplicate dates, drops exact duplicates,
  - drops non-finite values, and (for logdiff) non-positive values, recording what was dropped,
  - writes series/<id>.csv with header date,value,
  - runs the QC checks (range, duplicates, gaps, jumps, stale runs, head/tail) and writes qc/<id>.txt,
  - returns the catalog row with first/last/rows taken from the written file.
"""
import csv
import datetime as dt
import json
import math
import os
import re
import sys

import numpy as np

HOME = os.path.expanduser("~")
EXT = os.path.join(HOME, ".local/share/correlation2/external")
SERIES = os.path.join(EXT, "series")
RAW = os.path.join(EXT, "raw")
WORK = "/tmp/claude-1000/-home-bhoult-programming/0233028d-5555-4d00-84a1-6db185f7ac67/scratchpad/dm"
PARTS = os.path.join(WORK, "parts")
QCDIR = os.path.join(WORK, "qc")
CATALOG = os.path.join(EXT, "catalog_data-markets.csv")

CAT_COLS = ["id", "name", "category", "subcategory", "region", "source", "url", "frequency", "units",
            "transform", "known_after_days", "first", "last", "rows", "notes"]
CATEGORIES = {"weather", "climate_index", "energy", "commodity", "rates", "credit", "fx", "world_equity",
              "crypto", "volatility", "macro", "population", "space_weather", "geophysical", "astronomical",
              "attention", "health", "other"}
TRANSFORMS = {"logdiff", "diff", "anomaly", "anomaly_log1p", "level"}
FREQS = {"daily", "weekly", "monthly", "quarterly", "annual"}
GAP_DAYS = {"daily": 5, "weekly": 9, "monthly": 40, "quarterly": 100, "annual": 380}

for d in (SERIES, RAW, PARTS, QCDIR):
    os.makedirs(d, exist_ok=True)


def fmt(x):
    """Plain decimal text (no exponent), shortest round-trip for the value's own precision."""
    s = np.format_float_positional(x, trim="-", unique=True)
    if s.endswith("."):
        s = s[:-1]
    if s == "-0":
        s = "0"
    return s


def ymd(s):
    """'YYYYMMDD' or 'YYYY-MM-DD' -> 'YYYY-MM-DD' (validated)."""
    s = s.strip()
    if len(s) == 8 and s.isdigit():
        s = f"{s[:4]}-{s[4:6]}-{s[6:]}"
    dt.date.fromisoformat(s)
    return s


def month_end(s):
    """'YYYY-MM-DD' -> last calendar day of that month."""
    d = dt.date.fromisoformat(s)
    nxt = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return (nxt - dt.timedelta(days=1)).isoformat()


def _date(s):
    return dt.date.fromisoformat(s)


def qc(sid, dates, vals, freq, transform, jump=None, extra_lines=()):
    """Return (summary dict, list of text lines)."""
    n = len(vals)
    v = np.asarray(vals, dtype=np.float64)
    lines = [f"== {sid}  n={n}  {dates[0]}..{dates[-1]}  freq={freq} transform={transform}"]
    lines += list(extra_lines)
    summ = {"id": sid, "n": n, "first": dates[0], "last": dates[-1], "min": float(v.min()), "max": float(v.max())}
    lines.append(f"range: min={fmt(v.min())} max={fmt(v.max())} median={fmt(float(np.median(v)))}")
    sentinel = [d for d, x in zip(dates, vals) if x in (-9999, -999, 9999, 99999, -99999, -99)]
    if sentinel:
        lines.append(f"WARNING possible sentinels at {sentinel[:10]}")
    summ["sentinels"] = len(sentinel)
    # gaps
    ds = [_date(d) for d in dates]
    gaps = [((ds[i] - ds[i - 1]).days, dates[i - 1], dates[i]) for i in range(1, n)]
    thr = GAP_DAYS[freq]
    if freq == "daily" and n > 30:
        # 7-day series (crypto, some FRED) get a tighter threshold
        weekend = sum(1 for d in ds if d.weekday() >= 5) / n
        if weekend > 0.25:
            thr = 2
    big = sorted([g for g in gaps if g[0] > thr], reverse=True)
    summ["gaps"] = len(big)
    summ["max_gap"] = big[0][0] if big else (max(g[0] for g in gaps) if gaps else 0)
    lines.append(f"gaps > {thr} calendar days: {len(big)}" + (
        "; largest: " + ", ".join(f"{g[0]}d {g[1]}->{g[2]}" for g in big[:6]) if big else ""))
    # changes
    if n > 2:
        if transform == "logdiff":
            ch = np.diff(np.log(v))
        else:
            ch = np.diff(v)
        zero = float(np.mean(ch == 0))
        summ["zero_change_frac"] = zero
        # longest run of identical consecutive values
        best, cur, start, bstart = 0, 0, 0, 0
        for i in range(1, n):
            if vals[i] == vals[i - 1]:
                if cur == 0:
                    start = i - 1
                cur += 1
                if cur > best:
                    best, bstart = cur, start
            else:
                cur = 0
        summ["longest_flat"] = best
        lines.append(f"zero-change fraction={zero:.3f}; longest flat run={best} obs" +
                     (f" from {dates[bstart]} to {dates[bstart + best]}" if best else ""))
        if transform == "logdiff":
            thrj = jump if jump is not None else 0.25
            idx = np.argsort(-np.abs(ch))[:5]
            nj = int(np.sum(np.abs(ch) > thrj))
            lines.append(f"log-moves |d|>{thrj}: {nj}; top: " + ", ".join(
                f"{dates[i + 1]} {ch[i]:+.3f} ({fmt(v[i])}->{fmt(v[i + 1])})" for i in idx))
        else:
            mad = float(np.median(np.abs(ch - np.median(ch)))) * 1.4826
            sd = float(np.std(ch))
            scale = mad if mad > 0 else sd
            thrj = jump if jump is not None else 10 * scale
            idx = np.argsort(-np.abs(ch))[:5]
            nj = int(np.sum(np.abs(ch) > thrj)) if thrj > 0 else 0
            lines.append(f"moves: sd={sd:.4g} robust_sd={mad:.4g}; |d|>{thrj:.4g}: {nj}; top: " + ", ".join(
                f"{dates[i + 1]} {ch[i]:+.4g} ({fmt(v[i])}->{fmt(v[i + 1])})" for i in idx))
        summ["jumps"] = nj
    head = ", ".join(f"{d}={fmt(x)}" for d, x in zip(dates[:3], vals[:3]))
    tail = ", ".join(f"{d}={fmt(x)}" for d, x in zip(dates[-3:], vals[-3:]))
    lines.append(f"head: {head}")
    lines.append(f"tail: {tail}")
    return summ, lines


def write_series(meta, rows, jump=None, drop_nonpositive=None, min_rows=50):
    """meta: dict with catalog fields except first/last/rows. rows: iterable of (date 'YYYY-MM-DD', float).
    Returns (catalog row dict, qc summary dict, qc lines)."""
    sid = meta["id"]
    assert re.fullmatch(r"[a-z0-9_]+", sid), sid
    assert meta["category"] in CATEGORIES, meta
    assert meta["transform"] in TRANSFORMS, meta
    assert meta["frequency"] in FREQS, meta
    notes = [meta.get("notes", "")] if meta.get("notes") else []
    extra = []
    byd = {}
    conflicts = []
    exact_dups = 0
    nonfinite = []
    for d, x in rows:
        d = ymd(d)
        x = float(x)
        if not math.isfinite(x):
            nonfinite.append(d)
            continue
        if d in byd:
            if byd[d] == x:
                exact_dups += 1
                continue
            conflicts.append((d, byd[d], x))
            continue
        byd[d] = x
    if conflicts:
        raise ValueError(f"{sid}: conflicting duplicate dates {conflicts[:5]}")
    if exact_dups:
        extra.append(f"dropped {exact_dups} exact duplicate rows")
    if nonfinite:
        extra.append(f"dropped {len(nonfinite)} non-finite values: {nonfinite[:5]}")
    dates = sorted(byd)
    vals = [byd[d] for d in dates]
    if drop_nonpositive is None:
        drop_nonpositive = meta["transform"] == "logdiff"
    if drop_nonpositive:
        bad = [(d, x) for d, x in zip(dates, vals) if x <= 0]
        if bad:
            keep = [(d, x) for d, x in zip(dates, vals) if x > 0]
            dates = [d for d, _ in keep]
            vals = [x for _, x in keep]
            msg = f"dropped {len(bad)} non-positive value(s) for logdiff: " + ", ".join(
                f"{d}={fmt(x)}" for d, x in bad[:4]) + ("..." if len(bad) > 4 else "")
            extra.append(msg)
            notes.append(msg)
    if len(dates) < min_rows:
        raise ValueError(f"{sid}: only {len(dates)} rows")
    path = os.path.join(SERIES, sid + ".csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.write("date,value\n")
        for d, x in zip(dates, vals):
            f.write(f"{d},{fmt(x)}\n")
    os.replace(tmp, path)
    # re-read what was written, so first/last/rows come from the file itself
    with open(path) as f:
        r = list(csv.reader(f))
    assert r[0] == ["date", "value"]
    wd = [x[0] for x in r[1:]]
    wv = [float(x[1]) for x in r[1:]]
    assert wd == sorted(set(wd)), sid
    summ, lines = qc(sid, wd, wv, meta["frequency"], meta["transform"], jump=jump, extra_lines=extra)
    with open(os.path.join(QCDIR, sid + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    row = {k: meta.get(k, "") for k in CAT_COLS}
    row["first"], row["last"], row["rows"] = wd[0], wd[-1], len(wd)
    row["notes"] = "; ".join(n for n in notes if n)
    return row, summ, lines


class Part:
    """Collects catalog rows for one source block and saves them as parts/<name>.jsonl."""

    def __init__(self, name):
        self.name = name
        self.rows = []
        self.summ = []
        self.log = []

    def add(self, meta, rows, **kw):
        try:
            row, summ, lines = write_series(meta, rows, **kw)
        except Exception as e:  # noqa: BLE001 - report and continue with the other series
            msg = f"!! {meta.get('id')}: FAILED {e}"
            print(msg, file=sys.stderr)
            self.log.append(msg)
            return None
        self.rows.append(row)
        self.summ.append(summ)
        self.log += lines
        print("\n".join(lines))
        return row

    def save(self):
        with open(os.path.join(PARTS, self.name + ".jsonl"), "w") as f:
            for r in self.rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(PARTS, self.name + ".qc.txt"), "w") as f:
            f.write("\n".join(self.log) + "\n")
        with open(os.path.join(PARTS, self.name + ".summ.jsonl"), "w") as f:
            for s in self.summ:
                f.write(json.dumps(s) + "\n")
        merge_catalog()


def merge_catalog():
    rows = []
    for fn in sorted(os.listdir(PARTS)):
        if fn.endswith(".jsonl") and not fn.endswith(".summ.jsonl"):
            with open(os.path.join(PARTS, fn)) as f:
                rows += [json.loads(l) for l in f if l.strip()]
    ids = [r["id"] for r in rows]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate ids across parts: {sorted(dup)}")
    order = ["fx", "world_equity", "rates", "credit", "volatility", "energy", "commodity", "crypto", "macro",
             "attention", "other"]
    rows.sort(key=lambda r: (order.index(r["category"]) if r["category"] in order else 99, r["id"]))
    tmp = CATALOG + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAT_COLS, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, CATALOG)
    return len(rows)
