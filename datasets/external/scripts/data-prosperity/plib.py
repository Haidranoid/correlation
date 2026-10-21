"""Shared helpers for the data-prosperity agent: series writing, QC, catalog parts.

All scripts run with `python3 -I` from this directory's absolute path; they never execute
anything downloaded. Raw downloads live in ../../raw/<source>/, normalized series in ../../series/.
"""
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
import urllib.request

ROOT = os.path.expanduser("~/.local/share/correlation2/external")
RAW = os.path.join(ROOT, "raw")
SERIES = os.path.join(ROOT, "series")
HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(HERE, "parts")
QCDIR = os.path.join(HERE, "qc")
UA = f"correlation2-research/1.0 (personal research; contact {os.environ.get('C2_CONTACT', 'unset (set C2_CONTACT)')})"
CATALOG_COLS = ["id", "name", "category", "subcategory", "region", "source", "url", "frequency", "units",
                "transform", "known_after_days", "first", "last", "rows", "notes"]
STOCK_START, STOCK_END = "2001-12-31", "2025-06-27"

os.makedirs(SERIES, exist_ok=True)
os.makedirs(PARTS, exist_ok=True)
os.makedirs(QCDIR, exist_ok=True)


def fetch(url, dest, sleep=1.5, retries=3, headers=None, timeout=300):
    """Download url to dest (atomic), with a descriptive User-Agent. Returns bytes written."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            time.sleep(sleep)
            return len(data)
        except Exception as e:  # noqa: BLE001 - report and retry
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def month_end(y, m):
    if m == 12:
        return dt.date(y, 12, 31)
    return dt.date(y, m + 1, 1) - dt.timedelta(days=1)


def quarter_end(y, q):
    return month_end(y, 3 * q)


def fmt(v, nd=6):
    """Plain number text: no exponent, trailing zeros trimmed."""
    if v is None or not math.isfinite(v):
        raise ValueError(f"non-finite value {v}")
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def write_series(sid, rows, nd=6):
    """rows: iterable of (date: dt.date or 'YYYY-MM-DD', value). Sorted, de-duplicated (error on dups).
    Non-finite values are dropped. Returns (first, last, n)."""
    out = []
    for d, v in rows:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            continue
        ds = d.isoformat() if isinstance(d, dt.date) else str(d)
        out.append((ds, float(v)))
    out.sort()
    for a, b in zip(out, out[1:]):
        if a[0] == b[0]:
            raise ValueError(f"{sid}: duplicate date {a[0]}")
    if not out:
        raise ValueError(f"{sid}: no rows")
    path = os.path.join(SERIES, sid + ".csv")
    tmp = path + ".part"
    with open(tmp, "w", newline="") as f:
        f.write("date,value\n")
        for ds, v in out:
            f.write(f"{ds},{fmt(v, nd)}\n")
    os.replace(tmp, path)
    return out[0][0], out[-1][0], len(out)


def read_series(sid):
    with open(os.path.join(SERIES, sid + ".csv")) as f:
        next(f)
        return [(l.split(",")[0], float(l.split(",")[1])) for l in f if l.strip()]


def gap_report(dates, freq):
    """Missing periods between first and last for monthly/quarterly/annual series (dates = period ends)."""
    ds = [dt.date.fromisoformat(d) for d in dates]
    step = {"monthly": 1, "quarterly": 3, "annual": 12}.get(freq)
    if step is None:
        return []
    def idx(d):
        return d.year * 12 + d.month - 1
    have = {idx(d) for d in ds}
    lo, hi = idx(ds[0]), idx(ds[-1])
    missing = []
    for i in range(lo, hi + 1, step):
        if i not in have:
            missing.append(f"{i // 12}-{i % 12 + 1:02d}")
    return missing


class Part:
    """Catalog rows + QC lines for one source; written to parts/<name>.csv and qc/<name>.txt."""

    def __init__(self, name):
        self.name = name
        self.rows = []
        self.qc = []

    def add(self, sid, rows, *, name, subcategory, region, source, url, frequency, units, transform,
            known_after_days, notes, lo=None, hi=None, nd=6, category="prosperity"):
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in sid), sid
        assert frequency in ("daily", "weekly", "monthly", "quarterly", "annual"), frequency
        assert transform in ("logdiff", "diff", "anomaly", "anomaly_log1p", "level"), transform
        rows = [(d, v) for d, v in rows if v is not None and math.isfinite(v)]
        if transform == "logdiff":
            bad = [r for r in rows if r[1] <= 0]
            if bad:
                raise ValueError(f"{sid}: logdiff with non-positive values {bad[:3]}")
        first, last, n = write_series(sid, rows, nd)
        vals = [v for _, v in sorted((d.isoformat() if isinstance(d, dt.date) else d, v) for d, v in rows)]
        dates = sorted(d.isoformat() if isinstance(d, dt.date) else d for d, _ in rows)
        vmin, vmax = min(vals), max(vals)
        flags = []
        if lo is not None and vmin < lo:
            flags.append(f"BELOW {lo}: {vmin}")
        if hi is not None and vmax > hi:
            flags.append(f"ABOVE {hi}: {vmax}")
        if any(abs(v + 9999) < 1e-9 or abs(v + 999) < 1e-9 for v in vals):
            flags.append("SENTINEL")
        gaps = gap_report(dates, frequency)
        srt = sorted(zip(dates, vals))
        head = " ".join(f"{d}={fmt(v, 4)}" for d, v in srt[:3])
        tail = " ".join(f"{d}={fmt(v, 4)}" for d, v in srt[-3:])
        self.qc.append(f"{sid}\t{frequency}\t{first}..{last}\tn={n}\tmin={fmt(vmin, 4)}\tmax={fmt(vmax, 4)}"
                       f"\tgaps={len(gaps)}{(' ' + ','.join(gaps[:8]) + ('...' if len(gaps) > 8 else '')) if gaps else ''}"
                       f"\t{' '.join(flags) if flags else 'ok'}\thead: {head}\ttail: {tail}")
        if flags:
            print("QC FLAG", sid, flags, file=sys.stderr)
        self.rows.append({"id": sid, "name": name, "category": category, "subcategory": subcategory,
                          "region": region, "source": source, "url": url, "frequency": frequency,
                          "units": units, "transform": transform, "known_after_days": str(int(known_after_days)),
                          "first": first, "last": last, "rows": str(n), "notes": notes})

    def save(self):
        path = os.path.join(PARTS, self.name + ".csv")
        with open(path + ".part", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CATALOG_COLS, quoting=csv.QUOTE_MINIMAL)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        os.replace(path + ".part", path)
        with open(os.path.join(QCDIR, self.name + ".txt"), "w") as f:
            f.write("\n".join(self.qc) + "\n")
        print(f"part {self.name}: {len(self.rows)} series")


def build_catalog():
    """Merge parts/*.csv (in name order) into catalog_data-prosperity.csv and qc_data-prosperity.txt."""
    rows, qc = [], []
    seen = set()
    for fn in sorted(os.listdir(PARTS)):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(PARTS, fn), newline="") as f:
            for r in csv.DictReader(f):
                if r["id"] in seen:
                    raise ValueError(f"duplicate id {r['id']} in {fn}")
                seen.add(r["id"])
                rows.append(r)
        q = os.path.join(QCDIR, fn[:-4] + ".txt")
        if os.path.exists(q):
            qc.append(f"## {fn[:-4]}\n" + open(q).read())
    path = os.path.join(ROOT, "catalog_data-prosperity.csv")
    with open(path + ".part", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_COLS, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(path + ".part", path)
    with open(os.path.join(ROOT, "qc_data-prosperity.txt"), "w") as f:
        f.write("# QC per series: id, frequency, span, rows, min, max, missing periods, flags, first/last rows\n")
        f.write("\n".join(qc))
    print(f"catalog: {len(rows)} series -> {path}")
    return rows


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "build":
    build_catalog()
