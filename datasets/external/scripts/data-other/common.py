"""Shared helpers for the data-other download agent.

Every normalized series goes through write_series(), which sorts, checks for
duplicate dates and non-finite values, writes series/<id>.csv, prints a sanity
report, and records the metadata used later to build the catalog.
Run all scripts with `python3 -I`; scripts add their own directory to sys.path.
"""
import csv
import datetime as dt
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

EXT = os.path.expanduser("~/.local/share/correlation2/external")
RAW = os.path.join(EXT, "raw")
SERIES = os.path.join(EXT, "series")
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "meta")
UA = f"correlation2-research/1.0 (personal research; contact {os.environ.get('C2_CONTACT', 'unset (set C2_CONTACT)')})"

CATEGORIES = {"weather", "climate_index", "energy", "commodity", "rates", "credit", "fx",
              "world_equity", "crypto", "volatility", "macro", "population", "space_weather",
              "geophysical", "astronomical", "attention", "health", "other"}
FREQS = {"daily", "weekly", "monthly", "quarterly", "annual"}
TRANSFORMS = {"logdiff", "diff", "anomaly", "anomaly_log1p", "level"}
# expected spacing in days, used for the gap report
SPACING = {"daily": 1, "weekly": 7, "monthly": 31, "quarterly": 92, "annual": 366}


def fetch(url, dest, sleep=1.0, retries=3, timeout=120, headers=None, overwrite=False):
    """Download url to dest (bytes, unmodified). Polite: UA, sleep after each request."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not overwrite:
        return dest
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
            return dest
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404):
                raise
            time.sleep(sleep * (4 ** (attempt + 1)))
        except Exception as e:  # network hiccup
            last = e
            time.sleep(sleep * (4 ** (attempt + 1)))
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {last}")


def parse_date(s):
    return dt.date.fromisoformat(s)


def write_series(sid, rows, *, name, category, subcategory, region, source, url, frequency,
                 units, transform, known_after_days, notes, value_range=None,
                 allow_nonpositive=None, quiet=False):
    """rows: iterable of (date|str, float). Writes series/<sid>.csv and meta/<sid>.json."""
    import re
    assert re.fullmatch(r"[a-z0-9_]+", sid), sid
    assert category in CATEGORIES, category
    assert frequency in FREQS, frequency
    assert transform in TRANSFORMS, transform
    assert isinstance(known_after_days, int) and known_after_days >= 0
    clean = []
    for d, v in rows:
        if isinstance(d, str):
            d = parse_date(d)
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"{sid}: non-finite value on {d}")
        clean.append((d, v))
    clean.sort(key=lambda r: r[0])
    dates = [d for d, _ in clean]
    dups = sorted({d for i, d in enumerate(dates) if i and dates[i - 1] == d})
    if dups:
        raise ValueError(f"{sid}: duplicate dates {dups[:5]} ({len(dups)} total)")
    if not clean:
        raise ValueError(f"{sid}: no rows")
    vals = [v for _, v in clean]
    vmin, vmax = min(vals), max(vals)
    if value_range is not None:
        lo, hi = value_range
        bad = [(d, v) for d, v in clean if v < lo or v > hi]
        if bad:
            raise ValueError(f"{sid}: {len(bad)} values outside [{lo},{hi}], e.g. {bad[:5]}")
    if transform == "logdiff":
        nonpos = [(d, v) for d, v in clean if v <= 0]
        if nonpos:
            raise ValueError(f"{sid}: logdiff but {len(nonpos)} non-positive values, e.g. {nonpos[:5]}")
    for sentinel in (-9999, -999, -99, 9999, 99999):
        if any(v == sentinel for v in vals):
            print(f"  WARNING {sid}: value {sentinel} present (sentinel?)")
    os.makedirs(SERIES, exist_ok=True)
    path = os.path.join(SERIES, f"{sid}.csv")
    tmp = path + ".part"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "value"])
        for d, v in clean:
            w.writerow([d.isoformat(), fmt(v)])
    os.replace(tmp, path)
    meta = dict(id=sid, name=name, category=category, subcategory=subcategory, region=region,
                source=source, url=url, frequency=frequency, units=units, transform=transform,
                known_after_days=known_after_days, notes=notes)
    os.makedirs(META, exist_ok=True)
    with open(os.path.join(META, f"{sid}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    rep = check_series(sid, frequency, quiet=quiet)
    return rep


def fmt(v):
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    s = f"{v:.10g}"
    return s


def check_series(sid, frequency, quiet=False):
    """Re-read the written file and report range, gaps, first/last rows."""
    path = os.path.join(SERIES, f"{sid}.csv")
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        assert head == ["date", "value"], head
        rows = [(parse_date(a), float(b)) for a, b in r]
    dates = [d for d, _ in rows]
    assert all(dates[i] < dates[i + 1] for i in range(len(dates) - 1)), f"{sid}: not strictly ascending"
    vals = [v for _, v in rows]
    sp = SPACING[frequency]
    gaps = []
    for i in range(1, len(dates)):
        g = (dates[i] - dates[i - 1]).days
        if g > sp * 1.5 + (2 if frequency in ("monthly", "quarterly") else 0):
            gaps.append((dates[i - 1], dates[i], g))
    rep = dict(id=sid, first=dates[0].isoformat(), last=dates[-1].isoformat(), rows=len(rows),
               min=min(vals), max=max(vals), mean=sum(vals) / len(vals), n_gaps=len(gaps),
               largest_gap=max(gaps, key=lambda x: x[2]) if gaps else None,
               zeros=sum(1 for v in vals if v == 0))
    if not quiet:
        print(f"== {sid}: {rep['first']} .. {rep['last']}  rows={rep['rows']}  "
              f"min={rep['min']:.6g} max={rep['max']:.6g} mean={rep['mean']:.6g} zeros={rep['zeros']}")
        if gaps:
            big = sorted(gaps, key=lambda x: -x[2])[:5]
            print(f"   gaps>{sp}d-spacing: {len(gaps)}; largest: " +
                  "; ".join(f"{a}->{b} ({g}d)" for a, b, g in big))
        else:
            print("   no gaps beyond expected spacing")
        print("   head: " + "  ".join(f"{d} {fmt(v)}" for d, v in rows[:3]))
        print("   tail: " + "  ".join(f"{d} {fmt(v)}" for d, v in rows[-3:]))
    return rep


def daterange(a, b):
    d = a
    one = dt.timedelta(days=1)
    while d <= b:
        yield d
        d += one
