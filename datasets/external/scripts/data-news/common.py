"""Shared helpers for the data-news download agent (news headline sentiment).

Run every script with `python3 -I` (or the venv python with -I); scripts add their own
directory to sys.path. Downloads go to raw/<source>/ unmodified (some gzip-compressed,
which is lossless). write_series() sorts, checks for duplicate dates and non-finite
values, writes series/<id>.csv, prints a sanity report and records catalog metadata in
meta/<id>.json.
"""
import csv
import datetime as dt
import gzip
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

EXT = os.path.expanduser("~/.local/share/correlation2/external")
RAW = os.path.join(EXT, "raw")
SERIES = os.environ.get("NEWS_SERIES_DIR") or os.path.join(EXT, "series")  # override for dry runs
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.environ.get("NEWS_META_DIR") or os.path.join(HERE, "meta")
WORK = os.environ.get("NEWS_WORK_DIR") or os.path.join(HERE, "work")
LOGS = os.path.join(HERE, "logs")
UA = f"correlation2-research/1.0 (personal research on news headline sentiment; contact {os.environ.get('C2_CONTACT', 'unset (set C2_CONTACT)')})"

CATEGORIES = {"weather", "climate_index", "energy", "commodity", "rates", "credit", "fx",
              "world_equity", "crypto", "volatility", "macro", "population", "space_weather",
              "geophysical", "astronomical", "attention", "health", "politics", "news",
              "prediction_market", "other"}
FREQS = {"daily", "weekly", "monthly", "quarterly", "annual"}
TRANSFORMS = {"logdiff", "diff", "anomaly", "anomaly_log1p", "level"}
SPACING = {"daily": 1, "weekly": 7, "monthly": 31, "quarterly": 92, "annual": 366}
ID_PREFIXES = ("wikice_", "hn_", "abc_", "toi_", "irish_", "huff_", "atn_", "reut_")


def log(msg, fh=None):
    line = f"{dt.datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def http_get(url, headers=None, timeout=120, retries=6, sleep=1.0):
    """GET url politely; returns bytes. Backs off on 429/5xx and network errors.
    Honours Retry-After. Raises on 4xx other than 429."""
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=h)
            deadline = time.monotonic() + timeout  # total time, not just per socket read
            with urllib.request.urlopen(req, timeout=min(timeout, 60)) as r:
                chunks = []
                while True:
                    c = r.read(1 << 16)
                    if not c:
                        break
                    chunks.append(c)
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"total time > {timeout}s")
                data = b"".join(chunks)
                # Wayback replays the original Content-Encoding header even when the body it
                # serves is already decompressed, so check the gzip magic bytes.
                if r.headers.get("Content-Encoding") == "gzip" and data[:2] == b"\x1f\x8b":
                    data = gzip.decompress(data)
            time.sleep(sleep)
            return data
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404, 410):
                raise
            ra = e.headers.get("Retry-After") if e.headers else None
            wait = float(ra) if ra and ra.isdigit() else sleep * (3 ** (attempt + 1))
            time.sleep(min(wait, 600))
        except Exception as e:  # network hiccup
            last = e
            time.sleep(min(sleep * (3 ** (attempt + 1)), 600))
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {last}")


def save_bytes(dest, data, compress=False):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    if compress:
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            f.write(data)
    else:
        with open(tmp, "wb") as f:
            f.write(data)
    os.replace(tmp, dest)


def daterange(a, b):
    d = a
    one = dt.timedelta(days=1)
    while d <= b:
        yield d
        d += one


def fmt(v):
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.8g}"


def write_series(sid, rows, *, name, category, subcategory, region, source, url, frequency,
                 units, transform, known_after_days, notes, value_range=None, quiet=False):
    """rows: iterable of (date|str, float). Writes series/<sid>.csv and meta/<sid>.json."""
    assert re.fullmatch(r"[a-z0-9_]+", sid), sid
    assert sid.startswith(ID_PREFIXES), sid
    assert category in CATEGORIES, category
    assert frequency in FREQS, frequency
    assert transform in TRANSFORMS, transform
    assert isinstance(known_after_days, int) and known_after_days >= 0
    clean = []
    for d, v in rows:
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
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
    if value_range is not None:
        lo, hi = value_range
        bad = [(d, v) for d, v in clean if v < lo or v > hi]
        if bad:
            raise ValueError(f"{sid}: {len(bad)} values outside [{lo},{hi}], e.g. {bad[:5]}")
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
    return check_series(sid, frequency, quiet=quiet)


def check_series(sid, frequency, quiet=False):
    """Re-read the written file and report range, gaps, first/last rows."""
    path = os.path.join(SERIES, f"{sid}.csv")
    with open(path) as f:
        r = csv.reader(f)
        head = next(r)
        assert head == ["date", "value"], head
        rows = [(dt.date.fromisoformat(a), float(b)) for a, b in r]
    dates = [d for d, _ in rows]
    assert all(dates[i] < dates[i + 1] for i in range(len(dates) - 1)), f"{sid}: not strictly ascending"
    vals = [v for _, v in rows]
    sp = SPACING[frequency]
    gaps = []
    for i in range(1, len(dates)):
        g = (dates[i] - dates[i - 1]).days
        if g > sp * 1.5 + (2 if frequency in ("monthly", "quarterly") else 0):
            gaps.append((dates[i - 1], dates[i], g))
    span = (dates[-1] - dates[0]).days + 1
    rep = dict(id=sid, first=dates[0].isoformat(), last=dates[-1].isoformat(), rows=len(rows),
               min=min(vals), max=max(vals), mean=sum(vals) / len(vals), n_gaps=len(gaps),
               missing_days=(span - len(rows)) if frequency == "daily" else None,
               largest_gap=max(gaps, key=lambda x: x[2]) if gaps else None,
               zeros=sum(1 for v in vals if v == 0))
    if not quiet:
        print(f"== {sid}: {rep['first']} .. {rep['last']}  rows={rep['rows']}  "
              f"min={rep['min']:.5g} max={rep['max']:.5g} mean={rep['mean']:.5g} zeros={rep['zeros']}"
              + (f" missing_days={rep['missing_days']}" if rep['missing_days'] is not None else ""))
        if gaps:
            big = sorted(gaps, key=lambda x: -x[2])[:4]
            print(f"   gaps: {len(gaps)}; largest: " + "; ".join(f"{a}->{b} ({g}d)" for a, b, g in big))
        print("   head: " + "  ".join(f"{d} {fmt(v)}" for d, v in rows[:3]))
        print("   tail: " + "  ".join(f"{d} {fmt(v)}" for d, v in rows[-3:]))
    return rep
