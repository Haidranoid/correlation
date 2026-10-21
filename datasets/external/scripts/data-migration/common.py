"""Shared helpers for the data-migration download agent.

Every normalized series goes through write_series(), which sorts, checks for duplicate dates,
non-finite values and sentinels, enforces value ranges, writes series/<id>.csv, prints a sanity
report (first/last rows, gaps) and records the metadata used later by build_catalog.py.
Run every script with `python3 -I`; scripts add their own directory to sys.path.
Downloads are written unmodified under raw/<source>/ and are never executed.
"""
import csv
import datetime as dt
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

EXT = os.path.expanduser("~/.local/share/correlation2/external")
RAW = os.path.join(EXT, "raw")
SERIES = os.path.join(EXT, "series")
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "meta")
LOGS = os.path.join(HERE, "logs")
UA = f"correlation2-research/1.0 (personal research; contact {os.environ.get('C2_CONTACT', 'unset (set C2_CONTACT)')})"

CATEGORIES = {"weather", "climate_index", "energy", "commodity", "rates", "credit", "fx",
              "world_equity", "crypto", "volatility", "macro", "population", "space_weather",
              "geophysical", "astronomical", "attention", "health", "politics", "news",
              "prediction_market", "prosperity", "other", "migration"}
FREQS = {"daily", "weekly", "monthly", "quarterly", "annual"}
TRANSFORMS = {"logdiff", "diff", "anomaly", "anomaly_log1p", "level"}
# expected spacing in days, used for the gap report
SPACING = {"daily": 1, "weekly": 7, "monthly": 31, "quarterly": 92, "annual": 366}


def fetch(url, dest, sleep=1.5, retries=3, timeout=180, headers=None, overwrite=False, data=None):
    """Download url to dest (bytes, unmodified). Polite: UA, sleep after each request.
    Returns dest. Skips the download if dest already exists (unless overwrite)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not overwrite:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=h, data=data)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                lm = r.headers.get("Last-Modified")
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(body)
            os.replace(tmp, dest)
            # provenance: remember the URL and server Last-Modified next to the file
            with open(os.path.join(os.path.dirname(dest), "_downloads.tsv"), "a") as f:
                f.write(f"{dt.datetime.utcnow().isoformat(timespec='seconds')}Z\t"
                        f"{os.path.basename(dest)}\t{len(body)}\t{lm or ''}\t{url}\n")
            time.sleep(sleep)
            return dest
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 401, 403, 404, 410):
                raise
            time.sleep(sleep * (4 ** (attempt + 1)))
        except Exception as e:  # network hiccup
            last = e
            time.sleep(sleep * (4 ** (attempt + 1)))
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {last}")


def fmt(v):
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(round(v, 10))


def write_series(sid, rows, *, name, category, subcategory, region, source, url, frequency,
                 units, transform, known_after_days, notes, value_range=None, quiet=False,
                 allow_zero=False):
    """rows: iterable of (date|str, float). Writes series/<sid>.csv and meta/<sid>.json.
    Raises on duplicate dates, non-finite values, out-of-range values, non-positive values for
    logdiff (unless allow_zero, where zeros are dropped by the test's log anyway)."""
    assert re.fullmatch(r"[a-z0-9_]+", sid), sid
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
    vals = [v for _, v in clean]
    if value_range is not None:
        lo, hi = value_range
        bad = [(d, v) for d, v in clean if v < lo or v > hi]
        if bad:
            raise ValueError(f"{sid}: {len(bad)} values outside [{lo},{hi}], e.g. {bad[:5]}")
    if transform == "logdiff":
        nonpos = [(d, v) for d, v in clean if v <= 0]
        if nonpos and not (allow_zero and all(v == 0 for _, v in nonpos)):
            raise ValueError(f"{sid}: logdiff but {len(nonpos)} non-positive values, e.g. {nonpos[:5]}")
    for sentinel in (-9999, -999, -99, 9999, 99999, -1e9):
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
    # gap report
    exp = SPACING[frequency]
    gaps = []
    for i in range(1, len(dates)):
        g = (dates[i] - dates[i - 1]).days
        if g > exp * 1.5 + (3 if frequency == "daily" else 0):
            gaps.append((dates[i - 1].isoformat(), dates[i].isoformat(), g))
    meta = dict(id=sid, name=name, category=category, subcategory=subcategory, region=region,
                source=source, url=url, frequency=frequency, units=units, transform=transform,
                known_after_days=known_after_days, notes=notes,
                first=dates[0].isoformat(), last=dates[-1].isoformat(), rows=len(clean),
                vmin=min(vals), vmax=max(vals), n_gaps=len(gaps), gaps=gaps[:20],
                n_zero=sum(1 for v in vals if v == 0), n_neg=sum(1 for v in vals if v < 0))
    os.makedirs(META, exist_ok=True)
    with open(os.path.join(META, f"{sid}.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if not quiet:
        head = ", ".join(f"{d}={fmt(v)}" for d, v in clean[:3])
        tail = ", ".join(f"{d}={fmt(v)}" for d, v in clean[-3:])
        print(f"{sid}: {len(clean)} rows {dates[0]}..{dates[-1]} min={fmt(min(vals))} "
              f"max={fmt(max(vals))} gaps={len(gaps)}{' ' + str(gaps[:3]) if gaps else ''}\n"
              f"   first: {head}\n   last:  {tail}")
    return meta


def q(vals, p):
    """Quantile with linear interpolation (numpy-free, matches numpy's default)."""
    s = sorted(vals)
    if not s:
        return float("nan")
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)
