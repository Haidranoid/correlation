"""Shared helpers for the data-predictions download agent (prediction markets).

Run every script with `python3 -I`; scripts add their own directory to sys.path.

Conventions
- Raw downloads go to raw/<source>/... unmodified. Polymarket price windows are stored
  gzip-compressed (lossless; `zcat` gives the exact bytes the API returned).
- Every normalized series goes through write_series(), which sorts, rejects duplicate dates
  and non-finite values, writes series/<id>.csv, writes meta/<id>.json (used to build the
  catalog) and prints a sanity report (range, gaps, head/tail).
- Daily value = the as-of value at 16:00 America/New_York on that calendar date (all 7 days:
  these markets trade around the clock), computed from timestamped data.
"""
import csv
import datetime as dt
import gzip
import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
import zoneinfo

EXT = os.path.expanduser("~/.local/share/correlation2/external")
RAW = os.path.join(EXT, "raw")
SERIES = os.path.join(EXT, "series")
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "meta")
LOGS = os.path.join(HERE, "logs")
UA = f"correlation2-research/1.0 (personal research; contact {os.environ.get('C2_CONTACT', 'unset (set C2_CONTACT)')})"
NY = zoneinfo.ZoneInfo("America/New_York")
UTC = dt.timezone.utc

CATEGORIES = {"weather", "climate_index", "energy", "commodity", "rates", "credit", "fx",
              "world_equity", "crypto", "volatility", "macro", "population", "space_weather",
              "geophysical", "astronomical", "attention", "health", "politics", "news",
              "prediction_market", "other"}
FREQS = {"daily", "weekly", "monthly", "quarterly", "annual"}
TRANSFORMS = {"logdiff", "diff", "anomaly", "anomaly_log1p", "level"}
SPACING = {"daily": 1, "weekly": 7, "monthly": 31, "quarterly": 92, "annual": 366}

_last_request = {}


def _polite_wait(host, min_interval):
    t = _last_request.get(host)
    if t is not None:
        wait = min_interval - (time.monotonic() - t)
        if wait > 0:
            time.sleep(wait)


def fetch(url, dest, *, gz=False, min_interval=1.05, retries=4, timeout=90, overwrite=False,
          accept_status=(200,)):
    """Download url to dest unless cached. Returns the raw bytes (decompressed if gz).

    Polite: at most one request per `min_interval` seconds per host, our UA, retries with
    back-off on 429/5xx/network errors. 4xx other than 429 raise immediately (HTTPError).
    """
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not overwrite:
        with open(dest, "rb") as f:
            data = f.read()
        return gzip.decompress(data) if gz else data
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    host = urllib.request.urlparse(url).netloc
    last = None
    for attempt in range(retries):
        _polite_wait(host, min_interval)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            _last_request[host] = time.monotonic()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                f.write(gzip.compress(data, 6) if gz else data)
            os.replace(tmp, dest)
            return data
        except urllib.error.HTTPError as e:
            _last_request[host] = time.monotonic()
            last = e
            if e.code == 429 or e.code >= 500:
                ra = e.headers.get("Retry-After") if e.headers else None
                time.sleep(float(ra) if ra and ra.replace('.', '', 1).isdigit() else 5 * (2 ** attempt))
                continue
            raise
        except (ValueError, http.client.InvalidURL):
            raise  # malformed URL: retrying cannot help
        except Exception as e:  # network hiccup
            _last_request[host] = time.monotonic()
            last = e
            time.sleep(5 * (2 ** attempt))
    raise RuntimeError(f"fetch failed after {retries} tries: {url}: {last}")


def fetch_json(url, dest, **kw):
    return json.loads(fetch(url, dest, **kw))


def ny16(d):
    """Unix timestamp of 16:00 America/New_York on date d."""
    return int(dt.datetime(d.year, d.month, d.day, 16, 0, tzinfo=NY).timestamp())


def ts_to_nydate(ts):
    return dt.datetime.fromtimestamp(ts, NY).date()


def iso_to_ts(s):
    """Parse ISO-ish timestamps from the APIs ('2024-11-06T15:17:41Z', '2024-11-06 15:17:41+00')."""
    if s is None or s == "":
        return None
    s = s.strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if re.search(r"[+-]\d\d$", s):
        s = s + ":00"
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=UTC)
    return int(t.timestamp())


def daterange(a, b):
    d = a
    one = dt.timedelta(days=1)
    while d <= b:
        yield d
        d += one


def asof_daily(samples, first_ts=None, close_ts=None):
    """samples: sorted list of (ts, value). Returns {date: value} where value is the last
    sample with ts <= 16:00 NY of that date. A date is included only if 16:00 NY falls
    on/after the first sample (and first_ts, if given) and strictly before close_ts (the
    market was still trading at 16:00), if given."""
    out = {}
    if not samples:
        return out
    t0 = samples[0][0] if first_ts is None else max(first_ts, samples[0][0])
    d = ts_to_nydate(t0)
    last_ts = samples[-1][0]
    end_ts = close_ts if close_ts is not None else last_ts + 86400
    d_end = ts_to_nydate(end_ts)
    i = 0
    n = len(samples)
    cur = None
    for day in daterange(d, d_end):
        cut = ny16(day)
        if cut < t0:
            continue
        if close_ts is not None and cut >= close_ts:
            break
        if close_ts is None and cut > last_ts + 86400:
            break
        while i < n and samples[i][0] <= cut:
            cur = samples[i][1]
            i += 1
        if cur is not None:
            out[day] = cur
    return out


def pav_decreasing(xs, ps, ws=None):
    """Pool-adjacent-violators: closest non-increasing sequence to ps (ordered by xs asc)."""
    if ws is None:
        ws = [1.0] * len(ps)
    blocks = []  # [value, weight, count]
    for p, w in zip(ps, ws):
        blocks.append([p, w, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            wt = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / wt, wt, c1 + c2])
    out = []
    for v, w, c in blocks:
        out += [v] * c
    return out


def ladder_expectation(strikes, surv, grid=None, spacing=None):
    """Expected value of X from a ladder of 'X > K' probabilities.

    strikes ascending, surv = P(X > K) (already monotone). If grid is given, X lives on
    that grid and strikes are grid points: the bucket (K_i, K_{i+1}] is represented by the
    mean of its grid points, the lower tail (X <= K_1) by K_1 and the upper tail
    (X > K_n) by K_n + grid. Otherwise X is continuous: bucket midpoints, tails at
    K_1 - h/2 and K_n + h/2 with h = `spacing` (default: median strike gap).
    Returns (E[X], lower_tail_mass, upper_tail_mass)."""
    n = len(strikes)
    assert n >= 1 and n == len(surv)
    if grid is None:
        gaps = sorted(strikes[i + 1] - strikes[i] for i in range(n - 1))
        h = spacing if spacing is not None else (gaps[len(gaps) // 2] if gaps else 1.0)
        reps = [strikes[0] - h / 2] + [(strikes[i] + strikes[i + 1]) / 2 for i in range(n - 1)] + [strikes[-1] + h / 2]
    else:
        reps = [strikes[0]]
        for i in range(n - 1):
            lo = strikes[i] + grid
            hi = strikes[i + 1]
            reps.append((lo + hi) / 2 if hi >= lo else hi)
        reps.append(strikes[-1] + grid)
    masses = [1 - surv[0]] + [surv[i] - surv[i + 1] for i in range(n - 1)] + [surv[-1]]
    e = sum(m * r for m, r in zip(masses, reps))
    return e, masses[0], masses[-1]


def fmt(v):
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.8g}"


def write_series(sid, rows, *, name, category, subcategory, region, source, url, frequency,
                 units, transform, known_after_days, notes, value_range=None, quiet=False,
                 extra=None):
    """rows: iterable of (date|str, float). Writes series/<sid>.csv and meta/<sid>.json."""
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
    if value_range is not None:
        lo, hi = value_range
        bad = [(d, v) for d, v in clean if v < lo or v > hi]
        if bad:
            raise ValueError(f"{sid}: {len(bad)} values outside [{lo},{hi}], e.g. {bad[:5]}")
    if transform == "logdiff":
        nonpos = [(d, v) for d, v in clean if v <= 0]
        if nonpos:
            raise ValueError(f"{sid}: logdiff but {len(nonpos)} non-positive values")
    for sentinel in (-9999, -999, 9999, 99999):
        if any(v == sentinel for _, v in clean):
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
    if extra:
        meta["extra"] = extra
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
        if g > sp * 1.5:
            gaps.append((dates[i - 1], dates[i], g))
    rep = dict(id=sid, first=dates[0].isoformat(), last=dates[-1].isoformat(), rows=len(rows),
               min=min(vals), max=max(vals), mean=sum(vals) / len(vals), n_gaps=len(gaps),
               gap_days=sum(g - 1 for _, _, g in gaps),
               largest_gap=max(gaps, key=lambda x: x[2]) if gaps else None)
    if not quiet:
        print(f"== {sid}: {rep['first']} .. {rep['last']}  rows={rep['rows']}  "
              f"min={rep['min']:.6g} max={rep['max']:.6g} mean={rep['mean']:.6g}")
        if gaps:
            big = sorted(gaps, key=lambda x: -x[2])[:4]
            print(f"   gaps: {len(gaps)} ({rep['gap_days']} missing days); largest: " +
                  "; ".join(f"{a}->{b} ({g}d)" for a, b, g in big))
        else:
            print("   no gaps")
        print("   head: " + "  ".join(f"{d} {fmt(v)}" for d, v in rows[:3]))
        print("   tail: " + "  ".join(f"{d} {fmt(v)}" for d, v in rows[-3:]))
    return rep


def log(name, msg):
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, name), "a") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")
