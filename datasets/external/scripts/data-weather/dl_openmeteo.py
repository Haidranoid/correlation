"""Download Open-Meteo ERA5 daily archive for a handful of points, politely.

Usage: python3 -I dl_openmeteo.py [point_key ...]
Each point is fetched in two halves (2000-2012, 2013-2025), <=10 variables, models=era5, timezone=auto.
Weight per half ~= 4749 days / 14 ~= 339 calls; requests are spaced SLEEP seconds apart so the
hourly weighted total stays below ~4,600 (limit 5,000/h, 600/min, 10,000/day).
- HTTP 429 or any 'limit' message: stop at once.
- Server timeout / 5xx: wait RETRY_WAIT, retry once; if it fails again, split the range in two
  and fetch the halves (still spaced); a failure there stops the run.
- A running weighted-call estimate (failed attempts counted as spent) is kept in the log; no request
  is started that would take the day's total past BUDGET_CAP.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

OUT = os.path.expanduser("~/.local/share/correlation2/external/raw/openmeteo")
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openmeteo_download.log")
SLEEP = 300
RETRY_WAIT = 300
BUDGET_CAP = 7500          # self-imposed, well under the 10,000/day limit
PROBE_CALLS = 3            # three 1-call probes made by hand before the first download
VARS = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "sunshine_duration", "cloud_cover_mean",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm_mean",
        "soil_moisture_7_to_28cm_mean", "daylight_duration"]
POINTS = {
    "nyc": (40.78, -73.97),
    "il_corn": (40.48, -88.99),
    "ks_wheat": (38.36, -98.76),
    "ca_valley": (36.50, -120.00),
    "tx_lubbock": (33.58, -101.86),
    "br_coffee": (-21.55, -45.43),
    "br_soy": (-12.55, -55.72),
    "ar_pampas": (-33.89, -60.57),
    "tokyo": (35.68, 139.77),
}
HALVES = [("2000-01-01", "2012-12-31"), ("2013-01-01", "2025-12-31")]


class Stop(Exception):
    pass


class Transient(Exception):
    pass


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def weight(start, end):
    days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    return max(1.0, len(VARS) / 10) * max(1.0, days / 14)


def used_so_far():
    """Estimate weighted calls spent today from the log (every attempt counts)."""
    total = PROBE_CALLS
    if not os.path.exists(LOG):
        return total
    for line in open(LOG):
        m = re.search(r"weight=([\d.]+)", line)
        if m:
            total += float(m.group(1))
        elif re.search(r" (got|non-JSON|HTTP \d|API error|ERROR) ", line) and "have " not in line:
            total += weight("2000-01-01", "2012-12-31")  # all early requests were 13-year halves
    return total


_last_request = [0.0]


def fetch(key, start, end):
    """One polite request. Returns (parsed JSON, original response bytes) or raises Stop/Transient."""
    lat, lon = POINTS[key]
    w = weight(start, end)
    used = used_so_far()
    if used + w > BUDGET_CAP:
        raise Stop(f"budget cap: used ~{used:.0f}, next {w:.0f} would exceed {BUDGET_CAP}")
    wait = SLEEP - (time.time() - _last_request[0])
    if _last_request[0] and wait > 0:
        time.sleep(wait)
    _last_request[0] = time.time()
    url = ("https://archive-api.open-meteo.com/v1/archive?"
           f"latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           f"&daily={','.join(VARS)}&timezone=auto&models=era5")
    req = urllib.request.Request(url, headers={"User-Agent": "correlation2-research/1.0 (non-commercial)"})
    tag = f"{key} {start}..{end} weight={w:.1f}"
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            body, status = r.read(), r.status
    except urllib.error.HTTPError as e:
        body = e.read()
        if e.code == 429 or b"limit" in body.lower():
            log(f"HTTP {e.code} RATE LIMIT for {tag}: {body[:200]!r}; STOPPING")
            raise Stop("rate limit")
        if e.code >= 500:
            log(f"HTTP {e.code} transient for {tag}: {body[:200]!r}")
            raise Transient()
        log(f"HTTP {e.code} for {tag}: {body[:200]!r}; STOPPING")
        raise Stop("http error")
    except Exception as e:  # network error / socket timeout
        log(f"ERROR transient {e!r} for {tag}")
        raise Transient()
    try:
        js = json.loads(body)
    except ValueError:
        if b"timeout" in body.lower():
            log(f"non-JSON transient for {tag}: {body[:200]!r}")
            raise Transient()
        log(f"non-JSON response for {tag}: {body[:200]!r}; STOPPING")
        raise Stop("non-json")
    if js.get("error"):
        reason = str(js.get("reason"))
        log(f"API error for {tag}: {reason}; STOPPING")
        raise Stop(reason)
    n = len(js.get("daily", {}).get("time", []))
    exp = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    if n != exp:
        log(f"short response for {tag}: {n} of {exp} days; STOPPING")
        raise Stop("short")
    log(f"got {tag} HTTP {status} days={n} grid=({js.get('latitude')},{js.get('longitude')}) "
        f"elev={js.get('elevation')} tz={js.get('timezone')} used~{used + w:.0f}")
    return js, body


def save(key, start, end, body):
    """Write the response bytes exactly as received (provenance: raw files stay unmodified)."""
    path = os.path.join(OUT, f"{key}_{start}_{end}.json")
    with open(path + ".part", "wb") as f:
        f.write(body)
    os.replace(path + ".part", path)


def get_range(key, start, end, depth=0):
    """Fetch [start, end]; on a transient failure retry once, then split in two (one level)."""
    path = os.path.join(OUT, f"{key}_{start}_{end}.json")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        log(f"have {path}")
        return
    for attempt in (1, 2):
        try:
            save(key, start, end, fetch(key, start, end)[1])
            return
        except Transient:
            if attempt == 1:
                log(f"waiting {RETRY_WAIT}s before one retry of {key} {start}..{end}")
                time.sleep(RETRY_WAIT)
    if depth >= 1:
        log(f"{key} {start}..{end} failed twice after splitting; STOPPING")
        raise Stop("transient twice")
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    mid = a + (b - a) // 2
    log(f"splitting {key} {start}..{end} at {mid}")
    get_range(key, start, mid.isoformat(), depth + 1)
    get_range(key, (mid + dt.timedelta(1)).isoformat(), end, depth + 1)


def covered(key, start, end):
    """True if [start, end] is already covered by saved chunk files for this point."""
    need = set()
    d = dt.date.fromisoformat(start)
    while d <= dt.date.fromisoformat(end):
        need.add(d)
        d += dt.timedelta(1)
    for fn in os.listdir(OUT):
        m = re.fullmatch(rf"{re.escape(key)}_(\d{{4}}-\d\d-\d\d)_(\d{{4}}-\d\d-\d\d)\.json", fn)
        if m:
            a, b = dt.date.fromisoformat(m.group(1)), dt.date.fromisoformat(m.group(2))
            need = {x for x in need if not (a <= x <= b)}
    return not need


def main():
    os.makedirs(OUT, exist_ok=True)
    keys = sys.argv[1:] or [k for k in POINTS if k != "tokyo"]
    log(f"start run for {keys}; weighted calls used so far today ~{used_so_far():.0f} (cap {BUDGET_CAP})")
    try:
        for key in keys:
            for start, end in HALVES:
                if covered(key, start, end):
                    log(f"have {key} {start}..{end}")
                    continue
                get_range(key, start, end)
    except Stop as e:
        log(f"stopped: {e}; STOPPING")
        return 1
    log(f"done; weighted calls used today ~{used_so_far():.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
