"""Wikipedia "Portal:Current events" daily pages via the MediaWiki API.

Phase 1 (latest): the current revision of every daily page, one API call per calendar month
          (<= 31 titles), saved as raw/wikipedia_current_events/latest/YYYY-MM.json.
Phase 2 (asof):   for each day D from ASOF_START, the newest revision made at or before
          D+1 19:30 UTC (= 15:30 EDT / 14:30 EST, i.e. before the 4 pm New York close on D+1),
          one call per day, saved as raw/wikipedia_current_events/asof/YYYY/YYYY-MM-DD.json.
Titles come from a list=allpages enumeration of Portal namespace (100), prefix
"Current events/", non-redirects (saved in raw/ as allpages_portal_current_events_*.json).
Resumable: existing files are skipped. Sequential, ~1 request/second, maxlag=5.
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, LOGS, http_get, save_bytes, daterange, log

OUT = os.path.join(RAW, "wikipedia_current_events")
API = "https://en.wikipedia.org/w/api.php"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
ASOF_START = dt.date(2006, 1, 1)
END = dt.date(2026, 10, 16)


def title_map():
    path = os.path.join(OUT, "allpages_portal_current_events_2026-10-19.json")
    d = json.load(open(path))
    pat = re.compile(r"^Portal:Current events/(\d{4}) ([A-Z][a-z]+) (\d{1,2})$")
    out = {}
    for t in d["nonredirects"]:
        m = pat.match(t)
        if not m or m.group(2) not in MONTHS:
            continue
        try:
            day = dt.date(int(m.group(1)), MONTHS.index(m.group(2)) + 1, int(m.group(3)))
        except ValueError:
            continue
        assert day not in out, (day, t, out.get(day))
        out[day] = t
    return out


def api_call(params):
    params = dict(params)
    params.setdefault("format", "json")
    params.setdefault("formatversion", "2")
    params["maxlag"] = "5"
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        data = http_get(url, sleep=0.6)  # ~1 request/s including latency
        try:
            j = json.loads(data)
        except json.JSONDecodeError:
            time.sleep(5 * (attempt + 1))
            continue
        err = j.get("error", {}).get("code")
        if err == "maxlag":
            time.sleep(10 * (attempt + 1))
            continue
        if err:
            raise RuntimeError(f"API error {err}: {j['error']}")
        return data, j
    raise RuntimeError(f"gave up on {url}")


def phase_latest(tmap, fh):
    months = sorted({(d.year, d.month) for d in tmap})
    for (y, m) in months:
        dest = os.path.join(OUT, "latest", f"{y:04d}-{m:02d}.json")
        if os.path.exists(dest):
            continue
        titles = [tmap[d] for d in sorted(tmap) if (d.year, d.month) == (y, m) and d <= END]
        if not titles:
            continue
        base = dict(action="query", prop="revisions", rvprop="content|ids|timestamp",
                    rvslots="main", titles="|".join(titles))
        data, j = api_call(base)
        got = {p["title"] for p in j["query"]["pages"] if p.get("revisions")}
        k = 0
        while "continue" in j:  # result too big for one response: follow the continuation
            k += 1
            data_k, j = api_call({**base, **j["continue"]})
            save_bytes(dest.replace(".json", f".c{k}.json"), data_k)
            got |= {p["title"] for p in j["query"]["pages"] if p.get("revisions")}
        if len(got) != len(titles):
            log(f"latest {y}-{m:02d}: WARNING got {len(got)}/{len(titles)} pages", fh)
        save_bytes(dest, data)  # written last, so a crash mid-month refetches the month
        log(f"latest {y}-{m:02d}: {len(got)} pages ({k} continuations)", fh)


def phase_asof(tmap, fh):
    n = 0
    for d in daterange(ASOF_START, END):
        dest = os.path.join(OUT, "asof", f"{d.year:04d}", f"{d.isoformat()}.json")
        if os.path.exists(dest):
            continue
        t = tmap.get(d)
        if t is None:
            log(f"asof {d}: no page title", fh)
            continue
        cutoff = (d + dt.timedelta(days=1)).isoformat() + "T19:30:00Z"
        data, j = api_call(dict(action="query", prop="revisions", rvprop="content|ids|timestamp|size",
                                rvslots="main", rvlimit="1", rvdir="older", rvstart=cutoff, titles=t))
        save_bytes(dest, data)
        n += 1
        if n % 100 == 0:
            p = j["query"]["pages"][0]
            r = (p.get("revisions") or [{}])[0]
            log(f"asof {d}: done {n}; rev {r.get('timestamp')} size {r.get('size')}", fh)


def main():
    os.makedirs(LOGS, exist_ok=True)
    tmap = title_map()
    with open(os.path.join(LOGS, "fetch_wikice.log"), "a") as fh:
        log(f"titles: {len(tmap)} days {min(tmap)}..{max(tmap)}", fh)
        phase_latest(tmap, fh)
        phase_asof(tmap, fh)
        log("done", fh)


if __name__ == "__main__":
    main()
