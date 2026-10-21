"""Reuters US daily news archive pages (reuters.com/resources/archive/us/YYYYMMDD.html),
retrieved from the Internet Archive's Wayback Machine (the live pages were removed ~2019).

Step 1: CDX listing of every 200-status capture per year (saved raw, JSON).
Step 2: per day, choose the earliest capture taken >= 2 days after the day (so the day's list
        is complete); if none, the latest capture (flagged in the parser when it is too early).
Step 3: fetch with the `id_` modifier (original bytes, no Wayback rewriting), gzip-stored as
        raw/reuters_archive_wayback/YYYY/YYYYMMDD_<capture timestamp>.html.gz.
Scope 2007-01-01..2015-01-31 (coverage collapses after 2014). Sequential, ~1 request/second.
"""
import datetime as dt
import json
import os
import re
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, LOGS, http_get, save_bytes, log

OUT = os.path.join(RAW, "reuters_archive_wayback")
FIRST, LAST = dt.date(2007, 1, 1), dt.date(2015, 1, 31)


def cdx_year(y, fh):
    dest = os.path.join(OUT, "cdx", f"cdx_{y}.json")
    if not os.path.exists(dest):
        q = {"url": f"reuters.com/resources/archive/us/{y}", "matchType": "prefix",
             "filter": "statuscode:200", "fl": "timestamp,original,length,digest", "output": "json"}
        data = http_get("http://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode(q), timeout=300, sleep=2.0)
        save_bytes(dest, data)
        log(f"cdx {y}: {len(data)} bytes", fh)
    rows = json.load(open(dest))
    return rows[1:] if rows else []


def main():
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "fetch_reuters.log"), "a") as fh:
        caps = {}
        for y in range(FIRST.year, LAST.year + 1):
            for ts, orig, length, digest in cdx_year(y, fh):
                m = re.search(r"/archive/us/(\d{8})\.html$", orig)
                if not m:
                    continue
                d = dt.date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:]))
                if FIRST <= d <= LAST:
                    caps.setdefault(d, []).append((ts, orig))
        log(f"days with captures: {len(caps)}", fh)
        plan = {}
        for d, lst in caps.items():
            lst.sort()
            ok = [c for c in lst if c[0][:8] >= (d + dt.timedelta(days=2)).strftime("%Y%m%d")]
            plan[d] = ok[0] if ok else lst[-1]
        with open(os.path.join(OUT, "capture_plan.json"), "w") as f:
            json.dump({d.isoformat(): plan[d] for d in sorted(plan)}, f, indent=0)
        n = 0
        for d in sorted(plan):
            ts, orig = plan[d]
            dest = os.path.join(OUT, f"{d.year}", f"{d.strftime('%Y%m%d')}_{ts}.html.gz")
            if os.path.exists(dest):
                continue
            try:
                data = http_get(f"http://web.archive.org/web/{ts}id_/{orig}", timeout=180, sleep=1.0)
            except Exception as e:
                log(f"{d}: FAILED {ts} {orig}: {e}", fh)
                continue
            save_bytes(dest, data, compress=True)
            n += 1
            if n % 100 == 0:
                log(f"{d}: done {n}; {len(data)} bytes", fh)
        log("done", fh)


if __name__ == "__main__":
    main()
