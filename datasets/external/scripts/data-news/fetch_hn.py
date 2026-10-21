"""Hacker News stories with more than 10 points, one Algolia query per New York calendar day.

Endpoint: https://hn.algolia.com/api/v1/search_by_date?tags=story&numericFilters=
          created_at_i>=A,created_at_i<B,points>10&hitsPerPage=1000
A, B = midnight America/New_York at the start of day D and D+1 (DST-aware).
points>10 is a superset; the parser filters to >50 (and >20) locally.
If a window's nbHits exceeds the hits returned (1000 cap), the window is split in halves
recursively and each part saved separately (never needed in practice: <= ~200/day).
Raw responses are saved gzip-compressed (lossless) as raw/hn_algolia/YYYY/YYYY-MM-DD[_k].json.gz.
Resumable; sequential, ~1 request/second (Algolia's HN API allows 10,000 requests/hour).
"""
import datetime as dt
import json
import os
import sys
import urllib.parse
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, LOGS, http_get, save_bytes, daterange, log

OUT = os.path.join(RAW, "hn_algolia")
NY = ZoneInfo("America/New_York")
START, END = dt.date(2007, 2, 19), dt.date(2026, 10, 16)
ATTRS = "title,created_at_i,points,num_comments,url,author"


def epoch_midnight(d):
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=NY).timestamp())


def query(a, b):
    q = {"tags": "story", "numericFilters": f"created_at_i>={a},created_at_i<{b},points>10",
         "hitsPerPage": 1000, "attributesToRetrieve": ATTRS, "attributesToHighlight": ""}
    url = "https://hn.algolia.com/api/v1/search_by_date?" + urllib.parse.urlencode(q)
    data = http_get(url, sleep=0.3)  # ~1 request/s including ~0.8 s latency
    return data, json.loads(data)


def fetch_window(a, b, dest_base, parts, fh):
    data, j = query(a, b)
    if j["nbHits"] > len(j["hits"]):
        mid = (a + b) // 2
        log(f"split {dest_base}: nbHits {j['nbHits']} > {len(j['hits'])}", fh)
        fetch_window(a, mid, dest_base, parts, fh)
        fetch_window(mid, b, dest_base, parts, fh)
        return
    parts.append((data, len(j["hits"])))


def main():
    os.makedirs(LOGS, exist_ok=True)
    n = 0
    with open(os.path.join(LOGS, "fetch_hn.log"), "a") as fh:
        for d in daterange(START, END):
            dest = os.path.join(OUT, f"{d.year:04d}", f"{d.isoformat()}.json.gz")
            if os.path.exists(dest):
                continue
            a, b = epoch_midnight(d), epoch_midnight(d + dt.timedelta(days=1))
            parts = []
            fetch_window(a, b, dest, parts, fh)
            for k, (data, _) in enumerate(parts[1:], 1):
                save_bytes(dest.replace(".json.gz", f"_{k}.json.gz"), data, compress=True)
            save_bytes(dest, parts[0][0], compress=True)  # main file last (resume marker)
            n += 1
            if n % 100 == 0:
                log(f"{d}: done {n}; hits today {sum(h for _, h in parts)}", fh)
        log("done", fh)


if __name__ == "__main__":
    main()
