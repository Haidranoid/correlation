"""Second pass for the Reuters archive: some first-choice captures are truncated (e.g. the
2008-09-07 crawl of 2007 pages holds ~20% of the day's items). For every day whose parsed
headline count is < 25% of the median of same-type days (weekday vs weekend) within +-30 days,
look for a capture taken >= 2 days after the date that is > 1.5x larger (CDX length); if one
exists, fetch it and point capture_plan.json at it. The original plan is kept as
capture_plan_original.json; the earlier files stay in raw/ (the builder reads only planned files).
Needs work/reut_daily.json from a first build_reuters.py run.
"""
import datetime as dt
import json
import os
import re
import shutil
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, WORK, LOGS, http_get, save_bytes, log

OUT = os.path.join(RAW, "reuters_archive_wayback")
LAST = dt.date(2014, 12, 31)


def main():
    plan_p = os.path.join(OUT, "capture_plan.json")
    orig_p = os.path.join(OUT, "capture_plan_original.json")
    if not os.path.exists(orig_p):
        shutil.copy(plan_p, orig_p)
    plan = json.load(open(plan_p))
    caps = defaultdict(list)
    for y in range(2007, 2016):
        for ts, orig, length, digest in json.load(open(os.path.join(OUT, "cdx", f"cdx_{y}.json")))[1:]:
            m = re.search(r"/archive/us/(\d{8})\.html$", orig)
            if m:
                caps[m.group(1)].append((ts, orig, int(length) if length.isdigit() else -1))
    n = {dt.date.fromisoformat(k): v["n"] for k, v in json.load(open(os.path.join(WORK, "reut_daily.json"))).items()}
    days = sorted(n)
    with open(os.path.join(LOGS, "fetch_reuters.log"), "a") as fh:
        changed = 0
        for d in days:
            if d > LAST:
                continue
            peers = [n[e] for e in days if 0 < abs((e - d).days) <= 30 and (e.weekday() >= 5) == (d.weekday() >= 5)]
            if not peers or n[d] >= 0.25 * statistics.median(peers):
                continue
            key = d.strftime("%Y%m%d")
            chosen_ts = plan[d.isoformat()][0]
            chosen = [c for c in caps[key] if c[0] == chosen_ts][0]
            ok = [c for c in caps[key] if c[0][:8] >= (d + dt.timedelta(days=2)).strftime("%Y%m%d")]
            big = max(ok, key=lambda c: c[2])
            if big[2] <= 1.5 * chosen[2]:
                continue
            dest = os.path.join(OUT, f"{d.year}", f"{key}_{big[0]}.html.gz")
            if not os.path.exists(dest):
                data = http_get(f"http://web.archive.org/web/{big[0]}id_/{big[1]}", timeout=180, sleep=1.0)
                save_bytes(dest, data, compress=True)
            plan[d.isoformat()] = [big[0], big[1]]
            changed += 1
            log(f"fix {d}: count {n[d]}, capture {chosen_ts} ({chosen[2]} B) -> {big[0]} ({big[2]} B)", fh)
        with open(plan_p + ".part", "w") as f:
            json.dump(plan, f, indent=0)
        os.replace(plan_p + ".part", plan_p)
        log(f"fix pass: {changed} days re-pointed", fh)


if __name__ == "__main__":
    main()
