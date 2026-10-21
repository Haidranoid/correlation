"""Fetch Polymarket market metadata and CLOB price histories for the selected markets.

Price history: GET clob.polymarket.com/prices-history?market=<token>&startTs=..&endTs=..&fidelity=10
The endpoint refuses windows longer than ~15 days, so requests use fixed, aligned 14-day
windows (multiples of 14 days from 2020-01-06 00:00 UTC; samples then fall at hh:m0:02 UTC)
so that overlapping needs share the cache. Responses are saved unmodified (gzip) under
raw/polymarket/prices/<market_id>/.

Fed per-meeting events ("Fed Interest Rates: <Month> <Year>", "Fed decision in <Month>?")
are found in the tag crawl; for those only the window in which the meeting is the next one
(previous meeting - 2 days .. meeting + 1 day) is fetched.
"""
import datetime as dt
import gzip
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, RAW, fetch_json, iso_to_ts, log  # noqa: E402
from pm_select import INDIVIDUAL, ROLLING  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
PRAW = os.path.join(RAW, "polymarket")
E0 = int(dt.datetime(2020, 1, 6, tzinfo=dt.timezone.utc).timestamp())
W = 14 * 86400
FIDELITY = 10

# FOMC announcement dates (scheduled meetings; the decision day), 2021-2026
FOMC = ["2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22",
        "2021-11-03", "2021-12-15", "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
        "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14", "2023-02-01", "2023-03-22",
        "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18",
        "2024-11-07", "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18",
        "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
FED_TITLE = re.compile(r"^\s*fed (interest rates:\s*\w+ \d{4}|decision in \w+\??)\s*$", re.I)


def market_meta(mid):
    return fetch_json(f"{GAMMA}/markets/{mid}", os.path.join(PRAW, "markets", f"{mid}.json.gz"), gz=True)


def token_for(m, outcome):
    outs = json.loads(m["outcomes"]) if isinstance(m["outcomes"], str) else m["outcomes"]
    toks = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
    i = [o.strip().lower() for o in outs].index(outcome.strip().lower())
    return toks[i]


def windows(a, b):
    k0 = (a - E0) // W
    k1 = (b - E0) // W
    return [E0 + k * W for k in range(k0, k1 + 1)]


def price_path(mid, token, ws):
    return os.path.join(PRAW, "prices", str(mid), f"{token[:12]}_{ws}_f{FIDELITY}.json.gz")


def fetch_prices(mid, token, a, b):
    now = int(time.time())
    n = 0
    for ws in windows(a, min(b, now)):
        dest = price_path(mid, token, ws)
        if os.path.exists(dest):
            continue
        url = f"{CLOB}/prices-history?market={token}&startTs={ws}&endTs={ws + W}&fidelity={FIDELITY}"
        try:
            fetch_json(url, dest, gz=True)
        except Exception as e:
            log("pm_fetch.log", f"FAIL {mid} {ws}: {e}")
        n += 1
    return n


def load_prices(mid, token):
    import glob
    pts = {}
    for f in glob.glob(os.path.join(PRAW, "prices", str(mid), f"{token[:12]}_*_f{FIDELITY}.json.gz")):
        for x in json.loads(gzip.decompress(open(f, "rb").read())).get("history") or []:
            pts[int(x["t"])] = float(x["p"])
    return sorted(pts.items())


def fed_events():
    """[(meeting_date, prev_meeting_date, event_id, [market rows])] from the tag crawl index."""
    rows = json.load(gzip.open(os.path.join(HERE, "meta_pm", "markets_index.json.gz"), "rt"))
    ev = {}
    for r in rows:
        if FED_TITLE.match(r.get("event_title") or "") and r.get("neg_risk"):
            ev.setdefault(r["event_id"], []).append(r)
    out = []
    for eid, ms in ev.items():
        end = (ms[0].get("event_end") or "")[:10]
        try:
            e = dt.date.fromisoformat(end)
        except ValueError:
            continue
        # endDate is the meeting day (occasionally one day later): nearest FOMC date within 2 days
        near = [f for f in FOMC if abs((dt.date.fromisoformat(f) - e).days) <= 2]
        if not near:
            log("pm_fetch.log", f"fed event {eid} end {end} not near an FOMC date; skipped")
            continue
        i = FOMC.index(near[0])
        out.append((near[0], FOMC[i - 1], eid, ms))
    # one event per meeting: keep the highest-volume one if several map to the same date
    best = {}
    for meet, prev, eid, ms in out:
        vol = float(ms[0].get("event_volume") or 0)
        if meet not in best or vol > best[meet][0]:
            best[meet] = (vol, (meet, prev, eid, ms))
    return sorted(v for _, v in best.values())


def main():
    total = 0
    jobs = []
    for sid, (name, legs) in ROLLING.items():
        for mid, outcome, _ in legs:
            jobs.append((mid, outcome, None))
    for mid, outcome, sid, name, sub in INDIVIDUAL:
        jobs.append((mid, outcome, None))
    for meet, prev, eid, ms in fed_events():
        a = iso_to_ts(prev + "T00:00:00Z") - 2 * 86400
        b = iso_to_ts(meet + "T00:00:00Z") + 2 * 86400
        for r in ms:
            jobs.append((r["market_id"], "Yes", (a, b)))
    print(f"{len(jobs)} market jobs", flush=True)
    for i, (mid, outcome, rng) in enumerate(jobs):
        m = market_meta(mid)
        if not m.get("clobTokenIds"):
            log("pm_fetch.log", f"{mid}: no clob tokens; skipped")
            continue
        tok = token_for(m, outcome)
        st = iso_to_ts(m.get("startDate") or m.get("createdAt"))
        en = iso_to_ts(m.get("closedTime")) or iso_to_ts(m.get("endDate"))
        if rng:
            a, b = max(rng[0], st - 86400), min(rng[1], (en or rng[1]) + 86400)
        else:
            a, b = st - 86400, (en or int(time.time())) + 86400
        total += fetch_prices(mid, tok, a, b)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(jobs)} markets, {total} requests", flush=True)
    print(f"done: {total} requests", flush=True)
    log("pm_fetch.log", f"done {total} requests")


if __name__ == "__main__":
    main()
