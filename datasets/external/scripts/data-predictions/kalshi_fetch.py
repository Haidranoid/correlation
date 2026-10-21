"""Plan and fetch Kalshi hourly candles for the series we build.

Usage: python3 -I kalshi_fetch.py [--plan-only]

Plan (all hourly candles, <=4990 h per request, cached under raw/kalshi/candles/):
- "next event" ladders (CPI, CPI YoY, core CPI, payrolls, U3, GDP, AAA gas): each market from
  max(open, close-4990h) to close: one request covers the ~1-2 months the event is the
  nearest one plus several months of look-back for the sparse-candle as-of state.
- Fed ladder (KXFED): whole life (needed for the next-meeting and ~6-months-ahead series),
  events up to ~7 months ahead of today.
- S&P 500 year-end buckets (KXINXY, the highest-volume event per year): whole life.
- Rolling binaries (recession by year, shutdown by deadline) and the individual contracts
  listed in INDIVIDUAL: whole life.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import log  # noqa: E402
from kalshi_lib import (MAX_CANDLES, close_ts, events_of, fetch_candles, load_markets,  # noqa: E402
                        now_ts, open_ts, strike_of)

NEXT_LADDERS = ["KXCPI", "KXCPIYOY", "KXCPICORE", "KXPAYROLLS", "KXU3", "KXGDP", "KXAAAGASM"]
WHOLE_SERIES = ["KXRECSSNBER", "KXGOVSHUT", "KXGOVTSHUTDOWN"]
INXY_EVENTS = ["INXY-22DEC30", "INXY-23DEC29", "INXD-24DEC31", "KXINXY-25DEC31", "KXINXY-26DEC31H1600"]
# individual high-volume binaries (series, market ticker)
INDIVIDUAL = [
    ("KXRATECUT", "RATECUT-23DEC31"), ("KXRATECUT", "RATECUT-24MAR20"), ("KXRATECUT", "RATECUT-24MAY01"),
    ("KXRATECUT", "RATECUT-24JUN12"), ("KXRATECUT", "RATECUT-24JUL31"), ("KXRATECUT", "RATECUT-24SEP18"),
    ("KXRATECUT", "KXRATECUT-26DEC31"),
    ("KXSHUTDOWNBY", "SHUTDOWNBY-24"), ("KXSHUTDOWNBY", "KXSHUTDOWNBY-25"),
    ("KXDCEIL", "DCEIL-23JUN1"), ("KXDCEIL", "DCEIL-23JUN9"), ("KXDCEIL", "DCEIL-23DEC31"),
    ("KXDCEIL", "DCEIL-21DEC15"), ("KXDCEIL", "DCEIL-21OCT14"),
    ("KXFEDHIKE", None), ("KXDEFAULT", None),
]


def plan():
    now = now_ts()
    jobs = []  # (series, market, start, end)

    def add_whole(series, m):
        o, c = open_ts(m), close_ts(m)
        if o is None or c is None:
            return
        jobs.append((series, m, o, min(c, now)))

    for s in NEXT_LADDERS:
        for m in load_markets(s).values():
            if strike_of(m) is None:
                continue
            o, c = open_ts(m), close_ts(m)
            if o is None or c is None or o > now:
                continue
            end = min(c, now)
            jobs.append((s, m, max(o, end - MAX_CANDLES * 3600), end))
    horizon = now + 215 * 86400
    for m in load_markets("KXFED").values():
        if strike_of(m) is None or close_ts(m) is None or close_ts(m) > horizon:
            continue
        add_whole("KXFED", m)
    inx = load_markets("KXINXY")
    for m in inx.values():
        if m["event_ticker"] in INXY_EVENTS:
            add_whole("KXINXY", m)
    for s in WHOLE_SERIES:
        for m in load_markets(s).values():
            add_whole(s, m)
    for s, t in INDIVIDUAL:
        ms = load_markets(s)
        for m in ms.values():
            if t is None or m["ticker"] == t:
                add_whole(s, m)
    prio = {"KXFED": 0, "KXRECSSNBER": 1, "KXGOVSHUT": 1, "KXGOVTSHUTDOWN": 1, "KXCPI": 2,
            "KXPAYROLLS": 2, "KXU3": 2, "KXGDP": 2, "KXCPIYOY": 3, "KXCPICORE": 3}
    jobs.sort(key=lambda j: (prio.get(j[0], 4), j[3]))
    nreq = sum(max(1, -(-(e - st) // (MAX_CANDLES * 3600))) for _, _, st, e in jobs if e > st)
    return jobs, nreq


def main():
    jobs, nreq = plan()
    by = {}
    for s, *_ in jobs:
        by[s] = by.get(s, 0) + 1
    print(f"{len(jobs)} market jobs, ~{nreq} requests; per series: {by}", flush=True)
    if "--plan-only" in sys.argv:
        return
    done = 0
    for s, m, st, e in jobs:
        if e <= st:
            continue
        fetch_candles(m, s, st, e)
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{len(jobs)} markets", flush=True)
            log("kalshi_fetch.log", f"progress {done}/{len(jobs)}")
    print("done", flush=True)
    log("kalshi_fetch.log", "done")


if __name__ == "__main__":
    main()
