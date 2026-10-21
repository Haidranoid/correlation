"""Sanity check: does headline tone drop around known bad market days and recover?
For each source's work/<name>_daily.json and each event, the mean of a metric over a window of
days relative to the event is turned into a z-score against baseline windows of the same
length starting on the same weekday, taken from the 250 days ending 30 days before the event
(so day-of-week effects cancel). Windows: [-10,-1] before, [0,2] the event, [3,9] after,
[20,40] about a month later.
Usage: python3 -I sanity.py [--md] name [name ...]
"""
import datetime as dt
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK

EVENTS = [dt.date(2008, 9, 15), dt.date(2008, 10, 10), dt.date(2011, 8, 8), dt.date(2020, 3, 16)]
WINDOWS = [(-10, -1), (0, 2), (3, 9), (20, 40)]
METRICS = ["vader", "lm_tone", "lm_negshare", "crisis_any"]


def load(name):
    d = json.load(open(os.path.join(WORK, f"{name}_daily.json")))
    out = {}
    for k, v in d.items():
        if "cw_any" in v:
            v["crisis_any"] = v["cw_any"]
        out[dt.date.fromisoformat(k)] = v
    return out


def wmean(m, key, start, length):
    vals = [m[d][key] for d in (start + dt.timedelta(i) for i in range(length)) if d in m and key in m[d]]
    return statistics.mean(vals) if len(vals) * 2 >= length else None


def z(m, key, ev, a, b):
    length = b - a + 1
    start = ev + dt.timedelta(a)
    x = wmean(m, key, start, length)
    if x is None:
        return None
    base = []
    s = ev - dt.timedelta(280)
    while s + dt.timedelta(length - 1) <= ev - dt.timedelta(30):
        if s.weekday() == start.weekday():
            w = wmean(m, key, s, length)
            if w is not None:
                base.append(w)
        s += dt.timedelta(1)
    if len(base) < 15:
        return None
    sd = statistics.pstdev(base)
    return None if sd == 0 else (x - statistics.mean(base)) / sd


def main(names, md=False):
    if md:
        print("| source | event | " + " | ".join(f"{k} {a},{b}" for k in ("vader", "lm_tone") for a, b in WINDOWS)
              + " | crisis_any 0,2 |")
        print("|---|---|" + "---|" * (2 * len(WINDOWS) + 1))
    for name in names:
        m = load(name)
        if not md:
            print(f"=== {name}: z vs same-weekday baseline windows; windows " + " ".join(f"[{a},{b}]" for a, b in WINDOWS))
        for ev in EVENTS:
            if not any(abs((d - ev).days) <= 2 for d in m):
                continue
            if md:
                cells = []
                for key in ("vader", "lm_tone"):
                    for a, b in WINDOWS:
                        v = z(m, key, ev, a, b)
                        cells.append("" if v is None else f"{v:+.2f}")
                v = z(m, "crisis_any", ev, 0, 2)
                cells.append("" if v is None else f"{v:+.2f}")
                if any(cells):
                    print(f"| {name} | {ev} | " + " | ".join(cells) + " |")
                continue
            parts = []
            for key in METRICS:
                zs = [z(m, key, ev, a, b) for a, b in WINDOWS]
                if all(v is None for v in zs):
                    continue
                parts.append(f"{key}: " + " ".join("  .  " if v is None else f"{v:+5.2f}" for v in zs))
            print(f"  {ev}  " + " | ".join(parts))


if __name__ == "__main__":
    args = sys.argv[1:]
    md = "--md" in args
    main([a for a in args if a != "--md"], md=md)
