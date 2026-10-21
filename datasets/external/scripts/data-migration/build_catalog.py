"""Build catalog_data-migration.csv from meta/*.json, re-checking every normalized file, and write a QC
report (qc_data-migration.txt) with ranges, gaps, first/last rows and an estimate of which correlation
periods each series can join under the external test's rules (a re-implementation of its per-series
construction: transform, availability, live gap, gap bridging, trailing standardization, coverage)."""
import bisect
import calendar
import csv
import datetime as dt
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXT, SERIES, META

COLS = ["id", "name", "category", "subcategory", "region", "source", "url", "frequency", "units",
        "transform", "known_after_days", "first", "last", "rows", "notes"]
CAT = os.path.join(EXT, "catalog_data-migration.csv")
QC = os.path.join(EXT, "qc_data-migration.txt")

# ---------------------------------------------------------------- trading calendar and periods
cal = [dt.date.fromisoformat(r["date"]) for r in csv.DictReader(open(os.path.join(SERIES, "etf_qqq.csv")))]
DAYS = [d.toordinal() for d in cal]
PERIODS = {}
for r in csv.DictReader(open(os.path.expanduser("~/.local/share/correlation2/census/main-2025-06-27/windows.csv"))):
    if r["id"] in ("222", "242", "262", "282", "284"):
        a, b = dt.date.fromisoformat(r["first"]).toordinal(), dt.date.fromisoformat(r["last"]).toordinal()
        PERIODS[r["id"]] = (bisect.bisect_left(DAYS, a), bisect.bisect_right(DAYS, b))
LOOKBACK, MIN_OBS = 252, 126


def live_gap(freq, obs):
    named = {"daily": 10, "weekly": 15, "monthly": 50, "quarterly": 140, "annual": 300}.get(freq, 0)
    gaps = sorted(b[0] - a[0] for a, b in zip(obs[:100], obs[1:101]))
    if not gaps:
        inferred = 300
    else:
        g = gaps[len(gaps) // 2]
        inferred = 10 if g <= 4 else 15 if g <= 10 else 50 if g <= 45 else 140 if g <= 120 else 300
    return max(named, inferred)


def bridged(obs):
    out = [True] * len(obs)
    win = []
    for i in range(1, len(obs)):
        gap = obs[i][0] - obs[i - 1][0]
        if len(win) >= 20:
            s = sorted(win)
            p90 = s[min(len(s) * 9 // 10, len(s) - 1)]
            out[i] = gap <= (1 if p90 <= 1 else p90 + 2)
        win.append(gap)
        if len(win) > 250:
            win.pop(0)
    return out


def anomalies(obs, log1p):
    xs = {}
    for d, v in obs:
        if log1p:
            if v <= -1:
                continue
            xs[d] = math.log1p(v)
        else:
            xs[d] = v
    out = []
    for d, x in sorted(xs.items()):
        dd = dt.date.fromordinal(d)
        s, c, years = 0.0, 0, 0
        for yy in range(dd.year - 10, dd.year):
            centre = dt.date(yy, dd.month, min(dd.day, calendar.monthrange(yy, dd.month)[1])).toordinal()
            any_ = False
            for o in range(-7, 8):
                v = xs.get(centre + o)
                if v is not None:
                    s += v
                    c += 1
                    any_ = True
            years += any_
        if years >= 3:
            out.append((d, x - s / c))
    return out


def build(meta, obs_raw):
    tf, k, T = meta["transform"], meta["known_after_days"], len(DAYS)
    if tf == "logdiff":
        obs = [(d, math.log(v)) for d, v in obs_raw if v > 0]
    elif tf in ("diff", "level"):
        obs = list(obs_raw)
    else:
        obs = anomalies(obs_raw, tf == "anomaly_log1p")
    gap = live_gap(meta["frequency"], obs_raw)
    avail = [bisect.bisect_left(DAYS, d + k) for d, _ in obs]
    br = bridged(obs)
    x = [math.nan] * T
    raw = [math.nan] * T
    i, level, last_date, last = 0, None, None, None
    trail = []
    for row in range(T):
        sm, cnt, newest, broken = 0.0, 0, math.nan, False
        while i < len(obs) and avail[i] <= row:
            broken |= last_date is not None and not br[i]
            last_date = obs[i][0]
            sm += obs[i][1]
            cnt += 1
            newest = obs[i][1]
            i += 1
        if cnt:
            last = row
        live = row > 0 and last is not None and row - last <= gap
        while trail and trail[0][0] + LOOKBACK < row:
            trail.pop(0)
        if tf in ("logdiff", "diff"):
            if cnt:
                if level is not None and row > 0 and not broken:
                    x[row] = newest - level
                level = newest
            elif live and level is not None:
                x[row] = 0.0
            raw[row] = x[row]
        elif tf in ("anomaly", "anomaly_log1p"):
            if cnt and row > 0:
                x[row] = sm / cnt
            elif live:
                x[row] = 0.0
            raw[row] = x[row]
        else:
            enough = len(trail) >= 10
            if cnt:
                val = sm / cnt
                if row > 0 and enough:
                    x[row] = val - sum(t[1] for t in trail) / len(trail)
                    raw[row] = val
                trail.append((row, val))
            elif live and enough:
                x[row] = 0.0
                raw[row] = 0.0
    # standardization: finite where trailing 252 rows hold >= 126 finite values and >= 1 non-zero
    v = [False] * T
    fin = [0] * (T + 1)
    nz = [0] * (T + 1)
    for t in range(T):
        f = not math.isnan(x[t])
        fin[t + 1] = fin[t] + f
        nz[t + 1] = nz[t] + (f and x[t] != 0.0)
    for t in range(T):
        if math.isnan(x[t]):
            continue
        a = max(0, t - LOOKBACK)
        if fin[t] - fin[a] >= MIN_OBS and nz[t] - nz[a] >= 1:
            v[t] = True
    res = {}
    for pid, (a, b) in PERIODS.items():
        n = b - a
        have = sum(v[a:b])
        active = sum(1 for t in range(a, b) if not math.isnan(raw[t]) and raw[t] != 0.0)
        res[pid] = (have / n, active, have >= 0.9 * n and active >= 10)
    return res


# ---------------------------------------------------------------- other catalogs (id clashes)
other_ids = {}
for f in glob.glob(os.path.join(EXT, "catalog_*.csv")):
    if f == CAT:
        continue
    for r in csv.DictReader(open(f, newline="")):
        other_ids[r["id"]] = os.path.basename(f)

FLAG_WORDS = ["finite-life", "finite life", "single-contract", "single contract"]
metas = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(META, "*.json")))]
qc_lines, cat_rows, problems = [], [], []
summary = {}
for m in metas:
    sid = m["id"]
    path = os.path.join(SERIES, f"{sid}.csv")
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    if lines[0] != "date,value":
        problems.append(f"{sid}: bad header {lines[0]!r}")
    obs, prev = [], None
    for ln in lines[1:]:
        d, v = ln.split(",")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) or v.strip() == "":
            problems.append(f"{sid}: bad row {ln!r}")
            continue
        dd = dt.date.fromisoformat(d)
        fv = float(v)
        if not math.isfinite(fv):
            problems.append(f"{sid}: non-finite {ln!r}")
        if prev is not None and dd <= prev:
            problems.append(f"{sid}: dates not strictly ascending at {d}")
        prev = dd
        obs.append((dd.toordinal(), fv))
    first, last, rows = dt.date.fromordinal(obs[0][0]).isoformat(), dt.date.fromordinal(obs[-1][0]).isoformat(), len(obs)
    if (first, last, rows) != (m["first"], m["last"], m["rows"]):
        problems.append(f"{sid}: meta/file mismatch {first, last, rows} vs {m['first'], m['last'], m['rows']}")
    if sid in other_ids:
        problems.append(f"{sid}: id clash with {other_ids[sid]}")
    low = m["notes"].lower()
    for w in FLAG_WORDS:
        for mm in re.finditer(re.escape(w), low):
            before = low[:mm.start()].rstrip()
            if not (before.endswith("not a") or before.endswith("not") or before.endswith("no")):
                problems.append(f"{sid}: notes contain harness flag phrase {w!r}")
    if any(c in m["notes"] for c in "\n\r"):
        problems.append(f"{sid}: newline in notes")
    el = build(m, obs)
    joins = [p for p in ("222", "242", "262", "282", "284") if el[p][2]]
    summary[sid] = (m["frequency"], m["category"], joins)
    vals = [v for _, v in obs]
    head = ", ".join(f"{dt.date.fromordinal(d)}={v:g}" for d, v in obs[:3])
    tail = ", ".join(f"{dt.date.fromordinal(d)}={v:g}" for d, v in obs[-3:])
    qc_lines.append(
        f"{sid} [{m['category']}/{m['frequency']}/{m['transform']}/k={m['known_after_days']}] rows={rows} "
        f"{first}..{last} min={min(vals):g} max={max(vals):g} zeros={sum(1 for v in vals if v == 0)} "
        f"neg={sum(1 for v in vals if v < 0)} gaps={m['n_gaps']}{' ' + str(m['gaps'][:3]) if m['n_gaps'] else ''}\n"
        f"    first: {head}\n    last:  {tail}\n"
        "    est. coverage/active/joins: " + "  ".join(
            f"{p}:{100 * el[p][0]:.0f}%/{el[p][1]}/{'Y' if el[p][2] else '-'}" for p in ("222", "242", "262", "282", "284")))
    row = {c: m.get(c, "") for c in COLS}
    row["first"], row["last"], row["rows"] = first, last, rows
    cat_rows.append(row)

cat_rows.sort(key=lambda r: r["id"])
with open(CAT + ".part", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in cat_rows:
        w.writerow(r)
os.replace(CAT + ".part", CAT)

from collections import Counter
byfreq = Counter(v[0] for v in summary.values())
bycat = Counter((v[1], v[0]) for v in summary.values())
joined_any = Counter(v[0] for v in summary.values() if v[2])
full_only = Counter(v[0] for v in summary.values() if v[2] == ["284"])
none = [s for s, v in summary.items() if not v[2]]
head = [f"QC for data-migration, {dt.datetime.now().isoformat(timespec='seconds')}",
        f"series: {len(metas)}  by frequency: {dict(byfreq)}",
        f"by category/frequency: {dict(sorted(bycat.items()))}",
        "estimated test eligibility (periods 222=2005-09, 242=2010-14, 262=2015-19, 282=2020-24, 284=2005-2025 full):",
        f"  join at least one period: {dict(joined_any)}; join only the full-history period 284: {dict(full_only)}",
        f"  join no period ({len(none)}): {', '.join(none) if none else '-'}",
        f"problems: {len(problems)}"] + [f"  {p}" for p in problems] + [""]
with open(QC, "w") as f:
    f.write("\n".join(head + qc_lines) + "\n")
print("\n".join(head))
print(f"catalog: {CAT} ({len(cat_rows)} rows)\nqc: {QC}")
