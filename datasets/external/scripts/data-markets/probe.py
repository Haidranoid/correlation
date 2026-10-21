"""Timing probe: does a daily series' day-d value contain information from after the US close of day d?

For each daily series x, resample to US trading days (last observation on/before each SPY date, max 5 days old),
compute moves, and over 2005-01-03..2025-06-27 report:
  c0  = corr(x_d, spy_d)          same day
  cm1 = corr(x_d, spy_{d-1})      x reacts to the previous US session (typical for closes before the US close)
  pc1 = partial corr(x_d, spy_{d+1} | spy_d)   look-ahead indicator (should be ~0 for kad 0)
and the same against FRED DGS10 changes for rate series. Writes probe_results.tsv in the work dir.
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import dmlib  # noqa: E402

LO, HI = "2005-01-03", "2025-06-27"


def load(sid):
    with open(os.path.join(dmlib.SERIES, sid + ".csv")) as f:
        r = list(csv.reader(f))[1:]
    return [x[0] for x in r], np.array([float(x[1]) for x in r])


def to_days(dates, vals, days):
    """Last value on/before each target day (within 5 calendar days)."""
    out = np.full(len(days), np.nan)
    dd = np.array(dates, dtype="datetime64[D]")
    td = np.array(days, dtype="datetime64[D]")
    idx = np.searchsorted(dd, td, side="right") - 1
    ok = idx >= 0
    age = np.where(ok, (td - dd[np.clip(idx, 0, None)]).astype(int), 999)
    good = ok & (age <= 5)
    out[good] = vals[idx[good]]
    return out


def moves(v, tr):
    if tr == "logdiff":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.diff(np.log(v))
    return np.diff(v)


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 200:
        return float("nan"), int(m.sum())
    a, b = a[m], b[m]
    if a.std() == 0 or b.std() == 0:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(m.sum())


def partial(x, y, z):
    """corr(x, y | z) via residuals."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 200:
        return float("nan")
    x, y, z = x[m], y[m], z[m]
    def res(a):
        A = np.vstack([z, np.ones_like(z)]).T
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta
    rx, ry = res(x), res(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    cat = list(csv.DictReader(open(dmlib.CATALOG)))
    sd, sv = load("etf_spy")
    days = [d for d in sd if LO <= d <= HI]
    spy = moves(to_days(sd, sv, days), "logdiff")  # spy[i] = move ending days[i+1]
    gd, gv = load("fred_dgs10")
    g10 = moves(to_days(gd, gv, days), "diff")
    out = []
    for r in cat:
        if r["frequency"] != "daily":
            continue
        d, v = load(r["id"])
        x = moves(to_days(d, v, days), r["transform"])
        # x[i] and spy[i] are moves ending on days[i+1]
        c0, n = corr(x, spy)
        cm1, _ = corr(x[1:], spy[:-1])
        pc1 = partial(x[:-1], spy[1:], spy[:-1])
        g0, _ = corr(x, g10)
        gm1, _ = corr(x[1:], g10[:-1])
        gp1 = partial(x[:-1], g10[1:], g10[:-1])
        out.append((r["id"], r["category"], r["known_after_days"], n, c0, cm1, pc1, g0, gm1, gp1))
    path = os.path.join(dmlib.WORK, "probe_results.tsv")
    with open(path, "w") as f:
        f.write("id\tcategory\tkad\tn\tspy_c0\tspy_cm1\tspy_pc1\tdgs10_c0\tdgs10_cm1\tdgs10_pc1\n")
        for o in out:
            f.write("\t".join([o[0], o[1], str(o[2]), str(o[3])] + [f"{x:+.3f}" if not math.isnan(x) else "nan" for x in o[4:]]) + "\n")
    print(f"{'id':<32}{'cat':<13}{'kad':>3}{'n':>6}  spy:c0   cm1   pc1 | dgs10:c0  cm1   pc1")
    for o in out:
        flag = ""
        if o[2] in ("0", 0) and ((not math.isnan(o[6]) and o[6] > 0.05) or (o[1] in ("rates",) and not math.isnan(o[9]) and o[9] > 0.05)):
            flag = "  <-- look-ahead?"
        print(f"{o[0]:<32}{o[1]:<13}{o[2]:>3}{o[3]:>6}  {o[4]:+.2f} {o[5]:+.2f} {o[6]:+.3f} | {o[7]:+.2f} {o[8]:+.2f} {o[9]:+.3f}{flag}")


if __name__ == "__main__":
    main()
