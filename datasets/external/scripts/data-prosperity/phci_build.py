"""Philadelphia Fed state coincident indexes (FRED {ST}PHCI, keyless fredgraph.csv) -> 50 state
series and cross-state statistics. FRED dates the 1st of the month; re-dated to month end.
DC has no coincident index (FRED DCPHCI does not exist), so the state set is the 50 states.
"""
import csv
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402
from counties import STATES  # noqa: E402

RAWD = os.path.join(plib.RAW, "fred_phci")
K = 76
SRC = "Federal Reserve Bank of Philadelphia, State Coincident Indexes (via FRED)"
TIMING = ("known_after_days 76 = maximum lag verified from ALFRED vintages for every month 2008-11..2025-06, plus 1 day because the time of day of the release is unknown (typical "
          "22-30 days; January data wait for the BLS benchmark: 41-49 days in 2009-14, 56-75 since 2015, max Jan-2015 "
          "out 2015-04-16 = 75; Sep-2013 shutdown 57). ALFRED's release list bounds 2005-2008 below 75.")
REV = ("Model (dynamic factor on payrolls, unemployment rate, manufacturing hours, real wages) re-estimated every month, "
       "with annual benchmark revisions, so the whole history is revised; first published 2005-06-02, values before "
       "then are back-casts. FRED notes: copyrighted, citation required (Federal Reserve Bank of Philadelphia).")


def load(ab):
    rows = []
    with open(os.path.join(RAWD, f"{ab}PHCI.csv"), newline="") as f:
        r = csv.reader(f)
        next(r)
        for d, v in r:
            if v.strip() in ("", "."):
                continue
            y, m = int(d[:4]), int(d[5:7])
            rows.append((plib.month_end(y, m), float(v)))
    return rows


def main():
    part = plib.Part("20_philfed_coincident")
    series = {}
    for fips, (ab, nm) in STATES.items():
        if ab == "DC":
            continue
        rows = load(ab)
        series[ab] = dict(rows)
        part.add(f"phci_{ab.lower()}", rows, name=f"State coincident economic activity index, {nm}", subcategory="state",
                 region=f"US-{ab}", source=SRC, url=f"https://fred.stlouisfed.org/series/{ab}PHCI", frequency="monthly",
                 units="index 2007=100 (seasonally adjusted)", transform="logdiff", known_after_days=K,
                 notes=f"Date = last day of the month (FRED's first-of-month date moved). {TIMING} {REV}", lo=10, hi=400)
    assert len(series) == 50
    dates = sorted(set.intersection(*[set(s) for s in series.values()]))
    st = {k: [] for k in ["med", "p10", "p90", "rose"]}
    for a, b in zip(dates, dates[1:]):
        if (b.year * 12 + b.month) - (a.year * 12 + a.month) != 1:
            continue
        g = np.array([100.0 * np.log(series[s][b] / series[s][a]) for s in series])
        p10, p50, p90 = np.percentile(g, [10, 50, 90])
        st["med"].append((b, p50))
        st["p10"].append((b, p10))
        st["p90"].append((b, p90))
        st["rose"].append((b, 100.0 * float(np.mean(g > 0))))
    common = dict(subcategory="state_distribution", region="US states (50)", source=SRC,
                  url="https://fred.stlouisfed.org/release?rid=109", frequency="monthly", known_after_days=K)
    gn = (f"Monthly growth of each state's coincident index; already a change, so transform level (move = this month minus "
          f"its trailing-year mean) rather than the brief's diff. {TIMING} {REV}")
    part.add("phci_states_mom_median", st["med"], name="US states: median monthly growth of state coincident indexes",
             units="percent (100 x log change, month over month)", transform="level", notes=gn, lo=-20, hi=10, **common)
    part.add("phci_states_mom_p10", st["p10"], name="US states: 10th-percentile monthly growth of state coincident indexes",
             units="percent (100 x log change, month over month)", transform="level", notes=gn, lo=-50, hi=10, **common)
    part.add("phci_states_mom_p90", st["p90"], name="US states: 90th-percentile monthly growth of state coincident indexes",
             units="percent (100 x log change, month over month)", transform="level", notes=gn, lo=-20, hi=15, **common)
    part.add("phci_states_share_rising", st["rose"], name="US states: share of the 50 states whose coincident index rose this month (diffusion)",
             units="percent of states", transform="level",
             notes=f"Like the Philadelphia Fed's own one-month diffusion index (which nets rising minus falling). {gn}",
             lo=0, hi=100, **common)
    part.save()


if __name__ == "__main__":
    main()
