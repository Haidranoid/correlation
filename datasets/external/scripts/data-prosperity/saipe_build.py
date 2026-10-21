"""Census SAIPE county median household income and poverty rate (est{yy}all files, 1989, 1993, 1995,
1997-2024) -> cross-county distribution statistics (annual, dated December 31 of the income year).
Whitespace tokens: 0 state, 1 county, 2 poverty count (all ages), 5 poverty rate (all ages),
20 median household income; '.' = missing (Kalawao County, HI)."""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402

RAWD = os.path.join(plib.RAW, "census_saipe")
YEARS = [1989, 1993, 1995] + list(range(1997, 2025))
K = 1100
POP_FLOOR = 100_000


def fname(y):
    yy = f"{y % 100:02d}"
    return f"est{yy}all.dat" if y <= 2003 else f"est{yy}all.txt"


def num(t):
    try:
        return float(t)
    except ValueError:
        return None


def main():
    part = plib.Part("70_census_saipe")
    st = {k: [] for k in ["median", "p10", "p90", "min", "max", "ratio", "pov_median", "pov_p90"]}
    for y in YEARS:
        mhi, pov, big = [], [], []
        stamp = None
        with open(os.path.join(RAWD, fname(y)), encoding="latin-1") as f:
            for line in f:
                t = line.split()
                if len(t) < 25:
                    continue
                if t[0] == "00" and t[1] == "0":
                    stamp = t[-1]
                if t[1] == "0":
                    continue  # US and state rows
                m, p, n = num(t[20]), num(t[5]), num(t[2])
                if m is not None and m > 0:
                    mhi.append(m)
                    if p is not None and p > 0 and n is not None and n / (p / 100.0) >= POP_FLOOR:
                        big.append(m)
                if p is not None:
                    pov.append(p)
        d = dt.date(y, 12, 31)
        x = np.array(mhi)
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        st["median"].append((d, p50))
        st["p10"].append((d, p10))
        st["p90"].append((d, p90))
        st["ratio"].append((d, p90 / p10))
        st["min"].append((d, float(min(big))))
        st["max"].append((d, float(max(big))))
        st["pov_median"].append((d, float(np.median(pov))))
        st["pov_p90"].append((d, float(np.percentile(pov, 90))))
        print(y, fname(y), "counties", len(x), "big", len(big), "median MHI", round(p50), "median poverty", np.median(pov), "file stamp", stamp)
    notes = (f"known_after_days {K}: SAIPE lags were long in the 2000s (file stamps: 2000 data 2003-10-09; 2001-02 data "
             "2004-10-29; 2003 data revised 2006-09-21; 2004-06 data ~700 days), and since 2008 ~11-12 months (December "
             "of t+1; 2024 data 2026-01-27). One bound per series, so it is very conservative after 2008. Gaps: no county "
             "estimates for 1990-92, 1994, 1996. Method break: CPS-based to 2004, ACS-based from 2005 (model inputs). "
             "Connecticut = 8 counties to 2021, 9 planning regions from 2022. Not revised after release except 2003. "
             "ANNUAL: essentially untestable at the census move lengths; included because the user asked. US Census "
             "Bureau data, public domain.")
    common = dict(subcategory="county_distribution", region="US counties (SAIPE, 50 states + DC)",
                  source="US Census Bureau Small Area Income and Poverty Estimates (SAIPE)",
                  url="https://www2.census.gov/programs-surveys/saipe/datasets/", frequency="annual", known_after_days=K)
    fl = f"Over counties with an estimated population >= {POP_FLOOR:,} (poverty count / poverty rate). "
    part.add("saipe_cty_mhi_median", st["median"], name="US counties: median of county median household income (annual)",
             units="USD (nominal)", transform="logdiff", notes=notes, lo=5000, hi=300_000, **common)
    part.add("saipe_cty_mhi_p10", st["p10"], name="US counties: 10th percentile of county median household income (annual)",
             units="USD (nominal)", transform="logdiff", notes=notes, lo=5000, hi=300_000, **common)
    part.add("saipe_cty_mhi_p90", st["p90"], name="US counties: 90th percentile of county median household income (annual)",
             units="USD (nominal)", transform="logdiff", notes=notes, lo=5000, hi=300_000, **common)
    part.add("saipe_cty_mhi_min_pop100k", st["min"], name="US counties: minimum county median household income, counties with population >= 100k (annual)",
             units="USD (nominal)", transform="logdiff", notes=fl + notes, lo=5000, hi=300_000, **common)
    part.add("saipe_cty_mhi_max_pop100k", st["max"], name="US counties: maximum county median household income, counties with population >= 100k (annual)",
             units="USD (nominal)", transform="logdiff", notes=fl + notes, lo=5000, hi=500_000, **common)
    part.add("saipe_cty_mhi_p90_p10_ratio", st["ratio"], name="US counties: p90/p10 ratio of county median household income (annual)",
             units="ratio", transform="diff", notes=notes, lo=1, hi=10, **common)
    part.add("saipe_cty_poverty_median", st["pov_median"], name="US counties: median county poverty rate, all ages (annual)",
             units="percent of people", transform="diff", notes=notes, lo=0, hi=60, **common)
    part.add("saipe_cty_poverty_p90", st["pov_p90"], name="US counties: 90th-percentile county poverty rate, all ages (annual)",
             units="percent of people", transform="diff", notes=notes, lo=0, hi=70, **common)
    part.save()


if __name__ == "__main__":
    main()
