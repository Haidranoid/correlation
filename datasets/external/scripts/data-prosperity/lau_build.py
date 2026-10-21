"""BLS LAUS -> series: county distribution statistics, selected counties, states (SA) and
cross-state statistics. Reads work/*.npz from lau_parse.py.

County data are NSA only; the county unemployment rate is seasonally adjusted here with the causal
method in sa.py (median of the same month's deviation from a centred 2x12 MA over the previous
8 years, at least 3). Dates = last day of the reference month.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402
import sa  # noqa: E402
from counties import SELECTED, STATES  # noqa: E402

K_SA = 8
LF_FLOOR = 50_000
# Release lags measured from the BLS archive pages (raw/bls_release_archives/*_lags.tsv), stock window
# 2002-2025: county data ship with the Metropolitan Area release (median 32 days after month end,
# regular maximum 54 for January 2003 after the annual benchmark; September 2013 was folded into the
# October release on 2013-12-05 = 66 days). State data: median 20, max 48 (Jan 2003), Sep 2013 = 53.
K_COUNTY = 66
K_STATE = 53
URL_CTY = "https://download.bls.gov/pub/time.series/la/la.data.64.County"
URL_ST = "https://download.bls.gov/pub/time.series/la/la.data.3.AllStatesS"
REV = ("Current vintage (files of 2026-09-30); LAUS revises the prior month each release and the previous "
       "5 years each spring (annual benchmark), so values are revised, not first prints. October 2025 was "
       "never collected (2025 shutdown) and is missing.")
SA_NOTE = (f"County LAUS is published NSA only; seasonally adjusted here causally (median of the same month's "
           f"deviation from a centred 2x12 moving average over the previous {K_SA} years, >= 3; uses only data "
           f">= 6 months older than the month itself), so the SA value is computable when the month is published.")
TIMING_C = (f"known_after_days {K_COUNTY} = maximum observed lag of the county (metro) release in 2002-2025: "
            "typical 30-35 days, Jan data ~48-54 (annual benchmark), Sep-2013 data released 2013-12-05 (66).")
TIMING_S = (f"known_after_days {K_STATE} = maximum observed lag of the BLS state release in 2002-2025: typical "
            "17-22 days, Jan data ~40-48 (benchmark), Sep-2013 data released 2013-11-22 (53).")


def mdate(mi):
    return plib.month_end(int(mi) // 12, int(mi) % 12 + 1)


def main():
    part = plib.Part("10_bls_laus")
    z = np.load(os.path.join(plib.HERE, "work", "lau_county.npz"))
    fips = list(z["fips"])
    keep = np.array([not f.startswith("72") for f in fips])  # drop Puerto Rico municipios
    fips = [f for f, k in zip(fips, keep) if k]
    names = [n for n, k in zip(z["name"], keep) if k]
    months = z["months"]
    ur, emp, lf = z["ur"][keep], z["emp"][keep], z["lf"][keep]
    print("counties (50 states + DC):", len(fips))
    ur_sa = sa.causal_sa_matrix(ur, p=12, log=False, K=K_SA)
    np.savez_compressed(os.path.join(plib.HERE, "work", "lau_county_sa.npz"), fips=np.array(fips), months=months,
                        ur_sa=ur_sa)
    # trailing 12-month mean labor force (causal) for the size floor
    lf12 = np.full_like(lf, np.nan)
    with np.errstate(all="ignore"):
        for t in range(11, len(months)):
            w = lf[:, t - 11:t + 1]
            cnt = np.sum(np.isfinite(w), axis=1)
            tot = np.nansum(w, axis=1)
            lf12[:, t] = np.where(cnt >= 6, tot / np.maximum(cnt, 1), np.nan)

    stats = {k: [] for k in ["median", "p10", "p90", "max", "min", "ratio", "sd", "rose_mom", "rose_yoy",
                             "emp_yoy_med", "emp_grew_yoy", "n"]}
    for t, mi in enumerate(months):
        d = mdate(mi)
        x = ur_sa[:, t]
        ok = np.isfinite(x)
        if ok.sum() >= 0.9 * len(fips):
            v = x[ok]
            p10, p50, p90 = np.percentile(v, [10, 50, 90])
            stats["median"].append((d, p50))
            stats["p10"].append((d, p10))
            stats["p90"].append((d, p90))
            stats["ratio"].append((d, p90 / p10))
            stats["sd"].append((d, float(np.std(v))))
            big = ok & (np.nan_to_num(lf12[:, t]) >= LF_FLOOR)
            stats["max"].append((d, float(np.max(x[big]))))
            stats["min"].append((d, float(np.min(x[big]))))
            stats["n"].append((d, int(ok.sum()), int(big.sum())))
            if t >= 1:
                ok2 = ok & np.isfinite(ur_sa[:, t - 1])
                if ok2.sum() >= 0.9 * len(fips):
                    stats["rose_mom"].append((d, 100.0 * np.mean(x[ok2] > ur_sa[ok2, t - 1])))
        if t >= 12:
            a, b = ur[:, t], ur[:, t - 12]
            ok3 = np.isfinite(a) & np.isfinite(b)
            if ok3.sum() >= 0.9 * len(fips):
                stats["rose_yoy"].append((d, 100.0 * np.mean(a[ok3] > b[ok3])))
            e1, e0 = emp[:, t], emp[:, t - 12]
            ok4 = np.isfinite(e1) & np.isfinite(e0) & (e1 > 0) & (e0 > 0)
            if ok4.sum() >= 0.9 * len(fips):
                g = 100.0 * np.log(e1[ok4] / e0[ok4])
                stats["emp_yoy_med"].append((d, float(np.median(g))))
                stats["emp_grew_yoy"].append((d, 100.0 * np.mean(e1[ok4] > e0[ok4])))
    ns = stats["n"]
    print("counties per month (all, LF>=50k): first", ns[0], "last", ns[-1])

    common = dict(subcategory="county_distribution", region="US counties (50 states + DC)",
                  source="BLS Local Area Unemployment Statistics (LAUS)", url=URL_CTY, frequency="monthly",
                  known_after_days=K_COUNTY)
    base_notes = f"{SA_NOTE} {TIMING_C} {REV}"
    n_all = f"Unweighted across all ~{len(fips)} counties with a value that month."
    part.add("lau_cty_ur_median", stats["median"], name="US counties: median unemployment rate across counties (causal SA)",
             units="percent", transform="diff", notes=f"{n_all} {base_notes}", lo=0, hi=40, **common)
    part.add("lau_cty_ur_p10", stats["p10"], name="US counties: 10th-percentile unemployment rate across counties (causal SA)",
             units="percent", transform="diff", notes=f"{n_all} {base_notes}", lo=0, hi=40, **common)
    part.add("lau_cty_ur_p90", stats["p90"], name="US counties: 90th-percentile unemployment rate across counties (causal SA)",
             units="percent", transform="diff", notes=f"{n_all} {base_notes}", lo=0, hi=40, **common)
    part.add("lau_cty_ur_p90_p10_ratio", stats["ratio"], name="US counties: p90/p10 ratio of county unemployment rates (causal SA)",
             units="ratio", transform="diff", notes=f"{n_all} {base_notes}", lo=1, hi=10, **common)
    part.add("lau_cty_ur_sd", stats["sd"], name="US counties: cross-county standard deviation of unemployment rates (causal SA)",
             units="percentage points", transform="diff", notes=f"Dispersion measure. {n_all} {base_notes}", lo=0, hi=20, **common)
    floor = (f"Over counties whose trailing 12-month average labor force is >= {LF_FLOOR:,} (about 590 counties in 2025; "
             "the unrestricted max/min are tiny-county outliers).")
    part.add("lau_cty_ur_max_lf50k", stats["max"], name="US counties: maximum county unemployment rate, counties with labor force >= 50k (causal SA)",
             units="percent", transform="diff", notes=f"{floor} {base_notes}", lo=0, hi=60, **common)
    part.add("lau_cty_ur_min_lf50k", stats["min"], name="US counties: minimum county unemployment rate, counties with labor force >= 50k (causal SA)",
             units="percent", transform="diff", notes=f"{floor} {base_notes}", lo=-2, hi=20, **common)
    part.add("lau_cty_ur_share_rose_mom", stats["rose_mom"], name="US counties: share of counties whose unemployment rate rose this month (causal SA)",
             units="percent of counties", transform="level",
             notes=("Diffusion index; already a monthly change, so transform level (the move = this month's share minus its "
                    f"trailing-year mean) instead of the brief's diff. {n_all} {base_notes}"), lo=0, hi=100, **common)
    part.add("lau_cty_ur_share_rose_yoy", stats["rose_yoy"], name="US counties: share of counties whose unemployment rate is above its level 12 months earlier (NSA)",
             units="percent of counties", transform="diff",
             notes=("Year-over-year diffusion index on the NSA rates (no seasonal adjustment needed); ties count as not "
                    f"rising. Persistent, so diff. {n_all} {TIMING_C} {REV}"), lo=0, hi=100, **common)
    part.add("lau_cty_emp_yoy_median", stats["emp_yoy_med"], name="US counties: median year-over-year employment growth across counties (household survey basis, NSA)",
             units="percent (100 x log change over 12 months)", transform="diff",
             notes=f"LAUS resident employment, 12-month log change, median across counties. {n_all} {TIMING_C} {REV}",
             lo=-30, hi=30, **common)
    part.add("lau_cty_emp_share_grew_yoy", stats["emp_grew_yoy"], name="US counties: share of counties with employment above its level 12 months earlier (NSA)",
             units="percent of counties", transform="diff",
             notes=f"Year-over-year diffusion index of LAUS resident employment. {n_all} {TIMING_C} {REV}", lo=0, hi=100, **common)

    # selected counties: causal-SA unemployment rate
    fidx = {f: i for i, f in enumerate(fips)}
    for f, slug, label, why in SELECTED:
        i = fidx[f]
        rows = [(mdate(mi), ur_sa[i, t]) for t, mi in enumerate(months) if np.isfinite(ur_sa[i, t])]
        print(f"  {f} {label!r:45} BLS: {names[i]}")
        part.add(f"lau_ur_{slug}", rows, name=f"Unemployment rate, {label} (causal SA)",
                 subcategory="county", region=f"US-{label.split(', ')[-1] if f != '11001' else 'DC'} {label} (FIPS {f})",
                 source="BLS Local Area Unemployment Statistics (LAUS)", url=URL_CTY, frequency="monthly",
                 units="percent", transform="diff", known_after_days=K_COUNTY,
                 notes=f"Selected county: {why}. BLS name: {names[i]}. {SA_NOTE} {TIMING_C} {REV}", lo=0, hi=40)

    # states: BLS seasonally adjusted unemployment rate
    s = np.load(os.path.join(plib.HERE, "work", "lau_state_sa.npz"))
    sm = s["months"]
    sf = [f[:2] for f in s["fips"]]
    sur = s["ur"]
    st_rows = [i for i, f in enumerate(sf) if f in STATES]
    assert len(st_rows) == 51, len(st_rows)
    for i in st_rows:
        ab, nm = STATES[sf[i]]
        rows = [(mdate(mi), sur[i, t]) for t, mi in enumerate(sm) if np.isfinite(sur[i, t])]
        part.add(f"lau_ur_st_{ab.lower()}", rows, name=f"Unemployment rate, {nm} (BLS seasonally adjusted)",
                 subcategory="state", region=f"US-{ab}", source="BLS Local Area Unemployment Statistics (LAUS), states",
                 url=URL_ST, frequency="monthly", units="percent", transform="diff", known_after_days=K_STATE,
                 notes=f"BLS model-based, seasonally adjusted and smoothed. {TIMING_S} {REV}", lo=0, hi=35)
    S = sur[st_rows]
    srows = {k: [] for k in ["median", "max", "min", "sd", "rose"]}
    for t, mi in enumerate(sm):
        x = S[:, t]
        if np.all(np.isfinite(x)):
            d = mdate(mi)
            srows["median"].append((d, float(np.median(x))))
            srows["max"].append((d, float(np.max(x))))
            srows["min"].append((d, float(np.min(x))))
            srows["sd"].append((d, float(np.std(x))))
            if t >= 1 and np.all(np.isfinite(S[:, t - 1])):
                srows["rose"].append((d, 100.0 * np.mean(x > S[:, t - 1])))
    cs = dict(subcategory="state_distribution", region="US states (50 + DC)",
              source="BLS Local Area Unemployment Statistics (LAUS), states", url=URL_ST, frequency="monthly",
              known_after_days=K_STATE)
    sn = f"Across the 50 states + DC, BLS seasonally adjusted rates. {TIMING_S} {REV}"
    part.add("lau_st_ur_median", srows["median"], name="US states: median state unemployment rate", units="percent",
             transform="diff", notes=sn, lo=0, hi=30, **cs)
    part.add("lau_st_ur_max", srows["max"], name="US states: maximum state unemployment rate", units="percent",
             transform="diff", notes=sn, lo=0, hi=35, **cs)
    part.add("lau_st_ur_min", srows["min"], name="US states: minimum state unemployment rate", units="percent",
             transform="diff", notes=sn, lo=0, hi=20, **cs)
    part.add("lau_st_ur_sd", srows["sd"], name="US states: cross-state standard deviation of unemployment rates",
             units="percentage points", transform="diff", notes=f"Dispersion measure. {sn}", lo=0, hi=10, **cs)
    part.add("lau_st_ur_share_rose_mom", srows["rose"], name="US states: share of states whose unemployment rate rose this month",
             units="percent of states", transform="level",
             notes=f"Diffusion index (already a change), so transform level, not the brief's diff; ties (common at 0.1 pp resolution) count as not rising. {sn}",
             lo=0, hi=100, **cs)
    part.save()


if __name__ == "__main__":
    main()
