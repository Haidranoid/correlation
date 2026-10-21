"""Zillow county ZHVI (raw = not smoothed, not seasonally adjusted, mid-tier all homes) and ZORI
(smoothed, all homes + multifamily) -> selected counties and cross-county distribution series.

ZHVI is seasonally adjusted here with the causal method in sa.py (log, K=8). Distribution
statistics use a fixed panel: counties that already had a ZHVI value in January 2000 (membership
known at that time; ~940 mostly larger counties), so entries of small counties over time don't
move the statistics. Dates are Zillow's (month end).
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402
import sa  # noqa: E402
from counties import SELECTED  # noqa: E402

RAWD = os.path.join(plib.RAW, "zillow")
F_ZHVI = "County_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"
F_ZORI = "County_zori_uc_sfrcondomfr_sm_month.csv"
URL_ZHVI = "https://files.zillowstatic.com/research/public_csvs/zhvi/" + F_ZHVI
URL_ZORI = "https://files.zillowstatic.com/research/public_csvs/zori/" + F_ZORI
K = 35
LF_FLOOR = 50_000
TIMING = (f"known_after_days {K}: Zillow posts the month's data around the 16th of the next month (this file: "
          "2026-10-16 for September); earlier years' schedule was later in the month, so 35 is a conservative bound.")
REV = ("Revised and recomputed every month (the whole history can change; ZHVI methodology was overhauled in "
       "2019 and its history recomputed), so pre-2019 values are a later reconstruction, not what was known then. "
       "Data (c) Zillow, free for public use with attribution: 'Data provided by Zillow Research'.")
SA_NOTE = ("Raw (unsmoothed, NSA) ZHVI seasonally adjusted here causally (median of the same month's log deviation "
           "from a centred 2x12 moving average over the previous 8 years, >= 3; only data >= 6 months older).")


def load(fn):
    with open(os.path.join(RAWD, fn), newline="") as f:
        r = list(csv.reader(f))
    h = r[0]
    dates = h[9:]
    fips, names, X = [], [], []
    for x in r[1:]:
        if x[3] != "county":
            continue
        fips.append(x[7].zfill(2) + x[8].zfill(3))
        names.append(f"{x[2]}, {x[5]}")
        X.append([float(v) if v.strip() else np.nan for v in x[9:]])
    return dates, fips, names, np.array(X)


def lf12_by_fips(dates):
    """Trailing 12-month mean LAUS labor force per county, aligned to Zillow month-end dates."""
    z = np.load(os.path.join(plib.HERE, "work", "lau_county.npz"))
    lf, months = z["lf"], z["months"]
    out = {}
    midx = {int(m): i for i, m in enumerate(months)}
    # The floor for Zillow month t uses the 12 LAUS months ending at t-3: LAUS month t-3 is public ~66 days after it
    # ends, i.e. ~25 days before month t ends, so membership is known before the Zillow value (no look-ahead).
    want = [int(d[:4]) * 12 + int(d[5:7]) - 1 - 3 for d in dates]
    for f, row in zip(z["fips"], lf):
        v = np.full(len(dates), np.nan)
        for j, mi in enumerate(want):
            i = midx.get(mi)
            if i is not None and i >= 11:
                w = row[i - 11:i + 1]
                if np.sum(np.isfinite(w)) >= 6:
                    v[j] = np.nanmean(w)
        out[str(f)] = v
    return out


def main():
    part = plib.Part("30_zillow")
    dates, fips, names, X = load(F_ZHVI)
    print("ZHVI counties", len(fips), dates[0], dates[-1])
    SAX = sa.causal_sa_matrix(X, p=12, log=True, K=8)
    j2000 = dates.index("2000-01-31")
    panel = np.isfinite(X[:, j2000])
    print("panel (value in 2000-01):", int(panel.sum()))
    lf12 = lf12_by_fips(dates)
    LF = np.array([lf12.get(f, np.full(len(dates), np.nan)) for f in fips])

    st = {k: [] for k in ["med", "ratio", "g_med", "g_p10", "g_p90", "g_max", "g_min", "rose", "yoy_med"]}
    P = SAX[panel]
    Praw = X[panel]
    PLF = LF[panel]
    npan = int(panel.sum())
    for t, d in enumerate(dates):
        x = P[:, t]
        ok = np.isfinite(x)
        if ok.sum() >= 0.9 * npan:
            p10, p50, p90 = np.percentile(x[ok], [10, 50, 90])
            st["med"].append((d, p50))
            st["ratio"].append((d, p90 / p10))
        if t >= 1:
            g = 100.0 * np.log(P[:, t] / P[:, t - 1])
            okg = np.isfinite(g)
            if okg.sum() >= 0.9 * npan:
                gp = g[okg]
                a10, a50, a90 = np.percentile(gp, [10, 50, 90])
                st["g_med"].append((d, a50))
                st["g_p10"].append((d, a10))
                st["g_p90"].append((d, a90))
                st["rose"].append((d, 100.0 * np.mean(gp > 0)))
                big = okg & (np.nan_to_num(PLF[:, t]) >= LF_FLOOR)
                if big.sum() >= 100:
                    st["g_max"].append((d, float(np.max(g[big]))))
                    st["g_min"].append((d, float(np.min(g[big]))))
        if t >= 12:
            y = 100.0 * np.log(Praw[:, t] / Praw[:, t - 12])
            oky = np.isfinite(y)
            if oky.sum() >= 0.9 * npan:
                st["yoy_med"].append((d, float(np.median(y[oky]))))
    common = dict(subcategory="county_distribution", region=f"US counties (Zillow panel of {npan} counties with data since 2000-01)",
                  source="Zillow Home Value Index (ZHVI), county, raw", url=URL_ZHVI, frequency="monthly", known_after_days=K)
    nb = f"{SA_NOTE} {TIMING} {REV}"
    part.add("zillow_cty_zhvi_median", st["med"], name="US counties: median home value (ZHVI) across counties (causal SA)",
             units="USD", transform="logdiff", notes=f"Typical (35th-65th percentile) home value per county; median across the panel. {nb}",
             lo=10_000, hi=2_000_000, **common)
    part.add("zillow_cty_zhvi_p90_p10_ratio", st["ratio"], name="US counties: p90/p10 ratio of county home values (ZHVI, causal SA)",
             units="ratio", transform="diff", notes=f"Spread between expensive and cheap counties. {nb}", lo=1, hi=20, **common)
    gnote = ("Month-over-month change of each county's causal-SA ZHVI; already a change, so transform level (move = this month "
             "minus its trailing-year mean) rather than the brief's diff. ")
    part.add("zillow_cty_zhvi_mom_median", st["g_med"], name="US counties: median monthly home-value growth across counties (ZHVI, causal SA)",
             units="percent (100 x log change, month over month)", transform="level", notes=gnote + nb, lo=-10, hi=10, **common)
    part.add("zillow_cty_zhvi_mom_p10", st["g_p10"], name="US counties: 10th-percentile monthly home-value growth across counties (ZHVI, causal SA)",
             units="percent (100 x log change, month over month)", transform="level", notes=gnote + nb, lo=-15, hi=10, **common)
    part.add("zillow_cty_zhvi_mom_p90", st["g_p90"], name="US counties: 90th-percentile monthly home-value growth across counties (ZHVI, causal SA)",
             units="percent (100 x log change, month over month)", transform="level", notes=gnote + nb, lo=-10, hi=15, **common)
    fl = (f"Over panel counties whose LAUS labor force averaged >= {LF_FLOOR:,} over the 12 months ending 3 months earlier "
          "(about 450 counties; lagged so it is public before the Zillow month; unrestricted extremes are tiny-county noise). ")
    part.add("zillow_cty_zhvi_mom_max_lf50k", st["g_max"], name="US counties: maximum monthly home-value growth, counties with labor force >= 50k (ZHVI, causal SA)",
             units="percent (100 x log change, month over month)", transform="level", notes=fl + gnote + nb, lo=-10, hi=25, **common)
    part.add("zillow_cty_zhvi_mom_min_lf50k", st["g_min"], name="US counties: minimum monthly home-value growth, counties with labor force >= 50k (ZHVI, causal SA)",
             units="percent (100 x log change, month over month)", transform="level", notes=fl + gnote + nb, lo=-25, hi=10, **common)
    part.add("zillow_cty_zhvi_share_rose_mom", st["rose"], name="US counties: share of counties whose home value rose this month (ZHVI, causal SA)",
             units="percent of counties", transform="level", notes="Diffusion index (already a change), so transform level. " + nb,
             lo=0, hi=100, **common)
    part.add("zillow_cty_zhvi_yoy_median", st["yoy_med"], name="US counties: median year-over-year home-value growth across counties (ZHVI, raw)",
             units="percent (100 x log change over 12 months)", transform="diff",
             notes=f"12-month change of the raw ZHVI (no seasonal adjustment needed); persistent, so diff. {TIMING} {REV}",
             lo=-40, hi=40, **common)

    # selected counties
    fidx = {f: i for i, f in enumerate(fips)}
    for f, slug, label, why in SELECTED:
        zf = "09001" if f == "09190" else f
        if zf not in fidx:
            print("  no ZHVI for", f, label)
            continue
        i = fidx[zf]
        rows = [(dates[t], SAX[i, t]) for t in range(len(dates)) if np.isfinite(SAX[i, t])]
        geo = (" Zillow still publishes the old Fairfield County (09001), which contains the Western Connecticut and "
               "Greater Bridgeport planning regions used by BLS and BEA.") if f == "09190" else ""
        lbl = "Fairfield County, CT" if f == "09190" else label
        st_ab = lbl.split(", ")[-1] if f != "11001" else "DC"
        part.add(f"zillow_zhvi_{'ct_fairfield' if f == '09190' else slug}", rows, name=f"Home value index (ZHVI), {lbl} (causal SA)",
                 subcategory="county", region=f"US-{st_ab} {lbl} (FIPS {zf})", source="Zillow Home Value Index (ZHVI), county, raw",
                 url=URL_ZHVI, frequency="monthly", units="USD (typical home value, 35th-65th percentile)", transform="logdiff",
                 known_after_days=K, notes=f"Selected county: {why}. Zillow name: {names[i]}.{geo} {SA_NOTE} {TIMING} {REV}",
                 lo=10_000, hi=5_000_000)
        print(f"  {f} {lbl:42} first SA {rows[0][0]}  last {rows[-1][0]}  n={len(rows)}  Zillow: {names[i]}")

    # ZORI: median year-over-year rent growth across a fixed panel (counties with ZORI in Jan 2016)
    rd, rf, rn, R = load(F_ZORI)
    j0 = rd.index("2016-01-31")
    rpanel = np.isfinite(R[:, j0])
    rows = []
    RP = R[rpanel]
    for t in range(12, len(rd)):
        y = 100.0 * np.log(RP[:, t] / RP[:, t - 12])
        ok = np.isfinite(y)
        if ok.sum() >= 0.9 * rpanel.sum():
            rows.append((rd[t], float(np.median(y[ok]))))
    part.add("zillow_cty_zori_yoy_median", rows, name="US counties: median year-over-year rent growth across counties (ZORI)",
             subcategory="county_distribution", region=f"US counties (Zillow ZORI panel of {int(rpanel.sum())} counties with data in 2016-01)",
             source="Zillow Observed Rent Index (ZORI), county, smoothed", url=URL_ZORI, frequency="monthly",
             units="percent (100 x log change over 12 months)", transform="diff", known_after_days=K,
             notes=("SHORT: ZORI starts 2015 (introduced 2020, back-filled), so the 12-month change starts 2016. Zillow's ZORI is "
                    f"smoothed (Zillow-side, method not documented as one-sided). {TIMING} {REV}"), lo=-30, hi=40)
    part.save()


if __name__ == "__main__":
    main()
