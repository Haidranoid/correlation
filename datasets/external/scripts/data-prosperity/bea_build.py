"""BEA regional accounts -> state quarterly per-capita income distribution (SQINC1), county per-capita
income distribution and selected counties (CAINC1, annual), county real GDP growth distribution
(CAGDP1, annual). Quarterly dates = last day of the quarter; annual dates = December 31.
"""
import csv
import datetime as dt
import io
import os
import re
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402
from counties import SELECTED, STATES  # noqa: E402

RAWD = os.path.join(plib.RAW, "bea_regional")
TIMD = os.path.join(plib.RAW, "bea_release_timing")
POP_FLOOR = 100_000
LIC = "BEA statistics are US government works (public domain); cite the Bureau of Economic Analysis."


def read_table(zname, prefix):
    z = zipfile.ZipFile(os.path.join(RAWD, zname))
    n = [i.filename for i in z.infolist() if i.filename.startswith(prefix)][0]
    rows = list(csv.reader(io.TextIOWrapper(z.open(n), encoding="latin-1")))
    h = rows[0]
    periods = [p.strip() for p in h[8:]]
    out = {}
    names = {}
    for x in rows[1:]:
        if len(x) < 9:
            continue
        fips = x[0].strip().strip('"').strip()
        line = x[4].strip()
        vals = {}
        for p, v in zip(periods, x[8:]):
            v = v.strip()
            try:
                vals[p] = float(v)
            except ValueError:
                pass
        out[(fips, line)] = vals
        names[fips] = x[1].strip()
    return periods, out, names, n


def alfred_dates(rid):
    t = open(os.path.join(TIMD, f"release_dates_{rid}.txt")).read()
    return sorted({dt.date.fromisoformat(d) for d in re.findall(r"(\d{4}-\d{2}-\d{2})", t.split("Real Time Start")[1])})


def is_county(f):
    return len(f) == 5 and f[2:] != "000" and f[:2] in STATES


def sqinc(part):
    periods, T, names, member = read_table("SQINC.zip", "SQINC1__ALL")
    # measured lags: each quarter's first release >= 60 days after quarter end (ALFRED rid 110)
    rel = alfred_dates(110)
    lags = []
    for y in range(2003, 2026):
        for q in range(1, 5):
            if (y, q) < (2003, 3):
                continue  # ALFRED's release list starts 2004-01-27 (2003Q3)
            e = plib.quarter_end(y, q)
            r = next((d for d in rel if d >= e + dt.timedelta(days=60)), None)
            if r is not None and r <= dt.date(2025, 6, 27):
                lags.append(((r - e).days, f"{y}Q{q}", r.isoformat()))
    lags.sort()
    print("SQINC lags 2003Q3..: median", lags[len(lags) // 2][0], "max", lags[-3:])
    K = 120
    st_f = [f for f in STATES if (f + "000", "3") in T]
    assert len(st_f) == 51
    qs = []
    for p in periods:
        m = re.match(r"(\d{4}):Q(\d)", p)
        if m:
            qs.append((p, plib.quarter_end(int(m.group(1)), int(m.group(2)))))
    st = {k: [] for k in ["median", "p10", "p90", "min", "max", "ratio"]}
    for p, d in qs:
        x = [T[(f + "000", "3")].get(p) for f in st_f]
        if any(v is None for v in x):
            continue
        x = np.array(x)
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        st["median"].append((d, p50))
        st["p10"].append((d, p10))
        st["p90"].append((d, p90))
        st["min"].append((d, float(x.min())))
        st["max"].append((d, float(x.max())))
        st["ratio"].append((d, p90 / p10))
    notes = (f"Per capita personal income (SQINC1 line 3: personal income SAAR / mid-quarter population; published from "
             f"2010Q1 only), across the 50 states + DC; nominal dollars. known_after_days {K}: quarterly state personal income is released ~12-13 weeks "
             f"after the quarter (ALFRED release dates 2004-2025: median {lags[len(lags) // 2][0]} days, max "
             f"{lags[-1][0]} for {lags[-1][1]}; 2003-04 releases were ~118-119 days, 2009Q2 108 after a comprehensive "
             f"revision). Revised every release (the 2026-09-30 release revised 2009Q1-2026Q1). {LIC}")
    common = dict(subcategory="state_distribution", region="US states (50 + DC)",
                  source="BEA State quarterly personal income (SQINC1)", url="https://apps.bea.gov/regional/zip/SQINC.zip",
                  frequency="quarterly", known_after_days=K)
    for key, sid, nm in [("median", "median", "median"), ("p10", "p10", "10th-percentile"), ("p90", "p90", "90th-percentile"),
                         ("min", "min", "minimum"), ("max", "max", "maximum")]:
        part.add(f"bea_st_pcpi_q_{sid}", st[key], name=f"US states: {nm} per capita personal income across states (quarterly)",
                 units="USD per person (annual rate, nominal)", transform="logdiff", notes=notes, lo=1000, hi=300_000, **common)
    part.add("bea_st_pcpi_q_p90_p10_ratio", st["ratio"], name="US states: p90/p10 ratio of state per capita personal income (quarterly)",
             units="ratio", transform="diff", notes=notes, lo=1, hi=5, **common)
    # total personal income (line 1, from 1948): cross-state distribution of quarterly growth
    gmed, grose = [], []
    for (pa, da), (pb, db) in zip(qs, qs[1:]):
        a = [T[(f + "000", "1")].get(pa) for f in st_f]
        b = [T[(f + "000", "1")].get(pb) for f in st_f]
        if any(v is None or v <= 0 for v in a + b):
            continue
        g = 100.0 * np.log(np.array(b) / np.array(a))
        gmed.append((db, float(np.median(g))))
        grose.append((db, 100.0 * float(np.mean(g > 0))))
    gnotes = ("Quarterly growth of each state's total personal income (SQINC1 line 1, SAAR, nominal; per-capita figures exist "
              "only from 2010 because BEA dropped pre-2010 quarterly population). Growth is already a change, but the test's "
              "level transform needs >= 10 observations in the trailing 252 trading days, which a quarterly series never "
              "has, so diff (move = change in the growth rate / share). " + notes.split("nominal dollars. ", 1)[1])
    part.add("bea_st_pi_q_growth_median", gmed, name="US states: median quarterly personal income growth across states",
             units="percent (100 x log change, quarter over quarter)", transform="diff", notes=gnotes, lo=-15, hi=15, **common)
    part.add("bea_st_pi_q_share_rose", grose, name="US states: share of states whose personal income rose this quarter",
             units="percent of states", transform="diff", notes=gnotes, lo=0, hi=100, **common)
    return lags


def cainc(part):
    periods, T, names, member = read_table("CAINC1.zip", "CAINC1__ALL")
    rel = alfred_dates(175)
    K = 510
    tnote = (f"known_after_days {K}: county personal income for year t was released in April of t+2 until 2012 (2010 data "
             "on 2012-04-25 = 481 days), in mid-November of t+1 from 2012 (~320 days, ALFRED rid 175: "
             + ", ".join(d.isoformat() for d in rel if d.month == 11)[:120] +
             "...), and 2024 data on 2026-02-05 after the 2025 shutdown (401).")
    rev = ("Revised: each release revises recent years, and comprehensive updates (e.g. 2018, 2023) revise the whole "
           f"history. ANNUAL: essentially untestable at the census move lengths; included because the user asked. {LIC}")
    fips = sorted({f for (f, l) in T if l == "3" and is_county(f)})
    print("CAINC1 county-equivalents:", len(fips))
    st = {k: [] for k in ["median", "p10", "p90", "min", "max", "ratio"]}
    for y in periods:
        if not y.isdigit():
            continue
        x, big = [], []
        for f in fips:
            v = T[(f, "3")].get(y)
            pop = T.get((f, "2"), {}).get(y)
            if v is None:
                continue
            x.append(v)
            if pop is not None and pop >= POP_FLOOR:
                big.append(v)
        if len(x) < 2900:
            print("  CAINC1 year", y, "only", len(x))
            continue
        x = np.array(x)
        d = dt.date(int(y), 12, 31)
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        st["median"].append((d, p50))
        st["p10"].append((d, p10))
        st["p90"].append((d, p90))
        st["ratio"].append((d, p90 / p10))
        st["min"].append((d, float(min(big))))
        st["max"].append((d, float(max(big))))
    common = dict(subcategory="county_distribution", region="US counties (BEA county-equivalents, 50 states + DC)",
                  source="BEA County personal income (CAINC1)", url="https://apps.bea.gov/regional/zip/CAINC1.zip",
                  frequency="annual", known_after_days=K)
    base = (f"Per capita personal income (CAINC1 line 3), nominal dollars, unweighted across ~3,100 county-equivalents "
            f"(BEA combines some Virginia independent cities with their counties; Connecticut = 8 counties to 2023, "
            f"9 planning regions in 2024). {tnote} {rev}")
    fl = f"Over counties with BEA population >= {POP_FLOOR:,} (about 600). "
    part.add("bea_cty_pcpi_median", st["median"], name="US counties: median per capita personal income across counties (annual)",
             units="USD per person (nominal)", transform="logdiff", notes=base, lo=1000, hi=500_000, **common)
    part.add("bea_cty_pcpi_p10", st["p10"], name="US counties: 10th-percentile per capita personal income across counties (annual)",
             units="USD per person (nominal)", transform="logdiff", notes=base, lo=1000, hi=500_000, **common)
    part.add("bea_cty_pcpi_p90", st["p90"], name="US counties: 90th-percentile per capita personal income across counties (annual)",
             units="USD per person (nominal)", transform="logdiff", notes=base, lo=1000, hi=500_000, **common)
    part.add("bea_cty_pcpi_min_pop100k", st["min"], name="US counties: minimum per capita personal income, counties with population >= 100k (annual)",
             units="USD per person (nominal)", transform="logdiff", notes=fl + base, lo=1000, hi=500_000, **common)
    part.add("bea_cty_pcpi_max_pop100k", st["max"], name="US counties: maximum per capita personal income, counties with population >= 100k (annual)",
             units="USD per person (nominal)", transform="logdiff", notes=fl + base, lo=1000, hi=1_000_000, **common)
    part.add("bea_cty_pcpi_p90_p10_ratio", st["ratio"], name="US counties: p90/p10 ratio of county per capita personal income (annual)",
             units="ratio", transform="diff", notes=base, lo=1, hi=10, **common)
    for f, slug, label, why in SELECTED:
        bf = {"09190": "09001", "51059": "51919"}.get(f, f)
        vals = T[(bf, "3")]
        rows = [(dt.date(int(y), 12, 31), v) for y, v in vals.items() if y.isdigit()]
        geo = (" BEA publishes Connecticut's old counties through 2023 and only planning regions for 2024, so this is "
               "Fairfield County (09001) and ends with 2023 (not spliced).") if f == "09190" else (
               " BEA combines Fairfax County with Fairfax City and Falls Church (51919).") if f == "51059" else ""
        lbl = "Fairfield County, CT" if f == "09190" else label
        st_ab = lbl.split(", ")[-1] if f != "11001" else "DC"
        part.add(f"bea_pcpi_{'ct_fairfield' if f == '09190' else slug}", rows, name=f"Per capita personal income, {lbl} (annual)", subcategory="county",
                 region=f"US-{st_ab} {lbl} (FIPS {bf})", source="BEA County personal income (CAINC1)",
                 url="https://apps.bea.gov/regional/zip/CAINC1.zip", frequency="annual", units="USD per person (nominal)",
                 transform="logdiff", known_after_days=K,
                 notes=f"Selected county: {why}. BEA name: {names.get(bf)}.{geo} {tnote} {rev}", lo=1000, hi=1_000_000)


def cagdp(part):
    periods, T, names, member = read_table("CAGDP1.zip", "CAGDP1__ALL")
    K = 400
    fips = sorted({f for (f, l) in T if l == "1" and is_county(f)})
    print("CAGDP1 county-equivalents:", len(fips))
    yrs = [y for y in periods if y.isdigit()]
    med, share = [], []
    for y0, y1 in zip(yrs, yrs[1:]):
        g = []
        for f in fips:
            a, b = T[(f, "1")].get(y0), T[(f, "1")].get(y1)
            if a and b and a > 0 and b > 0:
                g.append(100.0 * np.log(b / a))
        if len(g) < 2900:
            print("  CAGDP1", y1, "only", len(g))
            continue
        d = dt.date(int(y1), 12, 31)
        med.append((d, float(np.median(g))))
        share.append((d, 100.0 * float(np.mean(np.array(g) > 0))))
    notes = ("BACK-CAST: county GDP was first published on 2019-12-12 (official series, 2001-2018 at once; a prototype for "
             "2012-2015 appeared in Dec 2018), so values before 2018 were never public at date + known_after_days; treat "
             "pre-2019 results as look-ahead. known_after_days 400: since 2019 released in early December of t+1 "
             "(339-346 days), 2024 data on 2026-02-04 (400). Real GDP in chained 2017 dollars (CAGDP1 line 1). Growth is "
             "already a change, but the test's level transform is never live for annual data (needs >= 10 observations in "
             "252 trading days), so diff. ANNUAL: essentially untestable; included because the user asked. " + LIC)
    common = dict(subcategory="county_distribution", region="US counties (BEA county-equivalents)",
                  source="BEA County GDP (CAGDP1)", url="https://apps.bea.gov/regional/zip/CAGDP1.zip", frequency="annual",
                  known_after_days=K)
    part.add("bea_cty_rgdp_growth_median", med, name="US counties: median real GDP growth across counties (annual)",
             units="percent (100 x log change, year over year)", transform="diff", notes=notes, lo=-30, hi=30, **common)
    part.add("bea_cty_rgdp_share_grew", share, name="US counties: share of counties whose real GDP grew (annual)",
             units="percent of counties", transform="diff", notes=notes, lo=0, hi=100, **common)


def main():
    part = plib.Part("60_bea_regional")
    sqinc(part)
    cainc(part)
    cagdp(part)
    part.save()


if __name__ == "__main__":
    main()
