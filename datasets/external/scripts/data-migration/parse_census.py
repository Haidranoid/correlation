"""Census Population Estimates Program: county and state population growth and components of change.

Inputs (raw/census_popest/, unmodified downloads):
  A   co-est00int-tot.csv           2000-2010 intercensal totals, July 1 2000..2010 (+ Census 2010)
  B   cc-est2020int-agesex-all.csv  2010-2020 intercensal (released Sep 2025), YEAR 1 = Apr 1 2010 base,
                                    2..11 = Jul 1 2010..2019, 12 = Apr 1 2020 Census; counties only
  C   co-est2025-alldata.csv        Vintage 2025 postcensal (released 2026-03-26): base Apr 2020,
                                    Jul 1 2020..2025, components 2021..2025
  V10 co-est2010-alldata.csv        Vintage 2010 evaluation estimates: components 2001..2010
  V20 co-est2020-alldata.csv        Vintage 2020 postcensal (released 2021-05-04): components 2011..2020

Growth for estimate year t (July 1 t-1 -> July 1 t), always within one file:
  2001..2010 A;  2011..2019 B;  2020 B(Jul2019->Apr2020) x C(Apr2020 base->Jul2020);  2021..2025 C.
Components (counts and R* rates per 1,000 mid-period population) for year t: V10 for 2001..2010,
V20 for 2011..2020, C for 2021..2025 (the R* columns partition the years with no overlap; the
partial-year 2000/2010/2020 count columns of each file are never used).

Outputs:
  derived/data-migration/census_popest_panel.csv : every county, state and the US, 2000..2025
  series/popest_*.csv for the distribution series, selected counties and states.
"""
import csv
import datetime as dt
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, EXT, write_series, q

R = os.path.join(RAW, "census_popest")
DERIVED = os.path.join(EXT, "derived", "data-migration")
URL = "https://www2.census.gov/programs-surveys/popest/datasets/"
K_PEP = 310  # days after July 1 (see notes)
YEARS = list(range(2001, 2026))


def read(name):
    with open(os.path.join(R, name), encoding="latin-1", newline="") as f:
        return list(csv.DictReader(f))


def key(r):
    return (int(r["STATE"]), int(r["COUNTY"]))


def num(x):
    x = (x or "").strip()
    if x in ("", "X", "(X)", "NA"):
        return None
    return float(x)


# ---------------------------------------------------------------- load
A = {}
names = {}
for r in read("co-est00int-tot.csv"):
    k = key(r)
    names.setdefault(k, (r["STNAME"], r["CTYNAME"]))
    A[k] = {y: num(r[f"POPESTIMATE{y}"]) for y in range(2000, 2011)}

B = {}
for r in read("cc-est2020int-agesex-all.csv"):
    if int(r["SUMLEV"]) != 50:
        continue
    k = key(r)
    names.setdefault(k, (r["STNAME"], r["CTYNAME"]))
    B.setdefault(k, {})[int(r["YEAR"])] = num(r["POPESTIMATE"])
# state rows for B are sums of their counties
for k in list(B):
    sk = (k[0], 0)
    d = B.setdefault(sk, {})
    for yc, v in B[k].items():
        d[yc] = d.get(yc, 0.0) + v

C, comp = {}, {}
COMPS = [("BIRTHS", "births"), ("DEATHS", "deaths"), ("NATURALINC", "natural"),
         ("INTERNATIONALMIG", "nim"), ("DOMESTICMIG", "ndm"), ("NETMIG", "netmig"),
         ("RESIDUAL", "residual")]
RATES = [("RBIRTH", "r_birth"), ("RDEATH", "r_death"), ("RNATURALINC", "r_natural"),
         ("RINTERNATIONALMIG", "r_nim"), ("RDOMESTICMIG", "r_ndm"), ("RNETMIG", "r_netmig")]


def load_components(rows, years, src, natural="NATURALINC"):
    for r in rows:
        k = key(r)
        names.setdefault(k, (r["STNAME"], r["CTYNAME"]))
        for y in years:
            d = comp.setdefault(k, {}).setdefault(y, {"src": src})
            for col, nm in COMPS:
                c = natural if col == "NATURALINC" else col
                d[nm] = num(r[f"{c}{y}"])
            for col, nm in RATES:
                c = "R" + natural if col == "RNATURALINC" else col
                d[nm] = num(r[f"{c}{y}"])
            d["pop_start"] = num(r.get(f"POPESTIMATE{y - 1}"))


load_components(read("co-est2010-alldata.csv"), range(2001, 2011), "V2010")
load_components(read("co-est2020-alldata.csv"), range(2011, 2021), "V2020")
rowsC = read("co-est2025-alldata.csv")
load_components(rowsC, range(2021, 2026), "V2025", natural="NATURALCHG")
for r in rowsC:
    k = key(r)
    names[k] = (r["STNAME"], r["CTYNAME"])  # latest names win
    C[k] = {"base": num(r["ESTIMATESBASE2020"])}
    for y in range(2020, 2026):
        C[k][y] = num(r[f"POPESTIMATE{y}"])


# ---------------------------------------------------------------- growth (within one file)
def growth(k, t):
    try:
        if 2001 <= t <= 2010:
            return A[k][t] / A[k][t - 1] - 1, "A"
        if 2011 <= t <= 2019:
            return B[k][t - 2008] / B[k][t - 2009] - 1, "B"
        if t == 2020:
            return (B[k][12] / B[k][11]) * (C[k][2020] / C[k]["base"]) - 1, "B+C"
        if 2021 <= t <= 2025:
            return C[k][t] / C[k][t - 1] - 1, "C"
    except (KeyError, TypeError, ZeroDivisionError):
        return None, None
    return None, None


def pop_start(k, t):
    """Population at July 1 of t-1 in the file used for year t's growth."""
    try:
        if 2001 <= t <= 2010:
            return A[k][t - 1]
        if 2011 <= t <= 2019:
            return B[k][t - 2009]
        if t == 2020:
            return B[k][11]
        return C[k][t - 1]
    except KeyError:
        return None


def chain_level(k):
    """Level series anchored on V2025 (2020..2025) and back-cast with within-file growth, so
    log changes equal each year's within-vintage growth. None if the chain breaks."""
    if k not in C:
        return None
    L = {2025: C[k][2025]}
    for t in range(2025, 2000, -1):
        g, _ = growth(k, t)
        if g is None:
            return L if len(L) > 1 else None
        L[t - 1] = L[t] / (1 + g)
    return L


counties = sorted(k for k in names if k[1] != 0)
states = sorted(k for k in names if k[1] == 0)

# ---------------------------------------------------------------- panel
os.makedirs(DERIVED, exist_ok=True)
panel_path = os.path.join(DERIVED, "census_popest_panel.csv")
cols = ["fips", "level", "state", "name", "year", "pop_jul1", "pop_source", "growth_pct",
        "growth_source", "chained_pop", "components_source", "births", "deaths", "natural",
        "nim", "ndm", "netmig", "residual", "r_birth", "r_death", "r_natural", "r_nim", "r_ndm",
        "r_netmig"]
nrow = 0
with open(panel_path, "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(cols)
    for k in states + counties:
        st, nm = names[k]
        L = chain_level(k) or {}
        for y in range(2000, 2026):
            if y <= 2009:
                p, ps = (A.get(k, {}).get(y), "intercensal2000-2010")
            elif y <= 2019:
                p, ps = (B.get(k, {}).get(y - 2008), "intercensal2010-2020")
            else:
                p, ps = (C.get(k, {}).get(y), "V2025")
            g, gs = growth(k, y) if y >= 2001 else (None, None)
            c = comp.get(k, {}).get(y, {})
            if p is None and g is None and not c:
                continue
            w.writerow([f"{k[0]:02d}{k[1]:03d}", "state" if k[1] == 0 else "county", st, nm, y,
                        "" if p is None else int(p), ps if p is not None else "",
                        "" if g is None else f"{100 * g:.5f}", gs or "",
                        "" if y not in L else f"{L[y]:.1f}", c.get("src", ""),
                        *["" if c.get(n) is None else c.get(n) for _, n in COMPS],
                        *["" if c.get(n) is None else c.get(n) for _, n in RATES]])
            nrow += 1
print(f"panel: {panel_path} {nrow} rows")


def jul1(y):
    return dt.date(y, 7, 1)


# ---------------------------------------------------------------- consistency checks
# 1. US NIM by year = sum of state NIM = sum of county NIM, per source file
for y in (2005, 2015, 2024):
    sn = sum(comp[k][y]["nim"] for k in states if y in comp.get(k, {}))
    cn = sum(comp[k][y]["nim"] for k in counties if y in comp.get(k, {}))
    print(f"check NIM {y}: states {sn:,.0f} counties {cn:,.0f}")
# 2. growth join smoothness for the US (sum of states, each file)
for t in (2009, 2010, 2011, 2019, 2020, 2021):
    num_ = den = 0.0
    for k in states:
        g, _ = growth(k, t)
        p0 = pop_start(k, t)
        if g is not None and p0:
            num_ += p0 * (1 + g)
            den += p0
    print(f"check US growth {t}: {100 * (num_ / den - 1):.3f}%")

# ---------------------------------------------------------------- distribution series
MINPOP = 10000
notes_pep = ("Census PEP. known_after_days=310 is the largest first-release lag across vintages "
             "(county files normally mid/late March of t+1, about 255-270 days after July 1; V2018 "
             "came 2019-04-18 (291 d), V2020 2021-05-04 (307 d)). Stored values are the latest "
             "revisions (intercensal for 2000-2019 levels; final postcensal vintages for components), "
             "not the first release, so every year carries revision look-ahead. ANNUAL: about 25 "
             "observations; flagged, nearly untestable at daily move lengths.")
dist = {nm: [] for nm in ["growth_median", "growth_p10", "growth_p90", "growth_max", "growth_min",
                          "nimr_median", "nimr_p10", "nimr_p90", "nimr_max", "nimr_min",
                          "ndmr_median", "ndmr_p10", "ndmr_p90", "ndmr_max", "ndmr_min",
                          "share_growing", "nim_top20_share", "us_nim"]}
universe = {}
for t in YEARS:
    gs = []
    for k in counties:
        g, _ = growth(k, t)
        p0 = pop_start(k, t)
        if g is not None and p0 is not None and p0 >= MINPOP:
            gs.append(100 * g)
    nimr, ndmr, nims = [], [], []
    for k in counties:
        c = comp.get(k, {}).get(t)
        if not c:
            continue
        if c["nim"] is not None:
            nims.append(c["nim"])
        if c["pop_start"] is not None and c["pop_start"] >= MINPOP:
            if c["r_nim"] is not None:
                nimr.append(c["r_nim"])
            if c["r_ndm"] is not None:
                ndmr.append(c["r_ndm"])
    d = jul1(t)
    universe[t] = (len(gs), len(nimr))
    for pre, xs in (("growth", gs), ("nimr", nimr), ("ndmr", ndmr)):
        dist[f"{pre}_median"].append((d, q(xs, 0.5)))
        dist[f"{pre}_p10"].append((d, q(xs, 0.1)))
        dist[f"{pre}_p90"].append((d, q(xs, 0.9)))
        dist[f"{pre}_max"].append((d, max(xs)))
        dist[f"{pre}_min"].append((d, min(xs)))
    dist["share_growing"].append((d, sum(1 for x in gs if x > 0) / len(gs)))
    tot = sum(nims)
    dist["nim_top20_share"].append((d, sum(sorted(nims, reverse=True)[:20]) / tot))
    us = sum(comp[k][t]["nim"] for k in states if t in comp.get(k, {}))
    dist["us_nim"].append((d, us))
print("universe sizes (growth, rates) by year:", {t: universe[t] for t in (2001, 2010, 2011, 2020, 2021, 2025)})

U = f"{MINPOP:,}"
desc = {
    "growth": ("annual population growth across US counties", "percent (July 1 to July 1)",
               "Growth within one file per year (A intercensal 2000s, B intercensal 2010s, "
               "2019->2020 chained through the April 2020 Census within B and V2025, C V2025 after). "
               f"Counties with at least {U} residents at the start of the year (about 2,400 counties)."),
    "nimr": ("net international migration rate across US counties", "per 1,000 mid-period residents",
             "Census R_INTERNATIONALMIG (V2010 for 2001-2010, V2020 for 2011-2020, V2025 for 2021-2025). "
             f"Counties with at least {U} residents at the start of the year. Net international migration "
             "is the Census 'immigration' component: foreign-born arrivals minus foreign-born emigration, "
             "plus net native movement abroad and Puerto Rico, plus Armed Forces moves. Quirks: in the V2010 "
             "vintage, troop deployments abroad count as emigration (2003: Chattahoochee GA -58, Onslow NC "
             "-38 per 1,000), and migrants held in ICE detention count as residents (Winn Parish LA and "
             "Adams County MS, +65 per 1,000 in 2024)."),
    "ndmr": ("net domestic migration rate across US counties", "per 1,000 mid-period residents",
             "Census R_DOMESTICMIG (same vintages as the NIM rate). "
             f"Counties with at least {U} residents at the start of the year."),
}
stat_name = {"median": "median", "p10": "10th percentile", "p90": "90th percentile",
             "max": "maximum", "min": "minimum"}
written = []
for pre, (what, units, extra) in desc.items():
    for st, sn in stat_name.items():
        sid = f"popest_dist_{pre}_{st}"
        rng = (-100, 100) if pre == "growth" else (-2000, 2000)
        written.append(write_series(
            sid, dist[f"{pre}_{st}"], name=f"County {what}: {sn} (annual)", category="population",
            subcategory="county_distribution", region="US", source="US Census Bureau PEP",
            url=URL, frequency="annual", units=units, transform="diff", known_after_days=K_PEP,
            notes=f"{extra} Dated July 1 of the estimate year (flows: July 1 t-1 to June 30 t). "
                  + ("Max/min are single-county extremes (disasters such as Katrina 2006, oil booms such as "
                     "the Bakken 2011-14), noisy by construction. " if st in ("max", "min") else "")
                  + notes_pep, value_range=rng))
written.append(write_series(
    "popest_dist_share_growing", dist["share_growing"],
    name="Share of US counties whose population grew (annual)", category="population",
    subcategory="county_distribution", region="US", source="US Census Bureau PEP", url=URL,
    frequency="annual", units="fraction of counties (0-1)", transform="diff", known_after_days=K_PEP,
    notes=f"Counties with at least {U} residents at the start of the year; growth as in "
          f"popest_dist_growth_median. " + notes_pep, value_range=(0, 1)))
written.append(write_series(
    "popest_dist_nim_top20_share", dist["nim_top20_share"],
    name="Concentration of US net international migration: share going to the top 20 counties (annual)",
    category="migration", subcategory="county_distribution", region="US",
    source="US Census Bureau PEP", url=URL, frequency="annual",
    units="fraction of national NIM (0-1)", transform="diff", known_after_days=K_PEP,
    notes="Sum of the 20 largest county NIM counts divided by the sum over all counties (= national NIM), "
          "all counties, same vintages as the rates. " + notes_pep, value_range=(0, 1)))
written.append(write_series(
    "popest_us_nim", dist["us_nim"],
    name="US net international migration, Census estimate (annual, July-June)", category="migration",
    subcategory="national", region="US", source="US Census Bureau PEP", url=URL, frequency="annual",
    units="persons per year", transform="logdiff", known_after_days=K_PEP,
    notes="Sum over states of INTERNATIONALMIG (V2010 2001-2010, V2020 2011-2020, V2025 2021-2025). "
          "Method breaks: V2025 adopted new NIM inputs, and the 2021-2024 surge reflects the 2023-24 "
          "humanitarian-migration revision. National counts are also in the state panels. " + notes_pep,
    value_range=(1e5, 5e6)))

# ---------------------------------------------------------------- selected counties
SELECTED = [
    # (fips state, county, slug, reason)
    (6, 37, "los_angeles_ca", "most populous; immigration gateway; domestic outflow"),
    (17, 31, "cook_il", "Chicago; gateway; domestic outflow"),
    (48, 201, "harris_tx", "Houston; fast growth and gateway"),
    (4, 13, "maricopa_az", "Phoenix; fastest numeric growth"),
    (6, 73, "san_diego_ca", "border metro"),
    (6, 59, "orange_ca", "large gateway suburb"),
    (12, 86, "miami_dade_fl", "largest Latin American gateway"),
    (48, 113, "dallas_tx", "Dallas; gateway"),
    (36, 47, "kings_ny", "Brooklyn; gateway"),
    (6, 65, "riverside_ca", "Inland Empire; domestic in-migration"),
    (36, 81, "queens_ny", "Queens; most foreign-born county"),
    (32, 3, "clark_nv", "Las Vegas; housing-cycle growth"),
    (53, 33, "king_wa", "Seattle; tech, H-1B gateway"),
    (48, 439, "tarrant_tx", "Fort Worth; fast growth"),
    (48, 29, "bexar_tx", "San Antonio"),
    (12, 11, "broward_fl", "gateway; Florida in-migration"),
    (6, 85, "santa_clara_ca", "Silicon Valley; tech gateway"),
    (26, 163, "wayne_mi", "Detroit; long decline"),
    (36, 61, "new_york_ny", "Manhattan"),
    (25, 17, "middlesex_ma", "Boston area; universities, biotech"),
    (42, 101, "philadelphia_pa", "large old-core city"),
    (48, 453, "travis_tx", "Austin; fast growth, tech"),
    (48, 85, "collin_tx", "Dallas suburb; among fastest-growing large counties"),
    (48, 121, "denton_tx", "Dallas suburb; fast growth"),
    (48, 157, "fort_bend_tx", "Houston suburb; fast growth, diverse"),
    (37, 183, "wake_nc", "Raleigh; fast growth"),
    (37, 119, "mecklenburg_nc", "Charlotte; banking hub"),
    (12, 57, "hillsborough_fl", "Tampa"),
    (12, 95, "orange_fl", "Orlando; Puerto Rican and Latin American inflows"),
    (12, 71, "lee_fl", "Fort Myers; retiree in-migration, hurricane Ian"),
    (4, 21, "pinal_az", "Phoenix exurb; housing boom and bust"),
    (49, 49, "utah_ut", "Provo; fast natural increase and tech"),
    (34, 17, "hudson_nj", "Jersey City; gateway"),
    (51, 59, "fairfax_va", "Washington suburb; federal economy, gateway"),
    (48, 329, "midland_tx", "Permian Basin oil boom"),
    (38, 105, "williams_nd", "Bakken oil boom and bust"),
    (22, 71, "orleans_la", "New Orleans; Hurricane Katrina"),
    (6, 75, "san_francisco_ca", "tech; pandemic outflow"),
    (39, 35, "cuyahoga_oh", "Cleveland; Rust Belt decline"),
    (8, 31, "denver_co", "Denver"),
]
for s, c, slug, why in SELECTED:
    k = (s, c)
    st, nm = names[k]
    L = chain_level(k)
    assert L and min(L) == 2000, (slug, sorted(L) if L else None)
    pop = [(jul1(y), round(v, 1)) for y, v in sorted(L.items())]
    written.append(write_series(
        f"popest_cty_{slug}_pop", pop, name=f"Population, {nm}, {st} (annual)",
        category="population", subcategory="county", region=f"US-{slug[-2:].upper()}",
        source="US Census Bureau PEP", url=URL, frequency="annual",
        units="persons (chain-linked, anchored on Vintage 2025)", transform="logdiff",
        known_after_days=K_PEP,
        notes=f"FIPS {s:02d}{c:03d} ({why}). Levels for 2020-2025 are V2025; earlier years are back-cast "
              "with within-file growth (2000s intercensal, 2010s intercensal, 2019->2020 chained through "
              "the April 2020 Census), so the log change of each year is that year's growth with no "
              "vintage seam; earlier levels differ from the published intercensal ones by the seam "
              "ratios (well under 1%). " + notes_pep, value_range=(1e4, 2e7)))
    for comp_nm, label, col in (("nimr", "net international migration rate", "r_nim"),
                                ("ndmr", "net domestic migration rate", "r_ndm")):
        rows = [(jul1(y), comp[k][y][col]) for y in YEARS if comp.get(k, {}).get(y, {}).get(col) is not None]
        written.append(write_series(
            f"popest_cty_{slug}_{comp_nm}", rows,
            name=f"{label[0].upper() + label[1:]}, {nm}, {st} (annual)",
            category="migration" if comp_nm == "nimr" else "population", subcategory="county",
            region=f"US-{slug[-2:].upper()}", source="US Census Bureau PEP", url=URL,
            frequency="annual", units="per 1,000 mid-period residents", transform="diff",
            known_after_days=K_PEP,
            notes=f"FIPS {s:02d}{c:03d} ({why}). Census R_{'INTERNATIONALMIG' if comp_nm == 'nimr' else 'DOMESTICMIG'}: "
                  "V2010 evaluation estimates for 2001-2010, V2020 for 2011-2020, V2025 for 2021-2025 "
                  "(method breaks at 2011 and 2021; county NIM is an allocation of the national total, "
                  "not a count; it includes Armed Forces moves abroad in 2001-2010 and ICE detainees). " + notes_pep, value_range=(-2000, 2000)))

# ---------------------------------------------------------------- states
ABBR = {1: "al", 2: "ak", 4: "az", 5: "ar", 6: "ca", 8: "co", 9: "ct", 10: "de", 11: "dc", 12: "fl",
        13: "ga", 15: "hi", 16: "id", 17: "il", 18: "in", 19: "ia", 20: "ks", 21: "ky", 22: "la",
        23: "me", 24: "md", 25: "ma", 26: "mi", 27: "mn", 28: "ms", 29: "mo", 30: "mt", 31: "ne",
        32: "nv", 33: "nh", 34: "nj", 35: "nm", 36: "ny", 37: "nc", 38: "nd", 39: "oh", 40: "ok",
        41: "or", 42: "pa", 44: "ri", 45: "sc", 46: "sd", 47: "tn", 48: "tx", 49: "ut", 50: "vt",
        51: "va", 53: "wa", 54: "wv", 55: "wi", 56: "wy"}
IN_DATA_OTHER = {"ca", "tx", "fl", "ny", "pa", "il", "oh", "ga", "nc", "mi"}
WRITE_STATE_POP = False
assert len(states) == 51 and all(k[0] in ABBR for k in states)
for k in states:
    a = ABBR[k[0]]
    st = names[k][0]
    # State population levels are kept in the derived panel (chained_pop) rather than as series: the 10
    # largest states and the 4 regions are already in data-other, and the series budget goes to
    # breadth (rates for all 51 states, counties, international). Set True to write them.
    if WRITE_STATE_POP and a not in IN_DATA_OTHER:
        L = chain_level(k)
        assert L and min(L) == 2000
        written.append(write_series(
            f"popest_st_{a}_pop", [(jul1(y), round(v, 1)) for y, v in sorted(L.items())],
            name=f"Population, {st} (annual)", category="population", subcategory="state",
            region=f"US-{a.upper()}", source="US Census Bureau PEP", url=URL, frequency="annual",
            units="persons (chain-linked, anchored on Vintage 2025)", transform="logdiff",
            known_after_days=K_PEP,
            notes="State = sum of its counties' estimates (intercensal 2010s state totals summed from "
                  "counties). Chain-linked like the county levels. The 10 largest states are in "
                  "data-other as census_pop_<st> (raw levels); these complete the 51. State totals are "
                  "first published each December (about 175-210 days), but 310 is kept for one "
                  "conservative rule across PEP series (V2020 came out 2021-05-04). " + notes_pep,
            value_range=(4e5, 5e7)))
    for comp_nm, label, col, cat in (("nimr", "net international migration rate", "r_nim", "migration"),
                                     ("ndmr", "net domestic migration rate", "r_ndm", "population")):
        rows = [(jul1(y), comp[k][y][col]) for y in YEARS if comp.get(k, {}).get(y, {}).get(col) is not None]
        written.append(write_series(
            f"popest_st_{a}_{comp_nm}", rows,
            name=f"{label[0].upper() + label[1:]}, {st} (annual)", category=cat,
            subcategory="state", region=f"US-{a.upper()}", source="US Census Bureau PEP", url=URL,
            frequency="annual", units="per 1,000 mid-period residents", transform="diff",
            known_after_days=K_PEP,
            notes="Census state R_" + ("INTERNATIONALMIG" if comp_nm == "nimr" else "DOMESTICMIG")
                  + ": V2010 evaluation estimates 2001-2010, V2020 2011-2020, V2025 2021-2025 (method "
                  "breaks at 2011 and 2021). State components are first released each December, about "
                  "175-210 days after July 1 (V2025 on 2026-01-27), earlier than counties; V2020 states came "
                  "2021-05-04 (307 days), so 310 keeps one conservative rule for every PEP series. "
                  + notes_pep, value_range=(-100, 100)))

print(f"written {len(written)} series")
with open(os.path.join(DERIVED, "census_selected_counties.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["fips", "slug", "state", "county", "reason"])
    for s, c, slug, why in SELECTED:
        w.writerow([f"{s:02d}{c:03d}", slug, names[(s, c)][0], names[(s, c)][1], why])
