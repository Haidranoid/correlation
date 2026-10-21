"""Population: FRED monthly (POPTHM, CNP16OV), World Bank annual world population,
Census annual state/region estimates (2000-2010 intercensal + 2010-2020 intercensal + Vintage 2025)."""
import csv, datetime as dt, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series

ANNUAL_WARN = ("ANNUAL: about 26 observations; a step series once aligned to trading days, so it "
               "cannot support daily lead-lag tests (only the release-day jump moves). Included "
               "because the user asked about population growth.")

def fred(sid):
    rows = []
    for r in csv.DictReader(open(os.path.join(RAW, "fred", f"{sid}.csv"))):
        v = r[sid].strip()
        if v in ("", "."):
            print(f"  {sid}: missing value on {r['observation_date']} (left out)")
            continue
        rows.append((r["observation_date"], float(v)))
    return rows

write_series("fred_popthm", fred("POPTHM"), name="US resident population incl. armed forces overseas (monthly)",
             category="population", subcategory="national", region="US", source="FRED (BEA)",
             url="https://fred.stlouisfed.org/series/POPTHM", frequency="monthly", units="thousands of persons",
             transform="logdiff", known_after_days=65, value_range=(150000, 400000),
             notes="Date = first day of the reference month (FRED convention); known_after_days counted from that "
                   "date. Published in BEA Personal Income and Outlays about 4 weeks after month end, i.e. about "
                   "56-60 days after the date; 65 is a conservative bound for normal months (government "
                   "shutdowns delayed it, e.g. Sep 2025 data appeared in Dec 2025). Very smooth (interpolated "
                   "from Census estimates) and revised at every Census vintage and intercensal update; values "
                   "are the latest revision, not what was known at the time. Usable only at 21/63-day move lengths.")
write_series("fred_cnp16ov", fred("CNP16OV"), name="US civilian noninstitutional population aged 16+ (monthly)",
             category="population", subcategory="national", region="US", source="FRED (BLS CPS)",
             url="https://fred.stlouisfed.org/series/CNP16OV", frequency="monthly", units="thousands of persons",
             transform="logdiff", known_after_days=42, value_range=(90000, 300000),
             notes="Date = first day of the reference month (FRED convention); known_after_days counted from that "
                   "date. Released in the BLS Employment Situation, usually the first Friday of the next month "
                   "(about 33-40 days after the date); 42 is a conservative bound for normal months (shutdown "
                   "delays: Sep 2013, Sep 2025). Oct 2025 is missing (no CPS during the 2025 shutdown). CPS "
                   "population controls are introduced each January WITHOUT revising earlier months, so logdiff "
                   "shows level breaks every January (e.g. Jan 2000, 2012, 2022, 2024, 2025, 2026); these are "
                   "methodology steps, not population moves. Usable only at 21/63-day move lengths.")

# ---- World Bank (UN WPP-based) world population ----
j = json.load(open(os.path.join(RAW, "worldbank", "WLD_SP.POP.TOTL.json")))
assert j[0]["total"] == len(j[1]), "World Bank response paginated unexpectedly"
wb = [(f"{int(r['date'])}-07-01", r["value"]) for r in j[1] if r["value"] is not None]
write_series("wb_pop_world", wb, name="World population (annual, mid-year)", category="population",
             subcategory="world", region="world", source="World Bank WDI (SP.POP.TOTL)",
             url="https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL?format=json",
             frequency="annual", units="persons", transform="logdiff", known_after_days=400,
             value_range=(2.5e9, 9e9),
             notes=ANNUAL_WARN + " Dated July 1 (mid-year estimate). WDI publishes year Y around mid Y+1, and the "
                   "whole series is revised with each UN World Population Prospects revision, so stored values "
                   "were not known at the time. 400 days is a conservative first-publication bound.")

# ---- Census state estimates, July 1 of each year ----
REGIONS = {"1": "northeast", "2": "midwest", "3": "south", "4": "west"}
state_pop = defaultdict(dict)    # state fips -> {year: pop}
state_name, state_region = {}, {}

# 2000-2009: 2000-2010 intercensal (totals rows: SEX=ORIGIN=RACE=AGEGRP=0)
v2010_a = {}
for r in csv.DictReader(open(os.path.join(RAW, "census", "st-est00int-alldata.csv"), encoding="latin-1")):
    if r["STATE"] == "0" or (r["SEX"], r["ORIGIN"], r["RACE"], r["AGEGRP"]) != ("0", "0", "0", "0"):
        continue
    fips = r["STATE"].zfill(2)
    state_name[fips], state_region[fips] = r["NAME"], r["REGION"]
    for y in range(2000, 2010):
        state_pop[fips][y] = int(r[f"POPESTIMATE{y}"])
    v2010_a[fips] = int(r["POPESTIMATE2010"])

# 2010-2019: 2010-2020 intercensal, sum over RACE (1-6) and single-year AGE for SEX=0, ORIGIN=0
acc = defaultdict(lambda: defaultdict(int))
for r in csv.DictReader(open(os.path.join(RAW, "census", "sc-est2020int-alldata6.csv"), encoding="latin-1")):
    if r["SUMLEV"] != "040" or r["SEX"] != "0" or r["ORIGIN"] != "0":
        continue
    fips = r["STATE"].zfill(2)
    for y in range(2010, 2020):
        acc[fips][y] += int(r[f"POPESTIMATE{y}"])
for fips, d in acc.items():
    for y, v in d.items():
        state_pop[fips][y] = v

# 2020-2025: Vintage 2025 (postcensal)
for r in csv.DictReader(open(os.path.join(RAW, "census", "NST-EST2025-ALLDATA.csv"), encoding="latin-1")):
    if r["SUMLEV"] != "040":
        continue
    fips = r["STATE"].zfill(2)
    if fips == "72":            # Puerto Rico: not part of the 50 states + DC
        continue
    for y in range(2020, 2026):
        state_pop[fips][y] = int(r[f"POPESTIMATE{y}"])

states = sorted(f for f in state_pop if f != "72")
assert len(states) == 51, len(states)
for f in states:
    assert sorted(state_pop[f]) == list(range(2000, 2026)), (f, sorted(state_pop[f]))
# consistency at the 2010 join: July 2010 in both intercensal files
diffs = [abs(v2010_a[f] - state_pop[f][2010]) / state_pop[f][2010] for f in states]
print(f"2010 join: max relative difference between the two intercensal files' July-2010 values: {max(diffs):.4%}")
# year-over-year growth at the joins (2009->2010, 2019->2020), US total
tot = {y: sum(state_pop[f][y] for f in states) for y in range(2000, 2026)}
for y in (2001, 2009, 2010, 2011, 2019, 2020, 2021, 2025):
    print(f"  US (sum of states) {y}: {tot[y]:,}  growth {100*(tot[y]/tot[y-1]-1):+.3f}%")

CENSUS_URL = "https://www2.census.gov/programs-surveys/popest/datasets/"
note_c = (ANNUAL_WARN + " July 1 resident population. Spliced from the 2000-2010 intercensal (2000-09), the "
          "2010-2020 intercensal released Sep 2025 (2010-19) and Vintage 2025 (2020-25, released 2026-01-27). "
          "All values are revised: the first-release vintage numbers published each December differed, so "
          "stored values carry look-ahead. known_after_days = 210 bounds the first release of each July-1 "
          "estimate (usually late December, about 170 days; V2025 came late, 210). V2024/V2025 revised net "
          "international migration up sharply for 2022-2024.")
reg = defaultdict(lambda: defaultdict(int))
for f in states:
    for y, v in state_pop[f].items():
        reg[state_region[f] if f in state_region else None][y] += v
assert None not in reg
for code, nm in REGIONS.items():
    write_series(f"census_pop_{nm}", [(f"{y}-07-01", v) for y, v in sorted(reg[code].items())],
                 name=f"Population, Census {nm.capitalize()} region (annual)", category="population",
                 subcategory="region", region=f"US-{nm}", source="US Census Bureau Population Estimates",
                 url=CENSUS_URL, frequency="annual", units="persons", transform="logdiff",
                 known_after_days=210, value_range=(4e7, 1.5e8), notes="Sum of the region's states. " + note_c)
ABBR = {"06": "ca", "48": "tx", "12": "fl", "36": "ny", "42": "pa", "17": "il", "39": "oh",
        "13": "ga", "37": "nc", "26": "mi"}
top = sorted(states, key=lambda f: -state_pop[f][2025])[:10]
assert set(top) == set(ABBR), [state_name.get(f) for f in top]
for f in top:
    write_series(f"census_pop_{ABBR[f]}", [(f"{y}-07-01", v) for y, v in sorted(state_pop[f].items())],
                 name=f"Population, {state_name[f]} (annual)", category="population", subcategory="state",
                 region=f"US-{ABBR[f].upper()}", source="US Census Bureau Population Estimates", url=CENSUS_URL,
                 frequency="annual", units="persons", transform="logdiff", known_after_days=210,
                 value_range=(5e6, 5e7),
                 notes="One of the 10 most populous states (all 51 are in raw/census). " + note_c)
