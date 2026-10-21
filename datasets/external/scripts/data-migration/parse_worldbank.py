"""World Bank WDI: population growth (SP.POP.GROW), net migration (SM.POP.NETM) and population
(SP.POP.TOTL) for 40 major economies -> derived panel; net migration for 10 of them -> series."""
import csv
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, EXT, write_series

R = os.path.join(RAW, "worldbank_migration")
DERIVED = os.path.join(EXT, "derived", "data-migration")
ECON = ["USA", "CHN", "JPN", "DEU", "IND", "GBR", "FRA", "ITA", "BRA", "CAN", "RUS", "KOR", "AUS", "MEX",
        "ESP", "IDN", "NLD", "SAU", "TUR", "CHE", "POL", "ARG", "SWE", "BEL", "THA", "ISR", "IRL", "NOR",
        "AUT", "ARE", "NGA", "ZAF", "EGY", "PHL", "VNM", "MYS", "SGP", "COL", "CHL", "PAK"]
data, names, updated = {}, {}, {}
for ind in ["SP.POP.GROW", "SM.POP.NETM", "SP.POP.TOTL"]:
    meta, rows = json.load(open(os.path.join(R, f"{ind}_all.json")))
    updated[ind] = meta["lastupdated"]
    for r in rows:
        c = r["countryiso3code"]
        if c in ECON and r["value"] is not None:
            data.setdefault((ind, c), {})[int(r["date"])] = float(r["value"])
            names[c] = r["country"]["value"]
assert all(c in names for c in ECON), [c for c in ECON if c not in names]
with open(os.path.join(DERIVED, "worldbank_population_migration_panel.csv"), "w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["iso3", "country", "year", "pop_growth_pct", "net_migration", "population", "wpp_projection_year"])
    for c in ECON:
        years = sorted(set().union(*[data.get((i, c), {}).keys() for i in ["SP.POP.GROW", "SM.POP.NETM", "SP.POP.TOTL"]]))
        for y in years:
            g = data.get(("SP.POP.GROW", c), {}).get(y)
            n = data.get(("SM.POP.NETM", c), {}).get(y)
            p = data.get(("SP.POP.TOTL", c), {}).get(y)
            w.writerow([c, names[c], y, "" if g is None else f"{g:.6f}", "" if n is None else int(n),
                        "" if p is None else int(p), int(y >= 2024)])
print("WB panel written; lastupdated", updated)

URL = "https://data.worldbank.org/indicator/SM.POP.NETM"
SEL = [("DEU", "de"), ("GBR", "gb"), ("FRA", "fr"), ("ITA", "it"), ("ESP", "es"), ("CAN", "ca"),
       ("AUS", "au"), ("JPN", "jp"), ("CHN", "cn"), ("MEX", "mx")]
for c, slug in SEL:
    s = sorted((dt.date(y, 7, 1), v) for y, v in data[("SM.POP.NETM", c)].items() if y <= 2023)
    write_series(
        f"wbmig_netm_{slug}", s, name=f"Net migration, {names[c]} (annual, World Bank/UN WPP)",
        category="migration", subcategory="international_net_migration", region=slug.upper(),
        source="World Bank WDI (UN World Population Prospects 2024)", url=URL, frequency="annual",
        units="persons per year (net; can be negative)", transform="diff", known_after_days=1100,
        notes=(f"WDI SM.POP.NETM, last updated {updated['SM.POP.NETM']}; dated July 1. MODELLED: the values are "
               "UN WPP 2024 estimates (a demographic reconstruction, smoothed and sometimes repeated across years), "
               "revised wholesale every two years; WDI published only 5-year totals before about 2023, so these "
               "annual values were never available in real time. Kept to estimate years (to 2023); WPP 2024 values "
               "for 2024-2025 are projections and are left out (they are in the derived panel, flagged). "
               "known_after_days=1100: a year's first WPP estimate can appear up to about 3 years later (biennial "
               "revisions). ANNUAL, flagged: nearly untestable at daily move lengths. Licence CC BY 4.0."),
        value_range=(-1e7, 1e7))
