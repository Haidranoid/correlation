"""World Bank WDI (API v2, keyless) -> cross-country statistics over 40 major economies and US/World
series. Annual, dated December 31 of the reference year."""
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402

RAWD = os.path.join(plib.RAW, "worldbank_wdi")
PANEL = ["ARG", "AUS", "BRA", "CAN", "CHN", "DEU", "FRA", "GBR", "IDN", "IND", "ITA", "JPN", "KOR", "MEX", "RUS", "SAU",
         "TUR", "USA", "ZAF", "ESP", "NLD", "CHE", "SWE", "POL", "BEL", "NOR", "AUT", "IRL", "ISR", "SGP", "THA", "MYS",
         "PHL", "VNM", "EGY", "NGA", "PAK", "BGD", "CHL", "COL"]
API = "https://api.worldbank.org/v2/country/{c}/indicator/{i}?format=json&per_page=20000&date=1960:2026"
LIC = "World Bank WDI, CC BY 4.0 (cite: World Bank, World Development Indicators)."
ANN = "ANNUAL: essentially untestable at the census move lengths; included because the user asked."


def load(ind):
    j = json.load(open(os.path.join(RAWD, ind + ".json")))
    out = {}
    for r in j[1]:
        if r["value"] is not None:
            out.setdefault(r["countryiso3code"], {})[int(r["date"])] = float(r["value"])
    return out, j[0].get("lastupdated")


def panel_stats(data, panel=PANEL):
    years = sorted(set.intersection(*[set(data.get(c, {})) for c in panel]))
    st = {k: [] for k in ["median", "p10", "p90", "ratio"]}
    for y in years:
        x = np.array([data[c][y] for c in panel])
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        d = dt.date(y, 12, 31)
        st["median"].append((d, p50))
        st["p10"].append((d, p10))
        st["p90"].append((d, p90))
        st["ratio"].append((d, p90 / p10))
    return st, years


def main():
    part = plib.Part("80_worldbank_wdi")
    gdp, lu = load("NY.GDP.PCAP.PP.KD")
    st, years = panel_stats(gdp)
    print("GDP pc PPP panel years", years[0], years[-1], len(years))
    K1 = 550
    n1 = (f"GDP per capita, PPP, constant 2021 international $ (NY.GDP.PCAP.PP.KD), across 40 major economies (G20 members "
          f"plus 21 others: {' '.join(PANEL)}), years where all 40 have data. known_after_days {K1}: WDI adds year t in "
          f"its July update of t+1 for most economies, later for some; 18 months is a conservative bound. Heavily "
          f"revised: every ICP round re-bases the whole PPP history (2021 PPPs exist only since May 2024). WDI "
          f"lastupdated {lu}. {ANN} {LIC}")
    common = dict(subcategory="country_distribution", region="40 major economies", source="World Bank World Development Indicators",
                  url=API.format(c=";".join(PANEL), i="NY.GDP.PCAP.PP.KD"), frequency="annual", known_after_days=K1)
    part.add("wb_gdppc_ppp_median", st["median"], name="Countries: median GDP per capita (PPP) across 40 major economies",
             units="international $ (2021 PPP, constant)", transform="logdiff", notes=n1, lo=500, hi=200_000, **common)
    part.add("wb_gdppc_ppp_p10", st["p10"], name="Countries: 10th-percentile GDP per capita (PPP) across 40 major economies",
             units="international $ (2021 PPP, constant)", transform="logdiff", notes=n1, lo=300, hi=200_000, **common)
    part.add("wb_gdppc_ppp_p90", st["p90"], name="Countries: 90th-percentile GDP per capita (PPP) across 40 major economies",
             units="international $ (2021 PPP, constant)", transform="logdiff", notes=n1, lo=500, hi=300_000, **common)
    part.add("wb_gdppc_ppp_p90_p10_ratio", st["ratio"], name="Countries: p90/p10 ratio of GDP per capita (PPP) across 40 major economies",
             units="ratio", transform="diff", notes=n1, lo=1, hi=50, **common)
    gni, lu2 = load("NY.GNP.PCAP.CD")
    st2, years2 = panel_stats(gni, [c for c in PANEL if c != "NGA"])
    print("GNI pc panel years", years2[0], years2[-1], len(years2))
    common2 = dict(common)
    common2["url"] = API.format(c=";".join(PANEL), i="NY.GNP.PCAP.CD")
    part.add("wb_gnipc_atlas_median", st2["median"], name="Countries: median GNI per capita (Atlas method, current US$) across 39 major economies",
             units="US$ (current, Atlas method)", transform="logdiff",
             notes=(f"GNI per capita, Atlas method (NY.GNP.PCAP.CD), median across the same economies except Nigeria (its Atlas "
                    f"GNI starts only in 2008), years where all 39 have data. known_after_days {K1}: published in the July WDI update of t+1 (income classifications). "
                    f"Revised. {ANN} {LIC}"), lo=50, hi=200_000, **common2)
    K2 = 1100
    surv = (f"From household surveys via the World Bank Poverty and Inequality Platform (US: Luxembourg Income Study). "
            f"known_after_days {K2}: survey-based numbers reach WDI ~1-3 years after the year (US 2024 was in by Oct 2026); "
            f"conservative. Revised when PPPs or survey processing change. {ANN} {LIC}")
    for ind, sid, nm, units, lo, hi in [
        ("SI.POV.GINI", "wb_gini_us", "Gini index of income, United States", "Gini index (0-100)", 20, 70),
        ("SI.DST.FRST.20", "wb_bottom20_share_us", "Income share held by the bottom 20%, United States", "percent of income", 1, 15),
    ]:
        data, _ = load(ind)
        rows = [(dt.date(y, 12, 31), v) for y, v in sorted(data["USA"].items())]
        part.add(sid, rows, name=f"{nm} (World Bank)", subcategory="country", region="USA",
                 source="World Bank World Development Indicators", url=API.format(c="USA", i=ind), frequency="annual",
                 units=units, transform="diff", known_after_days=K2, notes=f"{ind}. {surv}", lo=lo, hi=hi)
    pov, _ = load("SI.POV.DDAY")
    rows = [(dt.date(y, 12, 31), v) for y, v in sorted(pov["WLD"].items())]
    part.add("wb_poverty_3usd_world", rows, name="World poverty headcount ratio at $3.00 a day (2021 PPP)",
             subcategory="world", region="World", source="World Bank World Development Indicators",
             url=API.format(c="WLD", i="SI.POV.DDAY"), frequency="annual", units="percent of population", transform="diff",
             known_after_days=K2,
             notes=("SI.POV.DDAY for the World aggregate (lined-up estimates; recent years are nowcasts). 2019 is missing "
                    f"in this vintage. {surv}"), lo=0, hi=80)
    part.save()


if __name__ == "__main__":
    main()
