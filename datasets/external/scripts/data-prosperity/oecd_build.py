"""OECD harmonised monthly unemployment rates (Infra-annual labour statistics, DF_IALFS_UNE_M:
unemployment rate, % of labour force, seasonally adjusted, total, 15+) -> 10 countries and
cross-country distribution statistics over a fixed panel of 20 countries with monthly survey data.
Dates = last day of the reference month.
"""
import calendar
import csv
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402

RAWF = os.path.join(plib.RAW, "oecd", "oecd_ialfs_une_m.csv")
URL = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0/"
       ".UNE_LF_M.PT_LF_SUB._Z.Y._T.Y_GE15._Z.M?startPeriod=1990-01")
SRC = "OECD Infra-annual labour statistics (SDMX DF_IALFS_UNE_M)"
LIC = "OECD data: free under the OECD open-by-default policy (July 2024; CC BY 4.0 from 2024-07-01, OECD terms similar to CC BY before); cite OECD."
REV = ("Current vintage (download 2026-10-18), revised: seasonal factors are re-estimated, and for EU countries whose "
       "labour force survey is quarterly (e.g. France, Spain) Eurostat's monthly figures are model-based and revised "
       "when the quarterly survey arrives.")

# (code, slug, name, known_after_days, timing note)
COUNTRIES = [
    ("CAN", "ca", "Canada", 14, "Statistics Canada LFS out on the first or second Friday after the month (5-13 days)"),
    ("JPN", "jp", "Japan", 35, "MIC Labour Force Survey out at the end of the next month (~28-33 days)"),
    ("DEU", "de", "Germany", 40, "ILO rate via Destatis/Eurostat ~30-32 days after the month"),
    ("FRA", "fr", "France", 40, "Eurostat monthly estimate ~30-32 days after the month (model-based on the quarterly LFS)"),
    ("ITA", "it", "Italy", 40, "ISTAT/Eurostat ~30-32 days after the month"),
    ("GBR", "gb", "United Kingdom", 85,
     "ONS LFS is a rolling 3-month average; OECD dates it at the middle month, which is public ~6-7 weeks after the "
     "window ends (~75-80 days after the dated month)"),
    ("KOR", "kr", "South Korea", 20, "Statistics Korea survey out mid-following month (~10-17 days)"),
    ("AUS", "au", "Australia", 25, "ABS Labour Force out ~2-3 weeks after the month"),
    ("MEX", "mx", "Mexico", 35, "INEGI ENOE out ~4 weeks after the month (Apr-Jun 2020: telephone survey ETOE)"),
    ("ESP", "es", "Spain", 40, "Eurostat monthly estimate ~30-32 days after the month (model-based on the quarterly EPA)"),
]
PANEL = ["AUS", "AUT", "BEL", "CAN", "CZE", "DEU", "DNK", "ESP", "FIN", "FRA", "IRL", "ITA", "JPN", "KOR", "MEX",
         "NLD", "POL", "PRT", "SWE", "USA"]
K_PANEL = 45


def main():
    part = plib.Part("40_oecd_unemployment")
    data = {}
    with open(RAWF, newline="") as f:
        for r in csv.DictReader(f):
            y, m = map(int, r["TIME_PERIOD"].split("-"))
            data.setdefault(r["REF_AREA"], {})[(y, m)] = float(r["OBS_VALUE"])
    for code, slug, nm, k, tnote in COUNTRIES:
        rows = [(plib.month_end(y, m), v) for (y, m), v in sorted(data[code].items())]
        part.add(f"oecd_ur_{slug}", rows, name=f"Unemployment rate, {nm} (OECD harmonised, SA)", subcategory="country",
                 region=code, source=SRC, url=URL, frequency="monthly", units="percent of labour force", transform="diff",
                 known_after_days=k, notes=f"known_after_days {k}: {tnote}. {REV} {LIC}", lo=0, hi=40)
    months = sorted(set.intersection(*[set(data[c]) for c in PANEL]))
    st = {k: [] for k in ["median", "p10", "p90", "max", "min", "rose"]}
    prev = None
    for (y, m) in months:
        x = np.array([data[c][(y, m)] for c in PANEL])
        d = plib.month_end(y, m)
        p10, p50, p90 = np.percentile(x, [10, 50, 90])
        st["median"].append((d, p50))
        st["p10"].append((d, p10))
        st["p90"].append((d, p90))
        st["max"].append((d, float(x.max())))
        st["min"].append((d, float(x.min())))
        pm = (y, m - 1) if m > 1 else (y - 1, 12)
        if pm in months and prev is not None:
            px = np.array([data[c][pm] for c in PANEL])
            st["rose"].append((d, 100.0 * float(np.mean(x > px))))
        prev = (y, m)
    print("panel months", months[0], months[-1], len(months))
    pn = (f"Across a fixed panel of {len(PANEL)} OECD countries with monthly labour-force data ({', '.join(PANEL)}); "
          f"months where all {len(PANEL)} have a value. known_after_days {K_PANEL} covers the slowest member (Eurostat "
          f"countries ~30-32 days). The UK, Norway, Chile, Colombia, Costa Rica, Turkey (3-month averages) and small "
          f"Eurostat-derived series are left out. {REV} {LIC}")
    common = dict(subcategory="country_distribution", region=f"OECD panel of {len(PANEL)} countries", source=SRC, url=URL,
                  frequency="monthly", known_after_days=K_PANEL)
    part.add("oecd_ur_panel_median", st["median"], name="OECD countries: median unemployment rate across 20 countries",
             units="percent of labour force", transform="diff", notes=pn, lo=0, hi=30, **common)
    part.add("oecd_ur_panel_p10", st["p10"], name="OECD countries: 10th-percentile unemployment rate across 20 countries",
             units="percent of labour force", transform="diff", notes=pn, lo=0, hi=30, **common)
    part.add("oecd_ur_panel_p90", st["p90"], name="OECD countries: 90th-percentile unemployment rate across 20 countries",
             units="percent of labour force", transform="diff", notes=pn, lo=0, hi=30, **common)
    part.add("oecd_ur_panel_max", st["max"], name="OECD countries: maximum unemployment rate across 20 countries",
             units="percent of labour force", transform="diff", notes=pn, lo=0, hi=40, **common)
    part.add("oecd_ur_panel_min", st["min"], name="OECD countries: minimum unemployment rate across 20 countries",
             units="percent of labour force", transform="diff", notes=pn, lo=0, hi=20, **common)
    part.add("oecd_ur_panel_share_rose_mom", st["rose"], name="OECD countries: share of 20 countries whose unemployment rate rose this month",
             units="percent of countries", transform="level",
             notes=f"Diffusion index (already a change), so transform level; ties at 0.1 pp count as not rising. {pn}",
             lo=0, hi=100, **common)
    part.save()


if __name__ == "__main__":
    main()
