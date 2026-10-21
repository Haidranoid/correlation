"""World Inequality Database (bulk files) -> US top 1% / top 10% / bottom 50% shares of pre-tax national
income and net personal wealth, and cross-country medians over the same 40 economies as the WDI panel.
Variables: sptincj992 / shwealj992 (equal-split adults 20+). Annual, dated December 31.

Values flagged data_quality 0 and trailing years that just repeat the previous year's value (WID carries
the last estimate forward) are dropped, since they are not new observations."""
import datetime as dt
import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402
from wid_fetch import PANEL  # noqa: E402

RAWD = os.path.join(plib.RAW, "wid")
K = 1100
VARS = {"sptincj992": "pre-tax national income", "shwealj992": "net personal wealth"}
PCT = {"p99p100": "top 1%", "p90p100": "top 10%", "p0p50": "bottom 50%"}


def load(c):
    out = {}
    with gzip.open(os.path.join(RAWD, f"WID_data_{c}.csv.gz"), "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(";")
            if p[0] != c or p[1] not in VARS or p[2] not in PCT:
                continue
            q = p[7].strip() if len(p) > 7 else ""
            out.setdefault((p[1], p[2]), {})[int(p[3])] = (float(p[4]), q)
    return out


def clean(series):
    """{year: (value, quality)} -> sorted [(year, value)] without quality-0 points and trailing repeats."""
    ys = sorted(y for y, (v, q) in series.items() if q != "0")
    rows = [(y, series[y][0]) for y in ys]
    while len(rows) >= 2 and rows[-1][1] == rows[-2][1]:
        rows.pop()
    return rows


def main():
    part = plib.Part("90_wid")
    data = {c: load(c) for c in PANEL}
    timing = (f"known_after_days {K}: WID updates once a year and its latest genuine year trails by ~2-3 years (the "
              "2025-26 update's US data are real through 2022, later years carried forward and dropped here). BACK-CAST: "
              "the US distributional national accounts were first published in 2016-2017 (Piketty-Saez-Zucman), so "
              "earlier values were never public at date + known_after_days. Revised with every update. ANNUAL: "
              "essentially untestable at the census move lengths; included because the user asked. Source: WID.world "
              "free bulk download; cite WID.world and the papers in its metadata (licence page not reachable here).")
    us = data["US"]
    for var, vname in VARS.items():
        for pct, pname in PCT.items():
            rows = [(dt.date(y, 12, 31), 100.0 * v) for y, v in clean(us[(var, pct)]) if y >= 1913]
            sid = f"wid_us_{'top1' if pct == 'p99p100' else 'top10' if pct == 'p90p100' else 'bottom50'}_{'income' if var.startswith('sptinc') else 'wealth'}_share"
            part.add(sid, rows, name=f"United States: {pname} share of {vname} (WID)", subcategory="country", region="USA",
                     source="World Inequality Database (WID.world)", url="https://wid.world/bulk_download/wid_all_data.zip",
                     frequency="annual", units=f"percent of {vname}", transform="diff", known_after_days=K,
                     notes=(f"WID {var} {pct}: equal-split adults aged 20+. Annual from 1913 (earlier points are sparse "
                            f"benchmarks and left out). {timing}"), lo=-5, hi=100)
    # cross-country median of the top 10% income share over the economies whose WID series is mostly of good
    # quality (data_quality >= 3 in at least 30 of the 45 years 1980-2024); years where all of them have a value.
    var, pct = "sptincj992", "p90p100"
    good = [c for c in PANEL if sum(1 for y, (v, q) in data[c][(var, pct)].items() if y >= 1980 and q in ("3", "4", "5")) >= 30]
    good = [c for c in good if max(y for y, _ in clean(data[c][(var, pct)])) >= 2021]  # genuine values through 2021+
    cl = {c: dict(clean(data[c][(var, pct)])) for c in good}
    years = sorted(set.intersection(*[set(y for y in cl[c] if y >= 1980) for c in good]))
    rows = [(dt.date(y, 12, 31), 100.0 * float(np.median([cl[c][y] for c in good]))) for y in years]
    print("WID panel", len(good), good, "years", years[0], years[-1], len(years))
    part.add("wid_panel_top10_income_share_median", rows,
             name=f"Countries: median top 10% share of pre-tax national income across {len(good)} economies with good WID data",
             subcategory="country_distribution", region=f"{len(good)} economies", source="World Inequality Database (WID.world)",
             url="https://wid.world/bulk_download/wid_all_data.zip", frequency="annual", units="percent", transform="diff",
             known_after_days=K,
             notes=(f"WID {var} {pct}, median across {' '.join(good)}: the economies of the 40-country WDI panel whose series "
                    f"has WID data quality >= 3 in at least 30 of the years 1980-2024 (the others are mostly imputed) and genuine values "
                    f"through 2021 or later; years "
                    f"where all have a genuine value. A wealth-share panel was not built: wealth shares are quality 0 "
                    f"(imputed) for most of the panel. {timing}"), lo=0, hi=100)
    part.save()


if __name__ == "__main__":
    main()
