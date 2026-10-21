"""Normalize US Drought Monitor % area in D1-D4 (moderate drought or worse), weekly.

Usage: python3 -I parse_usdm.py
"""
import csv
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, drop_entries, qa_line  # noqa: E402

D = os.path.join(RAW, "usdm")
API = "https://usdmdataservices.unl.edu/api/"
AREAS = [("conus", "usdm_conus_cumulative.csv", "AreaOfInterest", "CONUS", "US CONUS",
          "USStatistics/GetDroughtSeverityStatisticsByAreaPercent?aoi=us&statisticsType=1"),
         ("ia", "usdm_IA_cumulative.csv", "StateAbbreviation", "IA", "US-IA Iowa", "aoi=19"),
         ("il", "usdm_IL_cumulative.csv", "StateAbbreviation", "IL", "US-IL Illinois", "aoi=17"),
         ("ne", "usdm_NE_cumulative.csv", "StateAbbreviation", "NE", "US-NE Nebraska", "aoi=31"),
         ("ks", "usdm_KS_cumulative.csv", "StateAbbreviation", "KS", "US-KS Kansas", "aoi=20"),
         ("tx", "usdm_TX_cumulative.csv", "StateAbbreviation", "TX", "US-TX Texas", "aoi=48"),
         ("ca", "usdm_CA_cumulative.csv", "StateAbbreviation", "CA", "US-CA California", "aoi=06")]


def main():
    drop_entries("usdm_")
    for key, fn, col, want, region, q in AREAS:
        rows = {}
        with open(os.path.join(D, fn), newline="", encoding="latin-1") as f:
            for r in csv.DictReader(f):
                if r[col] != want:
                    continue
                d = dt.datetime.strptime(r["MapDate"], "%Y%m%d").date()
                assert d.weekday() == 1, d  # Tuesday valid date
                vals = [float(r[k]) for k in ("D0", "D1", "D2", "D3", "D4")]
                assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])), (fn, d, vals)  # cumulative check
                assert abs(float(r["None"]) + vals[0] - 100) < 0.05, (fn, d)
                assert d not in rows, (fn, d)
                rows[d] = vals[1]
        url = API + (q if key == "conus" else
                     "StateStatistics/GetDroughtSeverityStatisticsByAreaPercent?" + q + "&statisticsType=1")
        url += "&startdate=1/1/2000&enddate=10/18/2026"
        m = dict(name=f"US Drought Monitor: % of {region.split(' ', 1)[1]} area in drought D1-D4",
                 category="weather", subcategory="drought", region=region,
                 source="US Drought Monitor (NDMC/USDA/NOAA)", url=url, frequency="weekly",
                 units="percent of area", transform="diff", known_after_days=2,
                 notes=("weekly map valid Tuesday 8 am ET (date = that Tuesday), released Thursday 8:30 am ET, so 2 days; "
                        "maps are not revised after release; value = cumulative D1 column (moderate drought or worse, "
                        "D1+D2+D3+D4) from statisticsType=1; checked None+D0=100 and D0>=D1>=...>=D4 on every row; "
                        "expert-blended analysis (not a pure physical measurement), weekly since 2000-01-04; request with header "
                        "Accept: text/csv; state queries need the FIPS code (aoi=19), abbreviations return nothing"))
        qa = write_series(f"usdm_{key}_d1plus", sorted(rows.items()), m, 0.0, 100.0, nd=2)
        print(qa_line(qa))


if __name__ == "__main__":
    main()
