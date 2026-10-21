"""Australia (ABS): quarterly net overseas migration and overseas migrant arrivals, from the ABS Data
API dataflow ERP_COMP_Q (Population and components of change - national, states and territories)."""
import csv
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from seasonality import seasonal_r2

F = os.path.join(RAW, "abs_migration", "ERP_COMP_Q_all_1981.csv")
m = {}
with open(F, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if not r["REGION: Region"].startswith("AUS"):
            continue
        assert r["UNIT_MULT: Unit of Multiplier"].startswith("3") or r["MEASURE: Measure"].startswith("13"), r
        y, qq = r["TIME_PERIOD: Time Period"].split("-Q")
        d = dt.date(int(y), 3 * (int(qq) - 1) + 1, 1)
        m.setdefault(r["MEASURE: Measure"], {})[d] = round(float(r["OBS_VALUE"]) * 1000.0)

URL = "https://data.api.abs.gov.au/rest/data/ABS,ERP_COMP_Q,1.0.0/all"
base = ("ABS 'National, state and territory population' components (Data API dataflow ERP_COMP_Q), "
        "Australia; dated the first day of the quarter; published in thousands to one decimal, converted "
        "to persons. known_after_days=275 from the quarter start: the quarterly release comes about 5.5-6 "
        "months after the quarter ends (e.g. December 2024 data in June 2025). Net overseas migration "
        "uses the '12/16-month rule', so preliminary quarters are model-based and are revised when final "
        "(about 18 months later); stored values are the latest revision. Category of travel changed in "
        "2006 (12/16-month rule replaced the 12/12 rule).")
nom = sorted(m["9: Net Overseas Migration"].items())
write_series("abs_nom", nom, name="Net overseas migration, Australia (quarterly)", category="migration",
             subcategory="net_migration", region="AU", source="Australian Bureau of Statistics", url=URL,
             frequency="quarterly", units="persons per quarter (net; can be negative)", transform="diff",
             known_after_days=275,
             notes=base + " Negative in 2020-Q2 to 2021 (border closure), record +165,500 in 2023-Q1. "
                          "Can be negative, hence diff.", value_range=(-2e5, 5e5))
arr = sorted(m["7: Overseas Arrivals"].items())
r2, n = seasonal_r2(arr, period="quarter")
write_series("abs_overseas_arrivals", arr, name="Overseas migrant arrivals, Australia (quarterly)",
             category="migration", subcategory="immigration", region="AU", source="Australian Bureau of Statistics",
             url=URL, frequency="quarterly", units="persons per quarter",
             transform="anomaly_log1p" if r2 >= 0.2 else "logdiff", known_after_days=275,
             notes=base + f" Overseas migrant arrivals (people who stay 12 of the next 16 months), the inflow "
                          f"side of NOM. Calendar quarters explain {100 * r2:.0f}% of quarterly log changes"
                          + (", hence anomaly_log1p (history from 1981)." if r2 >= 0.2 else ", hence logdiff."),
             value_range=(1, 1e6))
