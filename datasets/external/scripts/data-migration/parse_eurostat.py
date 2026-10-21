"""Eurostat monthly asylum applicants (migr_asyappctzm), all applicants (first-time + subsequent),
all citizenships, both sexes, all ages, persons. JSON-stat from the dissemination API."""
import collections
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from seasonality import seasonal_r2

F = os.path.join(RAW, "eurostat_asylum", "migr_asyappctzm_citizenTOTAL_sexT_ageTOTAL.json")
j = json.load(open(F))
dims, size = j["id"], j["size"]
geo = j["dimension"]["geo"]["category"]
tim = j["dimension"]["time"]["category"]
gi = {v: k for k, v in geo["index"].items()}
ti = {v: k for k, v in tim["index"].items()}
ai = {v: k for k, v in j["dimension"]["applicant"]["category"]["index"].items()}
strides = [1] * len(size)
for i in range(len(size) - 2, -1, -1):
    strides[i] = strides[i + 1] * size[i + 1]
pos = {d: i for i, d in enumerate(dims)}
data = collections.defaultdict(dict)
flags = collections.defaultdict(collections.Counter)
status = j.get("status", {})
for k, v in j["value"].items():
    kk = int(k)
    idx = [(kk // strides[i]) % size[i] for i in range(len(size))]
    if ai[idx[pos["applicant"]]] != "TOTAL":
        continue
    g = gi[idx[pos["geo"]]]
    t = ti[idx[pos["time"]]]
    data[g][dt.date(int(t[:4]), int(t[5:7]), 1)] = v
    if k in status:
        flags[g][status[k]] += 1

URL = "https://ec.europa.eu/eurostat/databrowser/view/migr_asyappctzm/default/table"
K = 110
base = ("Eurostat migr_asyappctzm, applicant=TOTAL (first-time + subsequent), citizen=TOTAL, sex=T, "
        "age=TOTAL, persons; dated the 1st of the month. Rounded to the nearest 5 by Eurostat; revised "
        f"as countries resubmit. Dataset updated {j['updated'][:10]}; known_after_days=110 from the 1st: "
        "member states must report within two months of month end and Eurostat posts mid-month, and on "
        "2026-10-16 the latest month was July 2026 for the EU aggregate, DE and ES (107 days). "
        "Ukrainians fleeing the 2022 invasion mostly received temporary protection, which is not counted "
        "here.")
SEL = [
    ("EU27_2020", "eu27", "European Union (27, 2020 composition)", "EU", "logdiff",
     "The EU aggregate starts only in 2013, too short for the 3-year anomaly warm-up, so logdiff "
     "(moderately seasonal: calendar months explain about 26% of monthly log changes)."),
    ("DE", "de", "Germany", "DE", "anomaly_log1p", ""),
    ("FR", "fr", "France", "FR", "anomaly_log1p", ""),
    ("IT", "it", "Italy", "IT", "anomaly_log1p", "Driven by Mediterranean sea arrivals (seasonal)."),
    ("ES", "es", "Spain", "ES", "anomaly_log1p", "Driven by Venezuelan and Colombian applicants after 2017."),
    ("EL", "el", "Greece", "GR", "anomaly_log1p", "Eastern Mediterranean route; one zero month."),
    ("AT", "at", "Austria", "AT", "anomaly_log1p", ""),
    ("SE", "se", "Sweden", "SE", "anomaly_log1p", ""),
    ("NL", "nl", "Netherlands", "NL", "anomaly_log1p", ""),
    ("BE", "be", "Belgium", "BE", "anomaly_log1p", ""),
    ("CH", "ch", "Switzerland", "CH", "anomaly_log1p", "Non-EU, Dublin associated."),
    ("UK", "uk", "United Kingdom", "GB", "anomaly_log1p", "Reported to Eurostat only 2009-01 to 2020-11 (Brexit); the series ends there."),
]
for g, slug, label, region, tf, extra in SEL:
    rows = sorted(data[g].items())
    r2, n = seasonal_r2(rows)
    if tf == "anomaly_log1p" and r2 < 0.2:
        tf = "logdiff"  # rule: anomaly_log1p only where calendar months explain >= 20% of log changes
    note = (base + f" Calendar months explain {100 * r2:.1f}% of the variance of monthly log changes. "
            + (("Seasonal (at least 20%) and long enough (from " + rows[0][0].isoformat()[:7] + "), hence "
                "anomaly_log1p; the test's 3-year warm-up means moves start 3 years after the first month. ")
               if tf == "anomaly_log1p" else
               ("Weakly seasonal (under 20%), hence logdiff. " if g != "EU27_2020" else "")) + extra
            + (f" Status flags: {dict(flags[g])}." if flags[g] else ""))
    zeros = [d.isoformat()[:7] for d, v in rows if v == 0]
    if tf == "logdiff" and zeros:
        note += (f" Zero month(s) {', '.join(zeros)} kept as observed (registration suspended); the test's log "
                 "transform drops them and does not bridge the gap, so no move is counted around them.")
    write_series(f"eurostat_asy_{slug}", rows, name=f"Asylum applicants, {label} (monthly)",
                 category="migration", subcategory="asylum", region=region, source="Eurostat",
                 url=URL, frequency="monthly", units="persons per month (first-time + subsequent applicants)",
                 transform=tf, known_after_days=K, notes=note, value_range=(0, 5e5), allow_zero=True)
