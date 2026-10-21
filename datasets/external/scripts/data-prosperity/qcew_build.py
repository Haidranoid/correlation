"""BLS QCEW county totals (all industries, all ownerships: agglvl 70, own 0, industry 10) -> cross-county
distribution of average weekly wage (causal SA) and employment growth. Quarterly, dated quarter end."""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402
import sa  # noqa: E402
from counties import STATES  # noqa: E402

RAWD = os.path.join(plib.RAW, "bls_qcew")
K = 240


def main():
    part = plib.Part("50_bls_qcew")
    manifest = json.load(open(os.path.join(RAWD, "manifest.json")))
    years = sorted(int(y) for y in manifest)
    print("QCEW years", years[0], years[-1], len(years))
    wage, emp = {}, {}
    for y in years:
        with open(os.path.join(RAWD, manifest[str(y)]["saved_as"]), newline="", encoding="latin-1") as f:
            for r in csv.DictReader(f):
                if r["agglvl_code"] != "70" or r["own_code"] != "0" or r["industry_code"] != "10":
                    continue
                a = r["area_fips"]
                if not (len(a) == 5 and a.isdigit() and a[:2] in STATES and not a.endswith("999") and not a.endswith("000")):
                    continue
                q = (int(r["year"]) - 1990) * 4 + int(r["qtr"]) - 1
                if r["disclosure_code"].strip() == "N":
                    continue
                w = float(r["avg_wkly_wage"] or 0)
                e = [float(r[k] or 0) for k in ("month1_emplvl", "month2_emplvl", "month3_emplvl")]
                if w > 0:
                    wage.setdefault(a, {})[q] = w
                if min(e) > 0:
                    emp.setdefault(a, {})[q] = sum(e) / 3.0
    areas = sorted(set(wage) | set(emp))
    nq = max(max(d) for d in wage.values()) + 1
    W = np.full((len(areas), nq), np.nan)
    E = np.full((len(areas), nq), np.nan)
    for i, a in enumerate(areas):
        for q, v in wage.get(a, {}).items():
            W[i, q] = v
        for q, v in emp.get(a, {}).items():
            E[i, q] = v
    print("counties", len(areas), "quarters", nq)
    WSA = sa.causal_sa_matrix(W, p=4, log=True, K=8)
    st = {k: [] for k in ["median", "p10", "p90", "ratio", "emp_yoy", "emp_grew"]}
    n_ok = []
    for q in range(nq):
        d = plib.quarter_end(1990 + q // 4, q % 4 + 1)
        x = WSA[:, q]
        ok = np.isfinite(x)
        if ok.sum() >= 0.9 * len(areas):
            p10, p50, p90 = np.percentile(x[ok], [10, 50, 90])
            st["median"].append((d, p50))
            st["p10"].append((d, p10))
            st["p90"].append((d, p90))
            st["ratio"].append((d, p90 / p10))
            n_ok.append(int(ok.sum()))
        if q >= 4:
            e1, e0 = E[:, q], E[:, q - 4]
            ok2 = np.isfinite(e1) & np.isfinite(e0)
            if ok2.sum() >= 0.9 * len(areas):
                g = 100.0 * np.log(e1[ok2] / e0[ok2])
                st["emp_yoy"].append((d, float(np.median(g))))
                st["emp_grew"].append((d, 100.0 * float(np.mean(g > 0))))
    print("counties with SA wage per quarter: min", min(n_ok), "max", max(n_ok))
    timing = (f"known_after_days {K}: county QCEW is released ~5-6 months after the quarter (BLS archive 2003-2025: median "
              "171 days, max 214 for 2003Q1; earlier releases undocumented, so margin added). Each quarter is revised in "
              "the following releases and finalised with the annual file.")
    url = "https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_by_industry.zip (member '... 10 Total, all industries.csv')"
    common = dict(subcategory="county_distribution", region="US counties (QCEW, 50 states + DC)",
                  source="BLS Quarterly Census of Employment and Wages (QCEW)", url=url, frequency="quarterly",
                  known_after_days=K)
    sa_note = ("Average weekly wage of all covered jobs in the county (NSA; Q1 carries bonuses), seasonally adjusted here "
               "causally (median of the same quarter's log deviation from a centred 2x4 moving average over the previous "
               "8 years, >= 3; only data >= 2 quarters older). Unweighted across ~3,150 counties; suppressed cells dropped. ")
    tail = f"{timing} Nominal dollars. BLS data, public domain."
    part.add("qcew_cty_wage_median", st["median"], name="US counties: median average weekly wage across counties (QCEW, causal SA)",
             units="USD per week (nominal)", transform="logdiff", notes=sa_note + tail, lo=100, hi=5000, **common)
    part.add("qcew_cty_wage_p10", st["p10"], name="US counties: 10th-percentile average weekly wage across counties (QCEW, causal SA)",
             units="USD per week (nominal)", transform="logdiff", notes=sa_note + tail, lo=100, hi=5000, **common)
    part.add("qcew_cty_wage_p90", st["p90"], name="US counties: 90th-percentile average weekly wage across counties (QCEW, causal SA)",
             units="USD per week (nominal)", transform="logdiff", notes=sa_note + tail, lo=100, hi=5000, **common)
    part.add("qcew_cty_wage_p90_p10_ratio", st["ratio"], name="US counties: p90/p10 ratio of county average weekly wages (QCEW, causal SA)",
             units="ratio", transform="diff", notes=sa_note + tail, lo=1, hi=5, **common)
    en = ("Average of the quarter's three monthly employment levels; 4-quarter log change (no seasonal adjustment needed). "
          "Unweighted across counties. ")
    part.add("qcew_cty_emp_yoy_median", st["emp_yoy"], name="US counties: median year-over-year employment growth across counties (QCEW)",
             units="percent (100 x log change over 4 quarters)", transform="diff", notes=en + tail, lo=-30, hi=30, **common)
    part.add("qcew_cty_emp_share_grew_yoy", st["emp_grew"], name="US counties: share of counties with employment above a year earlier (QCEW)",
             units="percent of counties", transform="diff", notes=en + tail, lo=0, hi=100, **common)
    part.save()


if __name__ == "__main__":
    main()
