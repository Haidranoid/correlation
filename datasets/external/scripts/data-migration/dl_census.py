"""Download the Census Population Estimates Program county and state files (raw, unmodified)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW

C = "https://www2.census.gov/programs-surveys/popest/datasets"
FILES = [
    # county, 2000-2010 intercensal (totals only, no components)
    ("co-est00int-tot.csv", f"{C}/2000-2010/intercensal/county/co-est00int-tot.csv"),
    # county, Vintage 2009 postcensal with components of change 2000-2009
    ("co-est2009-alldata.csv", f"{C}/2000-2009/counties/totals/co-est2009-alldata.csv"),
    # county, 2010-2020 intercensal (age/sex file holds the totals; released Sep 2025)
    ("cc-est2020int-agesex-all.csv", f"{C}/2010-2020/intercensal/county/asrh/cc-est2020int-agesex-all.csv"),
    # county, Vintage 2020 postcensal with components of change 2010-2020
    ("co-est2020-alldata.csv", f"{C}/2010-2020/counties/totals/co-est2020-alldata.csv"),
    # county, Vintage 2025 postcensal with components 2020-2025 (released Mar 2026)
    ("co-est2025-alldata.csv", f"{C}/2020-2025/counties/totals/co-est2025-alldata.csv"),
    # state, postcensal with components (V2009, V2020, V2025)
    ("NST_EST2009_ALLDATA.csv", f"{C}/2000-2009/state/totals/NST_EST2009_ALLDATA.csv"),
    ("nst-est2020-alldata.csv", f"{C}/2010-2020/state/totals/nst-est2020-alldata.csv"),
    ("NST-EST2025-ALLDATA.csv", f"{C}/2020-2025/state/totals/NST-EST2025-ALLDATA.csv"),
    # state, 2010-2020 intercensal totals
    ("st-est2020int-pop.csv", f"{C}/2010-2020/intercensal/state/st-est2020int-pop.csv"),
    # state, 2000-2010 intercensal totals
    ("st-est00int-01.csv", f"{C}/2000-2010/intercensal/state/st-est00int-01.csv"),
]
for name, url in FILES:
    dest = os.path.join(RAW, "census_popest", name)
    try:
        fetch(url, dest, sleep=1.5, timeout=900)
        print(f"ok   {name}  {os.path.getsize(dest)} bytes", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {e}", flush=True)
print("done", flush=True)
