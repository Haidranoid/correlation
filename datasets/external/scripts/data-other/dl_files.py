"""Download the single-file sources (raw, unmodified)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW

C = "https://www2.census.gov/programs-surveys/popest/datasets"
FILES = [
    # population
    ("fred", "POPTHM.csv", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=POPTHM"),
    ("fred", "CNP16OV.csv", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CNP16OV"),
    ("worldbank", "WLD_SP.POP.TOTL.json",
     "https://api.worldbank.org/v2/country/WLD/indicator/SP.POP.TOTL?format=json&per_page=200"),
    ("census", "st-est00int-alldata.csv", f"{C}/2000-2010/intercensal/state/st-est00int-alldata.csv"),
    ("census", "sc-est2020int-alldata6.csv",
     f"{C}/2010-2020/intercensal/state/asrh/sc-est2020int-alldata6.csv"),
    ("census", "NST-EST2025-ALLDATA.csv", f"{C}/2020-2025/state/totals/NST-EST2025-ALLDATA.csv"),
    # space weather
    ("gfz", "Kp_ap_Ap_SN_F107_since_1932.txt",
     "https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"),
    ("silso", "SN_d_tot_V2.0.csv", "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"),
    # health
    ("delphi", "fluview_nat_latest.json",
     "https://api.delphi.cmu.edu/epidata/fluview/?regions=nat&epiweeks=199740-202652"),
    ("delphi", "fluview_nat_lag0.json",
     "https://api.delphi.cmu.edu/epidata/fluview/?regions=nat&epiweeks=199740-202652&lag=0"),
    ("nyt_covid", "us.csv", "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv"),
    ("policyuncertainty", "All_Infectious_EMV_Data.csv",
     "https://www.policyuncertainty.com/media/All_Infectious_EMV_Data.csv"),
    # other
    ("gpr", "data_gpr_daily_recent.dta", "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta"),
    ("gpr", "data_gpr_daily_recent.xls", "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"),
    ("eia", "WGFUPUS2_leaf.html", "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WGFUPUS2&f=W"),
    ("eia", "WGFUPUS2w.xls", "https://www.eia.gov/dnav/pet/hist_xls/WGFUPUS2w.xls"),
    ("eia", "Region_US48.xlsx", "https://www.eia.gov/electricity/gridmonitor/knownissues/xls/Region_US48.xlsx"),
]
for sub, name, url in FILES:
    dest = os.path.join(RAW, sub, name)
    try:
        fetch(url, dest, sleep=1.5, timeout=600)
        print(f"ok   {sub}/{name}  {os.path.getsize(dest)} bytes", flush=True)
    except Exception as e:
        print(f"FAIL {sub}/{name}: {e}", flush=True)
print("done", flush=True)
