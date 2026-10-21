"""CBP: USBP monthly apprehensions/encounters by sector FY2000-FY2020 (PDF tables) and the
Nationwide Encounters CSVs (FY2020 onward). Also records the server Last-Modified of every monthly
CSV posted on the Nationwide Encounters page, as evidence of the release lag."""
import os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW, UA

D = os.path.join(RAW, "cbp")
B = "https://www.cbp.gov"
FILES = [
    ("usbp_monthly_encounters_fy2000-fy2020.pdf",
     B + "/sites/default/files/assets/documents/2021-Aug/U.S.%20Border%20Patrol%20Monthly%20Encounters%20%28FY%202000%20-%20FY%202020%29%20%28508%29.pdf"),
    ("usbp_monthly_apprehensions_fy2000-fy2019.pdf",
     B + "/sites/default/files/assets/documents/2020-Jan/U.S.%20Border%20Patrol%20Monthly%20Apprehensions%20%28FY%202000%20-%20FY%202019%29_1.pdf"),
    ("nationwide-encounters-data-dictionary.pdf",
     B + "/sites/default/files/assets/documents/2023-Sep/nationwide-encounters-data-dictionary.pdf"),
    ("nationwide-encounters-fy20-fy23-aor.csv",
     B + "/sites/default/files/assets/documents/2023-Nov/nationwide-encounters-fy20-fy23-aor.csv"),
    ("nationwide-encounters-fy21-fy24-aor.csv", B + "/sites/default/files/2024-10/nationwide-encounters-fy21-fy24-aor.csv"),
    ("nationwide-encounters-fy22-fy25-aor.csv", B + "/sites/default/files/2025-11/nationwide-encounters-fy22-fy25-aor.csv"),
    ("nationwide-encounters-fy23-fy26-aug-aor.csv", B + "/sites/default/files/2026-09/nationwide-encounters-fy23-fy26-aug-aor_0.csv"),
]
for name, url in FILES:
    dest = os.path.join(D, name)
    try:
        fetch(url, dest, sleep=2.0, timeout=600)
        print(f"ok   {name} {os.path.getsize(dest)}", flush=True)
    except Exception as e:
        print(f"FAIL {name}: {e}", flush=True)

# Release-lag evidence: the page lists one CSV per monthly update; HEAD each for Last-Modified.
page = os.path.join(D, "nationwide-encounters-page.html")
fetch(B + "/document/stats/nationwide-encounters", page, sleep=2.0)
links = sorted(set(re.findall(r'href="(/sites/default/files/[^"]*nationwide-encounters-[^"]*-aor[^"]*\.csv)"',
                              open(page, encoding="utf-8", errors="replace").read())))
out = os.path.join(D, "release_lag_evidence.tsv")
with open(out, "w") as f:
    f.write("file\tlast_modified\n")
    for l in links:
        req = urllib.request.Request(B + l, method="HEAD", headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                lm = r.headers.get("Last-Modified", "")
        except Exception as e:
            lm = f"ERR {e}"
        f.write(f"{l}\t{lm}\n")
        print(l, lm, flush=True)
        time.sleep(2.0)
print("done")
