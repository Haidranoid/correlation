"""Single-file downloads for Eurostat, IRCC, Statistics Canada, UK Home Office, ABS, OHSS, USCIS and the
World Bank (raw, unmodified; fetch() skips files already present). Each raw folder keeps a
_downloads.tsv with the URL, size and server Last-Modified of every file."""
import os, sys, urllib.parse, zipfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW

BROWSER = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
OHSS = "https://ohss.dhs.gov"
LIAS = [
    "/system/files/2026-06/2026_0604_ohss_legal-immigration-adjustment-of-status-fy-2025q4.xlsx",
    "/sites/default/files/2025-06/2025_0624_ohss_legal-immigration-adjustment-of-status-fy-2024q4_0.xlsx",
    "/sites/default/files/2024-06/2024_0507_ohss_legal-immigration-adjustment-of-status-fy-2023q4.xlsx",
    "/sites/default/files/2023-12/2023_0308_plcy_legal_immigration_adjustment_of_status_report_fy_2022q4_final_d_0.xlsx",
    "/sites/default/files/2023-12/22_0323_plcy_legal_immigration_and_adjustment_of_status_report_fy2021_q4_final_d_version.xlsx",
    "/sites/default/files/2023-12/fy_2020_q4_d_final_v2.pdf.xlsx",
    "/sites/default/files/2023-12/fy2019_q4_d_final.xlsx",
    "/sites/default/files/2023-12/fy2018_q1q2q3q4_tables_d.xls_.xlsx",
    "/sites/default/files/2023-12/fy2017_q1q2q3q4_tables_d.xlsx",
    "/sites/default/files/2023-12/fy2016_q1q2q3q4_tables_d.xlsx",
]
FILES = [
    ("eurostat_asylum", "migr_asyappctzm_citizenTOTAL_sexT_ageTOTAL.json",
     "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/migr_asyappctzm?format=JSON&lang=EN"
     "&citizen=TOTAL&sex=T&age=TOTAL&unit=PER&applicant=FRST&applicant=TOTAL", None),
    ("ircc", "EN_ODP-PR-ProvImmCat.xlsx", "https://www.ircc.canada.ca/opendata-donneesouvertes/data/EN_ODP-PR-ProvImmCat.xlsx", None),
    ("ircc", "ODP-PR-PT_IMMCAT.csv", "https://www.ircc.canada.ca/opendata-donneesouvertes/data/ODP-PR-PT_IMMCAT.csv", None),
    ("statcan_migration", "17100040-eng.zip", "https://www150.statcan.gc.ca/n1/tbl/csv/17100040-eng.zip", None),
    ("uk_homeoffice", "16_October_2026_Small_boats_-_time_series.ods",
     "https://assets.publishing.service.gov.uk/media/6ad1f5b949ba86d6c953cabd/16_October_2026_Small_boats_-_time_series.ods", None),
    ("uk_homeoffice", "asylum-claims-datasets-jun-2026.xlsx",
     "https://assets.publishing.service.gov.uk/media/6a85c2d7c9205b515d421eeb/asylum-claims-datasets-jun-2026.xlsx", None),
    ("uk_homeoffice", "asylum-summary-jun-2026-tables.ods",
     "https://assets.publishing.service.gov.uk/media/6a85c315b0504df9f2c89800/asylum-summary-jun-2026-tables.ods", None),
    ("uk_homeoffice", "visas-summary-jun-2026-tables.ods",
     "https://assets.publishing.service.gov.uk/media/6a85c6003d82f78d5c514552/visas-summary-jun-2026-tables.ods", None),
    ("abs_migration", "ERP_COMP_Q_all_1981.csv",
     "https://data.api.abs.gov.au/rest/data/ABS,ERP_COMP_Q,1.0.0/all?startPeriod=1981-Q1&dimensionAtObservation=AllDimensions",
     {"Accept": "application/vnd.sdmx.data+csv;labels=both"}),
    ("ohss", "ohss_monthly_tables_2024-11.xlsx",
     OHSS + "/sites/default/files/2025-01/2025_0116_ohss_immigration-enforcement-and-legal-processes-tables-november-2024.xlsx", BROWSER),
    ("uscis", "quarterly_all_forms_fy2026_q3_v1.xlsx",
     "https://www.uscis.gov/sites/default/files/document/data/quarterly_all_forms_fy2026_q3_v1.xlsx", BROWSER),
    ("uscis", "n400_performance_data_fy2026_q3_v1.xlsx",
     "https://www.uscis.gov/sites/default/files/document/data/n400_performance_data_fy2026_q3_v1.xlsx", BROWSER),
    ("uscis", "i485_performance_data_fy2026_q3_v1.xlsx",
     "https://www.uscis.gov/sites/default/files/document/data/i485_performance_data_fy2026_q3_v1.xlsx", BROWSER),
] + [("ohss", "lias_" + os.path.basename(p), OHSS + p, BROWSER) for p in LIAS] + [
    ("worldbank_migration", f"{ind}_all.json",
     f"https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date=1960:2025", None)
    for ind in ("SM.POP.NETM", "SP.POP.GROW", "SP.POP.TOTL")]

for sub, name, url, headers in FILES:
    dest = os.path.join(RAW, sub, name)
    try:
        fetch(url, dest, sleep=2.0, timeout=600, headers=headers)
        print(f"ok   {sub}/{name} {os.path.getsize(dest)}", flush=True)
    except Exception as e:
        print(f"FAIL {sub}/{name}: {e}", flush=True)
# StatCan zip: extract the CSV next to it (data only; nothing is executed)
z = os.path.join(RAW, "statcan_migration", "17100040-eng.zip")
out = os.path.join(RAW, "statcan_migration", "unz")
if not os.path.exists(os.path.join(out, "17100040.csv")):
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            assert "/" not in n and ".." not in n, n
        zf.extractall(out)
print("done")
