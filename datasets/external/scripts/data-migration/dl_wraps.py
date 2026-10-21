"""Refugee Processing Center (WRAPS): monthly refugee arrivals by region, one file per fiscal year
(FY2012 onward, archive page), plus the current Refugee Admissions Report."""
import os, sys, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW

D = os.path.join(RAW, "wrapsnet")
B = "https://www.wrapsnet.org"
FILES = [
    ("FY2012", "/documents/23OCT2012-Arrivals+by+Region.xls"),
    ("FY2013", "/documents/16MAY2013-ARR+by+Region+by+Month.xls"),
    ("FY2014", "/documents/Arrivals+by+PA+Region+by+Month+as+of+9.30.14.xls"),
    ("FY2015", "/documents/Arrivals+by+PA+Region+by+Month(FY15).xls"),
    ("FY2016", "/documents/Arrivals+by+PA+Region+by+Month(10.19.17(FY16)).xls"),
    ("FY2017", "/documents/Arrivals+by+PA+Region+by+Month(FY17).xls"),
    ("FY2018", "/documents/FY2018 Refugee Arrivals by Region.pdf"),
    ("FY2019", "/documents/Arrivals by PA Region by Month FY19REF.xlsx"),
    ("FY2020", "/documents/Arrivals by PA Region by Month as of 9-30-20.xlsx"),
    ("FY2021", "/documents/Refugee Arrivals by Region - Fiscal Year by Month as of 30 Sep 2021.pdf"),
    ("FY2022", "/documents/Refugee Arrivals by Region - Fiscal Year by Month as of 30 Sep 2022.pdf"),
    ("FY2023", "/documents/Refugee Arrivals by Region - Fiscal Year by Month as of 30 Sep 2023.pdf"),
    ("FY2024", "/documents/FY 2024 Arrivals by Region - Fiscal Year by Month as of 30 Sep 2024.pdf"),
    ("FY2025", "/documents/FY 2025 Arrivals by Region - Fiscal Year by Month as of 30 Sep 2025.pdf"),
    ("FY2026", "/documents/FY 2026 Arrivals by Region - Fiscal Year by Month as of 30 Sep 2026.pdf"),
    ("current", "/documents/Refugee Admissions Report as of September 30, 2026.xlsx"),
    ("since1975", "/documents/Refugee Admissions by Region since 1975 as of September 30, 2026.pdf"),
]
for tag, path in FILES:
    ext = os.path.splitext(path)[1]
    dest = os.path.join(D, f"{tag}_arrivals_by_region{ext}" if tag.startswith("FY") else f"{tag}{ext}")
    url = B + urllib.parse.quote(path, safe="/+()")
    try:
        fetch(url, dest, sleep=2.0, timeout=300)
        print(f"ok   {tag} {os.path.getsize(dest)}", flush=True)
    except Exception as e:
        print(f"FAIL {tag}: {e}", flush=True)
