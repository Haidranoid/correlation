import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW
for y in ["", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]:
    url = "https://www.tsa.gov/travel/passenger-volumes" + (f"/{y}" if y else "")
    dest = os.path.join(RAW, "tsa", f"passenger-volumes{'_' + y if y else '_current'}.html")
    try:
        fetch(url, dest, sleep=2)
        print("ok", url, os.path.getsize(dest))
    except Exception as e:
        print("FAIL", url, e)
