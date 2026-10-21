"""QC only: compare the monthly 'Grand Totals' of the Refugee Admissions Report (current.xlsx) with the
archived 'Arrivals by Region by Month' files (FY2012-FY2017 .xls via xlrd, FY2019-FY2020 .xlsx).
Run with the data venv: ~/programming/c2-wt/venv-data/bin/python -I check_refugee_archives.py"""
import os, sys
import xlrd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW
from xlsx_read import xlsx_rows, xlsx_sheets

R = os.path.join(RAW, "wrapsnet")
cur = {}
for s in xlsx_sheets(os.path.join(R, "current.xlsx")):
    if s.isdigit():
        rows = xlsx_rows(os.path.join(R, "current.xlsx"), s)
        hdr = next(r for r in rows if r and "OCT" in [str(c).strip() for c in r])
        io = [str(c).strip() for c in hdr].index("OCT")
        g = next(r for r in rows if r and any(isinstance(c, str) and c.strip().lower().startswith("grand total") for c in r))
        cur[int(s)] = [float(g[io + k] or 0) for k in range(12)]
MON = ["OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP"]


def by_header(rows):
    h = next(r for r in rows if r and sum(str(c).strip().upper() in MON for c in r) == 12)
    cols = [i for i, c in enumerate(h) if str(c).strip().upper() in MON]
    t = next(r for r in rows if r and any(str(c).strip() in ("Total", "Totals") for c in r[:4]))
    return [float(t[c] or 0) for c in cols]


for fy in range(2012, 2021):
    if fy == 2018:
        continue  # PDF only
    p = os.path.join(R, f"FY{fy}_arrivals_by_region.xls")
    if os.path.exists(p):
        sh = xlrd.open_workbook(p, logfile=open(os.devnull, "w")).sheets()[0]
        rows = [[sh.cell_value(i, j) for j in range(sh.ncols)] for i in range(sh.nrows)]
    else:
        p = os.path.join(R, f"FY{fy}_arrivals_by_region.xlsx")
        rows = xlsx_rows(p, xlsx_sheets(p)[0])
    a = by_header(rows)
    diff = [int(x - y) for x, y in zip(a, cur[fy])]
    print(f"FY{fy}: archive vs admissions report, monthly differences {diff}")
