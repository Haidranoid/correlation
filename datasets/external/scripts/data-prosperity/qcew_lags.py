"""QCEW county release lags from the BLS archive page (raw/bls_release_archives/cewqtr.htm): year from the
<h4>YYYY County Employment and Wages</h4> headings, quarter from the item text, release date from the file name."""
import datetime as dt, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib
R = os.path.join(plib.RAW, "bls_release_archives")
html = open(os.path.join(R, "cewqtr.htm"), encoding="utf-8", errors="replace").read()
Q = {"first": 1, "second": 2, "third": 3, "fourth": 4}
rows, year = [], None
for m in re.finditer(r"<h4[^>]*>\s*(\d{4})[^<]*</h4>|<li[^>]*>(.*?)</li>", html, re.S | re.I):
    if m.group(1):
        year = int(m.group(1)); continue
    body = m.group(2)
    d = re.search(r"cewqtr_(\d{8})\.(?:htm|pdf)", body)
    q = re.search(r"(first|second|third|fourth)", re.sub(r"<[^>]+>|&nbsp;", " ", body), re.I)
    if not (d and q and year):
        continue
    s = d.group(1); rel = dt.date(int(s[4:]), int(s[:2]), int(s[2:4]))
    qe = plib.quarter_end(year, Q[q.group(1).lower()])
    rows.append((f"{year}Q{Q[q.group(1).lower()]}", qe, rel, (rel - qe).days))
rows = sorted(set(rows))
with open(os.path.join(R, "cewqtr_lags.tsv"), "w") as f:
    f.write("period\tperiod_end\trelease\tlag_days\n")
    for r in rows: f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\n")
W = [r for r in rows if r[2] <= dt.date(2025, 6, 27) and r[1] >= dt.date(2001, 1, 1)]
lags = sorted(r[3] for r in W)
print(len(rows), "releases; window", len(W), "median", lags[len(lags)//2], "max", sorted(W, key=lambda r: -r[3])[:6])
print("first rows", rows[:3]); print("recent", rows[-6:])
