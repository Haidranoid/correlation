"""Other: GPR daily (Caldara-Iacoviello), EIA weekly gasoline product supplied,
EIA-930 US48 daily electricity demand (Eastern-time days), TSA checkpoint throughput."""
import datetime as dt, glob, html, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from dta118 import read_dta
from xlsx_read import iter_rows

# ---------------- GPR ----------------
names, types, rows = read_dta(os.path.join(RAW, "gpr", "data_gpr_daily_recent.dta"))
raw = open(os.path.join(RAW, "gpr", "data_gpr_daily_recent.dta"), "rb").read(400)
stamp = re.search(rb"<timestamp>.(.*?)</timestamp>", raw, flags=re.S).group(1).decode()
stamp_date = dt.datetime.strptime(stamp.strip(), "%d %b %Y %H:%M").date()
print("GPR file timestamp:", stamp, "->", stamp_date)
gpr = {"GPRD": [], "GPRD_THREAT": [], "GPRD_ACT": []}
for r in rows:
    d = dt.datetime.strptime(r["DAY"], "%Y%m%d").date()
    assert d == dt.date(1960, 1, 1) + dt.timedelta(days=int(r["date"])), (r["DAY"], r["date"])
    if d >= stamp_date:          # the update day itself may be partial
        continue
    for k in gpr:
        if r[k] is not None:
            gpr[k].append((d, r[k]))
GPR_URL = "https://www.matteoiacoviello.com/gpr.htm"
gpr_common = ("Caldara & Iacoviello (2022, AER) daily 'recent' GPR: share of articles in 10 major US/UK/Canadian "
              "newspapers about adverse geopolitical events, normalized to 100 = 1985-2019 average. Very noisy day "
              "to day (the authors also publish 7- and 30-day averages). The daily file is updated every Monday, so "
              "a day's value can take up to a week to appear: known_after_days 8 (dated vintages "
              "data_gpr_daily_recent_YYYYMMDD exist for a real-time reconstruction). Article counts can be revised "
              "slightly between vintages. The update day itself is dropped as possibly partial. "
              "GPR files normalized by data-other, exactly: raw/gpr/data_gpr_daily_recent.dta columns GPRD -> gpr_daily, "
              "GPRD_THREAT -> gpr_daily_threats, GPRD_ACT -> gpr_daily_acts. Not normalized: N10D, GPRD_MA7, "
              "GPRD_MA30, event, and all monthly/country GPR series (data_gpr_export was not downloaded); "
              "raw/gpr is left unmodified for other agents.")
write_series("gpr_daily", gpr["GPRD"], name="Geopolitical Risk index, daily", category="other",
             subcategory="geopolitical_risk", region="global (US/UK/CA press)", source="Caldara & Iacoviello",
             url=GPR_URL, frequency="daily", units="index (1985-2019 mean = 100)", transform="level",
             known_after_days=8, value_range=(0, 3000), notes=gpr_common)
write_series("gpr_daily_threats", gpr["GPRD_THREAT"], name="Geopolitical Risk index, threats component, daily",
             category="other", subcategory="geopolitical_risk", region="global (US/UK/CA press)",
             source="Caldara & Iacoviello", url=GPR_URL, frequency="daily", units="index (1985-2019 mean = 100)",
             transform="level", known_after_days=8, value_range=(0, 3000),
             notes="Threats sub-index (war threats, nuclear threats, terror threats...); the paper finds threats "
                   "depress investment and stocks more than acts. " + gpr_common)
write_series("gpr_daily_acts", gpr["GPRD_ACT"], name="Geopolitical Risk index, acts component, daily",
             category="other", subcategory="geopolitical_risk", region="global (US/UK/CA press)",
             source="Caldara & Iacoviello", url=GPR_URL, frequency="daily", units="index (1985-2019 mean = 100)",
             transform="level", known_after_days=8, value_range=(0, 3000),
             notes="Acts sub-index (war beginning, escalation, terror acts). " + gpr_common)

# ---------------- EIA weekly gasoline product supplied ----------------
s = open(os.path.join(RAW, "eia", "WGFUPUS2_leaf.html"), encoding="utf-8", errors="replace").read()
title = re.search(r"<title>(.*?)</title>", s, flags=re.S).group(1)
assert "Product Supplied of Finished Motor Gasoline" in title and "Thousand Barrels per Day" in title
MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                                   "Oct", "Nov", "Dec"], 1)}
gas = []
for row in re.findall(r"<tr>((?:(?!<tr>).)*?class='B6'.*?)</tr>", s, flags=re.S):
    cells = re.findall(r"<td class='(B\d)'>(.*?)</td>", row, flags=re.S)
    clean = [(c, html.unescape(v).replace("\xa0", " ").strip()) for c, v in cells]
    ym = clean[0][1]
    y, mname = ym.split("-")
    y = int(y)
    for (c1, dstr), (c2, val) in zip(clean[1::2], clean[2::2]):
        assert c1 == "B5" and c2 == "B3", (c1, c2)
        if not dstr:
            continue
        mm, dd = map(int, dstr.split("/"))
        yy = y if mm >= MON[mname] or MON[mname] != 1 else y - 1  # Jan row may hold a late-Dec week
        if MON[mname] == 1 and mm == 12:
            yy = y - 1
        d = dt.date(yy, mm, dd)
        if val in ("", "NA", "W", "--"):
            print("  gasoline: missing", d, repr(val))
            continue
        gas.append((d, float(val.replace(",", ""))))
assert all(d.weekday() == 4 for d, _ in gas), "EIA weekly dates should be Fridays"
rel = re.search(r"Release Date: ([\d/]+)", s).group(1)
print("EIA gasoline page release date:", rel)
write_series("eia_gasoline_supplied_wk", gas, name="US product supplied of finished motor gasoline (weekly)",
             category="energy", subcategory="demand", region="US", source="EIA Weekly Petroleum Status Report",
             url="https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WGFUPUS2&f=W", frequency="weekly",
             units="thousand barrels per day", transform="logdiff", known_after_days=6, value_range=(5000, 11000),
             notes="Proxy for US gasoline demand (disappearance from primary supply). Date = week-ending Friday. "
                   "Released Wednesday 10:30 ET for the week ending the previous Friday (5 days), Thursday in "
                   "holiday weeks (6); EIA's systems outage delayed releases in June 2022. Weekly estimates are "
                   "not merged with the later monthly figures. Seasonal (summer driving peak, holiday weeks) and "
                   "noisy week to week; transform logdiff, so compare hits against the seasonal pattern. COVID "
                   "collapse in Apr 2020. No-key history page (the API needs a free key).")

# ---------------- EIA-930 US48 demand ----------------
XL = os.path.join(RAW, "eia", "Region_US48.xlsx")
EXCEL0 = dt.date(1899, 12, 30)
hourly = defaultdict(float); hours = defaultdict(int); missing_h = defaultdict(int)
hdr = None
for rnum, row in iter_rows(XL, "Published Hourly Data"):
    if rnum == 1:
        hdr = row
        assert row["A"] == "Region" and row["C"] == "Local date" and row["H"] == "Demand" and row["F"] == "Time zone"
        continue
    assert row["A"] == "US48" and row["F"] == "Eastern", row
    d = EXCEL0 + dt.timedelta(days=int(row["C"]))
    if isinstance(row.get("H"), float):
        hourly[d] += row["H"]; hours[d] += 1
    else:
        missing_h[d] += 1
daily_pub = {}
for rnum, row in iter_rows(XL, "Published Daily Data"):
    if rnum == 1:
        assert row["B"] == "Local date" and row["E"] == "Demand"
        continue
    if row.get("A") != "US48":
        continue
    assert row["C"] == "Eastern"
    d = EXCEL0 + dt.timedelta(days=int(row["B"]))
    if isinstance(row.get("E"), float):
        daily_pub[d] = row["E"]
days = sorted(hourly)
mism = [(d, hourly[d], daily_pub.get(d)) for d in days if d in daily_pub and abs(hourly[d] - daily_pub[d]) > 1]
odd = [(d, hours[d]) for d in days if hours[d] not in (23, 24, 25)]
print(f"EIA-930: {len(days)} Eastern days from hourly ({days[0]}..{days[-1]}), published daily {len(daily_pub)}; "
      f"{len(mism)} days where hourly sum != published daily; days with !=23/24/25 hours: {len(odd)} {odd[:8]}")
print("  hour counts:", sorted(defaultdict(int, {h: sum(1 for d in days if hours[d] == h) for h in set(hours.values())}).items()))
if mism:
    print("  mismatches e.g.", mism[:5])
# keep complete days only; the last day may be partial
last_complete = max(d for d in days if hours[d] >= 23)
keep = [(d, daily_pub.get(d, hourly[d])) for d in days
        if hours[d] in (23, 24, 25) and missing_h[d] == 0 and d <= last_complete]
dropped = [d for d in days if d not in dict(keep)]
print(f"  dropped {len(dropped)} incomplete days: {dropped[:10]}")
write_series("eia930_us48_demand_daily", keep, name="US Lower-48 electricity demand, daily (Eastern-time days)",
             category="energy", subcategory="electricity_demand", region="US48",
             source="EIA-930 Hourly Electric Grid Monitor", url="https://www.eia.gov/electricity/gridmonitor/",
             frequency="daily", units="MWh per day", transform="anomaly", known_after_days=1,
             value_range=(6e6, 2e7),
             notes="SHORT: starts 2015-07-01. Sum of balancing-authority hourly demand over the Eastern-time calendar "
                   "day (23 or 25 hours on DST change days; incomplete days dropped). Matches EIA's published daily "
                   "sheet. Hourly data are posted in near real time, so the day is known by the next US close; "
                   "values are preliminary and BAs resubmit, so some revision look-ahead. Weather-driven (summer "
                   "and winter peaks, overlaps the weather series) with a strong weekday/weekend and holiday cycle; "
                   "transform anomaly (same-day-of-year), which does not remove the weekday cycle. Early months "
                   "(2015-16) have reporting gaps from some BAs.")

# ---------------- TSA checkpoint throughput ----------------
tsa = {}
for p in sorted(glob.glob(os.path.join(RAW, "tsa", "passenger-volumes_*.html"))):
    s = open(p, encoding="utf-8", errors="replace").read()
    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.S):
        c = [re.sub(r"<[^>]+>|\s+", " ", x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)]
        if len(c) >= 2 and re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", c[0]):
            m, d_, y = map(int, c[0].split("/"))
            d = dt.date(y, m, d_)
            v = float(c[1].replace(",", ""))
            if d in tsa and tsa[d] != v:
                raise ValueError(f"TSA conflicting values for {d}: {tsa[d]} vs {v}")
            tsa[d] = v
write_series("tsa_checkpoint_daily", sorted(tsa.items()), name="TSA checkpoint travel numbers (daily)",
             category="other", subcategory="air_travel", region="US", source="US Transportation Security Administration",
             url="https://www.tsa.gov/travel/passenger-volumes", frequency="daily", units="persons screened per day",
             transform="logdiff", known_after_days=4, value_range=(50000, 4e6),
             notes="SHORT: starts 2019-01-01. People screened at TSA checkpoints. Posted on weekday mornings for the "
                   "previous day; Friday and Saturday values appear on Monday, so a value can take 3 days (4 after a "
                   "Monday holiday): known_after_days 4. Strong weekday cycle (Tue/Wed/Sat lows) and holiday peaks; "
                   "COVID collapse in Mar-Apr 2020 dominates. Transform logdiff.")

# ---------------- SF Fed Daily News Sentiment Index ----------------
SF = os.path.join(RAW, "sffed", "news_sentiment_data.xlsx")
sent = []
for rnum, row in iter_rows(SF, "Data"):
    if rnum == 1:
        assert row["A"] == "date" and row["B"] == "News Sentiment", row
        continue
    if not isinstance(row.get("B"), float):
        continue
    sent.append((EXCEL0 + dt.timedelta(days=int(row["A"])), row["B"]))
write_series("sffed_news_sentiment", sent, name="SF Fed Daily News Sentiment Index", category="attention",
             subcategory="news_sentiment", region="US", source="Federal Reserve Bank of San Francisco",
             url="https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/",
             frequency="daily", units="sentiment score (day fixed effects; higher = more positive)",
             transform="diff", known_after_days=8, value_range=(-1.5, 1.5),
             notes="Shapiro, Sudhof & Wilson (2022) lexical sentiment of economics articles in 24 US newspapers. "
                   "The published series is a TRAILING weighted average with geometrically declining weights "
                   "(lag-1 autocorrelation 0.997), so it is smooth: transform diff; the daily change mostly "
                   "reflects the newest articles. No look-ahead from the smoothing (trailing). Updated weekly "
                   "(Mondays, through the previous Sunday), so a value can wait up to a week: known_after_days 8. "
                   "Daily values are revised as the article sample updates, and the whole series was recomputed "
                   "from Factiva in April 2021, so stored values differ from what was public at the time. "
                   "Stands in for GDELT news tone, which was skipped.")
