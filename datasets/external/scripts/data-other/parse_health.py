"""Health: CDC ILINet via Delphi (latest and first-release), NYT COVID, infectious-disease EMV."""
import csv, datetime as dt, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series

def mmwr_end(ew):
    """MMWR epiweek (Sun-Sat; week 1 contains Jan 4) -> week-ending Saturday."""
    y, w = divmod(ew, 100)
    jan4 = dt.date(y, 1, 4)
    w1 = jan4 - dt.timedelta(days=(jan4.weekday() + 1) % 7)
    return w1 + dt.timedelta(days=7 * (w - 1) + 6)

assert mmwr_end(202640) == dt.date(2026, 10, 10) and mmwr_end(202053) == dt.date(2021, 1, 2)
D = os.path.join(RAW, "delphi")
FLU_URL = "https://api.delphi.cmu.edu/epidata/fluview/?regions=nat&epiweeks=199740-202652"
lat = json.load(open(os.path.join(D, "fluview_nat_latest.json")))["epidata"]
lag0 = json.load(open(os.path.join(D, "fluview_nat_lag0.json")))["epidata"]
assert all(r["region"] == "nat" for r in lat + lag0) and all(r["lag"] == 0 for r in lag0)

flu_common = ("CDC ILINet national weighted influenza-like-illness percentage (wILI: share of outpatient "
              "visits for ILI, weighted by state population). Weekly, date = MMWR week-ending Saturday. "
              "CDC FluView publishes on the Friday after the week ends (6 days) in 94% of weeks; holiday "
              "weeks 8-12 days, so known_after_days 12 (the largest observed delay 2003-2026). Strong winter "
              "seasonality: transform logdiff (the epidemic growth rate; works at weekly spacing without "
              "day-of-year matching), so a hit should be compared against the seasonal pattern of the growth "
              "rate itself. 2009 H1N1 and 2020 COVID distort ILI.")
write_series("delphi_ili_wili", [(mmwr_end(r["epiweek"]), r["wili"]) for r in lat],
             name="US influenza-like illness, weighted % (latest revision)", category="health",
             subcategory="influenza", region="US", source="CDC ILINet via Delphi Epidata API",
             url=FLU_URL, frequency="weekly", units="percent of outpatient visits", transform="logdiff",
             known_after_days=12, value_range=(0.2, 15),
             notes="LATEST-REVISION values (backfilled for weeks to months after first release), so they "
                   "carry look-ahead; use delphi_ili_wili_first for a real-time test. Summer weeks 21-39 "
                   "missing 1998-2002 (no off-season reporting then). " + flu_common)
write_series("delphi_ili_wili_first", [(mmwr_end(r["epiweek"]), r["wili"]) for r in lag0],
             name="US influenza-like illness, weighted % (first release)", category="health",
             subcategory="influenza", region="US", source="CDC ILINet via Delphi Epidata API (lag=0 issues)",
             url=FLU_URL + "&lag=0", frequency="weekly", units="percent of outpatient visits",
             transform="logdiff", known_after_days=12, value_range=(0.2, 15),
             notes="FIRST-RELEASE values (each week's value as first published; the Delphi issue archive starts "
                   "2003w47), so no revision look-ahead. The logdiff of consecutive first releases mixes in the "
                   "revision of the prior week, a small difference from the true real-time growth. " + flu_common)

# ---- NYT COVID-19 (cumulative cases/deaths by report date) ----
rows = list(csv.DictReader(open(os.path.join(RAW, "nyt_covid", "us.csv"))))
dates = [dt.date.fromisoformat(r["date"]) for r in rows]
assert all((b - a).days == 1 for a, b in zip(dates, dates[1:])), "NYT dates not consecutive"
def ma7_new(col):
    cum = [int(r[col]) for r in rows]
    new = [cum[0]] + [b - a for a, b in zip(cum, cum[1:])]
    neg = [(dates[i], new[i]) for i in range(len(new)) if new[i] < 0]
    out = []
    for i in range(6, len(new)):
        out.append((dates[i], sum(new[i - 6:i + 1]) / 7))
    # start at the first day the 7-day average reaches 10/day (and stays positive after): earlier
    # values (0.1-few per day) give tiny-denominator log changes of +-1.5-2 that would dominate
    k = next(i for i, (d, v) in enumerate(out) if v >= 10)
    assert all(v > 0 for _, v in out[k:])
    return out[k:], neg
cases, negc = ma7_new("cases")
deaths, negd = ma7_new("deaths")
print(f"NYT: {len(negc)} days with negative daily cases, {len(negd)} with negative daily deaths "
      f"(corrections), e.g. {negc[:3]} {negd[:3]}")
covid_common = ("SHORT: 2020-2023 only (NYT stopped 2023-03-23). 7-day trailing average of daily new counts, "
                "from cumulative counts by REPORT date (not onset); trailing, so no look-ahead. The daily diffs "
                "include negative correction days and a weekday reporting cycle, which the 7-day window "
                "smooths. Counts for day D were in the NYT repository by the next morning, hence 1. "
                "Transform logdiff: the move is the growth rate of incidence. Starts on the first day the 7-day "
                "average reached 10 per day (earlier values of 0.1-a few per day are kept in raw/nyt_covid but left "
                "out here: their log changes are tiny-denominator noise). Testing regimes changed (home "
                "tests from 2022 undercount cases).")
write_series("nyt_covid_us_cases_7d", cases, name="US new COVID-19 cases per day, 7-day average",
             category="health", subcategory="covid", region="US", source="New York Times covid-19-data",
             url="https://github.com/nytimes/covid-19-data/blob/master/us.csv", frequency="daily",
             units="cases per day", transform="logdiff", known_after_days=1, value_range=(0.1, 2e6),
             notes=covid_common)
write_series("nyt_covid_us_deaths_7d", deaths, name="US new COVID-19 deaths per day, 7-day average",
             category="health", subcategory="covid", region="US", source="New York Times covid-19-data",
             url="https://github.com/nytimes/covid-19-data/blob/master/us.csv", frequency="daily",
             units="deaths per day", transform="logdiff", known_after_days=1, value_range=(0.1, 1e5),
             notes=covid_common)

# ---- Infectious Disease EMV tracker (Baker, Bloom, Davis et al.) ----
today_local = dt.date.today()
emv = []
for r in csv.DictReader(open(os.path.join(RAW, "policyuncertainty", "All_Infectious_EMV_Data.csv"))):
    d = dt.date(int(r["year"]), int(r["month"]), int(r["day"]))
    if d >= today_local:            # possibly partial day
        continue
    v = r["daily_infect_emv_index"].strip()
    if v in ("", "."):
        continue
    emv.append((d, float(v)))
write_series("emv_infectious_daily", emv, name="Daily Infectious Disease Equity Market Volatility tracker",
             category="health", subcategory="news_attention", region="US",
             source="Baker, Bloom, Davis, Kost, Sammon & Viratyosin (policyuncertainty.com)",
             url="https://www.policyuncertainty.com/infectious_EMV.html", frequency="daily",
             units="index (scaled share of newspaper articles)", transform="level", known_after_days=1,
             value_range=(0, 500),
             notes="Share of US newspaper articles that mention the economy, the stock market, volatility AND "
                   "infectious disease (epidemic, pandemic, virus, flu, ...), scaled to the EMV index. Zero on "
                   "most days before 2020; dominated by 2020-2022 COVID. Built from articles about market "
                   "volatility, so it partly reflects same-day market moves (contemporaneous, not a lead). "
                   "Updated daily; a day's value is complete the next day, hence 1. The newspaper sample can "
                   "be revised.")
