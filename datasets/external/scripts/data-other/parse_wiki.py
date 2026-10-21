"""Wikipedia daily views per topic = canonical article + all its redirects (agent=user)."""
import datetime as dt, glob, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, daterange

W = os.path.join(RAW, "wikimedia")
canon = json.load(open(os.path.join(W, "title_map.json")))

def safe(t):
    return "".join(c if c.isalnum() else "_" for c in t)

def load(p):
    out = {}
    for it in json.load(open(p))["items"]:
        assert it["agent"] == "user" and it["access"] == "all-access" and it["granularity"] == "daily"
        out[dt.date(int(it["timestamp"][:4]), int(it["timestamp"][4:6]), int(it["timestamp"][6:8]))] = it["views"]
    return out

IDS = {
    "Recession": ("wiki_recession", "economy"), "Great Recession": ("wiki_great_recession", "economy"),
    "Inflation": ("wiki_inflation", "economy"), "Stagflation": ("wiki_stagflation", "economy"),
    "Unemployment": ("wiki_unemployment", "economy"),
    "Unemployment benefits": ("wiki_unemployment_benefits", "economy"),
    "Layoff": ("wiki_layoff", "economy"), "Bankruptcy": ("wiki_bankruptcy", "economy"),
    "Foreclosure": ("wiki_foreclosure", "economy"), "Mortgage": ("wiki_mortgage", "economy"),
    "Interest rate": ("wiki_interest_rate", "economy"), "Federal Reserve": ("wiki_federal_reserve", "economy"),
    "Quantitative easing": ("wiki_quantitative_easing", "economy"), "Yield curve": ("wiki_yield_curve", "economy"),
    "Tariff": ("wiki_tariff", "economy"), "United States debt ceiling": ("wiki_us_debt_ceiling", "economy"),
    "Default (finance)": ("wiki_default_finance", "economy"), "Bank run": ("wiki_bank_run", "economy"),
    "Stock market": ("wiki_stock_market", "markets"), "Stock market crash": ("wiki_stock_market_crash", "markets"),
    "Bear market": ("wiki_market_trend", "markets"), "Wall Street Crash of 1929": ("wiki_crash_1929", "markets"),
    "Black Monday (1987)": ("wiki_black_monday_1987", "markets"),
    "Dow Jones Industrial Average": ("wiki_djia", "markets"), "S&P 500": ("wiki_sp500", "markets"),
    "VIX": ("wiki_vix", "markets"), "Short (finance)": ("wiki_short_selling", "markets"),
    "Gold as an investment": ("wiki_gold_investment", "markets"), "Price of oil": ("wiki_price_of_oil", "markets"),
    "Bitcoin": ("wiki_bitcoin", "markets"), "Hurricane": ("wiki_tropical_cyclone", "hazard"),
    "Earthquake": ("wiki_earthquake", "hazard"), "Pandemic": ("wiki_pandemic", "hazard"),
    "Influenza": ("wiki_influenza", "hazard"), "Terrorism": ("wiki_terrorism", "hazard"),
    "Cat": ("wiki_cat", "placebo"), "Philosophy": ("wiki_philosophy", "placebo"),
}
assert set(IDS) == set(canon), set(canon) ^ set(IDS)
MOVES = {  # renames inside the window, found from level breaks in the canonical title's views
    "Federal Reserve": "renamed from 'Federal Reserve System' in Oct 2018",
    "Mortgage": "article 'Mortgage loan' renamed to 'Mortgage' in Nov 2023",
    "Wall Street crash of 1929": "renamed from 'Wall Street Crash of 1929' in Nov 2024",
    "S&P 500": "titled 'S&P 500 Index' until Jan 2021",
}
ARTIFACTS = {
    "Mortgage": ("WARNING: level break 2024-05-07 to about early 2025: views of the article and of obscure "
                 "redirects (e.g. 'Islamic mortgage', 'Residential mortgages') rose 5-10x, almost entirely on "
                 "mobile-web (desktop under 2x, app flat), so likely a search-referral change or automated "
                 "traffic, not mortgage-market attention; treat that window with caution. "),
}
summary = []
last_common = None
for req, title in canon.items():
    sid, sub = IDS[req]
    main = load(os.path.join(W, f"views_{safe(title)}.json"))
    reds = sorted(glob.glob(os.path.join(W, f"rviews_{safe(title)}__*.json")))
    tot = defaultdict(int, main)
    red_total = 0
    for p in reds:
        for d, v in load(p).items():
            tot[d] += v
            red_total += v
    first, last = min(tot), max(tot)
    days = list(daterange(first, last))
    missing = [d for d in days if d not in tot]
    rows = [(d, tot[d]) for d in days if d in tot]
    share = red_total / sum(tot.values())
    summary.append((sid, title, len(reds), share, len(missing), min(v for _, v in rows)))
    note = (f"Daily English-Wikipedia views of '{title}' plus its {len(reds)} redirects (summing redirects keeps "
            f"pre-rename history; redirects are {100*share:.1f}% of the total). agent=user, all-access, UTC days. "
            + (f"Page {MOVES[title]}. " if title in MOVES else "")
            + ARTIFACTS.get(req, "")
            + ("PLACEBO topic with no plausible market link, a control for the attention series' shared "
               "artifacts (weekday cycle, bot spikes, platform trends). " if sub == "placebo" else "")
            + "SHORT: starts 2015-07-01 (REST API start; older pagecount dumps are not comparable). "
              "Strong weekday cycle and a long downward drift (search engines answering directly, mobile apps); "
              "undetected automated traffic causes one-off spikes, and the 'automated' agent class was split out "
              "of 'user' in 2020, a possible level shift. A UTC day's hourly dump files are normally public "
              "about 2 hours after each hour, so the day is known by the next US close (pipeline stalls of a "
              "few days happen; one was under way at download time). Moat et al. (2013) studied views of "
              "financial pages before market moves.")
    write_series(sid, rows, name=f"Wikipedia views: {title}", category="attention", subcategory=f"wikipedia_{sub}",
                 region="global (English Wikipedia)", source="Wikimedia Pageviews REST API",
                 url=("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/"
                      f"user/{title.replace(' ', '_')}/daily/20150701/20261017"),
                 frequency="daily", units="page views per day", transform="logdiff", known_after_days=1,
                 notes=note, quiet=True)
print(f"{'id':28s} {'title':30s} redirects share  missing_days min")
for sid, title, n, share, miss, mn in summary:
    print(f"{sid:28s} {title[:30]:30s} {n:4d} {100*share:6.2f}% {miss:4d} {mn}")
