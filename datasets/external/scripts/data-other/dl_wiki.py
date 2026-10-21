"""Wikipedia daily page views (agent=user, all-access) via the Wikimedia REST API.

Step 1: resolve redirects/normalization for the requested titles (one MediaWiki API call).
Step 2: fetch per-article daily views 2015-07-01 .. END for each canonical title.
Step 3: move check: if the canonical title's first 60 days are tiny relative to the whole,
        the page was probably moved during the window; fetch its redirects too (they hold
        the pre-move history) so the parser can sum them.
Raw JSON responses are saved unmodified.
"""
import json, os, sys, urllib.parse, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW

OUT = os.path.join(RAW, "wikimedia")
START, END = "20150701", "20261017"
REQUESTED = [
    # economy
    "Recession", "Great_Recession", "Inflation", "Stagflation", "Unemployment",
    "Unemployment_benefits", "Layoff", "Bankruptcy", "Foreclosure", "Mortgage",
    "Interest_rate", "Federal_Reserve", "Quantitative_easing", "Yield_curve", "Tariff",
    "United_States_debt_ceiling", "Default_(finance)", "Bank_run",
    # markets
    "Stock_market", "Stock_market_crash", "Bear_market", "Wall_Street_Crash_of_1929",
    "Black_Monday_(1987)", "Dow_Jones_Industrial_Average", "S&P_500", "VIX",
    "Short_(finance)", "Gold_as_an_investment", "Price_of_oil", "Bitcoin",
    # hazards / health
    "Hurricane", "Earthquake", "Pandemic", "Influenza", "Terrorism",
    # placebo controls (no plausible market link)
    "Cat", "Philosophy",
]

def api(params, dest):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    fetch(url, dest, sleep=1)
    with open(dest) as f:
        return json.load(f)

titles = [t.replace("_", " ") for t in REQUESTED]
res = api(dict(action="query", titles="|".join(titles), redirects=1, format="json",
               formatversion=2), os.path.join(OUT, "resolve_titles.json"))
q = res["query"]
mapping = {t: t for t in titles}
for n in q.get("normalized", []):
    for k, v in mapping.items():
        if v == n["from"]:
            mapping[k] = n["to"]
for r in q.get("redirects", []):
    for k, v in mapping.items():
        if v == r["from"]:
            mapping[k] = r["to"]
missing = [p["title"] for p in q["pages"] if p.get("missing")]
if missing:
    print("MISSING pages:", missing)
canon = {}
for req in titles:
    canon[req] = mapping[req]
with open(os.path.join(OUT, "title_map.json"), "w") as f:
    json.dump(canon, f, indent=1)
for k, v in canon.items():
    if k != v:
        print(f"redirect: {k!r} -> {v!r}")

def views_url(title):
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
            f"all-access/user/{t}/daily/{START}/{END}")

def safe(title):
    return "".join(c if c.isalnum() else "_" for c in title)

for req, title in canon.items():
    dest = os.path.join(OUT, f"views_{safe(title)}.json")
    fetch(views_url(title), dest, sleep=1)
    items = json.load(open(dest))["items"]
    vals = [it["views"] for it in items]
    first60 = statistics.median(vals[:60]) if vals else 0
    whole = statistics.median(vals) if vals else 0
    flag = "  <-- POSSIBLE MOVE" if (not vals or items[0]["timestamp"][:8] != START
                                    or first60 < 0.1 * whole) else ""
    print(f"{title}: n={len(vals)} first={items[0]['timestamp'][:8] if items else None} "
          f"median_first60={first60} median_all={whole}{flag}", flush=True)
    if flag:
        rd = api(dict(action="query", titles=title, prop="redirects", rdlimit="max",
                      rdnamespace=0, format="json", formatversion=2),
                 os.path.join(OUT, f"redirects_{safe(title)}.json"))
        reds = [r["title"] for p in rd["query"]["pages"] for r in p.get("redirects", [])]
        print(f"   {len(reds)} redirects: {reds[:15]}", flush=True)
        for r in reds:
            d2 = os.path.join(OUT, f"rviews_{safe(title)}__{safe(r)}.json")
            try:
                fetch(views_url(r), d2, sleep=0.5)
            except Exception as e:  # 404 = no views recorded for that title
                print(f"   redirect {r!r}: {e}")
print("done", flush=True)
