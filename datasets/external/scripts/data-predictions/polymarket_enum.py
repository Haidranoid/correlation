"""Enumerate Polymarket events (with nested markets) by topic tag via the Gamma API.

For each tag, page through /events?tag_slug=<tag>&closed=<true|false>&order=volume
(descending, 100 per page) until the page's smallest event volume falls below MIN_VOL or
MAX_PAGES is reached. Pages are saved unmodified (gzip) under raw/polymarket/events/.
Then writes a flat index of markets (raw/polymarket/markets_index.json is NOT written to
raw; the index goes to the scripts folder meta/ as a derived file).
"""
import gzip
import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, RAW, fetch_json  # noqa: E402

GAMMA = "https://gamma-api.polymarket.com"
OUT = os.path.join(RAW, "polymarket", "events")
TAGS = ["fed-rates", "economy", "recession", "inflation", "cpi", "gdp", "jobs-report",
        "unemployment", "tariffs", "trade-war", "government-shutdown", "shutdown", "debt-ceiling",
        "us-presidential-election", "elections", "midterms", "geopolitics", "ukraine", "russia",
        "israel", "middle-east", "iran", "china", "taiwan", "bitcoin", "ethereum", "indicies",
        "stocks", "oil", "commodities", "finance", "business"]
MIN_VOL = 250_000
MAX_PAGES = 4


def tag_pages(tag, closed):
    for p in range(MAX_PAGES):
        q = {"tag_slug": tag, "closed": "true" if closed else "false", "order": "volume",
             "ascending": "false", "limit": 100, "offset": 100 * p}
        url = f"{GAMMA}/events?" + urllib.parse.urlencode(q)
        dest = os.path.join(OUT, f"{tag}_{'closed' if closed else 'open'}_p{p}.json.gz")
        d = fetch_json(url, dest, gz=True)
        yield d
        if len(d) < 100:
            break
        vols = [float(e.get("volume") or 0) for e in d]
        if min(vols) < MIN_VOL:
            break


def main():
    events = {}
    for tag in TAGS:
        n = 0
        for closed in (True, False):
            for page in tag_pages(tag, closed):
                for e in page:
                    e.setdefault("_tags", [])
                    if e["id"] in events:
                        events[e["id"]]["_tags"].append(tag)
                    else:
                        e["_tags"].append(tag)
                        events[e["id"]] = e
                    n += 1
        print(f"{tag}: {n} events (unique so far {len(events)})", flush=True)
    # flat market index (derived, kept with the scripts, not in raw/)
    rows = []
    for e in events.values():
        for m in e.get("markets") or []:
            rows.append(dict(
                event_id=e["id"], event_title=e.get("title"), event_slug=e.get("slug"),
                event_start=e.get("startDate"), event_end=e.get("endDate"), event_volume=e.get("volume"),
                neg_risk=e.get("negRisk"), tags=e["_tags"], market_id=m.get("id"),
                question=m.get("question"), group_item=m.get("groupItemTitle"), slug=m.get("slug"),
                outcomes=m.get("outcomes"), tokens=m.get("clobTokenIds"), outcome_prices=m.get("outcomePrices"),
                start=m.get("startDate") or m.get("createdAt"), end=m.get("endDate"),
                closed=m.get("closed"), closed_time=m.get("closedTime"), uma_end=m.get("umaEndDate"),
                volume=m.get("volumeNum") or m.get("volume"), order_book=m.get("enableOrderBook"),
                created=m.get("createdAt"), description=(m.get("description") or "")[:600]))
    os.makedirs(os.path.join(HERE, "meta_pm"), exist_ok=True)
    with gzip.open(os.path.join(HERE, "meta_pm", "markets_index.json.gz"), "wt") as f:
        json.dump(rows, f)
    print(f"{len(events)} events, {len(rows)} markets indexed")


if __name__ == "__main__":
    main()
