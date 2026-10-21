"""Enumerate Kalshi markets (metadata only) for the series we use.

Kalshi splits data at a 'historical cutoff' (markets settled before it are served only by
/historical/...). For each series we page through both
  /historical/markets?series_ticker=S   and   /markets?series_ticker=S
and save every page unmodified under raw/kalshi/markets/. The KX-prefixed series ticker
also returns the legacy (pre-2024, un-prefixed) events, e.g. KXFED -> FED-23DEC.
"""
import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, fetch_json  # noqa: E402

API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = os.path.join(RAW, "kalshi")

SERIES = [
    "KXFED", "KXCPI", "KXCPIYOY", "KXCPICORE", "KXCPICOREYOY", "KXPAYROLLS", "KXU3", "KXGDP",
    "KXPCECORE", "KXJOBLESS", "KXRECSSNBER", "KXQRECESS", "KXAAAGASM", "KXINXY",
    "KXNASDAQ100Y", "KXGOVSHUT", "KXSHUTDOWNBY", "KXGOVTSHUTDOWN", "KXDCEIL", "KXDEFAULT",
    "KXTERMINALRATE", "KXRATECUT", "KXRATEHIKE", "KXFEDDECISION", "KXFEDHIKE", "KXFEDCUT",
    "KXLCPIMAX", "KXACPI", "KXU3MAX", "KXGDPUSMIN", "KXFEDFUNDSYEAR", "KXUSCPIYEAR",
]


def pages(kind, series):
    """kind: 'historical' or 'live'. Yields market lists, caching each page."""
    base = f"{API}/historical/markets" if kind == "historical" else f"{API}/markets"
    cursor = ""
    i = 0
    while True:
        q = {"series_ticker": series, "limit": 1000}
        if cursor:
            q["cursor"] = cursor
        url = base + "?" + urllib.parse.urlencode(q)
        dest = os.path.join(OUT, "markets", f"{series}_{kind}_p{i:02d}.json")
        d = fetch_json(url, dest)
        yield d.get("markets") or []
        cursor = d.get("cursor") or ""
        i += 1
        if not cursor or i > 30:
            break


def main():
    fetch_json(f"{API}/historical/cutoff", os.path.join(OUT, "historical_cutoff.json"))
    summary = {}
    for s in SERIES:
        fetch_json(f"{API}/series/{s}", os.path.join(OUT, "series", f"{s}.json"))
        n = {}
        for kind in ("historical", "live"):
            k = 0
            for ms in pages(kind, s):
                k += len(ms)
            n[kind] = k
        summary[s] = n
        print(s, n, flush=True)
    with open(os.path.join(OUT, "markets", "_enum_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
