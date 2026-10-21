"""Bitcoin on-chain statistics (blockchain.com charts API) and the Crypto Fear & Greed index (alternative.me).
Raw JSON saved unmodified in raw/blockchain_com/ and raw/alternative_me/ (downloaded 2026-10-19 ~00:59 UTC).
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmlib  # noqa: E402

BC = os.path.join(dmlib.RAW, "blockchain_com")
FNG = os.path.join(dmlib.RAW, "alternative_me", "fng_limit0.json")
FNG_KAD = int(os.environ.get("FNG_KAD", "0"))

ONCHAIN = [
    ("hash-rate", "bcinfo_btc_hash_rate", "Bitcoin network hash rate (daily estimate)", "TH/s", "estimated from blocks found that UTC day; noisy day to day (block luck)"),
    ("n-transactions", "bcinfo_btc_n_transactions", "Bitcoin confirmed transactions per day", "transactions per day", ""),
    ("n-unique-addresses", "bcinfo_btc_n_unique_addresses", "Bitcoin unique addresses used per day", "addresses per day", "a few missing days in the source"),
    ("estimated-transaction-volume-usd", "bcinfo_btc_tx_volume_usd", "Bitcoin estimated on-chain transaction value", "USD per day", "blockchain.com heuristic (excludes change outputs), valued at the day's BTC price"),
    ("transaction-fees-usd", "bcinfo_btc_fees_usd", "Bitcoin total transaction fees paid to miners", "USD per day", "excludes block subsidy; very spiky (fee events, ordinals/runes 2023-24)"),
]
START = "2011-01-01"


def main():
    part = dmlib.Part("crypto_extra")
    for chart, sid, name, units, extra in ONCHAIN:
        d = json.load(open(os.path.join(BC, chart + ".json")))
        assert d["status"] == "ok" and d["period"] == "day", (chart, d.get("period"))
        rows = []
        for v in d["values"]:
            t = dt.datetime.fromtimestamp(v["x"], dt.UTC)
            assert t.hour == 0 and t.minute == 0, (chart, t)
            if t.date().isoformat() >= START:
                rows.append((t.date().isoformat(), float(v["y"])))
        note = ("UTC-day aggregate (day ends 00:00 UTC = 19:00/20:00 New York, after the US close); the statistic is computable "
                "from the public blockchain right after the day ends, but blockchain.com's API itself lagged ~4-5 days when downloaded "
                "(2026-10-19), so 2 as a cushion (use a larger value if the API publication time matters); starts 2011-01-01; current "
                "values only (no vintages)")
        if extra:
            note += "; " + extra
        part.add(dict(id=sid, name=name, category="crypto", subcategory="onchain", region="Global",
                      source="blockchain.com charts API", url=f"https://api.blockchain.info/charts/{chart}?timespan=all&format=json&sampled=false",
                      frequency="daily", units=units, transform="logdiff", known_after_days=2, notes=note), rows, jump=1.0)
    data = json.load(open(FNG))["data"]
    rows = []
    for v in data:
        t = dt.datetime.fromtimestamp(int(v["timestamp"]), dt.UTC)
        assert t.hour == 0 and t.minute == 0, t
        rows.append((t.date().isoformat(), float(v["value"])))
    fng_note = ("0-100 sentiment index (volatility, momentum/volume, social media, dominance, trends); published at ~00:00 UTC and labelled "
                "with the publication date (the latest value, dated 2026-10-19, was online at 00:59 UTC that day), so it reflects the previous UTC "
                "day's data; ")
    fng_note += ("known_after_days=0: on history the day-d change correlates +0.54 with BTC's return over UTC day d-1 and only +0.07 "
                 "with UTC day d, confirming the labelling (value for d is public ~20 h before the US close of d)" if FNG_KAD == 0 else
                 "known_after_days=1 as a cushion (0 is defensible, see SOURCES)")
    fng_note += "; starts 2018-02-01; a few missing days"
    part.add(dict(id="altme_crypto_fear_greed", name="Crypto Fear & Greed Index (alternative.me)", category="crypto",
                  subcategory="sentiment", region="Global", source="alternative.me Fear & Greed API",
                  url="https://api.alternative.me/fng/?limit=0&format=json", frequency="daily", units="index 0-100",
                  transform="diff", known_after_days=FNG_KAD, notes=fng_note), rows, jump=None)
    part.save()
    print("catalog rows:", dmlib.merge_catalog())


if __name__ == "__main__":
    main()
