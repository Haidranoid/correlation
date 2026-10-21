"""PredictIt (final 90 days of key markets) and Manifold (play-money forecasts): fetch + build.

PredictIt: GET www.predictit.org/Resource/DownloadMarketChartData?marketid=<id>&timespan=90d
(the site's "download CSV") returns daily OHLC per contract for the last 90 days the market
traded (for a closed market, its final 90 days). Longer spans return nothing (and the JSON
chart API returns only 7 days), so each market gives at most 90 daily closes. Bars are New York calendar days (the 2024-11-05 bar
already shows the election-night result), so the close is known the next day:
known_after_days 1.

Manifold: GET api.manifold.markets/v0/bets?contractId=<id>&limit=1000[&before=<betId>] (newest
first) gives every bet with createdTime (ms) and probAfter, so the as-of probability at 16:00
New York is exact: known_after_days 0. Play money (mana), so these are forecasts, not
real-money market prices.
"""
import bisect
import csv
import datetime as dt
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, daterange, fetch, fetch_json, ny16, write_series  # noqa: E402

PI_MARKETS = [
    # (market id, contract name, series id, name, region)
    (1296, "Republican", "pi_2016_pres_gop", "PredictIt: Republican party wins the 2016 US presidential election"),
    (1250, None, "pi_2016_gop_trifecta", "PredictIt: Republicans control both Congress and the White House after 2016"),
    (2704, "Republican", "pi_2018_house_gop", "PredictIt: Republicans control the House after the 2018 midterms"),
    (2703, "Republican", "pi_2018_senate_gop", "PredictIt: Republicans control the Senate after the 2018 midterms"),
    (2721, "Republican", "pi_2020_pres_gop", "PredictIt: Republican party wins the 2020 US presidential election"),
    (4365, "Republican", "pi_2020_house_gop", "PredictIt: Republicans control the House after the 2020 election"),
    (4366, "Republican", "pi_2020_senate_gop", "PredictIt: Republicans control the Senate after the 2020 election"),
    (6892, "Republican", "pi_2022_house_gop", "PredictIt: Republicans win the House in the 2022 election"),
    # 6874 (Senate 2022) left out: 84 of its final 90 days are after the Dec 2022 runoff, flat at 0.01
]

MF_MARKETS = [
    ("dYPk10oxA9e02IFKp887", "mf_us_recession_by_2023jan01", "Manifold: will the US enter a recession by January 1, 2023?", "recession"),
    ("at4SfIwNAVDpHZCMUkQf", "mf_us_recession_by_end_2023", "Manifold: will the US be in a recession by the end of 2023?", "recession"),
    ("IMLre6PYIQgsesV6Wa6O", "mf_us_recession_before_end_2024", "Manifold: did the US enter a recession before the end of 2024?", "recession"),
    ("3szs86zpyB4QQY4zzP2S", "mf_debt_ceiling_by_2023jun30", "Manifold: will the US debt ceiling be raised by the end of June 2023?", "fiscal"),
    ("L5f8JDzueLshQQvWwaiI", "mf_banking_crisis_contained_2023", "Manifold: is the (March 2023) banking crisis contained?", "financial_stability"),
    ("3uHa1kljfNWFdxjKIFEK", "mf_fed_cut_2024mar_meeting", "Manifold: will the Fed cut rates at the March 2024 meeting?", "fed"),
    ("KWy1zy4UrkDJSaXvlEpV", "mf_fed_cut_2024sep_meeting", "Manifold: will the Fed cut rates at the September 2024 meeting?", "fed"),
]


def build_predictit():
    reps = []
    for mid, cname, sid, name in PI_MARKETS:
        meta = fetch_json(f"https://www.predictit.org/api/marketdata/markets/{mid}",
                          os.path.join(RAW, "predictit", "markets", f"{mid}.json"))
        raw = fetch(f"https://www.predictit.org/Resource/DownloadMarketChartData?marketid={mid}&timespan=90d",
                    os.path.join(RAW, "predictit", "csv", f"{mid}_90d.csv")).decode("utf-8-sig")
        rows = list(csv.DictReader(io.StringIO(raw)))
        names = sorted({r["ContractName"] for r in rows})
        if cname is None:
            cname = names[0]
        sel = [r for r in rows if r["ContractName"] == cname]
        if not sel:
            print("  no contract", mid, cname, names)
            continue
        out = {}
        for r in sel:
            mo, dd, yy = r["Date"].split(" ")[0].split("/")
            d = dt.date(int(yy), int(mo), int(dd))
            v = r.get("CloseSharePrice", "").replace("$", "").strip()
            if not v:
                continue
            if d in out:
                raise ValueError(f"{sid}: duplicate date {d}")
            out[d] = float(v)
        series = sorted(out.items())
        final = [c for c in meta.get("contracts") or [] if c.get("name") == cname]
        flat = 0  # trailing days already at the resolved price
        for _, v in reversed(series):
            if v <= 0.02 or v >= 0.98:
                flat += 1
            else:
                break
        reps.append(write_series(
            sid, series, name=name, category="prediction_market", subcategory="contract", region="US",
            source="PredictIt (Victoria University of Wellington / Aristotle; real-money, $850 cap per contract) public chart data",
            url=f"https://www.predictit.org/Resource/DownloadMarketChartData?marketid={mid}&timespan=90d (market page /markets/detail/{mid})",
            frequency="daily", units=f"probability (0-1), close price of the '{cname}' contract", transform="diff",
            known_after_days=1, value_range=(0, 1),
            notes=f"finite-life single contract, final 90 days only: \"{meta.get('name')}\" (PredictIt market {mid}); PredictIt serves no free "
                  f"history beyond the last 90 trading days of a market, so this is the run-up to resolution only. Daily close "
                  f"of the New York calendar day (bars include the evening/night), hence known_after_days 1. Thin, capped market "
                  f"($850 per trader), prices sum to >1 across contracts (fees). The last {flat} days sit at the "
                  f"resolved price (<=0.02 or >=0.98), i.e. carry no information. Final contract state: "
                  f"{final[0].get('lastTradePrice') if final else 'n/a'}."))
    return reps


def manifold_bets(cid):
    bets = []
    before = None
    for page in range(200):
        url = f"https://api.manifold.markets/v0/bets?contractId={cid}&limit=1000" + (f"&before={before}" if before else "")
        d = fetch_json(url, os.path.join(RAW, "manifold", "bets", cid, f"p{page:03d}.json"))
        if not d:
            break
        bets += d
        if len(d) < 1000:
            break
        before = d[-1]["id"]
    return bets


def build_manifold():
    reps = []
    for cid, sid, name, sub in MF_MARKETS:
        m = fetch_json(f"https://api.manifold.markets/v0/market/{cid}", os.path.join(RAW, "manifold", "markets", f"{cid}.json"))
        bets = manifold_bets(cid)
        pts = sorted((int(b["createdTime"]) // 1000, float(b["probAfter"])) for b in bets
                     if b.get("probAfter") is not None and not b.get("isCancelled"))
        if not pts:
            print("  no bets", cid)
            continue
        ts = [t for t, _ in pts]
        ps = [p for _, p in pts]
        start = int(m["createdTime"]) // 1000
        end = int(m.get("resolutionTime") or m.get("closeTime") or 0) // 1000
        close = int(m.get("closeTime") or 0) // 1000
        if close:
            end = min(end, close) if end else close
        d0 = dt.datetime.fromtimestamp(start, dt.timezone.utc).date()
        d1 = dt.datetime.fromtimestamp(end, dt.timezone.utc).date()
        rows = []
        for d in daterange(d0, d1):
            cut = ny16(d)
            if cut < ts[0] or cut >= end or cut > time.time():
                continue
            i = bisect.bisect_right(ts, cut)
            rows.append((d, ps[i - 1]))
        reps.append(write_series(
            sid, rows, name=name, category="prediction_market", subcategory="forecast_play_money", region="US",
            source="Manifold Markets public API (play money 'mana'; no key)",
            url=f"https://api.manifold.markets/v0/bets?contractId={cid}",
            frequency="daily", units="probability (0-1)", transform="diff", known_after_days=0, value_range=(0, 1),
            notes=f"finite-life single contract; play-money forecast, not a real-money market: \"{m.get('question')}\" ({m.get('uniqueBettorCount')} "
                  f"bettors; created {d0}, closes {dt.datetime.fromtimestamp(close, dt.timezone.utc).date() if close else 'n/a'}, "
                  f"resolution {m.get('resolution')}). Value = probability after the last bet at or before 16:00 New York "
                  f"(AMM price persists between bets, so quiet days repeat the last value), hence known_after_days 0. "
                  f"Creator-resolved; definitions are the creator's."))
    return reps


if __name__ == "__main__":
    build_predictit()
    build_manifold()
