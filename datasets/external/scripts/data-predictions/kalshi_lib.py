"""Kalshi helpers: load enumerated markets, group by event, parse strikes and candles.

Candle formats differ between endpoints:
  historical: price.close / price.previous, yes_bid.close, yes_ask.close, volume
  live:       price.close_dollars / price.previous_dollars, yes_bid.close_dollars, ...
Both are normalised by parse_candles() into (end_ts, bid, ask, last_trade) tuples.

Price at a timestamp (Polymarket's display rule, applied to Kalshi's book):
  bid = best yes bid (0 if none), ask = best yes ask (1 if none);
  if ask - bid <= 0.10: midpoint; elif a last trade exists: last trade price;
  else: unknown (the strike is left out that day).
"""
import collections
import glob
import gzip
import json
import os
import re
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, fetch_json, iso_to_ts, log  # noqa: E402

API = "https://api.elections.kalshi.com/trade-api/v2"
KRAW = os.path.join(RAW, "kalshi")
MAX_CANDLES = 4990  # API max is 5000 candles per request
SPREAD_MAX = 0.10


def load_markets(series):
    ms = {}
    for f in sorted(glob.glob(os.path.join(KRAW, "markets", f"{series}_*_p*.json"))):
        for m in json.load(open(f)).get("markets") or []:
            ms[m["ticker"]] = m
    return ms


def events_of(ms):
    ev = collections.defaultdict(list)
    for m in ms.values():
        ev[m["event_ticker"]].append(m)
    return ev


def strike_of(m):
    """Numeric strike of a 'greater than K' market (from floor_strike, else the ticker)."""
    st = m.get("strike_type")
    if st in ("greater", "greater_or_equal") and m.get("floor_strike") is not None:
        return float(m["floor_strike"])
    if st in (None, "", "greater"):
        mt = re.search(r"-T(-?[0-9.]+)$", m["ticker"])
        if mt:
            return float(mt.group(1))
    return None


def close_ts(m):
    return iso_to_ts(m.get("close_time"))


def open_ts(m):
    return iso_to_ts(m.get("open_time"))


def _num(x):
    if x is None or x == "":
        return None
    return float(x)


def _get(d, key):
    if not d:
        return None
    v = d.get(key)
    if v is None:
        v = d.get(key + "_dollars")
    return _num(v)


def parse_candles(objs):
    """objs: list of candle dicts (either format). Returns sorted list of
    (end_ts, bid, ask, last_trade, volume) with bid/ask as floats (None if missing)."""
    out = {}
    for c in objs:
        t = int(c["end_period_ts"])
        price = c.get("price") or {}
        last = _get(price, "close")
        if last is None:
            last = _get(price, "previous")
        bid = _get(c.get("yes_bid"), "close")
        ask = _get(c.get("yes_ask"), "close")
        vol = _num(c.get("volume", c.get("volume_fp")))
        out[t] = (t, bid, ask, last, vol)
    return [out[t] for t in sorted(out)]


def price_from(bid, ask, last):
    b = 0.0 if bid is None else bid
    a = 1.0 if ask is None else ask
    if a - b <= SPREAD_MAX + 1e-9:
        return (a + b) / 2
    if last is not None:
        return last
    return None


def candle_path(m, start, end):
    """Cache path (gzip). Older cache files may be plain .json (same name without .gz)."""
    return os.path.join(KRAW, "candles", m["event_ticker"], f"{m['ticker']}_{start}_{end}_60.json.gz")


def _read_cached(dest):
    for p, gz in ((dest, True), (dest[:-3], False)):
        if os.path.exists(p):
            data = open(p, "rb").read()
            return json.loads(gzip.decompress(data) if gz else data)
    return None


def fetch_candles(m, series, start, end, live_ok=True):
    """Fetch hourly candles for market m over [start, end] (unix s) in <=MAX_CANDLES-hour
    chunks, historical endpoint first and the live one if the market is not historical.
    Returns the raw candle dicts (all chunks)."""
    out = []
    tq = urllib.parse.quote(m["ticker"], safe="")  # a few legacy tickers contain spaces
    t = start
    while t < end:
        e = min(end, t + MAX_CANDLES * 3600)
        dest = candle_path(m, t, e)
        q = f"start_ts={t}&end_ts={e}&period_interval=60"
        d = _read_cached(dest)
        if d is None:
            try:
                d = fetch_json(f"{API}/historical/markets/{tq}/candlesticks?{q}", dest, gz=True)
            except Exception as ex:  # 404 -> not in the historical partition
                code = getattr(ex, "code", None)
                if code == 404 and live_ok:
                    try:
                        d = fetch_json(f"{API}/series/{series}/markets/{tq}/candlesticks?{q}", dest, gz=True)
                    except Exception as ex2:
                        log("kalshi_fetch.log", f"FAIL live {m['ticker']} {t}-{e}: {ex2}")
                        d = {"candlesticks": []}
                else:
                    log("kalshi_fetch.log", f"FAIL hist {m['ticker']} {t}-{e}: {ex}")
                    d = {"candlesticks": []}
        out += d.get("candlesticks") or []
        t = e
    return out


def cached_candles(m):
    """All cached candle chunks for a market, merged and parsed."""
    objs = []
    base = os.path.join(KRAW, "candles", m["event_ticker"], f"{glob.escape(m['ticker'])}_*_60.json")
    seen = set()
    for f in sorted(glob.glob(base + ".gz")) + sorted(glob.glob(base)):
        key = f[:-3] if f.endswith(".gz") else f
        if key in seen:
            continue
        seen.add(key)
        data = open(f, "rb").read()
        d = json.loads(gzip.decompress(data) if f.endswith(".gz") else data)
        objs += d.get("candlesticks") or []
    return parse_candles(objs)


def now_ts():
    return int(time.time())
