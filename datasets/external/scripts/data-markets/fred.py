"""FRED (St. Louis Fed) series via the keyless fredgraph CSV endpoint. Sequential, ~1.2 s between requests.

Usage: python3 -I fred.py [--refresh]
Raw CSVs are saved unmodified to raw/fred/<ID>.csv (skipped if already present unless --refresh).
Monthly series are re-dated from FRED's first-of-month convention to the last calendar day of the reference
month, so known_after_days counts from the end of the reference period (like the weekly series).
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmlib  # noqa: E402

URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
RAWDIR = os.path.join(dmlib.RAW, "fred")
VINTAGE = ("FRED current vintage (downloaded 2026-10-18), i.e. revised values, not as first published; "
           "ALFRED (free API key) has vintages/release dates")
SHUTDOWN = ("first-release delays: Oct-2013 shutdown (Sept-2013 data) and Dec-2018/Jan-2019 shutdown (Commerce data); "
            "the Oct-Nov 2025 shutdown delayed/cancelled releases after the stock-data window")

# (FRED id, name, category, subcategory, region, frequency, units, transform, known_after_days, notes)
SERIES = [
    # --- daily energy
    ("DCOILWTICO", "WTI crude oil spot price, Cushing OK", "energy", "oil", "US", "daily", "USD per barrel", "logdiff", 0,
     "EIA/Refinitiv spot assessment tied to the NYMEX settlement (14:30 New York); went negative (-36.98) on 2020-04-20"),
    ("DCOILBRENTEU", "Brent crude oil spot price, Europe (FOB)", "energy", "oil", "Global", "daily", "USD per barrel", "logdiff", 0,
     "EIA/Refinitiv assessment at the London close (before the New York afternoon)"),
    ("DHHNGSP", "Henry Hub natural gas spot price", "energy", "natgas", "US", "daily", "USD per million BTU", "logdiff", 1,
     "next-day gas traded in the morning; assessment/publication time not documented, so 1 to be safe; occasional extreme spikes (e.g. Feb-2021 Winter Storm Uri, Jan-2014)"),
    ("DGASNYH", "Conventional gasoline spot price, New York Harbor (regular)", "energy", "refined_products", "US", "daily", "USD per gallon", "logdiff", 0,
     "EIA/Refinitiv assessment at the NYMEX close (14:30 New York)"),
    ("DHOILNYH", "No. 2 heating oil spot price, New York Harbor", "energy", "refined_products", "US", "daily", "USD per gallon", "logdiff", 0,
     "EIA/Refinitiv assessment at the NYMEX close (14:30 New York)"),
    # --- daily rates
    ("DGS3MO", "3-month Treasury constant-maturity yield", "rates", "treasury", "US", "daily", "percent", "diff", 0,
     "H.15/Treasury CMT from indicative bid quotes at ~15:30 New York (observed before the close; published that evening)"),
    ("DGS2", "2-year Treasury constant-maturity yield", "rates", "treasury", "US", "daily", "percent", "diff", 0,
     "H.15/Treasury CMT from indicative bid quotes at ~15:30 New York"),
    ("DGS10", "10-year Treasury constant-maturity yield", "rates", "treasury", "US", "daily", "percent", "diff", 0,
     "H.15/Treasury CMT from indicative bid quotes at ~15:30 New York"),
    ("DGS30", "30-year Treasury constant-maturity yield", "rates", "treasury", "US", "daily", "percent", "diff", 0,
     "H.15/Treasury CMT at ~15:30 New York; 30-year not issued 2002-02-19..2006-02-08 (gap in FRED)"),
    ("T10Y2Y", "10-year minus 2-year Treasury spread", "rates", "curve", "US", "daily", "percentage points", "diff", 0,
     "DGS10 - DGS2 (both ~15:30 New York)"),
    ("T10Y3M", "10-year minus 3-month Treasury spread", "rates", "curve", "US", "daily", "percentage points", "diff", 0,
     "DGS10 - DGS3MO (both ~15:30 New York)"),
    ("T10YIE", "10-year breakeven inflation rate", "rates", "inflation_expectations", "US", "daily", "percent", "diff", 0,
     "DGS10 - DFII10 (both ~15:30 New York); starts 2003"),
    ("DFII10", "10-year TIPS (real) constant-maturity yield", "rates", "real_yield", "US", "daily", "percent", "diff", 0,
     "H.15 TIPS CMT at ~15:30 New York; starts 2003"),
    ("DFF", "Effective federal funds rate", "rates", "policy", "US", "daily", "percent", "diff", 1,
     "7-day series (weekends carry Friday's rate); the rate for day d is published by the NY Fed at 09:00 on the next business day; mostly steps at FOMC decisions"),
    # --- daily credit (ICE BofA OAS on FRED is truncated to the last 3 years, so Moody's spreads instead)
    ("BAA10Y", "Moody's seasoned Baa corporate yield minus 10-year Treasury", "credit", "corporate_spread", "US", "daily", "percentage points", "diff", 1,
     "Moody's daily seasoned long-term (20y+) Baa index minus DGS10; end-of-day bond prices, so 1; substitute for ICE BofA OAS, which FRED now serves only from 2023-10"),
    ("AAA10Y", "Moody's seasoned Aaa corporate yield minus 10-year Treasury", "credit", "corporate_spread", "US", "daily", "percentage points", "diff", 1,
     "Moody's daily seasoned long-term Aaa index minus DGS10; end-of-day, so 1"),
    # --- daily volatility (Cboe indices close at 16:15 New York)
    ("VIXCLS", "Cboe Volatility Index (VIX) close", "volatility", "equity_vol", "US", "daily", "index points (annualized % vol)", "logdiff", 1,
     "close is set at 16:15 New York (SPX options trade to 16:15), after the 16:00 stock close, so 1; the 16:00 value is nearly the same, so 0 would admit only a 15-minute window"),
    ("VXNCLS", "Cboe Nasdaq-100 Volatility Index (VXN) close", "volatility", "equity_vol", "US", "daily", "index points", "logdiff", 1,
     "close at 16:15 New York, so 1"),
    ("RVXCLS", "Cboe Russell 2000 Volatility Index (RVX) close", "volatility", "equity_vol", "US", "daily", "index points", "logdiff", 1,
     "close at 16:15 New York, so 1"),
    ("VXEEMCLS", "Cboe Emerging Markets ETF Volatility Index (VXEEM) close", "volatility", "equity_vol", "EM", "daily", "index points", "logdiff", 1,
     "close after 16:00 New York, so 1; starts 2011"),
    ("OVXCLS", "Cboe Crude Oil ETF Volatility Index (OVX) close", "volatility", "commodity_vol", "US", "daily", "index points", "logdiff", 1,
     "implied vol of USO options; close after 16:00 New York, so 1; spiked to 300+ in April 2020"),
    ("GVZCLS", "Cboe Gold ETF Volatility Index (GVZ) close", "volatility", "commodity_vol", "Global", "daily", "index points", "logdiff", 1,
     "implied vol of GLD options; close after 16:00 New York, so 1"),
    # --- daily FX (H.10 noon buying rates in New York)
    ("DTWEXBGS", "Nominal broad US dollar index (goods and services)", "fx", "dollar_index", "US", "daily", "index Jan-2006=100", "logdiff", 0,
     "Fed trade-weighted index built from the 12:00 New York H.10 rates: the inputs are market rates observed at noon (like CMT yields, hence 0), "
     "but the Fed publishes H.10 weekly on Mondays; if official publication is required use known_after_days=8"),
    ("DEXUSEU", "USD/EUR noon buying rate in New York", "fx", "g10", "EA", "daily", "USD per EUR", "logdiff", 0,
     "H.10 noon (12:00 New York) rate: a market rate observed before the close, hence 0; published weekly in H.10 (use 8 if official publication is required); "
     "earlier-timed counterpart of stooq_eurusd (kad 1)"),
    ("DEXJPUS", "JPY/USD noon buying rate in New York", "fx", "g10", "JP", "daily", "JPY per USD", "logdiff", 0,
     "H.10 noon (12:00 New York) rate, hence 0 (8 if official weekly publication is required)"),
    ("DEXCHUS", "CNY/USD noon buying rate in New York", "fx", "em", "CN", "daily", "CNY per USD", "logdiff", 0,
     "H.10 noon (12:00 New York) rate, hence 0 (8 if official weekly publication is required); hard peg 8.28 until 2005-07-21 (zero moves), managed band afterwards; replaces Stooq USDCNY, which is full of bad prints"),
    ("DEXCAUS", "CAD/USD noon buying rate in New York", "fx", "g10", "CA", "daily", "CAD per USD", "logdiff", 0,
     "H.10 noon (12:00 New York) rate, hence 0 (8 if official weekly publication is required)"),
    ("DEXMXUS", "MXN/USD noon buying rate in New York", "fx", "em", "MX", "daily", "MXN per USD", "logdiff", 0,
     "H.10 noon (12:00 New York) rate, hence 0 (8 if official weekly publication is required)"),
    ("DEXBZUS", "BRL/USD noon buying rate in New York", "fx", "em", "BR", "daily", "BRL per USD", "logdiff", 0,
     "H.10 noon (12:00 New York) rate, hence 0 (8 if official weekly publication is required)"),
    # --- daily other
    ("USEPUINDXD", "Daily news-based Economic Policy Uncertainty index (US)", "attention", "news_uncertainty", "US", "daily", "index (scaled newspaper article counts)", "logdiff", 2,
     "Baker-Bloom-Davis; built from that day's US newspaper articles, posted with a lag, so 2 to be safe; very noisy day to day (7-day series)"),
    # --- daily crypto (Coinbase)
    ("CBBTCUSD", "Coinbase Bitcoin price", "crypto", "coin_coinbase", "Global", "daily", "USD per BTC", "logdiff", 1,
     "FRED note: 'All data is as of 5 PM PST' (after the US close), so 1; 7-day series; Coinbase copyright: reproduction prohibited without permission (private research use)"),
    ("CBETHUSD", "Coinbase Ethereum price", "crypto", "coin_coinbase", "Global", "daily", "USD per ETH", "logdiff", 1,
     "as of 5 PM PST, so 1; Coinbase copyright: private research use"),
    ("CBLTCUSD", "Coinbase Litecoin price", "crypto", "coin_coinbase", "Global", "daily", "USD per LTC", "logdiff", 1,
     "as of 5 PM PST, so 1; Coinbase copyright: private research use"),
    ("CBBCHUSD", "Coinbase Bitcoin Cash price", "crypto", "coin_coinbase", "Global", "daily", "USD per BCH", "logdiff", 1,
     "as of 5 PM PST, so 1; Coinbase copyright: private research use"),
    # --- weekly
    ("ICSA", "Initial jobless claims (SA)", "macro", "labor", "US", "weekly", "claims", "logdiff", 5,
     "week ending Saturday, released the following Thursday 08:30 New York (Wednesday in holiday weeks), so 5; revised (seasonal factors re-estimated yearly); COVID spike to 6.1m in 2020"),
    ("CCSA", "Continued jobless claims (insured unemployment, SA)", "macro", "labor", "US", "weekly", "persons", "logdiff", 12,
     "week ending Saturday, released with the initial claims of the following week (Thursday, 12 days later), so 12"),
    ("MORTGAGE30US", "30-year fixed mortgage rate (Freddie Mac PMMS)", "rates", "mortgage", "US", "weekly", "percent", "diff", 0,
     "dated Thursday = release day (noon New York; Wednesday in holiday weeks), so 0; survey covers Mon-Wed"),
    ("GASREGW", "US regular gasoline retail price (all formulations)", "energy", "retail_fuel", "US", "weekly", "USD per gallon", "logdiff", 2,
     "EIA survey of ~900 stations at 08:00 Monday, published Monday afternoon (Tuesday after a holiday), so 2"),
    ("NFCI", "Chicago Fed National Financial Conditions Index", "credit", "financial_conditions", "US", "weekly", "index (0 = average, + = tighter)", "diff", 5,
     "week ending Friday, released Wednesday 08:30 New York, so 5; the whole history is re-estimated every week (model look-ahead in the current vintage)"),
    ("STLFSI4", "St. Louis Fed Financial Stress Index (v4)", "credit", "financial_conditions", "US", "weekly", "index (0 = normal stress)", "diff", 7,
     "week ending Friday, released the following Thursday (Friday after holidays), so 7; principal-component index re-estimated with new data; v4 history back-computed in 2022 (model look-ahead)"),
    ("WEI", "Weekly Economic Index (Lewis-Mertens-Stock)", "macro", "activity", "US", "weekly", "percent (4-quarter GDP growth scale)", "diff", 7,
     "week ending Saturday, published the following Thursday (later in holiday weeks), so 7; re-estimated and revised each week; starts 2008"),
    # --- monthly (re-dated to the last day of the reference month)
    ("UNRATE", "Unemployment rate", "macro", "labor", "US", "monthly", "percent", "diff", 23,
     "Employment Situation; usual release the first Friday after month end (<=10 days), bound 23 covers Sept-2013 data released 2013-10-22 (shutdown)"),
    ("PAYEMS", "All employees, total nonfarm (payrolls)", "macro", "labor", "US", "monthly", "thousands of persons", "logdiff", 23,
     "Employment Situation; usual release <=10 days after month end, bound 23 covers the 2013 shutdown; revised twice monthly and benchmarked yearly"),
    ("CPIAUCSL", "CPI-U all items (SA)", "macro", "inflation", "US", "monthly", "index 1982-84=100", "logdiff", 31,
     "usual release 10-21 days after month end, bound 31 covers Sept-2013 CPI released 2013-10-30 (shutdown); SA factors revised yearly"),
    ("INDPRO", "Industrial production index", "macro", "activity", "US", "monthly", "index 2017=100", "logdiff", 30,
     "Fed G.17, usual release ~15-18 days after month end; bound 30 covers the Sept-2013 G.17, postponed to late October 2013 because its BLS inputs were delayed by the shutdown; revised for several months"),
    ("RSAFS", "Advance retail sales: retail and food services", "macro", "consumption", "US", "monthly", "millions of USD (SA)", "logdiff", 46,
     "usual release 13-17 days after month end; bound 46 covers Dec-2018 data released 2019-02-14 (shutdown); revised"),
    ("HOUST", "Housing starts, total (SAAR)", "macro", "housing", "US", "monthly", "thousands of units (SAAR)", "logdiff", 60,
     "usual release 16-19 days after month end; bound 60 covers Dec-2018 data released 2019-02-26 (shutdown) and Sept/Oct-2013 (released Nov/Dec 2013); revised"),
    ("UMCSENT", "University of Michigan consumer sentiment (final)", "macro", "sentiment", "US", "monthly", "index 1966Q1=100", "diff", 3,
     "final reading is released on the 2nd-to-last or last Friday of the reference month itself (preliminary mid-month), so <=0 days after month end; 3 as a cushion; monthly from 1978"),
    ("PCE", "Personal consumption expenditures (nominal, SAAR)", "macro", "consumption", "US", "monthly", "billions of USD (SAAR)", "logdiff", 61,
     "usual release 27-33 days after month end; bound 61 covers Dec-2018 data released 2019-03-01 (shutdown); heavily revised (annual/comprehensive revisions)"),
]
DAILY_START = "1990-01-01"


def fetch(sid, refresh):
    """Download with curl over HTTP/1.1 (FRED resets HTTP/2 streams, and urllib stalled on it)."""
    dst = os.path.join(RAWDIR, sid + ".csv")
    if os.path.exists(dst) and not refresh:
        return dst, False
    last = None
    for attempt in range(4):
        tmp = dst + ".tmp"
        r = subprocess.run(["curl", "-sS", "--http1.1", "-m", "120", "-o", tmp, "-w", "%{http_code}",
                            URL.format(sid)], capture_output=True, text=True)
        code = r.stdout.strip()
        try:
            with open(tmp, "rb") as f:
                head = f.read(64)
        except FileNotFoundError:
            head = b""
        if r.returncode == 0 and code == "200" and head.startswith(b"observation_date,"):
            os.replace(tmp, dst)
            return dst, True
        last = f"curl rc={r.returncode} http={code} err={r.stderr.strip()[:200]} head={head[:60]!r}"
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{sid}: download failed: {last}")


def main():
    refresh = "--refresh" in sys.argv
    os.makedirs(RAWDIR, exist_ok=True)
    part = dmlib.Part("fred")
    for sid, name, cat, sub, region, freq, units, tr, kad, note in SERIES:
        path, downloaded = fetch(sid, refresh)
        if downloaded:
            time.sleep(1.2)
        with open(path) as f:
            lines = f.read().splitlines()
        assert lines[0] == f"observation_date,{sid}", lines[0]
        rows = []
        for ln in lines[1:]:
            d, v = ln.split(",")
            if v in ("", "."):
                continue
            if freq == "monthly":
                d = dmlib.month_end(d)
            if freq == "daily" and d < DAILY_START:
                continue
            if sid == "UMCSENT" and d < "1978-01-31":
                continue  # quarterly before 1978
            rows.append((d, float(v)))
        notes = note
        if freq == "monthly":
            notes = "date = last day of the reference month (FRED's first-of-month date moved to month end); " + note + "; " + SHUTDOWN
        if freq in ("monthly", "weekly") and sid not in ("MORTGAGE30US", "GASREGW"):
            notes += "; " + VINTAGE
        jump = {"daily": 0.25, "weekly": 0.5, "monthly": 0.2}[freq] if tr == "logdiff" else None
        if cat == "crypto":
            jump = 0.5
        part.add(dict(id="fred_" + sid.lower(), name=name, category=cat, subcategory=sub, region=region, source="FRED",
                      url=f"https://fred.stlouisfed.org/series/{sid}", frequency=freq, units=units, transform=tr,
                      known_after_days=kad, notes=notes), rows, jump=jump)
    part.save()
    print("catalog rows:", dmlib.merge_catalog())


if __name__ == "__main__":
    main()
