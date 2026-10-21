"""ETF / ETN daily closes from the correlation2 price store (read-only), as external market proxies."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import dmlib  # noqa: E402

STORE = os.path.expanduser("~/.local/share/correlation2/stores/stooq-2025-06-27")

# ticker: (name, category, subcategory, region, extra note)
ETFS = {
    "SPY": ("SPDR S&P 500 ETF", "world_equity", "us_broad_etf", "US", ""),
    "QQQ": ("Invesco QQQ (Nasdaq-100) ETF", "world_equity", "us_broad_etf", "US", ""),
    "IWM": ("iShares Russell 2000 ETF", "world_equity", "us_broad_etf", "US", ""),
    "DIA": ("SPDR Dow Jones Industrial Average ETF", "world_equity", "us_broad_etf", "US", ""),
    "XLE": ("Energy Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "XLF": ("Financial Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "XLK": ("Technology Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "XLU": ("Utilities Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "XLV": ("Health Care Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "XLI": ("Industrial Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "XLB": ("Materials Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "XLP": ("Consumer Staples Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "XLY": ("Consumer Discretionary Select Sector SPDR", "world_equity", "us_sector_etf", "US", ""),
    "VNQ": ("Vanguard Real Estate ETF (US REITs)", "world_equity", "us_sector_etf", "US", ""),
    "EEM": ("iShares MSCI Emerging Markets ETF", "world_equity", "intl_etf", "EM", "US-listed; trades while most underlying markets are closed, so it embeds US-session news"),
    "EFA": ("iShares MSCI EAFE ETF", "world_equity", "intl_etf", "DM ex-US", "US-listed; trades while most underlying markets are closed"),
    "FXI": ("iShares China Large-Cap ETF", "world_equity", "intl_etf", "CN", "US-listed; Hong Kong-listed holdings are closed during US hours"),
    "EWJ": ("iShares MSCI Japan ETF", "world_equity", "intl_etf", "JP", "US-listed; Tokyo is closed during US hours"),
    "EWZ": ("iShares MSCI Brazil ETF", "world_equity", "intl_etf", "BR", "US-listed; overlaps the B3 session"),
    "TLT": ("iShares 20+ Year Treasury Bond ETF", "rates", "treasury_etf", "US", ""),
    "IEF": ("iShares 7-10 Year Treasury Bond ETF", "rates", "treasury_etf", "US", ""),
    "SHY": ("iShares 1-3 Year Treasury Bond ETF", "rates", "treasury_etf", "US", ""),
    "TIP": ("iShares TIPS Bond ETF", "rates", "treasury_etf", "US", "inflation-linked Treasuries"),
    "HYG": ("iShares iBoxx USD High Yield Corporate Bond ETF", "credit", "corporate_etf", "US", ""),
    "LQD": ("iShares iBoxx USD Investment Grade Corporate Bond ETF", "credit", "corporate_etf", "US", ""),
    "EMB": ("iShares J.P. Morgan USD Emerging Markets Bond ETF", "credit", "em_sovereign_etf", "EM", ""),
    "GLD": ("SPDR Gold Shares", "commodity", "precious_metal_etf", "Global", "physically backed"),
    "SLV": ("iShares Silver Trust", "commodity", "precious_metal_etf", "Global", "physically backed"),
    "PPLT": ("abrdn Physical Platinum Shares ETF", "commodity", "precious_metal_etf", "Global", "physically backed"),
    "PALL": ("abrdn Physical Palladium Shares ETF", "commodity", "precious_metal_etf", "Global", "physically backed"),
    "USO": ("United States Oil Fund (WTI futures)", "energy", "oil_etf", "US", "holds near-month WTI futures: roll yield/contango makes it drift below spot over time; April 2020 1:8 reverse split and change of roll schedule"),
    "UNG": ("United States Natural Gas Fund (Henry Hub futures)", "energy", "natgas_etf", "US", "holds near-month NG futures: heavy contango decay; several reverse splits"),
    "URA": ("Global X Uranium ETF", "energy", "uranium_equity_etf", "Global", "mostly uranium-miner equities (non-US listed), not a uranium price"),
    "DBC": ("Invesco DB Commodity Index Tracking Fund", "commodity", "broad_etf", "Global", "futures basket (energy-heavy), optimized roll"),
    "DBA": ("Invesco DB Agriculture Fund", "commodity", "agriculture_etf", "Global", "futures basket"),
    "DBB": ("Invesco DB Base Metals Fund", "commodity", "industrial_metal_etf", "Global", "aluminium, zinc, copper futures"),
    "CPER": ("United States Copper Index Fund", "commodity", "industrial_metal_etf", "Global", "COMEX copper futures"),
    "CORN": ("Teucrium Corn Fund", "commodity", "agriculture_etf", "US", "CBOT corn futures; roll costs"),
    "WEAT": ("Teucrium Wheat Fund", "commodity", "agriculture_etf", "US", "CBOT wheat futures; roll costs"),
    "SOYB": ("Teucrium Soybean Fund", "commodity", "agriculture_etf", "US", "CBOT soybean futures; roll costs"),
    "CANE": ("Teucrium Sugar Fund", "commodity", "agriculture_etf", "Global", "ICE sugar #11 futures; roll costs"),
    "UUP": ("Invesco DB US Dollar Index Bullish Fund", "fx", "dollar_etf", "US", "long USDX futures; tracks the ICE dollar index plus T-bill carry"),
}


def main():
    syms = {}
    with open(os.path.join(STORE, "symbols.tsv")) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            syms.setdefault(p[1], []).append((int(p[0]), p))
    buf = np.memmap(os.path.join(STORE, "prices.bin"), dtype=np.uint8, mode="r")
    assert bytes(buf[:8]) == b"C2PRICE1"
    n = int(np.frombuffer(buf[12:16].tobytes(), "<u4")[0])
    rows = int(np.frombuffer(buf[16:24].tobytes(), "<u8")[0])
    offsets = np.frombuffer(buf[24:24 + 8 * (n + 1)].tobytes(), "<u8").astype(np.int64)
    pos = 24 + 8 * (n + 1)
    cols = []
    for _ in range(6):
        pos = -(-pos // 64) * 64
        cols.append(pos)
        pos += rows * 4
    assert pos <= len(buf)

    def col(c, dtype, a, b):
        return np.frombuffer(buf[cols[c] + 4 * a:cols[c] + 4 * b].tobytes(), dtype)

    part = dmlib.Part("etf")
    rawdir = os.path.join(dmlib.RAW, "store")
    os.makedirs(rawdir, exist_ok=True)
    for t, (name, cat, sub, region, extra) in ETFS.items():
        cand = [x for x in syms.get(t, []) if x[1][3] == "etf"]
        assert len(cand) == 1, (t, cand)
        i, p = cand[0]
        a, b = offsets[i], offsets[i + 1]
        d = col(0, "<i4", a, b)
        o, h, lo, c, v = (col(k, "<f4", a, b) for k in range(1, 6))
        dates = [(np.datetime64("1970-01-01") + np.timedelta64(int(x), "D")).astype(str) for x in d]
        assert dates[0] == p[4] and dates[-1] == p[5] and len(dates) == int(p[6]), (t, dates[0], p)
        with open(os.path.join(rawdir, f"{t}.csv"), "w") as f:
            f.write(f"# extracted read-only from {STORE}/prices.bin (symbol id {i}, {p[2]}, source {p[7]})\n")
            f.write("date,open,high,low,close,volume\n")
            for k in range(len(dates)):
                f.write(f"{dates[k]},{dmlib.fmt(o[k])},{dmlib.fmt(h[k])},{dmlib.fmt(lo[k])},{dmlib.fmt(c[k])},{dmlib.fmt(v[k])}\n")
        note = ("US close (4 pm New York); Stooq split- and dividend-adjusted closes (total-return-like), "
                "same June-2025 vintage as the stock panel; ETFs are excluded from the census universe")
        if extra:
            note += "; " + extra
        meta = dict(id=f"etf_{t.lower()}", name=f"{name} ({t}) close", category=cat, subcategory=sub,
                    region=region, source="Stooq US daily bulk (d_us_txt) via correlation2 store",
                    url=f"file://{STORE}/prices.bin#{t}", frequency="daily",
                    units="USD per share (split/dividend-adjusted)", transform="logdiff", known_after_days=0,
                    notes=note)
        part.add(meta, [(dates[k], float(dmlib.fmt(c[k]))) for k in range(len(dates))], jump=0.2)
    part.save()
    print("catalog rows:", dmlib.merge_catalog())


if __name__ == "__main__":
    main()
