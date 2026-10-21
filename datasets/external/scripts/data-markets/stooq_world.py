"""Stooq d_world_txt.zip (June 29 2025 snapshot): FX, precious metals, world indices, government bond yields,
money-market rates and cryptocurrencies. Extracts the selected files unmodified into raw/stooq/, then normalizes.

Usage: python3 -I stooq_world.py [world|crypto|all]
"""
import hashlib
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmlib  # noqa: E402

USB = "/run/media/bhoult/My Book/backups/popos-cosmic-2025/bhoult/programming/correlation"
ZIP = os.path.join(USB, "d_world_txt.zip")
RAWDIR = os.path.join(dmlib.RAW, "stooq", "d_world_txt")
URL = "https://stooq.com/db/h/ (bulk daily world archive d_world_txt.zip, local copy dated 2025-06-29)"
P = "data/daily/world/"

FX_NOTE = ("Stooq daily FX bar (Mon-Fri; Stooq's FX day ends late in the European evening / around the New York "
           "5 pm roll, i.e. at or after the US close), so known_after_days=1")
# file stem -> (id, name, subcategory, region, units)
FX = {
    "usd_i": ("stooq_usd_i", "Stooq US dollar index (Stooq's own currency basket, not ICE DXY)", "dollar_index", "US", "index points (Stooq methodology; ~60 in 2025, not DXY ~100)"),
    "eurusd": ("stooq_eurusd", "EUR/USD", "g10", "EA", "USD per EUR"),
    "usdjpy": ("stooq_usdjpy", "USD/JPY", "g10", "JP", "JPY per USD"),
    "gbpusd": ("stooq_gbpusd", "GBP/USD", "g10", "GB", "USD per GBP"),
    "usdchf": ("stooq_usdchf", "USD/CHF", "g10", "CH", "CHF per USD"),
    "audusd": ("stooq_audusd", "AUD/USD", "g10", "AU", "USD per AUD"),
    "usdcad": ("stooq_usdcad", "USD/CAD", "g10", "CA", "CAD per USD"),
    "nzdusd": ("stooq_nzdusd", "NZD/USD", "g10", "NZ", "USD per NZD"),
    "usdsek": ("stooq_usdsek", "USD/SEK", "g10", "SE", "SEK per USD"),
    "usdnok": ("stooq_usdnok", "USD/NOK", "g10", "NO", "NOK per USD"),
    "usdmxn": ("stooq_usdmxn", "USD/MXN", "em", "MX", "MXN per USD"),
    "usdbrl": ("stooq_usdbrl", "USD/BRL", "em", "BR", "BRL per USD"),
    "usdkrw": ("stooq_usdkrw", "USD/KRW", "em", "KR", "KRW per USD"),
    "usdinr": ("stooq_usdinr", "USD/INR", "em", "IN", "INR per USD"),
    "usdzar": ("stooq_usdzar", "USD/ZAR", "em", "ZA", "ZAR per USD"),
    "usdtry": ("stooq_usdtry", "USD/TRY", "em", "TR", "TRY per USD (post-2005 new lira; earlier values rescaled by Stooq)"),
    "usdrub": ("stooq_usdrub", "USD/RUB", "em", "RU", "RUB per USD"),
}
METALS = {
    "xauusd": ("stooq_xauusd", "Gold spot XAU/USD", "GLD"),
    "xagusd": ("stooq_xagusd", "Silver spot XAG/USD", "SLV"),
    "xptusd": ("stooq_xptusd", "Platinum spot XPT/USD", "PPLT"),
    "xpdusd": ("stooq_xpdusd", "Palladium spot XPD/USD", "PALL"),
}
# stem -> (id suffix, name, subcategory, region, close note, kad)
ASIA = "closes in the Asian session, hours before New York opens"
EUR = "closes in the European afternoon (~11:30-12:00 New York time)"
IDX = {
    "^nkx": ("nikkei225", "Nikkei 225", "asia_pacific", "JP", ASIA, 0),
    "^hsi": ("hangseng", "Hang Seng Index", "asia_pacific", "HK", ASIA, 0),
    "^shc": ("shanghai", "Shanghai Composite", "asia_pacific", "CN", ASIA, 0),
    "^kospi": ("kospi", "KOSPI", "asia_pacific", "KR", ASIA, 0),
    "^twse": ("taiex", "TAIEX (Taiwan Weighted)", "asia_pacific", "TW", ASIA, 0),
    "^aor": ("asx_allords", "S&P/ASX All Ordinaries", "asia_pacific", "AU", ASIA, 0),
    "^nz50": ("nzx50", "S&P/NZX 50", "asia_pacific", "NZ", ASIA, 0),
    "^sti": ("sti", "Straits Times Index", "asia_pacific", "SG", ASIA, 0),
    "^snx": ("sensex", "BSE Sensex", "asia_pacific", "IN", ASIA, 0),
    "^jci": ("jakarta", "Jakarta Composite", "asia_pacific", "ID", ASIA, 0),
    "^klci": ("klci", "FTSE Bursa Malaysia KLCI", "asia_pacific", "MY", ASIA, 0),
    "^set": ("set", "SET Index (Thailand)", "asia_pacific", "TH", ASIA, 0),
    "^dax": ("dax", "DAX", "europe", "DE", EUR, 0),
    "^ukx": ("ftse100", "FTSE 100", "europe", "GB", EUR, 0),
    "^cac": ("cac40", "CAC 40", "europe", "FR", EUR, 0),
    "^aex": ("aex", "AEX", "europe", "NL", EUR, 0),
    "^ibex": ("ibex35", "IBEX 35", "europe", "ES", EUR, 0),
    "^fmib": ("ftsemib", "FTSE MIB", "europe", "IT", EUR, 0),
    "^smi": ("smi", "Swiss Market Index", "europe", "CH", EUR, 0),
    "^omxs": ("omxs30", "OMX Stockholm 30", "europe", "SE", EUR, 0),
    "^oseax": ("oslo_allshare", "Oslo Bors All-share Index", "europe", "NO", EUR, 0),
    "^ath": ("athens", "Athens General Composite", "europe", "GR", EUR, 0),
    "^xu100": ("bist100", "BIST 100 (Istanbul)", "europe", "TR", "closes 18:00 Istanbul (~10:00-11:00 New York)", 0),
    "^moex": ("moex", "MOEX Russia Index", "europe", "RU", "main session closes 18:50 Moscow, but MOEX has run an evening session since 2023, so 1 to be safe; trading halted 2022-02-28..2022-03-23", 1),
    "^top40": ("jse_top40", "FTSE/JSE Top 40", "mideast_africa", "ZA", "closes 17:00 Johannesburg (~10:00-11:00 New York)", 0),
    "^tasi": ("tadawul", "Tadawul All Share (TASI)", "mideast_africa", "SA", "closes 15:00 Riyadh (~07:00-08:00 New York); trades Sunday-Thursday", 0),
    "^tsx": ("tsx", "S&P/TSX Composite", "americas", "CA", "closes 16:00 New York, simultaneous with the US close (closing auction at 4 pm)", 0),
    "^bvp": ("bovespa", "Bovespa (Ibovespa)", "americas", "BR", "B3 closes 17:00-18:00 Sao Paulo, which is at or after 16:00 New York for part of the year, so 1", 1),
    "^ipc": ("mexico_ipc", "S&P/BMV IPC (Mexico)", "americas", "MX", "BMV closes 15:00 Mexico City = 16:00-17:00 New York depending on DST, so 1", 1),
    "^ipsa": ("chile_ipsa", "S&P IPSA (Chile)", "americas", "CL", "Santiago closes 16:00 local = 14:00-16:00 New York depending on DST, so 1", 1),
}
# Stooq bond yield files: stem -> (id suffix, name, region, kad)
BONDS = {
    "2ydey": ("de2y", "Germany 2-year government bond yield", "DE"),
    "10ydey": ("de10y", "Germany 10-year Bund yield", "DE"),
    "30ydey": ("de30y", "Germany 30-year government bond yield", "DE"),
    "2yuky": ("uk2y", "UK 2-year gilt yield", "GB"),
    "10yuky": ("uk10y", "UK 10-year gilt yield", "GB"),
    "30yuky": ("uk30y", "UK 30-year gilt yield", "GB"),
    "2yjpy": ("jp2y", "Japan 2-year JGB yield", "JP"),
    "10yjpy": ("jp10y", "Japan 10-year JGB yield", "JP"),
    "30yjpy": ("jp30y", "Japan 30-year JGB yield", "JP"),
    "2yity": ("it2y", "Italy 2-year BTP yield", "IT"),
    "10yity": ("it10y", "Italy 10-year BTP yield", "IT"),
    "10yfry": ("fr10y", "France 10-year OAT yield", "FR"),
    "10yesy": ("es10y", "Spain 10-year Bono yield", "ES"),
    "10ycny": ("cn10y", "China 10-year government bond yield", "CN"),
    "10ycay": ("ca10y", "Canada 10-year government bond yield", "CA"),
    "10yauy": ("au10y", "Australia 10-year government bond yield", "AU"),
    "10ykry": ("kr10y", "South Korea 10-year government bond yield", "KR"),
    "10yiny": ("in10y", "India 10-year government bond yield", "IN"),
    "10ybry": ("br10y", "Brazil 10-year government bond yield", "BR"),
    "10ymxy": ("mx10y", "Mexico 10-year government bond yield", "MX"),
}
BOND_KAD = {}  # filled from the timing probe (see probe.py); default below
MONEY = {
    "money market/plopln3m": ("wibor3m", "WIBOR 3M (Warsaw interbank offered rate)", "PL", "fixing at 11:00 Warsaw", 0),
    "money market/huohuf3m": ("bubor3m", "BUBOR 3M (Budapest interbank offered rate)", "HU", "fixing at 11:00 Budapest", 0),
    "money market/cnocny3m": ("shibor3m", "SHIBOR 3M (Shanghai interbank offered rate)", "CN", "fixing published 11:00 Beijing; Stooq history starts 2020-04", 0),
}

# per-series repairs for the non-crypto block: id -> (start YYYYMMDD or None, [(drop_from, drop_to)], note)
WORLD_FIX = {
    "stooq_usdtry": ("20020101", [], "starts 2002 (post-float); 1990s Stooq data is stale and has alternating prints"),
    "stooq_usdrub": ("20000101", [], "starts 2000; 1998-99 Stooq data has many spike-and-revert prints"),
    "stooq_usdinr": ("19950101", [], "starts 1995; early-1990s data mostly unchanged (managed rate)"),
    "stooq_usdmxn": ("19950101", [], "starts 1995; 1990-94 data sparse/stale"),
    "stooq_bovespa": ("19950101", [], "starts 1995 (after the Real plan); earlier values reflect hyperinflation and redenominations"),
    "stooq_de2y": ("19990101", [], "starts 1999; 1990s data has spike-and-revert prints"),
    "stooq_uk2y": ("19990101", [], "starts 1999; 1990s data has spike-and-revert prints"),
    "stooq_es10y": ("19970101", [], "starts 1997"),
    "stooq_it10y": ("19970101", [], "starts 1997"),
    "stooq_bubor3m": (None, [("20041019", "20041019")], "2004-10-19 removed (bad print 11.55 between 10.6 and 10.54)"),
}

CRYPTO = {
    "btc": "Bitcoin", "eth": "Ethereum", "sol": "Solana", "xrp": "XRP", "doge": "Dogecoin", "bnb": "BNB",
    "ada": "Cardano", "link": "Chainlink", "avax": "Avalanche", "dot": "Polkadot", "fil": "Filecoin",
    "ltc": "Litecoin", "matic": "Polygon (MATIC, POL after the Sept-2024 1:1 migration)", "uni": "Uniswap",
    "near": "NEAR Protocol", "atom": "Cosmos", "ftm": "Fantom (Sonic S after the 2025 1:1 migration)",
    "trx": "TRON", "aave": "Aave", "xlm": "Stellar", "bch": "Bitcoin Cash", "crv": "Curve DAO",
    "etc": "Ethereum Classic", "sand": "The Sandbox", "chz": "Chiliz", "eos": "EOS (Vaulta after 2025 rebrand)",
    "mkr": "Maker", "grt": "The Graph", "xtz": "Tezos", "algo": "Algorand", "sushi": "SushiSwap",
    "xmr": "Monero", "rune": "THORChain", "vet": "VeChain", "mana": "Decentraland", "axs": "Axie Infinity",
    "hbar": "Hedera", "yfi": "yearn.finance", "comp": "Compound", "cake": "PancakeSwap", "lrc": "Loopring",
    "egld": "MultiversX (Elrond)", "ksm": "Kusama", "cro": "Cronos (Crypto.com)", "enj": "Enjin Coin",
    "zen": "Horizen", "zec": "Zcash", "neo": "NEO", "bat": "Basic Attention Token", "waves": "Waves",
    "dash": "Dash", "iota": "IOTA", "qtum": "Qtum", "xem": "NEM", "omg": "OMG Network", "dcr": "Decred",
    "icx": "ICON", "sc": "Siacoin", "kcs": "KuCoin Token",
}
STABLE = {"usdt": "Tether (USDT)", "usdc": "USD Coin (USDC)"}
# data repairs: (start date, rows to drop, note)
CRYPTO_FIX = {
    "btc": (None, [("20140207", "20140225")], "2014-02-07..2014-02-25 removed: Mt.Gox withdrawal-freeze/collapse prints (e.g. 111.92 on 2014-02-20 vs ~600 elsewhere); history before 2014-02-26 appears to be Mt.Gox-based (premium of up to ~10-20% vs other exchanges in late 2013)"),
    "xrp": ("20160101", [], "starts 2016-01-01: 2015 data has alternating thin-market prints"),
    "xmr": ("20160101", [], "starts 2016-01-01: 2015 data has alternating thin-market prints (0.278<->0.878)"),
    "dash": ("20160101", [], "starts 2016-01-01: 2014-2015 (Darkcoin era) data has round-number thin-market prints"),
    "dcr": ("20160301", [], "starts 2016-03-01: 2015-16 launch-period prints jump by 10x"),
    "zec": ("20161101", [], "starts 2016-11-01: launch days 2016-10-28..31 trade at 650-4000 USD"),
    "ltc": ("20140101", [("20140422", "20140422")], "starts 2014-01-01 (Oct-Dec 2013 thin, round-number prints); 2014-04-22 removed (bad print 5.0 between 12.73 and 12.18)"),
    "mkr": (None, [("20180407", "20180407")], "2018-04-07 removed (bad print 117.32 between 489 and 510)"),
    "sc": ("20160601", [("20190802", "20190802"), ("20190922", "20190922")], "starts 2016-06-01 (earlier prices sit on the 1e-6 tick); 2019-08-02 and 2019-09-22 removed (one-day bad prints 0.0072 and 0.0038)"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def extract(z, inner):
    """Copy one member unmodified into raw/stooq/d_world_txt/<inner>; return the local path."""
    dst = os.path.join(RAWDIR, inner)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    data = z.read(inner)
    with open(dst, "wb") as f:
        f.write(data)
    return dst


def read_stooq(path):
    """Stooq .txt -> list of (YYYYMMDD, close float, raw line)."""
    with open(path) as f:
        lines = f.read().splitlines()
    assert lines[0] == "<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>", lines[0]
    out = []
    for ln in lines[1:]:
        p = ln.split(",")
        assert p[1] == "D", ln
        out.append((p[2], float(p[7])))
    return out


def rows_from(path, start=None, drop=()):
    r = read_stooq(path)
    if start:
        r = [x for x in r if x[0] >= start]
    for a, b in drop:
        r = [x for x in r if not (a <= x[0] <= b)]
    return [(dmlib.ymd(d), c) for d, c in r]


def write_index(z):
    os.makedirs(os.path.dirname(RAWDIR), exist_ok=True)
    with open(os.path.join(dmlib.RAW, "stooq", "d_world_txt.zip.index.tsv"), "w") as f:
        f.write("name\tsize\tmodified\n")
        for i in z.infolist():
            f.write(f"{i.filename}\t{i.file_size}\t{'%04d-%02d-%02d %02d:%02d' % i.date_time[:5]}\n")


def add_fixed(part, meta, path, start=None, jump=None):
    meta["url"] = URL + " member " + os.path.relpath(path, RAWDIR)
    fx = WORLD_FIX.get(meta["id"])
    drop = []
    if fx:
        start = fx[0] or start
        drop = fx[1]
        meta["notes"] = (meta.get("notes", "") + "; " + fx[2]).strip("; ")
    part.add(meta, rows_from(path, start=start, drop=drop), jump=jump)


def world(z):
    part = dmlib.Part("stooq_world")
    for stem, (sid, name, sub, region, units) in FX.items():
        sub_dir = "currencies/major/" if os.path.basename(stem) in z_major else "currencies/other/"
        path = extract(z, P + sub_dir + stem + ".txt")
        add_fixed(part, dict(id=sid, name=name + " (Stooq daily close)", category="fx", subcategory=sub, region=region,
                      source="Stooq bulk d_world_txt", url=URL, frequency="daily", units=units, transform="logdiff",
                      known_after_days=1, notes=FX_NOTE), path, start="19900101", jump=0.15)
    for stem, (sid, name, etf) in METALS.items():
        path = extract(z, P + "currencies/major/" + stem + ".txt") if stem in ("xauusd", "xagusd") else \
            extract(z, P + "currencies/other/" + stem + ".txt")
        add_fixed(part, dict(id=sid, name=name + " (Stooq daily close)", category="commodity", subcategory="precious_metal",
                      region="Global", source="Stooq bulk d_world_txt", url=URL, frequency="daily",
                      units="USD per troy ounce", transform="logdiff", known_after_days=1,
                      notes=FX_NOTE.replace("FX bar", "spot-metal bar (FX-style 24h market)") +
                      f"; the US-listed ETF {etf} (etf_{etf.lower()}) is the exact-US-close version"),
                 path, start="19900101", jump=0.2)
    for stem, (suffix, name, sub, region, close, kad) in IDX.items():
        path = extract(z, P + "indices/" + stem + ".txt")
        add_fixed(part, dict(id="stooq_" + suffix, name=name + " (Stooq daily close)", category="world_equity",
                      subcategory=sub, region=region, source="Stooq bulk d_world_txt", url=URL, frequency="daily",
                      units="index points (local currency, price index)", transform="logdiff",
                      known_after_days=kad, notes=close + "; dated by local trading day"),
                 path, start="19900101", jump=0.15)
    path = extract(z, P + "indices/^cry.txt")
    add_fixed(part, dict(id="stooq_crb", name="Refinitiv/CoreCommodity CRB Index (Stooq daily close)", category="commodity",
                  subcategory="broad_index", region="Global", source="Stooq bulk d_world_txt", url=URL,
                  frequency="daily", units="index points", transform="logdiff", known_after_days=CRB_KAD,
                  notes="19-commodity futures index (energy-heavy) computed from US futures settlements (all by ~14:30 New York); "
                        "2005 methodology change (Reuters/Jefferies -> CoreCommodity)"),
             path, start="19940101", jump=0.1)
    for stem, (suffix, name, region) in BONDS.items():
        path = extract(z, P + "bonds/" + stem + ".b.txt")
        kad = BOND_KAD.get(suffix, DEFAULT_BOND_KAD.get(region, 1))
        add_fixed(part, dict(id="stooq_" + suffix, name=name + " (Stooq daily close)", category="rates",
                      subcategory="govt_" + suffix[2:], region=region, source="Stooq bulk d_world_txt", url=URL,
                      frequency="daily", units="percent (yield to maturity)", transform="diff", known_after_days=kad,
                      notes=bond_note(region, kad)),
                 path, start="19900101", jump=None)
    for stem, (suffix, name, region, note, kad) in MONEY.items():
        path = extract(z, P + stem + (".b.txt" if stem.startswith("bonds/") else ".txt"))
        if kad is None:
            kad = DEFAULT_BOND_KAD.get(region, 1)
            note = bond_note(region, kad)
        add_fixed(part, dict(id="stooq_" + suffix, name=name + " (Stooq daily)", category="rates", subcategory="money_market",
                      region=region, source="Stooq bulk d_world_txt", url=URL, frequency="daily", units="percent",
                      transform="diff", known_after_days=kad, notes=note),
                 path, start="19970101", jump=None)
    part.save()


def bond_note(region, kad):
    if region == "CN":
        return ("Stooq end-of-day yield, Asian session (closes hours before New York opens), so 0; illiquid quotes: "
                "11-40% of days unchanged in 2007-2013")
    if kad == 0:
        return ("Stooq end-of-day yield, local session closes before the New York afternoon, so 0; timing probe vs DGS10 (15:30 New York): "
                "moves pick up the previous US session (+0.1 to +0.3) and show no overlap with the US afternoon (look-ahead partial ~0, "
                "while IEF at 16:00 shows -0.11)")
    return ("Stooq end-of-day yield; Americas session ends at or after the US close (timing probe: look-ahead partial vs DGS10 "
            "+0.05 CA, +0.06 MX), so 1")


def crypto(z):
    part = dmlib.Part("stooq_crypto")
    note0 = ("Stooq daily crypto bar, 7 days a week; the bar closes after the US close (around 00:00 Warsaw/UTC, i.e. 18:00-20:00 "
             "New York; vs Coinbase on FRED at 5 PM PST the same-date moves correlate ~0.89 and Stooq closes 1-3 h earlier), so known_after_days=1; volume is reported only from 2021-12-26; Stooq outage gaps of 1-3 days around 2022-07-01..03")
    for stem, name in list(CRYPTO.items()) + list(STABLE.items()):
        path = extract(z, P + "cryptocurrencies/" + stem + ".v.txt")
        start, drop, fixnote = CRYPTO_FIX.get(stem, (None, [], ""))
        note = note0 + ("; " + fixnote if fixnote else "")
        sub = "coin"
        if stem in STABLE:
            sub = "stablecoin"
            note += ("; stablecoin kept for its depeg moves (USDT: 1.12 in Jan-2018, 0.90 in Apr-2017; "
                     "USDC: 0.956 close on 2023-03-11 during the SVB run); most days are ~1.000")
        part.add(dict(id="stooq_" + stem, name=name + " price (Stooq daily close)", category="crypto",
                      subcategory=sub, region="Global", source="Stooq bulk d_world_txt",
                      url=URL + " member " + os.path.relpath(path, RAWDIR), frequency="daily",
                      units="USD per coin", transform="logdiff", known_after_days=1, notes=note),
                 rows_from(path, start=start, drop=drop), jump=0.5)
    part.save()


# timing decisions (see SOURCES; bond/CRB values confirmed by probe.py)
DEFAULT_BOND_KAD = {"DE": 0, "GB": 0, "JP": 0, "IT": 0, "FR": 0, "ES": 0, "CN": 0, "AU": 0, "KR": 0, "IN": 0,
                    "CA": 1, "BR": 1, "MX": 1}
CRB_KAD = 0


def main():
    global z_major
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    z = zipfile.ZipFile(ZIP)
    z_major = {os.path.basename(n)[:-4] for n in z.namelist() if "/currencies/major/" in n and n.endswith(".txt")}
    write_index(z)
    if what in ("world", "all"):
        world(z)
    if what in ("crypto", "all"):
        crypto(z)
    print("catalog rows:", dmlib.merge_catalog())


z_major = set()
if __name__ == "__main__":
    main()
