# SOURCES: data-markets agent

Non-US-stock financial and economic series that could lead US stocks, normalized per `external_spec.md`.
Built 2026-10-18/19. Catalog: `catalog_data-markets.csv` (255 series). Series files: `series/<id>.csv`.

| Block | Source | Series | Frequency | Coverage |
|---|---|---|---|---|
| FX, precious metals, world indices, CRB, government yields, money market | Stooq bulk `d_world_txt.zip` (USB copy, 2025-06-29) | 75 | daily | 1990 (or listing date) .. 2025-06-27 |
| Cryptocurrencies (59 coins + USDT, USDC) | Stooq bulk `d_world_txt.zip` | 61 | daily, 7 days a week | 2010-07 (BTC) .. 2025-06-28 |
| Coinbase BTC/ETH/LTC/BCH | FRED | 4 | daily, 7 days a week | 2014-12 .. 2026-10-18 |
| Bitcoin on-chain activity | blockchain.com charts API | 5 | daily | 2011-01 .. 2026-10-14 |
| Crypto Fear & Greed | alternative.me API | 1 | daily | 2018-02-01 .. 2026-10-19 |
| Energy, rates, credit, vol, FX, EPU, claims, conditions, monthly macro | FRED (fredgraph CSV, no key) | 45 | 30 daily, 7 weekly, 8 monthly | 1990 (daily) / 1919-1978 (weekly, monthly) .. 2026-10 |
| US and big-economy macro not on FRED | Stooq bulk `d_macro_txt.zip` | 20 | monthly | 1963/1968/1985/2001/2010 .. 2025-06 |
| Crude-oil stocks, natural-gas storage | EIA history pages (+ Stooq before 2010 for gas) | 2 | weekly | 1982 / 1993 .. 2026-10-09 |
| ETF / ETN closes (market and commodity proxies) | correlation2 price store (Stooq US daily, same vintage as stocks) | 42 | daily | 1999/2005-2012 .. 2025-06-27 |

By category: crypto 71, world_equity 49 (30 indices + 19 US-listed equity ETFs), rates 37, macro 31, fx 25, commodity 17,
energy 11, credit 7, volatility 6, attention 1. `known_after_days`: 117 series at 0, 104 at 1, the rest 2..66 (weekly/monthly releases).

The count (255) is slightly above the spec's ~250 because the coordinator expanded crypto to 60-80 series. Near-duplicates were
trimmed elsewhere: no US indices from Stooq (ETFs instead), no Stooq US yields (FRED), 3 money-market rates, only key bond maturities.

## Files

- `raw/stooq/`: `SHA256SUMS.txt` (both zips), `d_world_txt.zip.index.tsv` (full zip index), the 136 selected world members
  extracted unmodified under `raw/stooq/d_world_txt/data/daily/world/...`, and the whole `d_macro_txt.zip` (0.9 MB).
  - Zip SHA-256: world `0642fa43...ef48e8`, macro `eab870b3...c2c1`; file dates 2025-06-29.
- `raw/fred/<ID>.csv`: FRED downloads, unmodified (49 files; `CNP16OV.csv` and `POPTHM.csv` there belong to data-other).
- `raw/eia/dm_WCESTUS1_leaf.html`, `raw/eia/dm_nw2_epg0_swo_r48_bcfw.htm` (the other `raw/eia` files are data-other's).
- `raw/blockchain_com/*.json`, `raw/alternative_me/fng_limit0.json`.
- `raw/store/<TICKER>.csv`: full OHLCV extract of each ETF from the read-only price store (provenance; not a download).
- `qc_data-markets.txt`: per-series QC log (range, duplicates, gaps, flat runs, largest moves, first/last rows).
- `timing_probe_data-markets.tsv`: the timing probe behind the `known_after_days` choices (see Timing).
- `scripts/data-markets/*.py`: the scripts that built everything (standard library + numpy, run with `python3 -I`; they were
  run from the session scratchpad, so `dmlib.WORK` must be repointed to re-run them).

## Sources, terms, what was taken

### 1. Stooq bulk archives (local USB copies, June 29 2025, the same vintage as the stock data)
`/run/media/bhoult/My Book/backups/popos-cosmic-2025/bhoult/programming/correlation/d_{world,macro}_txt.zip`, from https://stooq.com/db/h/.

**Terms.** Stooq's free bulk data is for private, non-commercial use; do not redistribute.

**Format.** Stooq daily CSV `<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`; the close is used.

**World archive selection (75 series):**
- **FX (17):** Stooq USD index `usd_i`, plus EURUSD, USDJPY, GBPUSD, USDCHF, AUDUSD, USDCAD, NZDUSD, USDSEK, USDNOK, USDMXN, USDBRL, USDKRW, USDINR, USDZAR, USDTRY, USDRUB.
  - `usd_i` is Stooq's own basket index (~60 in 2025), not ICE DXY. UUP (ETF) and DTWEXBGS (FRED) are the other dollar measures.
- **Precious metals (4):** XAU, XAG, XPT, XPD spot in USD.
- **Equity indices (30):**
  - Asia-Pacific: Nikkei 225, Hang Seng, Shanghai Composite, KOSPI, TAIEX, ASX All Ords, NZX 50, STI, Sensex, Jakarta, KLCI, SET.
  - Europe: DAX, FTSE 100, CAC 40, AEX, IBEX 35, FTSE MIB, SMI, OMXS30, Oslo all-share, Athens, BIST 100, MOEX.
  - Middle East/Africa: JSE Top 40, Tadawul.
  - Americas: TSX, Bovespa, Mexico IPC, Chile IPSA.
- **Commodity index (1):** Refinitiv/CoreCommodity CRB (`^cry`).
- **Government yields (20):** DE 2/10/30y, UK 2/10/30y, JP 2/10/30y, IT 2/10y, FR, ES, CN, CA, AU, KR, IN, BR, MX 10y.
  - Several series start only 2005-11-28 (UK/JP/FR/CA/AU 10y); BR 10y starts 2010, MX 10y 2008.
- **Money market (3):** WIBOR 3M (PL), BUBOR 3M (HU), SHIBOR 3M (CN, from 2020-04). The archive's money-market folder holds only PL, HU and CN.
- **Crypto (61):** see the Crypto section.

**Macro archive selection (20).** Stooq dates monthly values on the last business day of the reference month (weekly: the week-end
day), i.e. the reference period, not the release. To be safe, each monthly date was moved to the month's last weekday if Stooq's
date was earlier: 3 Chicago PMI dates, e.g. 2002-12-29. `known_after_days` comes from each release calendar (see the catalog notes).
- **US (12):**
  - not on FRED: ISM manufacturing and ISM services (both 2010+, FRED dropped ISM for licensing reasons), Chicago PMI (2001+), Conference Board confidence (2010+), Conference Board LEI m/m (2010+);
  - regional Fed surveys: Philadelphia Fed (1968+), Empire State (2010+), Richmond Fed (2010+);
  - housing: NAHB (1985+), NAR existing home sales (1999+; FRED carries only 13 months), new home sales (1963+);
  - ADP employment change (2001+).
- **Big economies (8):** China CPI y/y and industrial production y/y; Ifo; ZEW sentiment; euro-area HICP y/y and manufacturing PMI; Japan industrial production y/y; UK CPI y/y (all 2010+).
- **Not taken:**
  - US series that duplicate FRED: claims, CPI, payrolls, unemployment, IP, starts, permits, sentiment.
  - Most m/m and y/y variants of the same data; quarterly GDP/ULC/PCE-q.
  - S&P Global US PMIs: only from 2016-17, and the newest values are flash readings.
  - China PMI `pmmncn`: recent values match neither NBS nor Caixin.
  - GfK Germany: forward-dated.
  - Stooq's EIA weekly crude and gas files: replaced by EIA itself, since Stooq misses weeks, e.g. 13 weeks in 2017.

### 2. FRED, https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID> (no key)
Downloaded sequentially with curl over HTTP/1.1 (FRED resets HTTP/2 streams; Python's urllib stalled), ~1.2 s apart.

**Terms.** FRED data are free, but many series are third-party copyrighted: FRED flags "data in this graph are copyrighted"
on Moody's, Cboe, NFCI, STLFSI4, WEI and EPU. Coinbase's notice: "Reproduction of Coinbase data in any form is prohibited except with
the prior written permission of Coinbase". Use them for private research with citation and don't redistribute.

**Taken (49):**
- **Daily energy:** WTI and Brent spot, Henry Hub spot, NY Harbor gasoline and heating-oil spot.
- **Daily rates:** DGS3MO/2/10/30, T10Y2Y, T10Y3M, T10YIE, DFII10, DFF.
- **Daily credit:** BAA10Y, AAA10Y.
- **Daily vol:** VIX, VXN, RVX, VXEEM, OVX, GVZ.
- **Daily FX:** DTWEXBGS, DEXUSEU/JPUS/CHUS/CAUS/MXUS/BZUS.
- **Daily EPU:** USEPUINDXD.
- **Coinbase:** CBBTC/ETH/LTC/BCHUSD.
- **Weekly:** ICSA, CCSA, MORTGAGE30US, GASREGW, NFCI, STLFSI4, WEI.
- **Monthly:** UNRATE, PAYEMS, CPIAUCSL, INDPRO, RSAFS, HOUST, UMCSENT (monthly from 1978), PCE.

Daily series start 1990-01-01 (or the series start). Population (POPTHM, CNP16OV) is data-other's and was not taken.

**Monthly dates.** FRED dates the 1st of the reference month; I re-dated each observation to the last calendar day of the month,
so `known_after_days` counts from the end of the reference period, like the weekly series. Note that data-other's FRED population
series keep the first-of-month date and count their `known_after_days` from it.

**Vintage.** All FRED values are the current vintage (2026-10), i.e. revised. ALFRED (https://alfred.stlouisfed.org, free API key)
has every vintage and exact release dates; with it, monthly `known_after_days` could be exact per observation and values unrevised.

**Not taken from FRED:**
- ICE BofA OAS (BAMLH0A0HYM2, BAMLC0A0CM): FRED now serves only the last 3 years (from 2023-10-17, ICE licensing). Moody's spreads, HYG and LQD are used instead.
- WCESTUS1: not on FRED (404), so taken from EIA.
- LBMA gold: removed from FRED; Stooq XAUUSD and GLD instead.

### 3. EIA, https://www.eia.gov (public domain; citation requested)
- **Crude stocks:** `WCESTUS1` weekly U.S. ending stocks of crude oil excluding the SPR, 1982-08-20.., in thousand barrels, from the leaf page
  https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTUS1&f=W.
- **Gas storage:** Lower-48 working gas in underground storage, from https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm.
  - EIA's page starts 2010-01-01; 1993-12-31..2009-12-25 is prepended from Stooq `nginus.m`.
  - The two sources agree within 1 Bcf on 701 of 766 overlapping weeks (the rest are small revisions).

### 4. correlation2 price store (read-only), `~/.local/share/correlation2/store/prices.bin` + `symbols.tsv`
- **What:** 42 ETFs/ETNs, which the census excludes: SPY QQQ IWM DIA, XLE XLF XLK XLU XLV XLI XLB XLP XLY VNQ, EEM EFA FXI EWJ EWZ, TLT IEF SHY TIP,
  HYG LQD EMB, GLD SLV PPLT PALL, USO UNG URA, DBC DBA DBB CPER CORN WEAT SOYB CANE, UUP.
- **Prices:** Stooq split- and dividend-adjusted closes at the 4 pm New York close, so `known_after_days` 0. Most start 2005-02-25, the start of
  Stooq's free ETF history; QQQ starts 1999.
- **Parser:** binary layout per `crates/c2-data/src/store.rs`; dates are days since 1970-01-01.
- **Not taken:** other long-history candidates were skipped as near-duplicates (DBO, DBE, GSG, DJP, UGA, BNO, JNK, FXE, FXY), as stock baskets (GDX, XOP, OIH, SMH, KRE)
  or as too short (BITO 2021, IBIT/GBTC 2024, VXX 2018).

### 5. blockchain.com charts API and alternative.me
- **blockchain.com:** `https://api.blockchain.info/charts/<name>?timespan=all&format=json&sampled=false` (free, no key; blockchain.com terms, attribute).
  - Taken: hash-rate, n-transactions, n-unique-addresses, estimated-transaction-volume-usd, transaction-fees-usd, from 2011-01-01.
  - Difficulty skipped (a step function).
- **alternative.me:** Crypto Fear & Greed, `https://api.alternative.me/fng/?limit=0&format=json` (free; attribution "Crypto Fear & Greed Index by alternative.me").
- **Not available without a key:** total crypto market cap and BTC dominance history. CoinGecko's global history needs a paid key (a free demo key gives 365 days); CoinMarketCap history is paid.
- **Free but not taken (count budget):** DefiLlama (no key) has total stablecoin market cap and DeFi TVL.

## Timing (`known_after_days`)
Rule: calendar days after `date` until the value was public by 4 pm New York. Where a market quantity is observed before the close
(Treasury CMT at 15:30, FX noon rates), it counts as known that day even if the official table is published later, as in the spec's
Treasury example.

| Group | known_after_days | Why |
|---|---|---|
| Asian, European, Middle East and African indices; European and Asian government yields; money-market fixings; TSX; CRB | 0 | close before New York's close (TSX at the same moment, 4 pm) |
| Bovespa, Mexico IPC, Chile IPSA, MOEX (evening session since 2023); CA/BR/MX 10y | 1 | close at or after 4 pm New York for part of the year |
| Stooq FX and spot metals | 1 | Stooq's FX day ends at or after the US close |
| FRED H.10 noon FX and DTWEXBGS | 0 | 12:00 New York market rates; the Fed's H.10 table is published weekly, so use 8 if official publication is required |
| US Treasury CMT, spreads, breakeven, TIPS | 0 | quotes at ~15:30 New York |
| WTI, Brent, NY Harbor products | 0 | assessments at the NYMEX/London close (<= 14:30 New York) |
| Henry Hub spot | 1 | assessment time undocumented. The probe correlates it with gas producers' previous day (+0.1), so the date may be the flow day and 1 is generous |
| Cboe vol indices (VIX, VXN, RVX, VXEEM, OVX, GVZ) | 1 | close at 16:15. Set 0 if a 15-minute window is acceptable; the same-day VIX/SPY control needs that |
| DFF, Moody's spreads | 1 | published the next morning / end-of-day prices |
| Crypto prices (Stooq, Coinbase) | 1 | daily bars close after the US close (00:00 Warsaw/UTC for Stooq; "5 PM PST" for Coinbase) |
| Bitcoin on-chain | 2 | UTC-day aggregates. Computable from the chain at 00:00 UTC, but the API lagged ~4-5 days at download |
| Fear & Greed | 0 | published ~00:00 UTC and dated that day, so it reflects the previous UTC day (proof below) |
| EPU daily | 2 | built from that day's newspapers, posted with a lag |
| Weekly | ICSA 5, CCSA 12, MORTGAGE30US 0 (dated the Thursday release day), GASREGW 2, NFCI 5, STLFSI4 7, WEI 7, EIA crude 6, EIA gas 7 | release day minus week-end day, plus holiday slack |
| Monthly (FRED; dates = month end) | UNRATE/PAYEMS 23, CPI 31, INDPRO 30, RSAFS 46, HOUST 60, PCE 61, UMCSENT 3 | conservative bounds that cover the shutdown delays (see below) |
| Monthly (Stooq; dates = last weekday) | 0 for surveys released inside the month (Philly/Empire/Richmond, Conference Board, NAHB, Ifo, ZEW); ISM 6/8, Chicago PMI 1, EZ PMI 5, ADP 10, China CPI/IP 20, EZ HICP 20, UK CPI 23, JP IP 33, existing home sales 30, LEI 30, new home sales 78 | release calendars plus slack |

**Shutdown delays covered by the monthly bounds.** Usual lags are given in each catalog note; use them to relax the bounds.
- Oct-2013 shutdown, September 2013 data:
  - Employment Situation released 2013-10-22 (22 days after month end);
  - CPI released 2013-10-30;
  - retail sales released 2013-10-29;
  - personal income/PCE released 2013-11-08;
  - the Fed's G.17 industrial production postponed to late October 2013 (its BLS inputs were delayed);
  - new home sales (Sept + Oct) released 2013-12-04.
- Dec-2018/Jan-2019 shutdown (Commerce Department), December 2018 data:
  - retail sales released 2019-02-14 (45 days after month end);
  - housing starts released 2019-02-26;
  - PCE released 2019-03-01;
  - new home sales released 2019-03-04 (November-2018 new home sales were also delayed into 2019; the bound of 78 has margin for both).
- The Oct-Nov 2025 shutdown falls after the stock window; FRED has no October-2025 UNRATE or CPI.

**Timing probe** (`timing_probe_data-markets.tsv`, `probe.py`, 2005-01-03..2025-06-27). Each daily series is resampled to US trading days and compared with SPY and with DGS10 changes:
- c0: same-day correlation;
- cm1: correlation with the previous US session;
- pc1: partial correlation with the next US session given today's. This is the look-ahead detector: it is about 0 when the value is fixed before 4 pm.

| series | SPY c0 | SPY cm1 | SPY pc1 | DGS10 c0 | DGS10 cm1 | DGS10 pc1 | reading |
|---|---|---|---|---|---|---|---|
| Nikkei 225 | +0.16 | +0.49 | +0.01 | | | | closes before the US session: reacts to the previous US day |
| Hang Seng | +0.23 | +0.34 | -0.01 | | | | same |
| DAX / FTSE 100 | +0.60 / +0.57 | +0.18 / +0.21 | +0.01 / -0.01 | | | | closes in the New York morning |
| TSX | +0.77 | +0.04 | -0.11 (-0.03 winsorized) | | | | simultaneous close; the negative pc1 comes from 2008/2020 reversals, not look-ahead (look-ahead would be positive) |
| Stooq AUDUSD / USDMXN | +0.55 / -0.54 | +0.02 / +0.06 | +0.01 / -0.07 | | | | bar closes after 4 pm: contains post-close moves, so kad 1 |
| FRED DEXMXUS / DTWEXBGS (noon) | -0.34 / -0.29 | -0.19 / -0.24 | -0.02 / -0.00 | | | | noon rates: no post-close information, so kad 0 |
| IEF ETF (16:00, calibration) | | | | -0.95 | +0.04 | -0.11 | a 30-minute overlap with the 15:30 CMT is clearly detected |
| Stooq DE / UK 10y | | | | +0.62 / +0.60 | +0.15 / +0.15 | +0.00 / +0.02 | close before the US afternoon, so kad 0 |
| Stooq JP 10y | | | | +0.14 | +0.33 | -0.01 | Asian session, so kad 0 |
| Stooq CA / MX 10y | | | | +0.81 / +0.23 | +0.04 / +0.03 | +0.05 / +0.07 | overlap with the US afternoon, so kad 1 |

Noise floor: US ETFs that close exactly at 4 pm show pc1 between -0.05 and +0.09 against SPY, so only larger or systematic values matter.

**Crypto timing cross-checks:**
- Stooq vs Coinbase same-date daily moves correlate 0.87-0.90 (BTC, ETH, LTC, BCH since 2017), with +0.09..+0.12 into Stooq's next day and a median price gap of 0.4-0.7%. Stooq's bar therefore closes 1-3 hours before Coinbase's "5 PM PST", i.e. around 18:00-20:00 New York.
- Coinbase BTC also correlates +0.10..+0.12 with crypto stocks' (COIN, MSTR, MARA, RIOT) next-day returns, so it really contains post-close information.
- Fear & Greed change on day d vs BTC: +0.54 with UTC day d-1 and +0.07 with day d. So the value dated d is computed from data up to ~00:00 UTC of d and is public long before the US close of d.

## Transforms
- `logdiff`: prices, index levels, FX, ETF closes, vol indices, claims, crude stocks, payrolls, CPI, IP, sales, starts, PCE, home sales, on-chain counts and values.
- `diff`: yields, spreads, rates, breakevens, mortgage rate, NFCI/STLFSI4/WEI, diffusion and sentiment indices, y/y inflation and growth rates, UNRATE, Fear & Greed.
- `level`: ADP monthly change and LEI m/m, which are already changes.
- `anomaly`: gas storage (strong injection/withdrawal cycle).

## Data-quality checks and repairs
Each series was checked by `dmlib.write_series` and the scan scripts; per-series output is in `qc_data-markets.txt`. Checks:
- duplicate and conflicting dates (none conflicting);
- non-finite values;
- sentinels (none; ADP -99 in Dec-2009 is a real value);
- non-positive values for logdiff;
- gaps by frequency;
- flat runs and the zero-change share;
- largest moves;
- spike-and-revert patterns (one-day bad prints);
- first and last rows.

The catalog was validated against the files: first, last and rows match; ids are unique and don't clash with the other agents' catalogs; values are plain decimals.

**Repairs (all noted in the series' catalog notes):**
- **Dropped series:**
  - Stooq USDCNY: 222 spike-and-revert moves over 1% after 2005, e.g. 7.29->7.11->7.27; FRED DEXCHUS used instead.
  - Stooq DE and UK 3-month bill yields: alternating prints, e.g. 0.31->0.44->0.35.
- **Late starts:**
  - USDTRY 2002 (stale 1990s, redenomination era);
  - USDRUB 2000 (1998-99 bad prints);
  - USDINR and USDMXN 1995 (stale);
  - Bovespa 1995 (hyperinflation, redenominations);
  - DE 2y and UK 2y 1999 (1990s spikes);
  - ES and IT 10y 1997.
- **Single bad prints removed:** BUBOR 3M 2004-10-19.
- **WTI:** the negative 2020-04-20 settlement (-36.98) was dropped, because logdiff can't take it; the move across the gap is 04-17 to 04-21.
- **Crypto:** see the Crypto section.
- **Monthly re-dating:** Stooq dates moved to the month's last weekday where earlier (3 Chicago PMI rows).

**Kept on purpose:**
- Genuine crisis moves: 1997-98 Asia, 2008, 2020, the gilt crisis of 2022-09-28, the Japan crash of 2024-08-05, the Athens closure of 2015 (38-day gap), the MOEX halt of 2022 (27-day gap), and Henry Hub cold-snap spikes (2003-02, 2021-02 Uri, 2024-01, 2025-01).

**Known gaps and staleness:**
- Lunar New Year and Golden Week gaps of 7-20 days in Asian indices (not forward-filled).
- Henry Hub spot is missing many days in 1997-2007 (EIA), e.g. 28 gaps in 2005.
- CPER (2012-14) and CANE (2012-13) have illiquid-day gaps; PALL is missing 2025-05-22/23.
- Chinese 10y is stale (11-40% unchanged days in 2007-13).
- WIBOR and BUBOR are flat for years at a time (policy holds), and DEXCHUS shows the hard peg until 2005-07-21. These give many zero moves.
- Several Stooq monthly series miss a few months: Richmond Fed 10, China IP every January (Jan-Feb are combined), ISM 1.
- FRED UNRATE and CPI miss October 2025 (shutdown).
- The on-chain tx-volume series has a strong weekday cycle: weekend values are 3-5x lower, giving large Monday/Saturday moves.

## Crypto (71 series, category `crypto`)

**Selection:**
- Stooq coins with history starting before 2021 and real trading.
- Ranked by median 2022-2025 dollar volume (Stooq volume is reported only from 2021-12-26): the top 45, plus 14 long-history coins from 2014-2017 (Dash, XMR, ZEC, ZEN, NEO, BAT, WAVES, IOTA, QTUM, XEM, OMG, DCR, ICX, SC, KCS).

**Excluded:**
- Stablecoins as price series, except USDT and USDC, kept for their real depegs: USDT closed at 1.12 in Jan-2018 and 0.904 in Apr-2017; USDC at 0.956 on 2023-03-11 (SVB).
  - DAI has a gap from 2018-01 to 2020-04; TUSD and USDD were also skipped.
- Wrapped/derivative tokens: WBTC, stETH, PAXG, XAUT (BTC/ETH/gold duplicates).
- Data-error coins: STX (755 missing days, ticker reuse), SCRT (2020 relabel, x25 jump), HNT (x95 jump), HOT and ONE (bad prints and ticker reuse), CEL and BTG (2025 gaps), XDC and OCEAN (long gaps).
- ~130 coins listed only from 2022-05-14 (≤3.1 years).

**Survivorship caution:** selecting by 2022-25 liquidity favours coins that survived.

**Timing:** Stooq bars run 7 days a week; the daily bar closes after the US close (see Timing), so `known_after_days` is 1. The last bar is 2025-06-28.

**Gaps:** Stooq has a 1-3 day outage around 2022-07-01..03 in many coins; otherwise 0-3 missing days per coin.

**Repairs:**
- BTC: removed 2014-02-07..2014-02-25, the Mt.Gox freeze/collapse prints (111.92 on 2014-02-20 vs ~600 elsewhere). Earlier BTC history appears to be Mt.Gox-based, at a premium of up to ~10-20% in late 2013.
- Later starts: XRP and XMR 2016-01-01 and DASH 2016-01-01 (thin 2014-15 prints); DCR 2016-03-01; ZEC 2016-11-01 (launch prices 650-4000); LTC 2014-01-01 (Oct-Dec 2013 thin); SC 2016-06-01 (1e-6 tick).
- One-day bad prints removed: LTC 2014-04-22, MKR 2018-04-07, SC 2019-08-02 and 2019-09-22.

Coverage of every crypto series:

| id | coin | first | last | rows | missing days |
|---|---|---|---|---|---|
| stooq_btc | Bitcoin | 2010-07-17 | 2025-06-28 | 5442 | 19 (Mt.Gox window removed) |
| stooq_ltc | Litecoin | 2014-01-01 | 2025-06-28 | 4194 | 3 |
| stooq_eth | Ethereum | 2015-08-07 | 2025-06-28 | 3614 | 0 |
| stooq_xrp | XRP | 2016-01-01 | 2025-06-28 | 3464 | 3 |
| stooq_xmr | Monero | 2016-01-01 | 2025-06-28 | 3467 | 0 |
| stooq_dash | Dash | 2016-01-01 | 2025-06-28 | 3467 | 0 |
| stooq_dcr | Decred | 2016-03-01 | 2025-06-28 | 3407 | 0 |
| stooq_sc | Siacoin | 2016-06-01 | 2025-06-28 | 3312 | 3 |
| stooq_doge | Dogecoin | 2016-07-01 | 2025-06-28 | 3285 | 0 |
| stooq_etc | Ethereum Classic | 2016-07-27 | 2025-06-28 | 3259 | 0 |
| stooq_zec | Zcash | 2016-11-01 | 2025-06-28 | 3162 | 0 |
| stooq_xlm | Stellar | 2017-01-17 | 2025-06-28 | 3085 | 0 |
| stooq_usdt | Tether (stablecoin) | 2017-03-05 | 2025-06-28 | 3038 | 0 |
| stooq_bat | Basic Attention Token | 2017-06-01 | 2025-06-28 | 2948 | 2 |
| stooq_xem | NEM | 2017-06-02 | 2025-06-28 | 2949 | 0 |
| stooq_zen | Horizen | 2017-06-06 | 2025-06-28 | 2945 | 0 |
| stooq_iota | IOTA | 2017-06-13 | 2025-06-28 | 2938 | 0 |
| stooq_eos | EOS (Vaulta) | 2017-06-29 | 2025-06-28 | 2922 | 0 |
| stooq_omg | OMG Network | 2017-07-14 | 2025-06-28 | 2907 | 0 |
| stooq_bch | Bitcoin Cash | 2017-08-01 | 2025-06-28 | 2889 | 0 |
| stooq_waves | Waves | 2017-08-04 | 2025-06-28 | 2885 | 1 |
| stooq_neo | NEO | 2017-08-05 | 2025-06-28 | 2885 | 0 |
| stooq_bnb | BNB | 2017-09-08 | 2025-06-28 | 2851 | 0 |
| stooq_icx | ICON | 2017-09-23 | 2025-06-28 | 2836 | 0 |
| stooq_link | Chainlink | 2017-09-28 | 2025-06-28 | 2829 | 2 |
| stooq_trx | TRON | 2017-10-10 | 2025-06-28 | 2819 | 0 |
| stooq_enj | Enjin Coin | 2017-11-01 | 2025-06-28 | 2797 | 0 |
| stooq_mkr | Maker | 2017-11-23 | 2025-06-28 | 2772 | 3 |
| stooq_qtum | Qtum | 2017-11-25 | 2025-06-28 | 2773 | 0 |
| stooq_kcs | KuCoin Token | 2017-12-13 | 2025-06-28 | 2755 | 0 |
| stooq_ada | Cardano | 2017-12-30 | 2025-06-28 | 2738 | 0 |
| stooq_fil | Filecoin | 2018-02-12 | 2025-06-28 | 2684 | 10 (2018-04) |
| stooq_xtz | Tezos | 2018-07-05 | 2025-06-28 | 2551 | 0 |
| stooq_vet | VeChain | 2018-08-04 | 2025-06-28 | 2521 | 0 |
| stooq_usdc | USD Coin (stablecoin) | 2018-12-14 | 2025-06-28 | 2389 | 0 |
| stooq_ftm | Fantom (Sonic S) | 2018-12-19 | 2025-06-28 | 2384 | 0 |
| stooq_cro | Cronos | 2019-03-16 | 2025-06-28 | 2297 | 0 |
| stooq_matic | Polygon (MATIC/POL) | 2019-04-30 | 2025-06-28 | 2252 | 0 |
| stooq_atom | Cosmos | 2019-05-01 | 2025-06-28 | 2251 | 0 |
| stooq_algo | Algorand | 2019-07-01 | 2025-06-28 | 2190 | 0 |
| stooq_chz | Chiliz | 2019-09-04 | 2025-06-28 | 2122 | 3 |
| stooq_hbar | Hedera | 2019-09-20 | 2025-06-28 | 2109 | 0 |
| stooq_rune | THORChain | 2019-12-18 | 2025-06-28 | 2020 | 0 |
| stooq_ksm | Kusama | 2020-02-13 | 2025-06-28 | 1963 | 0 |
| stooq_sol | Solana | 2020-04-10 | 2025-06-28 | 1906 | 0 |
| stooq_mana | Decentraland | 2020-04-15 | 2025-06-28 | 1901 | 0 |
| stooq_lrc | Loopring | 2020-04-18 | 2025-06-28 | 1896 | 2 |
| stooq_comp | Compound | 2020-06-19 | 2025-06-28 | 1836 | 0 |
| stooq_yfi | yearn.finance | 2020-07-22 | 2025-06-28 | 1800 | 3 |
| stooq_crv | Curve DAO | 2020-08-14 | 2025-06-28 | 1780 | 0 |
| stooq_sand | The Sandbox | 2020-08-20 | 2025-06-28 | 1774 | 0 |
| stooq_dot | Polkadot | 2020-08-21 | 2025-06-28 | 1770 | 3 |
| stooq_sushi | SushiSwap | 2020-09-01 | 2025-06-28 | 1762 | 0 |
| stooq_egld | MultiversX | 2020-09-03 | 2025-06-28 | 1760 | 0 |
| stooq_uni | Uniswap | 2020-09-17 | 2025-06-28 | 1746 | 0 |
| stooq_avax | Avalanche | 2020-09-22 | 2025-06-28 | 1740 | 1 |
| stooq_cake | PancakeSwap | 2020-09-30 | 2025-06-28 | 1733 | 0 |
| stooq_aave | Aave | 2020-10-10 | 2025-06-28 | 1723 | 0 |
| stooq_near | NEAR Protocol | 2020-10-15 | 2025-06-28 | 1718 | 0 |
| stooq_axs | Axie Infinity | 2020-11-04 | 2025-06-28 | 1698 | 0 |
| stooq_grt | The Graph | 2020-12-18 | 2025-06-28 | 1654 | 0 |
| fred_cbbtcusd | Coinbase BTC | 2014-12-01 | 2026-10-18 | 4305 | 35 (start-up gaps Dec-2014/Jan-2015) |
| fred_cbethusd | Coinbase ETH | 2016-05-18 | 2026-10-18 | 3803 | 3 |
| fred_cbltcusd | Coinbase LTC | 2016-08-17 | 2026-10-18 | 3713 | 2 |
| fred_cbbchusd | Coinbase BCH | 2017-12-20 | 2026-10-18 | 3219 | 6 |
| bcinfo_btc_hash_rate | BTC hash rate | 2011-01-01 | 2026-10-14 | 5763 | 3 |
| bcinfo_btc_n_transactions | BTC transactions/day | 2011-01-01 | 2026-10-14 | 5763 | 3 |
| bcinfo_btc_n_unique_addresses | BTC addresses/day | 2011-01-02 | 2026-10-14 | 5747 | 18 |
| bcinfo_btc_tx_volume_usd | BTC on-chain value, USD | 2011-01-01 | 2026-10-15 | 5745 | 22 (15 in Aug-2011) |
| bcinfo_btc_fees_usd | BTC fees, USD | 2011-01-01 | 2026-10-14 | 5766 | 0 |
| altme_crypto_fear_greed | Fear & Greed 0-100 | 2018-02-01 | 2026-10-19 | 3179 | 4 |

## Positive controls (same-day correlation with US stocks, 2005..2025-06-27, from the store, `controls.py`)
These should co-move with specific US stocks on the same day (offset 0), so a correct pipeline must find them.

| External series | US stocks | Same-day correlation | Notes |
|---|---|---|---|
| GLD / XAUUSD | gold miners NEM, AEM, KGC, RGLD | +0.61 to +0.69 | |
| XAGUSD | silver miners PAAS, HL, AG | +0.59 to +0.64 | |
| URA | CCJ | +0.82 | URA holds CCJ |
| URA | UEC | +0.67 | |
| USO | XOM, OXY, APA | +0.54 to +0.58 | |
| WTI spot | XOM, CVX, COP, OXY, SLB | +0.41 to +0.47 | |
| CPER | FCX, SCCO | +0.55 | |
| UNG | EQT, RRC, AR | +0.35 to +0.37 | Henry Hub spot only +0.12 |
| BTC (Stooq) | MSTR, RIOT, MARA, COIN | +0.73, +0.66, +0.64, +0.59 | 2021 on; kad 1 |
| Bovespa | ITUB, PBR, VALE | +0.79, +0.75, +0.73 | kad 1 |
| DAX | SAP, DB | +0.55 to +0.61 | |
| FTSE 100 | BP, HSBC | +0.54 to +0.59 | |
| Sensex | IBN, HDB | +0.51 to +0.56 | |
| DGS10 | JPM, BAC, SCHW | +0.28 to +0.34 | |
| VIX | broad stocks | about -0.5 | needs kad 0 to be testable at offset 0 |
| DEXMXUS | AMX | -0.41 | |

**Asian indices (reverse direction).** Same-day correlation with ADRs is +0.27 to +0.43 (TAIEX-TSM +0.31, Nikkei-TM +0.30, Hang Seng-BABA +0.43). The ADR's previous day predicts the index even better: +0.28 to +0.36 (TSM d-1 -> TAIEX d). This is the cleanest "stocks lead external" control.

**Null controls.** Asian index day d vs the ADR's next day is about 0 (TAIEX->TSM -0.06), as expected.

## What the test should be careful about
1. **Timing.** `known_after_days` is per series and conservative. Stooq FX/crypto (1) and FRED noon FX (0) carry the same currencies at different times of day. The weekly and monthly lags use upper bounds; the usual lags are in the notes. US-listed ETFs and US-hours series (VIX, Treasury, energy, FX noon) overlap the stock session itself, so only offset >= 1 after `known_after_days` is a real lead.
2. **Revisions.** FRED values are the 2026 vintage. Payrolls, retail sales, PCE, IP and housing are revised. NFCI, STLFSI4 and WEI re-estimate their whole history every week (model look-ahead). Stooq macro mixes revised history with first prints for the latest months; ADP rebuilt its history in 2022. ALFRED would fix this.
3. **Mechanical overlap.** SPY/QQQ/IWM/DIA, the sector SPDRs and VNQ are built from US stocks in the census universe, so they co-move mechanically. EEM/EFA/FXI/EWJ/EWZ trade during US hours, not their home markets' hours.
4. **ETF quirks.** USO and UNG lose value to contango and have had reverse splits (prices adjusted). CPER and CANE were illiquid early. Closes are dividend-adjusted.
5. **Seasonality and calendars.** Gas storage is seasonal (anomaly). Crude stocks are mildly seasonal. On-chain tx volume has a weekday cycle. Crypto, DFF, EPU, Fear & Greed and on-chain series include weekends. Foreign holidays are left missing.
6. **Pegs, flat periods and noise.** CNY (to 2005-07), WIBOR and BUBOR holds, DFF steps and the JGB 2y near zero give many zero moves. EPU daily is extremely noisy.
7. **Short histories.** Stooq macro outside the long US series starts 2010. Several Stooq bond series start 2005-11, crypto 2014-2020, F&G 2018, SHIBOR 2020, VXEEM 2011. Gas storage before 2010 comes from Stooq.
8. **Crypto survivorship.** Coins were chosen by 2022-25 liquidity, so the set favours survivors (FTT, LUNA and CEL are absent).
9. **Data beyond the stock window.** FRED, EIA and on-chain series run to Oct 2026. Stooq series stop at the 2025-06-27/28 snapshot.

## Reproduction
Scripts are in `scripts/data-markets/`. They were run as `python3 -I <script>` in this order:
1. `etf.py`
2. `stooq_world.py all`
3. `fred.py`
4. `stooq_macro.py`
5. `eia.py`
6. `crypto_extra.py`
7. `probe.py` and `probe2.py` (timing evidence)
8. `controls.py`
9. `validate.py`

`dmlib.py` holds the shared writer and QC. Nothing downloaded was executed.
