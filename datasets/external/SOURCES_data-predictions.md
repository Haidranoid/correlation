# SOURCES: data-predictions (prediction markets)

Agent `data-predictions`, built 2026-10-18. Catalog: `catalog_data-predictions.csv` (category `prediction_market` for every row). Scripts: `scripts/data-predictions/` (run with `python3 -I`, standard library plus numpy only). Raw downloads: `raw/kalshi/`, `raw/polymarket/`, `raw/predictit/`, `raw/manifold/`.

The Iowa Electronic Markets are covered by `data-politics` (the `iem_*` series) and are not duplicated here.

## Bottom line on coverage

Prediction-market history that is free and keyless starts in mid-2021 (Kalshi) and is only rich from 2023 onward (Polymarket's order book, Kalshi's fuller ladders). Of the census window (stocks to 2025-06-27), only about 2021-07 to 2025-06 is testable, and for most Polymarket series only about 2024 to 2025-06. Before Kalshi (mid-2021), the only data is four 90-day PredictIt windows around the 2016, 2018, 2020 and 2022 elections. The series also run past 2025-06-27 to 2026-10-18, but that part cannot be tested against the current stock panel.

## Timing convention (applies to every series unless the notes say otherwise)

- One row per **calendar day**, weekends included: these markets trade around the clock, as do the existing crypto series. Align to trading days as-of in the test; don't diff raw rows across weekends as if they were trading days.
- **Value = state at 16:00 America/New_York** on `date`, computed from timestamped data, with `known_after_days` 0:
  - Polymarket: the last 10-minute `prices-history` sample at or before 16:00 ET (in practice the 15:50:02 ET sample). It must be no older than 6 hours, otherwise the day is missing.
  - Kalshi: hourly candles are sparse (a candle exists only when the book or trades change). The state at 16:00 ET is the close of the last candle ending at or before 16:00 ET. Price = yes bid/ask midpoint if the spread is at most $0.10 (no bid counts as 0, no ask as 1); otherwise the last trade price, but only if that trade happened in the previous 7 days; otherwise unknown (the strike is left out that day).
  - Manifold: probability after the last bet at or before 16:00 ET.
  - PredictIt is the exception: its free data has only daily bars (New York calendar day, close at midnight ET), so `known_after_days` is 1.
- A day is included only while the market was still trading at 16:00 ET (resolved markets stop).
- Transform: `diff` for probabilities and expected prints; `logdiff` only for `ks_spx_yearend_exp` (index points).

## Rolling (continuous) series: rules

Each contract lives only until it resolves, so the continuous series stitch contracts together. Days that fall in a roll gap (the next contract not yet listed or not priced) are missing, never filled.

- **Nearest unresolved event** (all Kalshi ladders and binaries): on date D, use the nearest event whose markets are still trading at 16:00 ET on D. Kalshi closes macro markets at 08:25-08:29 ET on release day and Fed markets at about 13:55 ET on FOMC day, minutes before the release. So on a CPI, jobs, GDP or FOMC day the Kalshi series already uses the *following* event. A meeting or month for which Kalshi listed nothing (e.g. the Nov 2021 FOMC) is a gap.
- **Polymarket Fed (`pm_fed_next_*`)**: from the day after the previous FOMC meeting through the meeting day, as requested ("roll the day after each meeting"). Polymarket keeps these markets open after the 14:00 announcement until resolution (about 17:00-19:00 ET), so the meeting-day 16:00 value already shows the decision (near 0/1).
- **Expected values from strike ladders** (Kalshi `*_exp`, `ks_fed_*`):
  - Kalshi lists "above K" contracts. Their prices are made monotone (isotonic, pool-adjacent-violators).
  - For grid-valued prints (CPI and U-3 to 0.1, the Fed to 0.25), each bucket between strikes is valued at its mean grid point, the lower tail at the lowest strike, and the upper tail at the highest strike plus one grid step. For continuous quantities (payrolls, gasoline) buckets are valued at their midpoints.
  - Days with fewer than 3 priced strikes, or more than 30% probability in either truncated tail, are left out (`ks_fed_next_*` excepted, see below).
- **Fed (Kalshi)**: U_cur is the upper bound of the target range before the meeting. It comes from the FOMC record, and every settled Kalshi event matched it (0 mismatches). From the ladder:
  - P(hike) = price of the strike at U_cur;
  - P(cut) = 1 - price of the strike at U_cur - 0.25 (missing while that strike wasn't listed, 2021 to early 2022);
  - expected change = 100 * (E[upper] - U_cur) bp, requiring the lowest priced strike to be at or below U_cur;
  - `ks_fed_6m_exp_upper` uses the meeting closest to D + 182 days (within 45 days) that has a usable ladder.
- **Calendar-year and cycle series**: recession this year (Kalshi RECSSNBER-yy; Polymarket 2024/2025/2026 markets); House and Senate control (Polymarket 2024 cycle through election day 2024-11-05, then the 2026 cycle from its listing in July 2025).
- Roll days coincide with FOMC days, macro release days and Jan 1, all of which carry known stock-return calendar effects. A census hit concentrated on roll days is a calendar effect, not a lead.

## Sources

### Kalshi (CFTC-regulated exchange), 2021-07 onward
- Endpoints (base `https://api.elections.kalshi.com/trade-api/v2`), all public with no key:
  - `/series`, `/series/{ticker}`;
  - `/historical/cutoff` (at download: markets settled before 2026-08-19 are served only by `/historical/...`);
  - `/historical/markets?series_ticker=` and `/markets?series_ticker=` (metadata; the KX-prefixed series also return the legacy un-prefixed events, e.g. KXFED returns FED-21JUL);
  - `/historical/markets/{ticker}/candlesticks` and `/series/{series}/markets/{ticker}/candlesticks` (hourly, at most 5000 candles per request).
- The two candle endpoints use different field names (`close` vs `close_dollars`); both are parsed. A few legacy tickers contain spaces (`GDP-232022 Q4-...`) and are URL-encoded.
- Terms: Kalshi's public market data, used for personal non-commercial research. The API is free without registration for read-only market data; trading needs an account.
- Downloads: 5,133 candle responses (one request each), sequential, at about 1 request per second. Candle files are stored gzip-compressed (lossless) under `raw/kalshi/candles/<event>/`; metadata pages are under `raw/kalshi/markets/`.

### Polymarket, CLOB era (about 2023-01 onward; liquid from 2024)
- Gamma API `https://gamma-api.polymarket.com`: `/events?tag_slug=<tag>&closed=..&order=volume` (32 topic tags crawled, 4,782 events and 40,663 markets indexed), `/events/{id}`, `/markets/{id}`, `/public-search`.
- CLOB `https://clob.polymarket.com/prices-history?market=<token>&startTs&endTs&fidelity=10`:
  - Windows longer than about 15 days are refused, so requests use aligned 14-day windows.
  - `interval=max` works only at fidelity of 720 minutes or more, and its samples fall at 00:00/12:00 UTC (after the US close), so it wasn't used.
  - 1,451 requests, 0 failures.
- Public endpoints, no key, used read-only for research. Raw pages are stored gzip-compressed under `raw/polymarket/{events,markets,prices}/`.
- What `p` is: Polymarket does not document it. The samples match the price Polymarket charts and displays (book midpoint, or the last trade when the spread is wide); this was not independently verified.
- **Not available: the 2020 to late-2022 AMM era.** Pre-order-book markets (the 2020 presidential election, the 2022 midterms, 2021 BTC markets) return an empty history from the CLOB. The public Goldsky subgraphs that held their on-chain trades now answer "paused and deprecated ... data is stale and incorrect", and the replacement (Goldsky Turbo Pipelines) needs an account, so no AMM-era Polymarket series were made.
- The tag crawl, ordered by volume, misses some low-volume or untagged events; specific events were fetched by id where needed.

### PredictIt, 2016-2022 snapshots only
- No free bulk archive exists:
  - The official API (`/api/marketdata/all/`, `/api/marketdata/markets/{id}`) gives only current prices. Its licence is non-commercial use, with PredictIt credited as the source.
  - The per-market chart CSV (`/Resource/DownloadMarketChartData?marketid={id}&timespan=90d`, the site's "download data" link) returns the **final 90 days** of daily OHLC per contract. `180d` and `365d` return nothing, and the JSON chart API (`/api/Public/GetMarketChartData/{id}`) returns only 7 days; those 7-day JSON files under `raw/predictit/charts/` are superseded by the CSVs in `raw/predictit/csv/`.
  - A GitHub search found only scrapers and API wrappers (e.g. AlanJSayler/predictithistorical, danielkovtun/rpredictit), no data mirror.
  - Any researcher data-sharing arrangement with PredictIt would need an application; not pursued (and not verified to exist).
- Market ids were found by search and a sparse, polite id scan (379 metadata requests at 1 per second; raw under `raw/predictit/markets/`).
- 8 markets were kept: the 2016 party and trifecta markets, the 2018 House and Senate, the 2020 party, House and Senate, and the 2022 House. The 2022 Senate market was dropped because 84 of its 90 days come after the December runoff, flat at 0.01.

### Manifold Markets (play money), 2022 onward
- `https://api.manifold.markets/v0/search-markets`, `/v0/market/{id}`, `/v0/bets?contractId=&limit=1000&before=` (full bet history with timestamps). No key; the documented limit is 500 requests per minute, and this agent used about 1 per second.
- Flagged as **play-money forecasts, not real-money markets** (subcategory `forecast_play_money`). They were chosen to cover 2022-2023, when real-money coverage was thin: recession by 2023 and 2024, the 2023 debt ceiling, the March 2023 banking crisis, and Fed cuts at the March and September 2024 meetings.

### Metaculus: not used
- `https://www.metaculus.com/api/posts/` now answers 403 "The API is only available to authenticated users". **Available with a free account/API token**; this agent did not register.

## Skipped, and why
- Kalshi, enumerated but not built:
  - KXFEDDECISION (direct hike/hold/cut outcomes, 2023-05 onward): redundant with the KXFED ladder, which runs from 2021-07.
  - KXJOBLESS: 2021-22 only, mostly single-strike.
  - KXPCECORE: gappy before Nov 2024.
  - KXCPICOREYOY: near-duplicate of core CPI MoM.
  - KXNASDAQ100Y: near-duplicate of the S&P year-end series.
  - KXTERMINALRATE, KXRATEHIKE, KXACPI, KXU3MAX, KXGDPUSMIN: few, low-volume contracts.
  - KXFEDFUNDSYEAR, KXUSCPIYEAR: listed from 2026-07 only.
  - KXDEFAULT, KXFEDHIKE: candles fetched, not built (thin).
  - Kalshi's election markets: listed only from Oct 2024; Polymarket covers 2023-2024 better.
- Polymarket:
  - Monthly CPI, jobs, unemployment and GDP markets (2024-2026): thin, and duplicative of Kalshi's longer ladders.
  - 2028 candidate markets: no party-level 2028 market was found, and the candidate markets carry inflated volume on long-shot names.
  - Near-certain non-US elections (Germany 2025, Mexico 2024, South Korea 2025): little variation.
  - Stock-specific and "largest company" markets, gold and oil price-target markets: mechanically the asset price.
  - Short-dated (under 30 days) daily or weekly crypto and index "up or down" markets.

## What was built (111 series, all `prediction_market`, all daily)

| group | series | span | notes |
|---|---|---|---|
| Kalshi rolling | 14 | 2021-07-02 .. 2026-10-18 | Fed next meeting: P(hike), P(cut), expected change bp; Fed ~6 months ahead (expected upper bound); next CPI MoM, CPI YoY, core CPI MoM, payrolls, U-3 and GDP prints (expected values); AAA gas at month end; S&P 500 at year end; recession this year; shutdown at the next deadline |
| Polymarket rolling | 6 | 2023-09-20 .. 2026-10-18 | Fed next meeting: P(cut), P(hike), expected change bp (from 2024-02); recession this year (from 2024-08); House and Senate control, Republican, by cycle |
| Kalshi contracts | 17 | 2022-08-30 .. 2026-10-18 | finite-life: recession 2023/2024/2025/2026; first Fed cut by Dec 2023, Mar/May/Jun/Jul/Sep 2024; shutdown in 2024, 2025, and on Oct 2 2023, Nov 17 2023, Jan 31 2026; debt ceiling by Jun 9 and Dec 31 2023 |
| Polymarket contracts | 59 | 2023-01-09 .. 2026-02-28 | finite-life: Fed (10), macro/markets (5), tariffs and trade (9), US election 2024 (5), non-US elections (4), geopolitics (14), crypto (6), shutdown and debt ceiling (5), TikTok (1) |
| PredictIt contracts | 8 | 2016-08-12 .. 2022-11-22 | finite-life, final 90 days only: 2016 president (party) and GOP trifecta; 2018 House and Senate; 2020 president, House and Senate; 2022 House |
| Manifold contracts | 7 | 2022-06-30 .. 2025-12-31 | finite-life, play money: recession by 2023/2024 (3), debt ceiling Jun 2023, banking crisis 2023, Fed cut at the Mar and Sep 2024 meetings |

Single contracts (91) say "finite-life" in `notes` and have subcategory `contract` or `forecast_play_money`. The 20 rolling series don't say "finite-life".

## Data-quality checks

- **Every series, on write:** sorted, strictly ascending unique dates, finite values only, probabilities checked to lie in [0, 1] (and `bp` within +/-150), no sentinels. A sanity print of range, gaps, and the first and last 3 rows is in `scripts/data-predictions/logs/*_build.out`. `build_catalog.py` re-reads every file and re-checks header, order, finiteness and range: 0 problems.
- **No future rows:** a date is written only once its 16:00 New York has passed.
- **FOMC path:** the hard-coded upper bound after each meeting (2021-2025) agreed with every settled Kalshi FED event (0 mismatches); 2026 meetings were taken from Kalshi settlements.
- **Known events reproduce:**
  - Kalshi expected change for the June 2022 meeting went from +52 to +71 bp between the May CPI release and the 75 bp hike.
  - Kalshi P(hike) fell from 0.995 to 0.69 between 2023-03-09 and 2023-03-13 (SVB).
  - Before the 50 bp cut in September 2024: Kalshi -40.6 bp, Polymarket -38.3 bp.
  - Polymarket 2024 election contracts move on the 2024-06-27 debate (Biden-nominee odds 0.895 to 0.755), the 2024-07-13 assassination attempt (Trump 0.595 to 0.685) and the 2024-07-21 withdrawal (Biden-nominee odds 0.0015).
- **Two contracts that look wrong but are right:** `pmc_shutdown_by_2024mar09` and `pmc_funding_lapse_2025mar15` resolved Yes on technical funding lapses (bills signed after midnight on Saturday 2024-03-09 and 2025-03-15).
- **Cross-venue agreement** (roll days, i.e. FOMC days, the day after and Jan 1-2, excluded because the venues roll on different days):
  - Fed cut at next meeting, Kalshi ladder vs Polymarket: 928 common days 2024-02-06..2026-10-18; level corr 0.991, mean abs diff 0.018, daily-change corr 0.648 (n=904)
  - Fed hike at next meeting, Kalshi vs Polymarket: 927 common days 2024-02-06..2026-10-18; level corr 0.996, mean abs diff 0.009, daily-change corr 0.909 (n=902)
  - expected change at next meeting (bp), Kalshi vs Polymarket: 927 common days 2024-02-06..2026-10-18; level corr 0.992, mean abs diff 0.852, daily-change corr 0.685 (n=902)
  - US recession this year, Kalshi vs Polymarket: 762 common days 2024-08-05..2026-10-18; level corr 0.987, mean abs diff 0.019, daily-change corr 0.617 (n=742)
  - recession in 2023, Kalshi vs Manifold (play money): 389 common days 2022-11-18..2023-12-31; level corr 0.957, mean abs diff 0.081, daily-change corr 0.014 (n=378)
  - recession by end of 2024, Kalshi vs Manifold: 532 common days 2023-07-19..2025-01-28; level corr 0.952, mean abs diff 0.068, daily-change corr -0.019 (n=517)
  - 2024: Republican party vs Trump (Polymarket): 294 common days 2024-01-05..2024-11-05; level corr 0.917, mean abs diff 0.018, daily-change corr 0.602 (n=287)
  - Fed cut by Jun 12 2024, Kalshi vs Polymarket: 127 common days 2024-02-02..2024-06-11; level corr 0.998, mean abs diff 0.013, daily-change corr 0.896 (n=124)
  - shutdown Oct 1 2025, Kalshi rolling vs Polymarket: only 13 common days
- **Forecast vs outcome** (the value on the last day before each release vs the print the settled ladder pins down):
  - CPI MoM (%): 50 releases; corr(expected, realized) 0.934; mean error (realized-expected) +0.004; mean abs error 0.065
  - CPI YoY (%): 46 releases; corr(expected, realized) 0.997; mean error (realized-expected) -0.023; mean abs error 0.070
  - core CPI MoM (%): 51 releases; corr(expected, realized) 0.646; mean error (realized-expected) -0.006; mean abs error 0.070
  - unemployment rate (%): 44 releases; corr(expected, realized) 0.912; mean error (realized-expected) -0.026; mean abs error 0.111
  - GDP advance (%), realized = centre of its <=0.5-wide strike bucket: 13 releases; corr(expected, realized) 0.579; mean error (realized-expected) -0.199; mean abs error 0.531
  - payrolls (k), realized = centre of its <=50k-wide strike bucket: 24 releases; corr(expected, realized) 0.594; mean error (realized-expected) +1.814; mean abs error 55.542
- **Reading the checks:**
  - Kalshi and Polymarket agree closely in level and co-move day to day.
  - Manifold's play-money levels agree, but its daily changes are essentially uncorrelated with Kalshi's: treat Manifold day-to-day moves as noise.
  - The expected-print series track the realized prints: CPI YoY very tightly; payrolls and GDP far less (their surprises are large).

### Per-series QC table

Gaps are missing calendar days inside a series' span (roll gaps and unpriced days, never filled). The zero-move share is the share of day-to-day changes that are exactly 0, i.e. flat stretches, pinned 0.005/0.995 quotes or Manifold days without bets.

| id | first | last | rows | gaps (missing days) | rows in census panel 2005-01-03..2025-06-27 | of which 2020-01-01 on | min | max | share of zero day-to-day moves |
|---|---|---|---|---|---|---|---|---|---|
| ks_core_cpi_mom_next_exp | 2022-07-01 | 2026-10-18 | 1483 | 19 (88) | 1050 | 1050 | 0.1415 | 0.6405 | 0.13 |
| ks_cpi_mom_next_exp | 2021-11-28 | 2026-10-18 | 1669 | 20 (117) | 1202 | 1202 | -0.1685 | 1.062 | 0.07 |
| ks_cpi_yoy_next_exp | 2022-12-08 | 2026-10-18 | 1305 | 23 (106) | 847 | 847 | 2.291 | 7.205 | 0.08 |
| ks_fed_6m_exp_upper | 2022-02-24 | 2026-10-13 | 1519 | 24 (174) | 1080 | 1080 | 1.38 | 5.723 | 0.22 |
| ks_fed_next_expchg_bp | 2021-07-02 | 2026-10-18 | 1863 | 4 (72) | 1385 | 1385 | -49.5 | 89.25 | 0.24 |
| ks_fed_next_pcut | 2022-03-16 | 2026-10-18 | 1678 | 0 (0) | 1200 | 1200 | 0.005 | 0.995 | 0.57 |
| ks_fed_next_phike | 2021-07-02 | 2026-10-18 | 1863 | 4 (72) | 1385 | 1385 | 0.005 | 0.995 | 0.57 |
| ks_gas_month_exp | 2024-05-13 | 2026-10-18 | 802 | 14 (87) | 330 | 330 | 2.786 | 4.774 | 0.03 |
| ks_gdp_next_exp | 2021-12-07 | 2026-10-18 | 1511 | 22 (266) | 1075 | 1075 | -0.3465 | 6.266 | 0.06 |
| ks_payrolls_next_exp | 2023-03-11 | 2026-10-18 | 1262 | 8 (56) | 805 | 805 | -10.88 | 398 | 0.07 |
| ks_recession_cur_year | 2022-07-01 | 2026-10-18 | 1414 | 1 (157) | 936 | 936 | 0.005 | 0.85 | 0.42 |
| ks_shutdown_next_deadline | 2021-09-28 | 2026-10-18 | 843 | 9 (1004) | 608 | 608 | 0.005 | 0.965 | 0.33 |
| ks_spx_yearend_exp | 2022-07-07 | 2026-10-18 | 1533 | 11 (32) | 1068 | 1068 | 3537 | 7719 | 0.03 |
| ks_u3_next_exp | 2022-02-10 | 2026-10-18 | 1527 | 25 (185) | 1070 | 1070 | 3.457 | 4.582 | 0.16 |
| ksc_debt_ceiling_by_2023dec31 | 2022-09-27 | 2023-06-04 | 251 | 0 (0) | 251 | 251 | 0.725 | 0.995 | 0.52 |
| ksc_debt_ceiling_by_2023jun09 | 2023-01-13 | 2023-06-04 | 143 | 0 (0) | 143 | 143 | 0.055 | 0.995 | 0.25 |
| ksc_fed_cut_by_2023dec31 | 2022-08-30 | 2023-12-31 | 489 | 0 (0) | 489 | 489 | 0.005 | 0.6 | 0.42 |
| ksc_fed_cut_by_2024jul31 | 2023-08-29 | 2024-07-31 | 318 | 2 (20) | 318 | 318 | 0.005 | 0.94 | 0.42 |
| ksc_fed_cut_by_2024jun12 | 2023-08-29 | 2024-06-12 | 283 | 2 (6) | 283 | 283 | 0.005 | 0.915 | 0.43 |
| ksc_fed_cut_by_2024mar20 | 2023-08-29 | 2024-03-20 | 204 | 1 (1) | 204 | 204 | 0.005 | 0.635 | 0.52 |
| ksc_fed_cut_by_2024may01 | 2023-08-29 | 2024-05-01 | 246 | 1 (1) | 246 | 246 | 0.005 | 0.875 | 0.42 |
| ksc_fed_cut_by_2024sep18 | 2023-08-29 | 2024-09-18 | 374 | 1 (13) | 374 | 374 | 0.3 | 0.995 | 0.37 |
| ksc_recession_2023 | 2022-11-18 | 2024-01-24 | 433 | 0 (0) | 433 | 433 | 0.005 | 0.495 | 0.59 |
| ksc_recession_2025 | 2024-08-05 | 2026-01-30 | 544 | 0 (0) | 327 | 327 | 0.005 | 0.695 | 0.35 |
| ksc_recession_2026 | 2025-07-14 | 2026-10-18 | 462 | 0 (0) | 0 | 0 | 0.035 | 0.41 | 0.39 |
| ksc_recession_by_end_2024 | 2023-07-19 | 2025-01-30 | 562 | 0 (0) | 562 | 562 | 0.005 | 0.75 | 0.51 |
| ksc_shutdown_in_2024 | 2024-03-15 | 2024-12-30 | 291 | 0 (0) | 291 | 291 | 0.005 | 0.415 | 0.50 |
| ksc_shutdown_in_2025 | 2024-11-05 | 2025-09-30 | 330 | 0 (0) | 235 | 235 | 0.01 | 0.9 | 0.31 |
| ksc_shutdown_on_2023nov17 | 2023-10-01 | 2023-11-16 | 47 | 0 (0) | 47 | 47 | 0.005 | 0.095 | 0.63 |
| ksc_shutdown_on_2023oct02 | 2023-01-13 | 2023-10-01 | 262 | 0 (0) | 262 | 262 | 0.005 | 0.875 | 0.40 |
| ksc_shutdown_on_2026jan31 | 2025-11-11 | 2026-01-30 | 81 | 0 (0) | 0 | 0 | 0.095 | 0.795 | 0.14 |
| mf_banking_crisis_contained_2023 | 2023-03-15 | 2023-12-31 | 292 | 0 (0) | 292 | 292 | 0.1246 | 0.97 | 0.73 |
| mf_debt_ceiling_by_2023jun30 | 2023-04-24 | 2023-06-03 | 41 | 0 (0) | 41 | 41 | 0.4881 | 0.9958 | 0.10 |
| mf_fed_cut_2024mar_meeting | 2023-12-14 | 2024-03-21 | 99 | 0 (0) | 99 | 99 | 0.0007066 | 0.6955 | 0.42 |
| mf_fed_cut_2024sep_meeting | 2023-12-14 | 2024-09-17 | 279 | 0 (0) | 279 | 279 | 0.2672 | 0.9854 | 0.68 |
| mf_us_recession_before_end_2024 | 2023-02-09 | 2025-12-31 | 1057 | 0 (0) | 870 | 870 | 0.01 | 0.7488 | 0.37 |
| mf_us_recession_by_2023jan01 | 2022-06-30 | 2024-12-31 | 916 | 0 (0) | 916 | 916 | 0.01738 | 0.8985 | 0.83 |
| mf_us_recession_by_end_2023 | 2022-11-02 | 2024-01-01 | 426 | 0 (0) | 426 | 426 | 0.005128 | 0.5978 | 0.70 |
| pi_2016_gop_trifecta | 2016-08-12 | 2016-11-09 | 90 | 0 (0) | 90 | 0 | 0.1 | 0.99 | 0.18 |
| pi_2016_pres_gop | 2016-08-13 | 2016-11-10 | 90 | 0 (0) | 90 | 0 | 0.18 | 0.99 | 0.20 |
| pi_2018_house_gop | 2018-08-10 | 2018-11-07 | 90 | 0 (0) | 90 | 0 | 0.02 | 0.4 | 0.29 |
| pi_2018_senate_gop | 2018-08-10 | 2018-11-07 | 90 | 0 (0) | 90 | 0 | 0.68 | 0.99 | 0.35 |
| pi_2020_house_gop | 2020-10-06 | 2021-01-03 | 90 | 0 (0) | 90 | 90 | 0.01 | 0.18 | 0.48 |
| pi_2020_pres_gop | 2020-09-20 | 2020-12-18 | 90 | 0 (0) | 90 | 90 | 0.05 | 0.55 | 0.25 |
| pi_2020_senate_gop | 2020-11-04 | 2021-02-01 | 90 | 0 (0) | 90 | 90 | 0.01 | 0.84 | 0.34 |
| pi_2022_house_gop | 2022-08-25 | 2022-11-22 | 90 | 0 (0) | 90 | 90 | 0.72 | 0.99 | 0.44 |
| pm_fed_next_expchg_bp | 2024-02-06 | 2026-10-18 | 972 | 2 (14) | 494 | 494 | -49.93 | 24.94 | 0.03 |
| pm_fed_next_pcut | 2024-02-06 | 2026-10-18 | 972 | 2 (14) | 494 | 494 | 0.0009935 | 0.999 | 0.04 |
| pm_fed_next_phike | 2024-02-06 | 2026-10-18 | 972 | 2 (14) | 494 | 494 | 0.0004822 | 0.9985 | 0.07 |
| pm_house_gop_cur_cycle | 2024-02-22 | 2026-10-18 | 716 | 1 (254) | 258 | 258 | 0.065 | 0.555 | 0.65 |
| pm_senate_gop_cur_cycle | 2023-09-20 | 2026-10-18 | 871 | 1 (254) | 413 | 413 | 0.345 | 0.845 | 0.68 |
| pm_us_recession_cur_year | 2024-08-05 | 2026-10-18 | 798 | 1 (7) | 320 | 320 | 0.0015 | 0.665 | 0.29 |
| pmc_argentina2023_milei | 2023-05-30 | 2023-11-19 | 174 | 0 (0) | 174 | 174 | 0.075 | 0.78 | 0.39 |
| pmc_biden_drops_out | 2023-09-21 | 2024-07-21 | 305 | 0 (0) | 305 | 305 | 0.125 | 0.997 | 0.53 |
| pmc_btc_100k_2024 | 2024-03-04 | 2024-12-04 | 276 | 0 (0) | 276 | 276 | 0.1 | 0.9255 | 0.18 |
| pmc_btc_150k_2025 | 2024-12-31 | 2025-12-31 | 366 | 0 (0) | 179 | 179 | 0.0005 | 0.58 | 0.15 |
| pmc_btc_reserve_first_100d | 2024-11-07 | 2025-04-29 | 174 | 0 (0) | 174 | 174 | 0.02 | 0.475 | 0.20 |
| pmc_btc_spot_etf_by_2024jan15 | 2023-12-01 | 2024-01-10 | 41 | 0 (0) | 41 | 41 | 0.645 | 0.975 | 0.00 |
| pmc_canada2025_carney_pm | 2025-01-15 | 2025-04-29 | 105 | 0 (0) | 105 | 105 | 0.11 | 0.993 | 0.10 |
| pmc_china_invade_taiwan_2024 | 2024-01-15 | 2024-12-31 | 352 | 0 (0) | 352 | 352 | 0.0015 | 0.195 | 0.47 |
| pmc_china_invade_taiwan_2025 | 2025-01-29 | 2025-12-31 | 337 | 0 (0) | 150 | 150 | 0.0015 | 0.165 | 0.45 |
| pmc_courts_block_tariffs_by_2025jun | 2025-04-12 | 2025-05-28 | 47 | 0 (0) | 47 | 47 | 0.0155 | 0.4 | 0.13 |
| pmc_debt_ceiling_by_inauguration | 2024-12-19 | 2025-01-19 | 32 | 0 (0) | 32 | 32 | 0.006 | 0.645 | 0.06 |
| pmc_dem2024_biden_nominee | 2024-01-13 | 2024-08-05 | 206 | 0 (0) | 206 | 206 | 0.0005 | 0.905 | 0.66 |
| pmc_eth_ath_2024 | 2024-02-13 | 2024-12-31 | 323 | 0 (0) | 323 | 323 | 0.001 | 0.805 | 0.13 |
| pmc_eth_etf_by_2024may31 | 2024-01-10 | 2024-05-23 | 135 | 0 (0) | 135 | 135 | 0.055 | 0.715 | 0.22 |
| pmc_fed_2024sep_cut50plus | 2024-07-29 | 2024-09-18 | 52 | 0 (0) | 52 | 52 | 0.085 | 0.999 | 0.06 |
| pmc_fed_2cuts_2025 | 2024-12-30 | 2025-12-10 | 346 | 0 (0) | 180 | 180 | 0.0005 | 0.66 | 0.29 |
| pmc_fed_cut_2025 | 2025-03-05 | 2025-09-17 | 197 | 0 (0) | 115 | 115 | 0.7 | 0.9995 | 0.27 |
| pmc_fed_cut_by_2024jun12 | 2024-02-01 | 2024-06-12 | 133 | 0 (0) | 133 | 133 | 0.003 | 0.98 | 0.08 |
| pmc_fed_cut_by_2024may01 | 2023-12-06 | 2024-05-01 | 148 | 0 (0) | 148 | 148 | 0.0005 | 0.88 | 0.19 |
| pmc_fed_emergency_cut_2024 | 2024-08-02 | 2024-12-31 | 152 | 0 (0) | 152 | 152 | 0.0015 | 0.285 | 0.36 |
| pmc_fed_hike_2024 | 2024-04-25 | 2024-12-18 | 238 | 0 (0) | 238 | 238 | 0.0005 | 0.175 | 0.43 |
| pmc_fed_hike_2025 | 2024-12-30 | 2025-12-10 | 346 | 0 (0) | 180 | 180 | 0.0005 | 0.255 | 0.42 |
| pmc_funding_lapse_2025mar15 | 2025-01-10 | 2025-03-18 | 68 | 0 (0) | 68 | 68 | 0.15 | 0.9995 | 0.22 |
| pmc_funding_lapse_2025oct01 | 2025-03-26 | 2025-09-30 | 189 | 0 (0) | 94 | 94 | 0.265 | 0.948 | 0.48 |
| pmc_gdp_negative_2025q1 | 2025-03-04 | 2025-04-29 | 57 | 0 (0) | 57 | 57 | 0.155 | 0.65 | 0.11 |
| pmc_gdp_negative_2025q2 | 2025-04-04 | 2025-07-29 | 117 | 0 (0) | 85 | 85 | 0.0065 | 0.735 | 0.16 |
| pmc_inflation_above3_2025 | 2025-02-07 | 2026-01-12 | 340 | 0 (0) | 141 | 141 | 0.0015 | 0.95 | 0.20 |
| pmc_iran_close_hormuz_2025 | 2025-01-23 | 2025-12-31 | 343 | 0 (0) | 156 | 156 | 0.0005 | 0.52 | 0.34 |
| pmc_iran_strike_isr_2024 | 2024-10-26 | 2024-12-31 | 67 | 0 (0) | 67 | 67 | 0.003 | 0.65 | 0.14 |
| pmc_isr_hamas_ceasefire_2024 | 2024-08-07 | 2024-12-31 | 147 | 0 (0) | 147 | 147 | 0.0025 | 0.67 | 0.06 |
| pmc_isr_hamas_ceasefire_by_2025jul | 2025-03-20 | 2025-06-30 | 103 | 0 (0) | 100 | 100 | 0.003 | 0.76 | 0.04 |
| pmc_isr_hezbollah_ceasefire_2024 | 2024-09-24 | 2024-12-04 | 72 | 0 (0) | 72 | 72 | 0.14 | 0.9995 | 0.07 |
| pmc_isr_strike_iran_by_2025jul | 2025-03-31 | 2025-06-12 | 74 | 0 (0) | 74 | 74 | 0.18 | 0.525 | 0.11 |
| pmc_isr_strike_iran_nuclear_2024 | 2024-10-02 | 2024-11-18 | 48 | 0 (0) | 48 | 48 | 0.115 | 0.998 | 0.04 |
| pmc_large_tariffs_first_6m | 2024-11-18 | 2025-07-29 | 254 | 0 (0) | 222 | 222 | 0.165 | 0.981 | 0.18 |
| pmc_nyse_circuit_breaker_2025 | 2025-01-01 | 2025-12-31 | 365 | 0 (0) | 178 | 178 | 0.0015 | 0.305 | 0.45 |
| pmc_pres2024_gop_party | 2023-01-09 | 2024-11-14 | 676 | 0 (0) | 676 | 676 | 0.27 | 0.9995 | 0.62 |
| pmc_pres2024_trump | 2024-01-05 | 2024-11-05 | 306 | 0 (0) | 306 | 306 | 0.405 | 0.715 | 0.42 |
| pmc_pres2024_trump_popular_vote | 2024-01-10 | 2024-11-11 | 307 | 0 (0) | 307 | 307 | 0.205 | 0.9995 | 0.50 |
| pmc_recession_by_2025may | 2024-09-26 | 2025-04-30 | 217 | 0 (0) | 217 | 217 | 0.001 | 0.37 | 0.27 |
| pmc_remove_10pct_tariff_by_2025jul | 2025-04-07 | 2025-06-30 | 85 | 0 (0) | 82 | 82 | 0.004 | 0.327 | 0.26 |
| pmc_ru_ua_ceasefire_2024 | 2024-09-16 | 2024-12-31 | 107 | 0 (0) | 107 | 107 | 0.0015 | 0.225 | 0.10 |
| pmc_ru_ua_ceasefire_2025 | 2024-12-30 | 2025-12-31 | 367 | 0 (0) | 180 | 180 | 0.0005 | 0.775 | 0.22 |
| pmc_ru_ua_ceasefire_by_2025jul | 2025-01-22 | 2025-06-30 | 160 | 0 (0) | 157 | 157 | 0.0015 | 0.635 | 0.11 |
| pmc_shutdown_by_2023nov19 | 2023-10-04 | 2023-11-18 | 46 | 0 (0) | 46 | 46 | 0.0005 | 0.685 | 0.04 |
| pmc_shutdown_by_2024mar09 | 2024-01-26 | 2024-03-11 | 46 | 0 (0) | 46 | 46 | 0.026 | 0.9965 | 0.36 |
| pmc_taiwan2024_lai | 2023-10-27 | 2024-01-12 | 78 | 0 (0) | 78 | 78 | 0.45 | 0.79 | 0.32 |
| pmc_tariff25_mex_can_by_2025feb | 2024-11-26 | 2025-02-03 | 70 | 0 (0) | 70 | 70 | 0.005 | 0.43 | 0.20 |
| pmc_tariff40_china_first_100d | 2024-11-20 | 2025-04-02 | 134 | 0 (0) | 134 | 134 | 0.0295 | 0.325 | 0.17 |
| pmc_tariff_mexico_up_by_2025may | 2025-02-03 | 2025-05-03 | 90 | 0 (0) | 90 | 90 | 0.015 | 0.9995 | 0.13 |
| pmc_tariff_revenue_250b_2025 | 2025-04-16 | 2026-02-28 | 319 | 0 (0) | 73 | 73 | 0.0045 | 0.35 | 0.24 |
| pmc_tiktok_ban_by_2025may | 2024-09-18 | 2025-01-21 | 126 | 0 (0) | 126 | 126 | 0.125 | 0.9935 | 0.18 |
| pmc_trump_ends_ukraine_war_90d | 2024-10-23 | 2025-04-23 | 183 | 0 (0) | 183 | 183 | 0.0005 | 0.5 | 0.14 |
| pmc_trump_fire_powell_by_2025jun | 2025-04-08 | 2025-05-31 | 54 | 0 (0) | 54 | 54 | 0.0015 | 0.195 | 0.13 |
| pmc_turkey2023_erdogan | 2023-02-07 | 2023-05-28 | 111 | 0 (0) | 111 | 111 | 0.3225 | 0.998 | 0.17 |
| pmc_us_china_deal_by_2025jun | 2025-04-08 | 2025-05-11 | 34 | 0 (0) | 34 | 34 | 0.145 | 0.75 | 0.06 |
| pmc_us_eu_deal_by_2025jul | 2025-04-15 | 2025-06-30 | 77 | 0 (0) | 74 | 74 | 0.0115 | 0.635 | 0.05 |
| pmc_us_strike_iran_by_2025jul | 2025-03-31 | 2025-06-21 | 83 | 0 (0) | 83 | 83 | 0.0385 | 0.6815 | 0.10 |
| pmc_ust10y_above5_by_2025jun30 | 2024-12-30 | 2025-07-01 | 184 | 0 (0) | 180 | 180 | 0.0005 | 0.605 | 0.19 |

## Cautions for the test

1. **Coverage:** effectively testable only up to the 2025-06-27 end of the stock panel. Start dates: Kalshi Fed 2021-07; Kalshi macro prints CPI MoM 2021-11, GDP 2021-12, U-3 2022-02, core CPI 2022-07, CPI YoY 2022-12, payrolls 2023-03, gas 2024-05; Polymarket 2023 to 2024. That's 1.5 to 4 years, so 63-day moves and long offsets have very few independent observations. The PredictIt pieces are four separate 90-day windows (2016, 2018, 2020, 2022).
2. **Calendar rows:** 7 days a week; align as-of to trading days. Weekend values are real (24/7 trading) but stocks don't trade then.
3. **Rolls and calendar effects:** rolling series jump at each roll (new meeting, month, year or cycle), and roll days are FOMC, CPI, jobs, GDP release days and Jan 1, all of which carry known stock-return calendar effects. Kalshi rolls on the release or meeting day itself; Polymarket's Fed series rolls the day after, with the meeting-day value showing the decision. Hits concentrated on those dates are calendar effects, not leads.
4. **Bounded, sticky probabilities:** values live in [0, 1], pin at 0.005/0.995 for long stretches (see the zero-move column), and converge to 0/1 at resolution; finite-life contracts end abruptly. `diff` on a probability near 0 or 1 is tiny by construction.
5. **Mechanical copies of prices already in the panel:** `ks_spx_yearend_exp` is essentially the S&P 500 plus a drift; `ks_gas_month_exp` is the AAA spot gas price; the BTC/ETH contracts track crypto prices. A hit with stocks there is the underlying asset, not new information.
6. **Revisions and definitions:** expected prints refer to the first release (BLS/BEA advance numbers, later revised). Recession contracts differ by venue and year (two negative BEA quarters vs NBER; see each series' notes), so the rolling recession series changes definition at each Jan 1.
7. **Liquidity:** 2021-2022 Kalshi ladders are thin (wide spreads, few strikes). Days that fail the ladder rules are dropped, so early series are gappy, and the first days after any listing can be unreliable.
8. **Manifold is play money** and PredictIt is capped at $850 per trader; weight their evidence accordingly.
9. **Selection bias in the contracts:** the 91 finite-life contracts were chosen by topic and volume with hindsight (e.g. "Fed cuts 50+ bps in Sept 2024" was picked knowing it happened). Read contract-level census hits as browsing, not evidence; the 20 rolling series are the systematic ones.
10. **Overlaps:** the `ksc_recession_20xx` contracts contain the segments that make up `ks_recession_cur_year`, and several Polymarket Fed contracts are components of `pm_fed_next_*`. Don't count a hit in both as independent.

## Files
- Catalog: `~/.local/share/correlation2/external/catalog_data-predictions.csv` (111 rows).
- Series: `~/.local/share/correlation2/external/series/{ks_,ksc_,pm_,pmc_,pi_,mf_}*.csv`.
- Scripts (rerunnable, cache-aware): `scripts/data-predictions/`:
  - `kalshi_enum.py`, `kalshi_fetch.py` and `kalshi_build.py`, with `kalshi_lib.py` and `gzip_candles.py`;
  - `polymarket_enum.py`, `polymarket_fetch.py` and `polymarket_build.py`, with `pm_select.py`;
  - `pi_mf_build.py`;
  - `build_catalog.py`;
  - `qc_report.py`.
- Logs: `scripts/data-predictions/logs/`; derived indexes in `meta_pm/` and `meta_qc/`.

