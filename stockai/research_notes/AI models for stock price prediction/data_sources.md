# Data sources for training and honestly testing a stock price-prediction model (as of 2026-10-18)

Scope: what a hobbyist can get, at what cost and with what biases, to (1) extend daily US history past 2025-06-27 for a clean never-seen test period, (2) remove survivorship bias by adding delisted stocks, and (3) optionally add intraday bars, within 171 GB of free disk (`df` on 2026-10-18 shows 171G free on `/`).

How to read the prices: pages marked "fetched 2026-10-18" are vendor pages I read directly that day. "Via search summary" means the claim comes from a search engine's digest of the page, either because the direct fetch failed (404, 403, 429 or a JavaScript wall) or because I did not fetch it. Treat those as less certain.

## 1. Stooq in 2026: access rules, bulk snapshots, adjustment, data quality, delisted coverage

### Takeaway
Since about March 2026, Stooq's per-ticker CSV downloads need a personal API key, obtained through a CAPTCHA, with an unpublished daily hit limit. The whole site, including the bulk page `stooq.com/db/h/`, now sits behind a JavaScript proof-of-work browser check, so bulk zips realistically have to be downloaded by hand in a browser.

Third-party evidence suggests `d_us_txt.zip` was still being downloaded in October 2026, but I could not confirm its current end date.

The data in hand is split-adjusted and also back-adjusted for dividends, so old price levels are not the prices that actually traded. It contains no delisted companies, so a fresh Stooq snapshot would carry survivorship bias into the test period as well.

### Cited Findings
- On 2026-10-18 a plain HTTP request to the bulk page returned only an anti-bot page: "This site requires JavaScript to verify your browser. Please enable JavaScript and reload." It contains a SHA-256 proof-of-work script that POSTs to `/__verify`. The file list and dates were not visible without a real browser. — [Stooq bulk page, fetched 2026-10-18](https://stooq.com/db/h/)
- pandas-datareader issue #1012, opened 2026-04-13: "As March 2026, Stooq requires an API key for data downloads." `DataReader(ticker,'stooq',...)` now fails with a `ParserError` because Stooq returns an HTML page explaining how to get a key instead of CSV. The key page is `https://stooq.com/q/d/?s=spy.us&get_apikey`. The issue was still open, with an unmerged PR. — [GitHub pandas-datareader #1012](https://github.com/pydata/pandas-datareader/issues/1012)
- April 2026 forum posts report that Stooq requires both an API key and JavaScript browser verification. The described procedure is to open the `get_apikey` page for any ticker, solve the CAPTCHA, and copy the CSV link, which then carries an `apikey` parameter. A per-key daily quota exists but is not published. Going over it returns the text "Exceeded the daily hits limit" with HTTP 200, so scripts must check the response body. — [lite-C/Zorro forum, "Stooq now requires an API key"](https://opserver.de/ubb7/ubbthreads.php?ubb=showflat&Number=489362); [api-evangelist/stooq profile](https://github.com/api-evangelist/stooq). Flag: via search summary. A direct fetch of the forum returned HTTP 429, and neither source is official Stooq documentation.
- Daily download limits predate 2026. An older AmiBroker forum thread shows a Polish-language daily-limit violation message from Stooq. — [AmiBroker forum](https://forum.amibroker.com/t/stooq-download-range-control-and-download-limit-violation/1167)
- Bulk zips still appear to be in use in October 2026. A GitHub issue opened 2026-10-07 states "Stooq's real bulk downloads are multi-symbol bundles", namely `d_us_txt.zip` (daily) and `h_us_txt.zip` (hourly), "~13.3k members each", laid out as `data/daily/us/nyse etfs/2/spy.us.txt`. Its acceptance criteria say "Verified once against the real d_us_txt.zip". It does not state the last data date or how the file was obtained. — [GitHub rustyeddy/trader #462](https://github.com/rustyeddy/trader/issues/462)
- A third-party profile describes the `/db/h/` snapshots as ZIP-compressed CSV covering 12,000+ securities at daily, weekly and monthly frequency. — [api-evangelist/stooq](https://github.com/api-evangelist/stooq)
- Local inspection of the file in hand, `/home/bhoult/programming/correlation/data/stooq/d_us_txt.zip`:
  - 523,004,504 bytes compressed, 1.80 GB uncompressed, 11,563 files, member timestamps 2025-06-28.
  - Header: `<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`.
  - AAPL on 2020-08-28 (the last day before its 4:1 split): close 121.699, volume 192,421,436.
  - SPY: close 207.27 on 2020-03-23, 584.323 on 2024-12-31 and 614.91 on 2025-06-27.
  - No files exist for SIVB, BBBY, FRC or TWTR, all of which delisted in 2022-2023.
  - Source: [local Stooq d_us_txt.zip](file:///home/bhoult/programming/correlation/data/stooq/d_us_txt.zip)
- AAPL closed at $499.23 on Friday 2020-08-28, before the 4:1 split; the split-adjusted equivalent is about $124.81. — [InvestingCube](https://www.investingcube.com/apple-share-price-gains-in-premarket-trading-as-stock-split-takes-effect/) (via search summary)
- I found no official Stooq documentation of its adjustment method and no published error reports. One undated, anecdotal Wealth-Lab forum user found Stooq "doesn't have data" for the symbols they tested. — [Wealth-Lab forum](https://wl6.wealth-lab.com/Forum/Posts/Data-provider-alternatives-with-dividend-adjusted-data-38814)
- Vendor-quality context: split and dividend records are the hardest data to clean, and most third-party feeds that aggregate splits and dividends have misreporting problems such as wrong ex-dates and wrong ratios, especially for small caps. — [FirstRate Data, Data Quality](https://firstratedata.com/about/dataquality)

### Inferences
- **Adjustment.** Stooq's 121.70 for AAPL on 2020-08-28 is about 2.5% below the split-adjusted $124.81, so Stooq's series is split-adjusted and also back-adjusted for later dividends. The 192M volume is consistent with split-adjusted volume, but I did not check raw pre-split volume against a source. Consequences:
  - Old prices change between snapshots as new dividends are applied. A 2026 snapshot will not match the 2025 file for overlapping dates, so stitch by daily returns, not by price level.
  - Closes behave like total-return series, which is good for return labels but distorts price-level features such as "under $5" filters.
- **Survivorship.** The snapshot contains only symbols alive at download time, and the missing 2022-2023 delisted names confirm the bias the brief describes. A new snapshot taken in October 2026 would also omit names delisted between mid-2025 and October 2026, making any long-only test over that window look better than it really was.
- **Maintenance.** The roughly 13.3k members reported in October 2026, against 11,563 files in the June 2025 file, hint that the bulk file is still maintained and has grown. Unverified: the count may include directories or another snapshot.
- **Practical route.** Open `stooq.com/db/h/` in a normal browser, where the proof-of-work check passes automatically, download `d_us_txt.zip` by hand, and check the last `<DATE>` inside. Do not script around the check.

### Gaps
- The current file list, sizes and update dates on `/db/h/` were not visible because of the JavaScript challenge. Whether `d_us_txt.zip` currently runs into September or October 2026 is unconfirmed.
- Whether the bulk zips themselves need an API key or CAPTCHA, or count toward the daily limit, is unconfirmed; no source addressed it.
- The exact daily hit limit per key is unpublished.
- How quickly Stooq applies new corporate actions is undocumented. As an unverified check from memory, SPY's raw closes of about $586 on 2024-12-31 and about $223 on 2020-03-23 would imply roughly 0.3% and 7% dividend adjustment respectively. The 0.3% looks like only one of SPY's two 2025 dividends was applied, which may indicate adjustment lag.

## 2. Fresh end-of-day US data after 2025-06-27 (free or cheap)

### Takeaway
Daily data for the test window (2025-06-30 to 2026-10-16) is cheap. The cheapest options:
- **Sharadar "Prices"**, personal-use license: $9/month for the 5-year tier.
- **Massive (Polygon.io was renamed Massive)**: free Basic plan with 2 years of history at 5 calls a minute, or Starter at $29/month with flat files.
- **Tiingo**: Power plan at $30/month for about 110k symbols, personal use.
- **EODHD**: Historian plan at $19.99/month.
- **MarketParquet**: $79 one-time, data through 2026-10-16. A new, unverified vendor.

Alpaca's free tier is IEX-only, a single venue, so its bars are not comparable to consolidated bars. yfinance still works but is unofficial, "personal use only" and rate-limited. Three sources claim delisted (survivorship-free) coverage of the test window itself: Sharadar, MarketParquet, and EODHD. EODHD serves delisted tickers via `delisted=1`, with EOD, splits and dividends for companies delisted after 2018; that claim is via search summary because the page returned 404 (see section 3).

### Cited Findings
- **Sharadar** (fetched 2026-10-18), USD:

  | Tier | History | Monthly | Annual |
  |---|---|---|---|
  | Prices | 5 years | $9 | $99 |
  | Prices | 10 years | $19 | $199 |
  | Prices | Full history | $39 | $299 |
  | Bundle (14 tables) | Full history | $69 | $499 |

  The Prices tier includes the tables descriptions, tickers, actions, sp500, stocks, funds and metrics. Bulk downloads match the purchased history tier, and subscriptions are under a "Personal Use License". — [Sharadar subscribe page](https://sharadar.com/subscribe)
- **Tiingo** (fetched 2026-10-18):

  | | Starter | Power |
  |---|---|---|
  | Price | $0/month | $30/month |
  | Requests per hour / per day | 50 / 1,000 | 10,000 / 100,000 |
  | Unique symbols per month | 500 | 110,328 |
  | Bandwidth per month | 1 GB | 40 GB |
  | End-of-day history | "30+ Years" | "30+ Years" |

  The license is "Internal Use Only", meaning "you may only use the data for your own personal use". Business internal use costs $50/month or $499/year. — [Tiingo pricing](https://www.tiingo.com/about/pricing)
- **Massive.com** (fetched 2026-10-18). `polygon.io/pricing` returns a 301 redirect to `massive.com/pricing`, confirming the rename.

  | Plan | Price | History | API calls | Data timing | Flat files |
  |---|---|---|---|---|---|
  | Stocks Basic | $0/month | 2 years | 5/minute | End of day | Not listed |
  | Stocks Starter | $29/month | 5 years | Unlimited | 15-minute delayed | Included |
  | Stocks Developer | $79/month | 10 years | Unlimited | 15-minute delayed | Included |
  | Stocks Advanced | $199/month | 20+ years | Unlimited | Real-time | Included |

  All plans include minute aggregates and are labelled "Individual use"; annual billing saves 20%. — [Massive pricing](https://massive.com/pricing)
- **EODHD** personal plans (fetched 2026-10-18):

  | Plan | Monthly | Yearly | Limits |
  |---|---|---|---|
  | Free | $0 | $0 | 20 API calls/day; 1 year of EOD |
  | Historian (EOD) | $19.99 | $199 | 100,000 calls/day, 1,000/minute |
  | Active Trader (EOD + intraday) | $29.99 | $299.90 | 100,000 calls/day, 1,000/minute |
  | ALL-IN-ONE | $99.99 | $999.90 | 100,000 calls/day, 1,000/minute |

  Paid plans include "30+ yrs" of US EOD history, back to 1972 for the oldest names. — [EODHD pricing](https://eodhd.com/pricing)
- **Alpaca** (fetched 2026-10-18):
  - Basic is free: 200 historical calls/minute, "for equities only the IEX exchange", "latest 15 minutes" historical restriction, history "Since 2016".
  - Algo Trader Plus is $99/month: all US exchanges, 10,000 calls/minute.
  - The consolidated CTA and UTP feeds "encompass 100% of market volume".
  - Sources: [Alpaca market data](https://alpaca.markets/data); [Alpaca docs](https://docs.alpaca.markets/docs/about-market-data-api)
- **MarketParquet** (fetched 2026-10-18): data "current through 2026-10-16". It includes delisted names, uses split-adjusted prices, and costs $79 one-time with an optional $15/month "Keep Current" add-on. Details are in sections 3 and 4. — [MarketParquet historical data](https://marketparquet.com/historical-market-data)
- **Alpha Vantage** (fetched 2026-10-18): free tier of 25 requests/day. Premium runs from $49.99/month (75 requests/minute) to $249.99/month (1,200 requests/minute), with annual plans "2 months off". — [Alpha Vantage premium](https://www.alphavantage.co/premium/)
- **Financial Modeling Prep**: the pricing page returned HTTP 403, so there are no confirmed 2026 prices. — [FMP pricing](https://site.financialmodelingprep.com/pricing-plans)
- **yfinance**:
  - README: "the Yahoo! finance API is intended for personal use only". yfinance "is not affiliated, endorsed, or vetted by Yahoo", and users should refer to Yahoo's terms of use for their rights to the data. — [yfinance README](https://raw.githubusercontent.com/ranaroussi/yfinance/main/README.md)
  - Rate limits (anecdotal, 2024-2025 threads): users report `YFRateLimitError` ("Too Many Requests") and IP blocks lasting hours to more than 24 hours. One user managed about 100 requests and then needed a 30-second pause. Failures occurred on versions 0.2.57-0.2.59 even from different IPs. — [yfinance discussion #2431](https://github.com/ranaroussi/yfinance/discussions/2431); [issue #2422](https://github.com/ranaroussi/yfinance/issues/2422); [issue #2480](https://github.com/ranaroussi/yfinance/issues/2480)
- **Norgate** EOD-only tiers without delisted names: Silver $148.50 per 6 months (10 years of history); Gold $198 per 6 months (20 years). — [Norgate packages](https://norgatedata.com/stockmarketpackages.php)
- **Databento** (fetched 2026-10-18): pay-as-you-go priced per GB, or Standard at $199/month; $125 in free credits for new users, expiring 6 months after signup. — [Databento pricing](https://databento.com/pricing)

### Inferences
- **Cheapest bulk route.** One month of Sharadar Prices (5-year tier, $9) gives the whole test window plus about four years of overlap with the Stooq data, useful for checking the splice. It also includes the `actions` and `tickers` tables and claims survivorship-free coverage (section 3).
- **Massive Basic (free).** Its 2-year window covers the test period, but a per-ticker loop at 5 calls/minute for 5,000 names takes about 17 hours. A whole-market-per-date call would need only about 330 calls, but I did not confirm Basic includes one.
- **Tiingo.** The free tier's 500-symbol cap is too small for about 5,000 names.
- **Alpaca IEX.** IEX-only bars are a single-venue sample. Volume, and sometimes high, low and close, differ from consolidated bars. Do not use free Alpaca bars in a test period if training used consolidated Stooq bars.
- **Adjustment consistency.** Each vendor's adjustment differs: Stooq back-adjusts for dividends, MarketParquet is split-only, Massive flat files are unadjusted. Rebuild training and test data from one vendor, or splice using returns.

### Gaps
- FMP 2026 prices and limits (page returned 403).
- Whether Massive Basic includes the grouped-daily (whole market per date) endpoint or day-aggregate flat files.
- Alpha Vantage: whether daily adjusted data is premium-only in 2026.
- Databento prices and history depth for US equities. Its `get_cost` API gives exact quotes per request but needs an account.
- No 2026-dated reports on yfinance reliability were found, and it is unclear whether Yahoo serves delisted tickers.

## 3. Survivorship-bias-free US history including delisted companies

### Takeaway
Options, cheapest first:
- **Sharadar Prices, full history**: $39 for one month or $299/year, personal license. EOD from 1998, claimed survivorship-free, with corporate actions and S&P 500 membership tables. The cheapest documented source.
- **MarketParquet**: $79 one-time with a perpetual license. Claims 8,500 delisted stocks from 2000, but its operator and data provenance are undisclosed.
- **EODHD Historian**: $19.99/month. Delisted tickers via `delisted=1`, with fuller coverage only for delistings after 2018.
- **Norgate Platinum**: $346.50 for 6 months. The most explicit survivorship-free product (back to 1990, delisted securities and historical index constituents), but its updater is Windows-only (a virtual machine is allowed).

CRSP via WRDS is effectively unavailable to unaffiliated individuals.

### Cited Findings
- **Norgate** US Stocks packages (fetched 2026-10-18), USD:

  | Package | 6 months | 12 months | History | Delisted securities and historical index constituents |
  |---|---|---|---|---|
  | Silver | $148.50 | $270.00 | 10 years | No |
  | Gold | $198.00 | $360.00 | 20 years | No |
  | Platinum | $346.50 | $630.00 | Back to 1990 | "Also includes access to delisted securities and historical index constituents" |
  | Diamond | $433.13 | $787.50 | Back to 1950 | Yes |

  — [Norgate stock market packages](https://norgatedata.com/stockmarketpackages.php)
- **Norgate Data Updater** requirements (fetched 2026-10-18):
  - Windows 10, Windows 11, Windows 365, or Windows Server 2022/2025 (single user); 64-bit CPU, at least 4 GB RAM, .NET Framework 4.7.2 or higher. "VM usage is allowed."
  - US disk usage: Platinum is a 2 GB download and 9.1 GB on disk; Diamond 2.3 GB and 10.3 GB; Silver 911 MB and 5.3 GB.
  - Source: [Norgate NDU installation](https://norgatedata.com/ndu-installation.php)
- Wine is reported as unsupported for Norgate's software, with a Windows VM (licensed Windows needed) as the alternative. — [Premium Data (Norgate) downloads FAQ](https://www.premiumdata.net/support/downloads-faq.php) (via search summary)
- The `norgatedata` Python package's latest version is reported as 1.0.74 (2023-10-04) on piwheels, while the PyPI user page says "Last released Jan 11, 2024". — [piwheels norgatedata](https://www.piwheels.org/project/norgatedata); [PyPI NorgateData](https://www.pypi.org/user/NorgateData/) (via search summary; the two dates conflict)
- A third-party listing quotes Norgate at "$35 / month, 21-day trial", with the tier unspecified and probably outdated. — [NexusFi](https://nexusfi.com/d/data-providers/norgate-data/)
- **Sharadar**:
  - Personal Prices tiers are $9, $19 or $39 per month for 5 years, 10 years or full history; the full-history Prices tier is $299/year. Tables include tickers, actions, sp500, stocks and funds, under a "Personal Use License". The page does not explicitly mention delisted coverage. — [Sharadar subscribe page](https://sharadar.com/subscribe)
  - Sharadar's site calls its fundamentals "nearly completely free of survivorship bias" and, in a blog excerpt, its stock price data "survivorship bias free". Coverage "extends back to the 90s". — [Sharadar home](https://sharadar.com/)
  - QuantRocket lists Sharadar "End-of-Day US Stock and Fund Prices" with history since 1998 and S&P 500 Constituents since 1957. — [QuantRocket Sharadar](https://www.quantrocket.com/sharadar/)
  - Professional users must buy through Nasdaq Data Link, according to a reseller page. — [QuantRocket Sharadar pricing](https://www.quantrocket.com/pricing/data/sharadar/) (via search summary)
  - The Nasdaq Data Link SEP page renders via JavaScript and returned no content. — [Nasdaq Data Link SEP](https://data.nasdaq.com/databases/SEP)
- **MarketParquet** (fetched 2026-10-18):
  - Coverage: "16,000+ cumulative stocks" of which 8,500 are delisted (suffixed "-DELISTED"), plus 5,000+ ETFs. Stocks and ETFs from 2000-01-03, current through 2026-10-16.
  - Bars: 1-minute, 5-minute, 30-minute, hourly and daily. Prices are split-adjusted, "Dividends are not embedded in prices", and zero-volume bars are kept.
  - Price: $79 one-time; Keep Current $15/month; free daily sample and a one-week, 30-symbol intraday sample.
  - Source: [MarketParquet historical data](https://marketparquet.com/historical-market-data)
- MarketParquet's homepage names no company, location or upstream data source; it says only "We couldn't find reliable split-adjusted Parquet data for backtesting, so we built it." — [MarketParquet home](https://marketparquet.com/)
- MarketParquet license terms (fetched 2026-10-18):
  - Permitted: "personal trading and investment research"; backtesting and modeling "for yourself or within your own company"; "Feed the data to AI/ML models and coding agents you operate"; publishing derived results.
  - Prohibited: redistributing raw files.
  - The license is perpetual for data already downloaded, even after cancelling, and the data is provided as-is with no accuracy warranty.
  - Source: [MarketParquet data license](https://marketparquet.com/data-license)
- **FirstRate Data**: "16327 stock tickers" from January 2000 at 1-minute to daily resolution, "including over 7000 delisted tickers", "adjusted for splits and dividends"; no prices on the page. — [FirstRate Data](https://firstratedata.com/)
- **EODHD**: delisted tickers are listed with the Exchange Symbol List endpoint using `delisted=1`. Companies delisted after 2018 have EOD, fundamentals, splits and dividends; those delisted after 2021 also have intraday; pre-2018 delistings have EOD only. EODHD warns that remembered tickers may now belong to other companies. — [EODHD delisted data](https://eodhd.com/financial-apis/review-delisted-stock-companies-data) (via search summary; the page returned 404 on direct fetch)
- **Massive / Polygon**: survivorship evidence is anecdotal and conflicting.
  - Positive: Polygon reportedly claims to be free of survivorship bias, and a GitHub developer describes its data as point-in-time by ticker. — [Wealth-Lab forum](https://wl6.wealth-lab.com/Forum/Posts/Survivorship-bias-Delisted-symbols-Historical-index-components-33495); [shinathan/polygon.io-stock-database](https://github.com/shinathan/polygon.io-stock-database)
  - Negative: users report that reused tickers return only the newest company's history; for HLTH, only Cue Health from 2021-09-24. — [Alpaca forum](https://forum.alpaca.markets/t/using-polygon-to-get-historical-data-for-changed-tickers/1802); [Stonks Capital, Massive Problems Part 1](https://stonkscapital.substack.com/p/massive-problems-part-1)
  - All via search summary.
- Massive data-quality report (Niv Goren, 2026-01-22): "Some splits in their database aren't real"; "The flat files API only provides unadjusted prices"; late prints (for example BTTC) can distort backtests. — [Stonks Capital, Massive Problems Part 2](https://stonkscapital.substack.com/p/massive-problems-part-2)
- **CRSP via WRDS**: access is through institutional subscriptions. University guides state that alumni and unaffiliated people do not qualify, and licenses restrict use to internal academic purposes. — [University of Denver WRDS guide](https://libguides.du.edu/wrds); [Maastricht University Library, CRSP](https://library.maastrichtuniversity.nl/database/crsp/)

### Inferences
- **Sharadar.** At $39 for one month, full-history Prices is roughly 9x cheaper than Norgate Platinum's 6-month minimum and runs natively on Linux (bulk download, no Windows updater). Its survivorship-free claim is the vendor's own wording, and the delisted count is not stated. Check the `tickers` table for delisted names (for example SIVB, BBBY, FRC, TWTR) on day one, inside the paid month.
- **Norgate.** The strongest explicit guarantee, but on this Linux workstation it needs a Windows VM plus a Windows license, and it is a subscription.
- **MarketParquet.** A similar product shape to FirstRate (about 16k stocks, about 7-8.5k delisted, from January 2000, the same five bar intervals), but it is an anonymous operator with unstated provenance. Its quality and the legitimacy of its licensing chain are unverifiable, so test the free samples before paying.
- **Massive flat files.** One file per day of all tickers that traded, so names that later delisted should be present for the days they traded, by construction. That is unverified. Mapping tickers to companies across reuse is the user's job, and prices are unadjusted (apply splits yourself, after validating them).
- **Test window.** Survivorship-free coverage of 2025-06-30 to 2026-10-16 matters more for an honest test than pre-2005 depth: a long-only model tested on survivors only gets a free tailwind.

### Gaps
- Sharadar: explicit delisted counts, adjustment method (adjusted vs unadjusted close columns), and whether personal-license data may be kept after cancelling. The terms page was not read.
- Norgate: personal-use license text, trial length, and whether downloaded data stays usable after the subscription lapses.
- EODHD delisted coverage (the page returned 404) and Tiingo delisted coverage. The riingo docs show Tiingo ticker lists with `startDate` and `endDate` columns, which are not confirmed to mark delistings. — [riingo](https://github.com/business-science/riingo)
- FirstRate prices; CRSP individual pricing (not published).

## 4. Intraday and minute data: prices, history depth, and size vs the 171 GB budget

### Takeaway
Two cheap routes fit the 171 GB budget:
- **Massive flat files** (consolidated tape, unadjusted): Starter at $29/month gives 5 years, about 28 GB gzipped; Developer at $79/month gives 10 years, about 50 GB.
- **MarketParquet**: $79 one-time for 1-minute stock bars from 2000 to 2026-10-16, including delisted names, 89.3 GB in Parquet (vendor-stated). Unverified vendor.

Kibot's all-stocks 1-minute package costs $3,000. Ten years of dense 1-minute arrays for 5,000 stocks take 49 GB (float16) to 157 GB (float32 plus timestamps). Use float16 or 5-minute bars, or keep only one copy of the raw files.

### Cited Findings
- **Massive flat files**, `us_stocks_sip/minute_aggs_v1`: one gzipped CSV per trading day with per-minute OHLC, volume and transactions for all US equities.
  - Reported daily sizes: October 2025 21.6-26.2 MB; November 2025 21.8-25.3 MB (the half-day 2025-11-28 was 12.6 MB); January 2020 about 15-16 MB; February 2022 about 19-25 MB.
  - Records go back to 2003-09-10, and files update around 11 a.m. ET for the prior day.
  - Access: Starter 5 years, Developer 10 years, Advanced all history.
  - Sources: [Massive docs 2025/10](https://massive.com/docs/flat-files/stocks/minute-aggregates/2025/10); [2025/11](https://massive.com/docs/flat-files/stocks/minute-aggregates/2025/11); [2020/01](https://massive.com/docs/flat-files/stocks/minute-aggregates/2020/01); [2022/02](https://massive.com/docs/flat-files/stocks/minute-aggregates/2022/02). Via search summary: the 2025/10 and 2026/09 pages returned 404 on direct fetch on 2026-10-18.
- Massive flat files hold only unadjusted prices. — [Stonks Capital, Massive Problems Part 2](https://stonkscapital.substack.com/p/massive-problems-part-2)
- **MarketParquet** (fetched 2026-10-18): 95,791 Parquet files, one per trading day per dataset, totalling 168 GB:

  | Dataset | Size |
  |---|---|
  | stock_1min | 89.3 GB |
  | stock_daily | 1.2 GB |
  | etf_1min | 15.3 GB |
  | etf_daily | 351 MB |
  | futures_1min | 3.3 GB |
  | futures_daily | 45 MB |

  $79 one-time. — [MarketParquet historical data](https://marketparquet.com/historical-market-data)
- **Kibot** (fetched 2026-10-18):
  - "All Stocks" (18,000+ symbols): 30-minute $600; 1-minute $3,000; 1-minute plus bid/ask $4,500; tick plus bid/ask $9,000.
  - S&P 500 package: 30-minute $150; 1-minute $750.
  - "One-minute history goes back to 1998"; tick and bid/ask data from 2009.
  - Unadjusted, split-adjusted and fully adjusted series; updates via Premium at $139/month. Delisted coverage is not stated.
  - Source: [Kibot buy page](http://www.kibot.com/buy.aspx)
- **FirstRate Data**: about 16k stocks at 1-minute to daily resolution from January 2000, including more than 7,000 delisted; adjusted for splits and dividends; zipped CSV; prices not shown. — [FirstRate Data](https://firstratedata.com/)
- **EODHD** Active Trader ($29.99/month): US 1-minute from 2004; "other intervals" from October 2020. — [EODHD pricing](https://eodhd.com/pricing)
- **Alpaca**: bars "Since 2016"; IEX-only on Basic, SIP on Algo Trader Plus at $99/month. — [Alpaca docs](https://docs.alpaca.markets/docs/about-market-data-api)
- **Databento**: $125 in free credits; `get_cost` returns exact request costs. EQUS.SUMMARY is confirmed for OHLCV-1d, but I found no confirmation of 1-minute depth. — [Databento pricing](https://databento.com/pricing); [Databento docs, finding an equities dataset](https://databento.com/docs/examples/equities/equities-introduction/finding-an-equities-dataset) (via search summary)
- **Stooq** has an hourly bundle, `h_us_txt.zip`, with about 13.3k members. — [GitHub rustyeddy/trader #462](https://github.com/rustyeddy/trader/issues/462)

### Inferences
Size arithmetic, my own calculation. Assumptions: 5,000 stocks, 390 regular-session minutes, 252 days a year, and a dense grid (an upper bound).

| Item | Size |
|---|---|
| Bars per year | 491.4M |
| Bars, 5 years / 10 years | 2.46B / 4.91B |
| float32 OHLCV + int64 timestamp + int32 ticker id (32 bytes/bar), 5 / 10 years | 79 GB / 157 GB |
| Dense float32 OHLCV tensor (20 bytes/bar), 10 years | 98 GB |
| float16 OHLCV (10 bytes/bar), 10 years | 49 GB |
| 5-minute bars, float32, 10 years | about 20 GB |
| Massive gzipped flat files, 5 years (2021-2026) | about 28 GB |
| Massive gzipped flat files, 10 years | about 50 GB (assumes 2016-2019 files are about 2020 size) |

- **Cross-check.** MarketParquet's 89.3 GB for about 26 years of 1-minute stock bars is about 3.4 GB a year, in line with Massive's roughly 4-6 GB a year gzipped.
- **Budget.**
  - Fits: about 50 GB of compressed Massive files for 10 years plus 49 GB of float16 arrays, leaving about 70 GB.
  - Does not fit: raw files plus float32 arrays with timestamps for 10 years (about 207 GB). Convert year by year and delete the raw files, or memory-map from Parquet.
  - MarketParquet's full 168 GB archive does not fit alongside anything else; stock_1min plus stock_daily (90.5 GB) does.
- **Choice.** Massive is the known, SIP-sourced choice, but you must validate and apply splits yourself. MarketParquet is far cheaper per year of history but unverified.

### Gaps
- FirstRate and Databento US-equities prices; vendor-stated totals for Massive.
- Whether MarketParquet allows selective download by dataset or year, and whether its 1-minute bars include extended hours.
- Whether Massive's license lets you keep flat files after cancelling.
- Kibot's delisted coverage.

## 5. Point-in-time index membership (S&P 500, Russell)

### Takeaway
Free S&P 500 point-in-time membership is available from community datasets: GitHub hanshof/sp500_constituents (daily from 1996), or a reconstruction from Wikipedia's "Selected Changes" table. Paid and maintained options:
- Sharadar's `sp500` table, already included in the $9-$39 personal Prices tiers; history since 1957.
- Norgate Platinum and Diamond ("historical index constituents").
- EODHD's S&P 500 constituents from 2000, in a paid fundamentals package.

I found no free point-in-time Russell membership source.

### Cited Findings
- hanshof/sp500_constituents: daily S&P 500 membership CSVs from 1996, 833 commits; license and last update not checked. — [GitHub hanshof/sp500_constituents](https://github.com/hanshof/sp500_constituents)
- Robot Wealth (2020): rebuild historical S&P 500 membership for free by scraping Wikipedia's current-constituents and "Selected Changes" tables. — [Robot Wealth](https://robotwealth.com/how-to-get-historical-spx-constituents-data-for-free/)
- Sharadar's Prices tier includes an `sp500` table. — [Sharadar subscribe](https://sharadar.com/subscribe). QuantRocket says the S&P 500 Constituents dataset covers additions and removals since 1957. — [QuantRocket Sharadar](https://www.quantrocket.com/sharadar/)
- EODHD offers historical S&P 500 constituents from January 2000 (more than 90,000 rows) in a paid Fundamentals package. — [EODHD blog](https://eodhistoricaldata.com/blog/sp-500-historical-constituents-data)
- Norgate Platinum and Diamond include "historical index constituents"; the packages page does not name the indices. — [Norgate packages](https://norgatedata.com/stockmarketpackages.php)

### Inferences
- Sharadar's `sp500` table comes free with the Prices tier recommended for needs 1 and 2, so a separate membership purchase is unnecessary for S&P 500 universes.
- Membership history only helps when you also have prices for removed members. Without delisted prices, it just lists what is missing.

### Gaps
- Which indices Norgate's constituents cover (Russell 1000, 2000, 3000?).
- Any free or cheap point-in-time Russell membership history.
- The license and freshness of hanshof/sp500_constituents.

## 6. Pretraining corpus sizes of finance foundation models vs what is realistically obtainable here

### Takeaway
Kronos (arXiv 2025-08-02; NeurIPS 2025) was pretrained on more than 12 billion K-line (OHLCV) records from 45 global exchanges.
- The in-hand Stooq US daily set (26.6M rows) is about 450x smaller.
- Ten years of 1-minute bars for 5,000 US stocks (up to 4.9B bars) reaches the same order of magnitude as Kronos's corpus.
- About 26 years of 1-minute bars for about 16k stocks (MarketParquet or FirstRate class) would likely be comparable or larger, not verified.

### Cited Findings
- Kronos: "pre-train Kronos using an autoregressive objective on a massive, multi-market corpus of over 12 billion K-line records from 45 global exchanges". A tokenizer quantizes OHLCV into hierarchical discrete tokens for a decoder-only Transformer. The authors claim a 93% RankIC improvement over the leading time-series foundation model, 9% lower volatility MAE, and 22% better generative fidelity. — [NeurIPS 2025 listing](https://neurips.cc/virtual/2025/132504)
- The Kronos paper was submitted 2025-08-02. The abstract does not state data frequencies, asset classes, time span or model sizes. — [arXiv 2508.02739](https://arxiv.org/abs/2508.02739)
- Kronos weights (for example Kronos-small and Kronos-base) are on Hugging Face. — [Hugging Face Kronos-base copy](https://huggingface.co/jeckmy/Kronos-base)

### Inferences
- Kronos was submitted in August 2025, so its pretraining data cannot extend past about mid-2025. Data from after about 2025-07 is a clean test for any Kronos-derived model and for a self-trained one.
- All of Stooq's daily markets together are very likely well under 100M rows, roughly two orders of magnitude below Kronos. Minute bars are the only realistic way to get near 12B records on this machine, and the data-loading cost is disk and RAM, not GPU.

### Gaps
- Kronos's exact data cutoff, its daily vs intraday mix, and its share of US equities (not in the abstract; full paper not read).
- Corpus sizes of other finance foundation models were not researched.

## 7. Recommended cheapest-sufficient option for each need

### Takeaway
1. **Clean test period (daily, 2025-06-30 onward):** Sharadar Prices, 5-year tier, $9 for one month (personal license). Bulk-download everything, including `actions`, `tickers` and `sp500`. Free cross-check: a hand-downloaded Stooq `d_us_txt.zip`.
2. **Survivorship-free history:** Sharadar Prices, full history, $39 for one month (or $299/year). This also covers need 1, so buy this tier instead of the 5-year one if doing both. The fallback with the most explicit guarantee is Norgate Platinum, $346.50 for 6 months, which needs a Windows VM.
3. **Intraday:** the known-vendor choice is Massive Starter, $29 for one month (5 years of SIP minute files, about 28 GB gzipped) or Developer, $79 (10 years, about 50 GB). The cheapest-on-paper choice is MarketParquet, $79 one-time (1-minute from 2000, including delisted names, 89.3 GB, perpetual license), but only after its free samples check out.

### Cited Findings
- Sharadar personal Prices: $9, $19 or $39 per month for 5 years, 10 years or full history ($99, $199 or $299 a year); tables include tickers, actions, sp500, stocks and funds; "Personal Use License". — [Sharadar subscribe](https://sharadar.com/subscribe)
- Sharadar describes its price data as "survivorship bias free" — [Sharadar home](https://sharadar.com/) — with EOD history since 1998 — [QuantRocket Sharadar](https://www.quantrocket.com/sharadar/)
- Norgate Platinum: $346.50 for 6 months, back to 1990, delisted securities and historical index constituents. — [Norgate packages](https://norgatedata.com/stockmarketpackages.php). Its updater runs on Windows only, and "VM usage is allowed". — [Norgate NDU installation](https://norgatedata.com/ndu-installation.php)
- Massive Starter is $29/month (5 years, flat files) and Developer $79/month (10 years), "Individual use". — [Massive pricing](https://massive.com/pricing)
- MarketParquet: $79 one-time; 8,500 delisted stocks; 1-minute from 2000-01-03 to 2026-10-16; license perpetual for downloaded data, with AI/ML use permitted. — [MarketParquet](https://marketparquet.com/historical-market-data); [MarketParquet license](https://marketparquet.com/data-license)
- Stooq bulk page behind a JavaScript check; API key and CAPTCHA required since about March 2026. — [Stooq](https://stooq.com/db/h/); [pandas-datareader #1012](https://github.com/pydata/pandas-datareader/issues/1012)

### Inferences
- **Need 1 (Sharadar 5-year, $9).**
  - Covers: test-window daily bars with claimed delisted coverage (so the test is not survivor-only), plus about four years of overlap with Stooq for checking the splice.
  - Does not cover: intraday. Retention of data after cancelling is unconfirmed (license terms not read).
  - Fallbacks: the free Stooq snapshot (survivor-only, back-adjusted, currency unconfirmed) or Massive Basic (free, 2 years, 5 calls/minute).
- **Need 2 (Sharadar full history, $39).**
  - Does not cover: pre-1998 history, Russell membership, or a vendor-stated delisted count. Verify on day one that SIVB, BBBY, FRC and TWTR are present.
  - Norgate gives explicit delisted and index guarantees back to 1990, at about 9x the cost, plus Windows-VM friction.
  - CRSP is not an option.
- **Need 3.**
  - Massive does not cover: delisted-name identity (ticker reuse is your job), adjusted prices (validate and apply splits yourself), or a known retention policy.
  - MarketParquet does not cover: provenance, quality assurance or dividend adjustment, and it is an anonymous operator.
  - Either fits in 171 GB only if training arrays are float16 or 5-minute bars, or the raw files are deleted after conversion.
- **Cheapest combined plan for all three needs:** about $68 (one month of Sharadar full-history Prices at $39 plus one month of Massive Starter at $29), or $79 for MarketParquet alone if it passes sample checks. Even $118 for both Sharadar and MarketParquet costs about a third of Norgate's minimum $346.50.
