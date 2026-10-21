# External data, `data-other` share: sources, checks and caveats

Written by the `data-other` agent, 2026-10-18/19 (UTC). The catalog is `catalog_data-other.csv` (79 series). The normalized files are `series/<id>.csv` and the raw downloads are under `raw/<source>/`. Copies of the download and parse scripts are in `scripts/data-other/`; they parse with the standard library plus numpy and run with `python3 -I`.

## What is here

| category | n | frequency | span | ids |
|---|---|---|---|---|
| space_weather | 5 | daily | 1932 (SN 1818) to 2026-10 | `gfz_ap`, `gfz_kp_max`, `gfz_storm_ap30`, `gfz_f107_obs`, `silso_sn_daily` |
| astronomical (computed) | 4 | daily | 1990-01-01 to 2026-10-18 | `astro_moon_illum`, `astro_moon_age`, `astro_daylight_nyc`, `astro_sad_nyc` |
| geophysical | 4 | daily | 1995-01-01 to 2026-10-17 | `usgs_world_m5_count`, `usgs_world_m6_count`, `usgs_world_max_mag`, `usgs_ca_m3_count` |
| attention | 38 | daily | Wikipedia 2015-07-01 to 2026-10-15; SF Fed 1980- | 37 `wiki_*` topics (2 of them placebos), `sffed_news_sentiment` |
| health | 5 | weekly/daily | flu 1997-; COVID 2020-2023; EMV 1985- | `delphi_ili_wili`, `delphi_ili_wili_first`, `nyt_covid_us_cases_7d`, `nyt_covid_us_deaths_7d`, `emv_infectious_daily` |
| energy | 2 | weekly/daily | gasoline 1991-; electricity 2015-07- | `eia_gasoline_supplied_wk`, `eia930_us48_demand_daily` |
| other | 4 | daily | GPR 1985-; TSA 2019- | `gpr_daily`, `gpr_daily_threats`, `gpr_daily_acts`, `tsa_checkpoint_daily` |
| population | 17 | monthly (2), annual (15) | 1948/1959-; annual 2000-2025 (world 1960-) | `fred_popthm`, `fred_cnp16ov`, `wb_pop_world`, `census_pop_{northeast,midwest,south,west}`, `census_pop_{ca,tx,fl,ny,pa,il,oh,ga,nc,mi}` |

59 series are daily, 3 weekly, 2 monthly and 15 annual.

**Shared raw folders.** `raw/fred/` and `raw/eia/` also hold the markets agent's downloads. Mine are `raw/fred/POPTHM.csv` and `raw/fred/CNP16OV.csv`, and `raw/eia/Region_US48.xlsx`, `raw/eia/WGFUPUS2_leaf.html` and `raw/eia/WGFUPUS2w.xls`. Every other raw folder listed below is mine alone.

## Conventions used (read before testing)

- **Dates.**
  - Daily: the calendar day the value describes. UT/UTC days for GFZ, SILSO, USGS and Wikipedia; Eastern-time days for EIA-930; local reporting days for TSA, NYT, GPR, SF Fed and EMV.
  - Weekly: the **end** of the week (MMWR Saturday for flu, Friday for EIA gasoline), as in the spec's jobless-claims example.
  - Monthly (FRED): the **first** day of the reference month, the FRED convention, so `known_after_days` is counted from the 1st, not from month end.
  - Annual population: July 1 of the year, the Census and UN mid-year reference date.
- **Counts carry explicit zeros.** USGS counts have a row for every day, with 0 on quiet days. They are observations, not gaps.
- **Missing observations are left out**, never filled: GFZ and SILSO `-1`, the CNP16OV blank for Oct 2025, F10.7 instrument outages.
- **`known_after_days`** is a bound on when the value was public by the 4 pm New York close. Where the delay varies I used the largest normal delay, not the typical one. The basis for each value is in the series' notes and in the table at the end.
- **Revisions.** Many of these series are stored as the latest revision. The notes say so wherever a value was not what was public at the time. The one fully first-release series is `delphi_ili_wili_first`.

## Sources

### GFZ Potsdam: Kp, ap, Ap, SN, F10.7 (`raw/gfz/`)
- File: `https://kp.gfz.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt`. The old `kp.gfz-potsdam.de` URL redirects there.
- Licence: CC BY 4.0, except the sunspot numbers inside it, which are CC BY-NC 4.0 (SILSO). Cite Matzka et al. (2021), Space Weather, doi:10.1029/2020SW002641.
- Series:
  - `gfz_ap`: daily Ap. I checked that Ap equals the mean of the eight 3-hour ap values, within rounding, on every day.
  - `gfz_kp_max`: maximum 3-hour Kp of the day.
  - `gfz_storm_ap30`: 0/1 storm day, Ap ≥ 30 (NOAA "minor storm or worse"). There are 462 storm days since 2000, 4.7% of days; Kp_max ≥ 5 would flag about twice as many.
  - `gfz_f107_obs`: observed F10.7 flux. GFZ recommends the observed value rather than the 1 AU adjusted one.
- Checks:
  - `-1` sentinels removed.
  - The UT day in progress at download is excluded; at 00:38 UTC on 10-19 the 2026-10-18 day was complete.
  - Kp is definitive through 2026-09-30 and nowcast after (flag D=0). SN is definitive through 2026-03-31.
  - F10.7 has 17 gaps of 2-3 days since 2000.
  - Six F10.7 days exceed 500 sfu (2001-04-06, 2001-12-28, 2003-11-04, 2005-09-09, 2006-12-06, 2011-03-07, max 938.6). These are solar radio bursts during the noon measurement on major-flare days, real and kept as published.
- Literature: Krivelyova & Robotti (2003) find lower stock returns in the days after geomagnetic storms.

### SILSO daily sunspot number (`raw/silso/`)
- File: `https://www.sidc.be/SILSO/INFO/sndtotcsv.php` (SN_d_tot_V2.0). Licence CC BY-NC 4.0, WDC-SILSO, Royal Observatory of Belgium.
- Series: `silso_sn_daily`, transform `diff` because it can be 0.
- Cross-check against GFZ's copy: identical on every day from 2000-01-01 to 2026-03-31. The 61 differences from 2026-04 on are GFZ's preliminary values, at most 7 apart.
- Caveats:
  - Values from 2026-07-01 on are provisional.
  - The whole series was rescaled in 2015 (v2.0), a scale change that is harmless for correlation.
  - Values before 1849 are sparse.
- **Placebo.** No plausible mechanism links sunspots (or F10.7) to US stock moves days later. A "hit" here measures the test's false-positive rate, so read the other space-weather results against it.

### Computed astronomical series (no download)
- Lunar series:
  - `astro_moon_illum`: illuminated fraction.
  - `astro_moon_age`: days since the last new moon, a sawtooth.
  - Both are evaluated at 16:00 New York time each day. Method: Meeus, *Astronomical Algorithms* (2nd ed.), ch. 49 for new-moon instants and ch. 48 for the phase angle.
  - Validation: five published new moons (2000-01-06, 2008-10-28, 2017-08-21, 2024-01-11, 2025-06-25) match to within 0.5 minutes. Illumination is 0.0000 at a new moon and 1.0000 at a full moon. Meeus's worked example is reproduced to 0.0015.
- Daylight series:
  - `astro_daylight_nyc`: NOAA sunrise equation at 40.7128N 74.0060W with -0.833 deg altitude. It matches published NYC solstice day lengths (15h05m and 9h15m) to within a minute.
  - `astro_sad_nyc`: the Kamstra-Kramer-Levi (2003) SAD variable, night hours minus 12 from 21 Sep to 20 Mar and 0 otherwise. It is slightly negative for a few days next to each equinox because refraction lengthens the day.
- Timing: deterministic and known years in advance, so `known_after_days` = 0.
- These are seasonality or calendar regressors, not forecasts. Lunar phase is effectively a second placebo: Yuan, Zheng & Zhu (2006) report a lunar effect, but a hit at an offset unrelated to the 29.5-day cycle is noise. Kamstra et al. stress the change in daylight, so `diff` may suit `astro_daylight_nyc` better than `level`.

### USGS ComCat earthquakes (`raw/usgs/`)
- Query: `https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&eventtype=earthquake&...`, run by calendar year for 1995 to 2026-10-17. Each year was checked first with the `count` endpoint (all under the 20,000 cap), and the rows saved equal the counted events every year. Public domain (USGS).
- Downloads:
  - World M≥4.5: 209,593 events, 122 of them still "automatic" (unreviewed) near the end.
  - California box (32-42N, 125-114W), M≥3.0: 15,433 events.
  - `eventtype=earthquake` excludes explosions, quarry blasts and nuclear tests. One event returned by the M≥4.5 query had a preferred magnitude of 3.38 (nc21364840, 2004-05-17) and was dropped.
- Series (UTC days):
  - `usgs_world_m5_count` and `usgs_world_m6_count`.
  - `usgs_world_max_mag`: every day has at least one M≥4.5 event, so it is defined daily.
  - `usgs_ca_m3_count`: M3 is well above regional completeness. A CONUS box was avoided because the 2014-16 Oklahoma induced seismicity would dominate it.
- Sanity: the largest days are 2011-03-11 and 2004-12-26 (M9.1), 2025-07-29 and 2010-02-27 (M8.8), and 2012-04-11 and 2005-03-28 (M8.6). The busiest California days are 2019-07-06 (Ridgecrest, 605 events), 2010-04-05 (El Mayor-Cucapah) and 1999-10-16 (Hector Mine).
- Caveat: the catalog is revised (magnitudes reviewed, events added or removed), so counts near a threshold differ slightly from real time.

### Wikipedia page views (`raw/wikimedia/`)
- API: Wikimedia REST, `.../pageviews/per-article/en.wikipedia/all-access/user/<title>/daily/20150701/20261017`, called with a descriptive User-Agent that includes a contact address. Data licence CC0. Redirects were resolved with the MediaWiki API.
- **Each series is the canonical article plus all its redirects** (976 redirect titles across the 37 topics). That keeps the pre-rename history of pages moved during the window:
  - Federal Reserve: titled "Federal Reserve System" until Oct 2018.
  - S&P 500: titled "S&P 500 Index" until Jan 2021.
  - Mortgage: the article was "Mortgage loan" until Nov 2023.
  - Wall Street crash of 1929: retitled from "...Crash..." in Nov 2024.
  - Requested titles that are redirects map to their targets: Bear_market → Market trend, Hurricane → Tropical cyclone.
  - After summing, no topic is missing a day, and every value is > 0, so `logdiff` is safe.
- Topics:
  - Economy: Recession, Great Recession, Inflation, Stagflation, Unemployment, Unemployment benefits, Layoff, Bankruptcy, Foreclosure, Mortgage, Interest rate, Federal Reserve, Quantitative easing, Yield curve, Tariff, US debt ceiling, Default (finance), Bank run.
  - Markets: Stock market, Stock market crash, Market trend (bear/bull market), Wall Street crash of 1929, Black Monday (1987), DJIA, S&P 500, VIX, Short (finance), Gold as an investment, Price of oil, Bitcoin.
  - Hazards: Tropical cyclone, Earthquake, Pandemic, Influenza, Terrorism.
  - **Placebo controls: Cat, Philosophy.** These share the attention series' artifacts (weekday cycle, platform drift, bot spikes), so a finance topic should beat them, not just beat zero.
- Level-break screen: every break above 4x between consecutive 30-day medians lines up with a real event, except the one warned below. The real ones are the 2020-03 COVID crash, Harvey/Irma 2017-08, the 2023-06 debt-ceiling resolution, the 2025-01 tariffs and a 2026-03 oil spike. For the oil spike I checked that desktop, mobile-web and app views all rose together.
- **Warning: `wiki_mortgage`, 2024-05-07 to about early 2025.** Views rose 5-10x, almost entirely on mobile-web (desktop under 2x, app flat), and obscure redirects such as "Islamic mortgage" rose with it. That points to a search-referral change or automated traffic, not attention to the mortgage market.
- General caveats:
  - Strong weekday cycle and a long downward drift (search engines answering directly).
  - The "automated" agent class was split out of "user" in 2020, a possible level shift.
  - SHORT: starts 2015-07; the pre-2015 pagecount dumps are not comparable.
- Timing: hourly dump files are normally public about 2 hours after each hour, so a UTC day is complete by about 02:30 UTC the next day: `known_after_days` = 1. When I downloaded, the pipeline was stalled (the last hourly file was 2026-10-16 19:00 UTC), and the REST API's daily data ended 2026-10-15. Stalls of a few days happen.
- Literature: Moat et al. (2013), views of financial pages before market moves.

### SF Fed Daily News Sentiment Index (`raw/sffed/`)
- File: `https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx`. Free with citation: Shapiro, Sudhof & Wilson (2022), "Measuring News Sentiment".
- Series: `sffed_news_sentiment`.
- The published series is a trailing, geometrically weighted average (lag-1 autocorrelation 0.997), so the transform is `diff`.
- Updated weekly, on Mondays through the previous Sunday, so `known_after_days` = 8.
- Daily values are revised as the article sample updates, and the series was recomputed from Factiva in April 2021.
- It has 18 gaps of 2-3 days, all in 1980.
- It stands in for GDELT news tone (see "Skipped").

### CDC ILINet via Delphi Epidata (`raw/delphi/`)
- API: `https://api.delphi.cmu.edu/epidata/fluview/?regions=nat&epiweeks=199740-202652`, free, no key. CDC surveillance data are public domain; cite Delphi Epidata (CMU).
- Series:
  - `delphi_ili_wili`: latest revision, 1997w40 to 2026w40.
  - `delphi_ili_wili_first`: `&lag=0`, each week's value as first published, 2003w47 to 2026w40.
- Dates are MMWR epiweeks (Sun-Sat; week 1 contains Jan 4) converted to the week-ending Saturday. The conversion is checked against Delphi (202640 → 2026-10-10; 2020 has 53 weeks).
- **Publication delay, measured from Delphi's `release_date` on the 1,074 first-release issues:** 6 days (the Friday) in 94% of weeks, 8-12 days in holiday weeks, maximum 12. Hence `known_after_days` = 12.
- Gaps: summer weeks 21-39 are missing in 1998-2002 in the latest series, and in 2004-2008 in the first-release series (no off-season reporting then).
- Transform `logdiff`, the epidemic growth rate. It works at weekly spacing; a day-of-year anomaly on Saturday-dated weekly data would misalign by about a day per year. The growth rate is itself seasonal, so judge hits against that seasonal pattern.

### NYT COVID-19 (`raw/nyt_covid/`)
- File: `https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv`, cumulative US cases and deaths by report date, 2020-01-21 to 2023-03-23. Licence CC BY-NC 4.0.
- Series:
  - `nyt_covid_us_cases_7d` and `nyt_covid_us_deaths_7d`: 7-day trailing average of daily new counts, with `logdiff` as the incidence growth rate.
  - Each starts on the first day the 7-day average reached 10 per day (2020-03-03 for cases, 2020-03-17 for deaths). The earlier values, 0.1 to a few per day, are in the raw file; their log changes of ±1.5-2 are tiny-denominator noise.
  - The daily differences include 3 negative correction days each, which the 7-day window absorbs.
- SHORT (about 3 years). Testing regimes changed (home tests from 2022).

### Infectious Disease EMV tracker (`raw/policyuncertainty/`)
- File: `https://www.policyuncertainty.com/media/All_Infectious_EMV_Data.csv` (Baker, Bloom, Davis, Kost, Sammon & Viratyosin). Free with citation.
- Series: `emv_infectious_daily`, 1985-01-01 to 2026-10-17. Today's possibly partial row was dropped.
- Zero on 54% of days, dominated by 2020-2022.
- It counts articles about stock-market volatility, so it partly reflects same-day market moves.

### Caldara-Iacoviello Geopolitical Risk, daily (`raw/gpr/`)
- File: `https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.dta`; the `.xls` copy is also kept. Free with citation: Caldara & Iacoviello (2022), AER.
- Parsing: the `.dta` (Stata format 118) is read by `scripts/data-other/dta118.py`, which decodes bytes only. The Stata date column was checked against the DAY column on every row.
- **Normalized, exactly:**
  - GPRD → `gpr_daily`
  - GPRD_THREAT → `gpr_daily_threats`
  - GPRD_ACT → `gpr_daily_acts`
- **Not normalized:** N10D, GPRD_MA7, GPRD_MA30, event, and all monthly or country-level GPR (`data_gpr_export` was not downloaded). `raw/gpr` is left unmodified; data-politics keeps its country GPR in its own `raw/gpr_country/`.
- Timing: the daily file is "updated every Monday" (site text), so `known_after_days` = 8. The update day itself (2026-10-13) was dropped as possibly partial. Dated vintages (`data_gpr_daily_recent_YYYYMMDD`) exist if anyone wants a real-time reconstruction.

### EIA weekly gasoline product supplied (`raw/eia/`)
- Page: `https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WGFUPUS2&f=W`, the no-key history page, parsed as HTML. The `.xls` copy is kept. Public domain. The EIA API needs a free key and was not used.
- Series: `eia_gasoline_supplied_wk`, thousand barrels per day, week-ending Fridays, 1991-02-08 to 2026-10-09. There are no gaps, and every date is checked to be a Friday.
- Timing: the WPSR is released Wednesday at 10:30 ET (5 days after the Friday), Thursday in holiday weeks (6). The page showed a Thursday release on 2026-10-15. So `known_after_days` = 6. EIA's June 2022 systems outage delayed a few weeks further.
- Seasonal: summer peak; COVID low of 5,065 in April 2020.

### EIA-930 US48 electricity demand (`raw/eia/Region_US48.xlsx`)
- File: `https://www.eia.gov/electricity/gridmonitor/knownissues/xls/Region_US48.xlsx` (37 MB). Public domain. Parsed by `scripts/data-other/xlsx_read.py` (zipfile plus iterparse).
- Series: `eia930_us48_demand_daily`, the sum of hourly demand over the **Eastern-time** day.
- Checks:
  - My hourly sums equal EIA's published daily sheet on all 4,126 days.
  - Hours per day: 23 on 11 days and 25 on 11 days (the DST changes), otherwise 24. No hours are missing.
- SHORT: starts 2015-07-01.
- Transform `anomaly` (same day of year). It is weather-driven, so it overlaps the weather series, and its strong weekday cycle survives the anomaly.
- Data are preliminary and balancing authorities resubmit. `known_after_days` = 1.

### TSA checkpoint numbers (`raw/tsa/`)
- Pages: `https://www.tsa.gov/travel/passenger-volumes` plus `/2019` to `/2025`, parsed as HTML. Public domain.
- Series: `tsa_checkpoint_daily`, 2019-01-01 to 2026-10-15, no gaps, with a conflict check across pages.
- SHORT (2019-).
- Posted on weekday mornings, with Friday and Saturday values appearing on Monday, so `known_after_days` = 4 (a Monday holiday adds a day).

### Population (`raw/fred/`, `raw/census/`, `raw/worldbank/`)
- `fred_popthm`: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=POPTHM`, from the BEA Personal Income and Outlays release.
  - Monthly, 1959-01 to 2026-08.
  - Published about 4 weeks after month end, about 56-60 days after FRED's first-of-month date: `known_after_days` = 65. Shutdowns pushed it later (the Sep 2025 data came out in Dec 2025).
- `fred_cnp16ov`: from the BLS Employment Situation, usually the first Friday of the next month.
  - `known_after_days` = 42 from the 1st.
  - Oct 2025 is missing (no CPS during the shutdown).
  - **The CPS population controls enter each January without revising earlier months, so `logdiff` shows methodology breaks every January** (e.g. 2026-01: 274,816 → 274,676).
- `wb_pop_world`: World Bank WDI `SP.POP.TOTL`, the world aggregate, 1960-2025, dated July 1. Licence CC BY 4.0. It is revised with every UN WPP revision, so `known_after_days` = 400 as a first-publication bound.
- Census (public domain), July 1 estimates:
  - Sources: the 2000-2010 intercensal file (`st-est00int-alldata.csv`) for 2000-09; the 2010-2020 intercensal file (`sc-est2020int-alldata6.csv`, released Sep 2025, summed over race and age) for 2010-19; Vintage 2025 (`NST-EST2025-ALLDATA.csv`, released 2026-01-27) for 2020-25.
  - The two intercensal files' July 2010 values differ by at most 0.20% for any state. US growth is smooth across both joins (2009 0.88%, 2010 0.85%; 2019 0.52%, 2020 0.41%).
  - Normalized: the 4 Census regions (sums of their states) and the 10 most populous states. All 51 states plus DC are in `raw/census` if wanted.
  - `known_after_days` = 210, the first release of each July-1 estimate, usually in late December. Stored values are later revisions; the per-vintage directories at `www2.census.gov/programs-surveys/popest/datasets/2000-2001/`, `.../2000-2002/`, ... would allow a real-time reconstruction.
- **Annual series cannot support daily lead-lag tests.** About 26 points, aligned to trading days, form a step function that moves once a year. They are included only because the user asked about population growth.
- The monthly series are very smooth and usable only at the 21- and 63-day move lengths.

## Skipped or not available

| item | why |
|---|---|
| Google Trends | Handed to the dedicated `data-trends` agent at the coordinator's request. **I made no Google Trends requests and hold no Trends data.** |
| GDELT news tone | No small, stable daily file. The DOC API is query-dependent, reprocessed and covers only 2017 on. Replaced by the SF Fed sentiment index. |
| Daily Economic Policy Uncertainty | Covered by the markets agent (FRED `USEPUINDXD`). The coordinator said to skip it. |
| Country-level monthly GPR (GPRC_*), state/country/Twitter/UK EPU, Partisan Conflict | Covered by `data-politics` in its own `raw/gpr_country/` and `raw/policyuncertainty_politics/`. I normalized none of these. |
| Monthly global GPR (GPR/GPRT/GPRA, `data_gpr_export`) | Mine by assignment, but not taken: it aggregates the same article counts as the daily `gpr_daily*` series, which supersede it at these move lengths. |
| Wikipedia before 2015-07 | Only the legacy hourly pagecount dumps, which are huge and use a different definition (no agent split). |
| EIA API v2 | Needs a free key; the no-key history page and grid-monitor file were used instead. |
| JHU CSSE COVID | Redundant with NYT. |
| FEMA disaster declarations | Free (OpenFEMA, no key), but marginal and overlapping the weather agent's hurricanes. Not taken. |
| Calendar effects (holidays, DST weekends, turn of month) | Computable from the calendar; not downloaded, as instructed. |
| All 51 states as separate annual series | Kept in `raw/census`; only regions and the top 10 states were normalized to avoid 50 near-duplicate annual series. |
| Real-time (first-vintage) Census estimates | The per-vintage files exist; not reconstructed (annual series, little value to the test). |

## Data-quality checks run on every series

All in `scripts/data-other/common.py` and `build_catalog.py`:
- strictly ascending, unique dates;
- finite values;
- `logdiff` series strictly positive;
- explicit plausible-range assertions per series, which caught the F10.7 flare days and the sub-threshold USGS event;
- a sentinel scan (-9999, -999, 9999, ...). The only flag was a genuine 9,999-view day for Influenza on 2020-02-14;
- a gap report against the expected spacing, and first/last rows printed;
- catalog `first`/`last`/`rows` computed from the written files;
- an ID-clash check against the other agents' catalogs, with none found.

Every series has at least 974 observations in the census window (2005-01-03 to 2025-06-27), except the annual ones (20) and the monthly ones (245).

## Notes for whoever runs the test

1. **Timing.** Use `date + known_after_days`.
   - Already conservative: flu 12 (maximum observed), GPR and SF Fed 8 (weekly updates), TSA 4 (weekends), EIA gasoline 6 (holiday weeks).
   - Monthly and annual lags are counted from the first-of-month or July-1 date.
2. **Revision look-ahead** is present in the population, USGS, latest flu, EIA-930, GPR and SF Fed series (all stored as latest revisions). For flu, prefer `delphi_ili_wili_first`.
3. **Seasonality and calendar effects:**
   - `astro_*` series are pure calendar regressors.
   - Electricity, gasoline, flu and TSA are seasonal with weekday or holiday cycles.
   - Wikipedia series have a strong weekday cycle.
   - Any hit should be compared against a seasonal or day-of-week baseline.
4. **Placebos.** Use `silso_sn_daily`, `gfz_f107_obs`, `astro_moon_*` and `wiki_cat`/`wiki_philosophy` to calibrate false positives: they should not "lead" stocks.
5. **Short series:**
   - Wikipedia and EIA-930 start in 2015.
   - TSA starts in 2019.
   - COVID covers only 2020-2023.
   - These offer far fewer independent correlation periods.
6. **Level breaks to beware of:**
   - `fred_cnp16ov` every January.
   - `wiki_mortgage` from 2024-05.
   - COVID-era shifts in TSA, gasoline and every attention series.
