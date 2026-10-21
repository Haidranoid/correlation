# SOURCES: data-prosperity (category `prosperity`)

The user asked to include "prosperity statistics for countries down to the county level if possible. Like max min median income, etc."
This catalog holds US county, US state and country prosperity series: unemployment, income, wages, home values, inequality.

- Catalog: `catalog_data-prosperity.csv` (342 series).
- QC report: `qc_data-prosperity.txt`.
- Scripts: `scripts/data-prosperity/`. Run with `python3 -I`; numpy is the only non-stdlib dependency. Order:
  1. `lau_parse.py`, then `lau_build.py`;
  2. `phci_build.py`, `zillow_build.py`, `oecd_build.py`;
  3. `qcew_fetch.py 1990 2026`, then `qcew_build.py`;
  4. `bea_build.py`, `saipe_build.py`, `wdi_build.py`, `wid_fetch.py`, `wid_build.py`;
  5. `plib.py build` merges `parts/*.csv` into the catalog.
- Raw downloads: `raw/<source>/`, unmodified, each folder with a `SHA256SUMS.txt`. 1.4 GB in total: QCEW 828 MB, LAUS 349 MB, WID 162 MB, the rest under 35 MB each.

Overlap was checked against the other catalogs:
- National US macro (UNRATE, PAYEMS, CPI, housing starts, ...) belongs to data-markets.
- Population (US, the 10 largest states, regions, world) belongs to data-other.
- County population and migration belong to data-migration.

None of those are repeated here. The county series are unemployment, income, wages and home values, not population.

## What is here (342 series)

| Group | Frequency | Count | Span | Testable? |
|---|---|---|---|---|
| County distributions: LAUS unemployment and employment | monthly | 11 | 1991/1993 .. 2026-08 | yes (21/63-day moves) |
| County distributions: Zillow home values and rents | monthly | 10 | 1998/2000 .. 2026-09 (rents 2016+) | yes |
| 50 selected counties: unemployment rate (LAUS, causal SA) | monthly | 50 | 1993 .. 2026-08 | yes |
| 50 selected counties: home value index (Zillow ZHVI, causal SA) | monthly | 50 | 1999 (some 2001-2008) .. 2026-09 | yes |
| 51 states + DC: unemployment rate (BLS SA) | monthly | 51 | 1976 .. 2026-08 | yes |
| State distributions: unemployment rate | monthly | 5 | 1976 .. 2026-08 | yes |
| 50 states: Philadelphia Fed coincident index | monthly | 50 | 1979 .. 2026-08 | yes |
| State distributions: coincident index growth | monthly | 4 | 1979 .. 2026-08 | yes |
| 10 countries: OECD harmonised unemployment rate | monthly | 10 | 1990 .. 2026-06/09 | yes |
| 20-country OECD panel: unemployment distribution | monthly | 6 | 1997 .. 2026-08 | yes |
| County distributions: QCEW weekly wage and employment | quarterly | 6 | 1991/1993 .. 2026Q1 | yes (63-day moves) |
| State distributions: BEA quarterly personal income | quarterly | 8 | per capita 2010Q1 ..; growth 1950 .. 2026Q2 | yes |
| 50 selected counties: per capita personal income (BEA) | annual | 50 | 1969 .. 2024 | essentially no |
| County distributions: BEA per capita income (6), county GDP growth (2) | annual | 8 | 1969 / 2002 .. 2024 | essentially no |
| County distributions: SAIPE median household income and poverty | annual | 8 | 1989 .. 2024 (gaps 1990-92, 94, 96) | essentially no |
| Countries: World Bank WDI | annual | 8 | 1963-1992 .. 2024/2025 | essentially no |
| Countries: WID income and wealth shares | annual | 7 | 1913/1980 .. 2021-2023 | essentially no |

- Monthly 247 + quarterly 14 = 261 testable.
- Annual 81. The test only lets a series into a 1260-day period when at least 10 of its trading days carry a non-zero move. An annual series has about 5 in such a period, so annual series can only enter the full-history period. They are flagged "ANNUAL: essentially untestable" in their notes and kept because the user asked for income "down to the county level".

**Count note.** The brief asked for about 150-300 series. Its explicit lists alone come to about 250:
- 50 counties x 3 series;
- 50 coincident indexes plus 51 state unemployment rates;
- the requested median/max/min/p10/p90/p90-p10 sets.

That is why the total is 342. Ways to trim if needed:
- the 50 annual county incomes;
- the cross-country min/max-style statistics;
- `lau_st_ur_median`, which nearly duplicates data-markets' UNRATE.

## Conventions

- **Dates.** The last day of the reference period: month end, quarter end, or December 31 for annual data. `known_after_days` counts from that date, as data-markets does for its FRED macro. (data-other's FRED population series use the first of the month.)
- **`known_after_days`.** One number per series, so each is the maximum observed publication lag over the stock window (2002-2025), not the typical lag. The evidence is listed per source below, and the files used for it are kept in `raw/`. As a result, most monthly series carry the January benchmark delay or the September-2013 shutdown delay for every month.
- **Revisions.** Every value is the current (October 2026) vintage, not the first print. LAUS, QCEW, BEA, Zillow, the coincident indexes, WDI and WID all revise history. Several series did not exist in real time before a certain date (they are back-casts), and their notes say so:
  - coincident indexes before June 2005;
  - county GDP before December 2019;
  - WID's US distributional accounts before 2016-17;
  - Zillow's whole ZHVI history, recomputed after the 2019 method change;
  - ZORI before 2020.

  A value from before a series' first publication is look-ahead no matter what `known_after_days` says.
- **Seasonal adjustment (causal).** County LAUS, Zillow raw ZHVI and QCEW wages are published not seasonally adjusted (NSA). I adjust them with `scripts/data-prosperity/sa.py`:
  - The seasonal factor for month m of year y is the median of the same month's deviation from a centred 2x12 moving average (2x4 for quarters) over years y-8..y-1, using at least 3 years.
  - It is additive for rates and works on logs for levels.
  - It uses only data at least 6 months (2 quarters) older than the month itself, so the adjusted value can be computed on the day the month is published.
  - The factors are deliberately not renormalised within a year, because that would mix in factors built from later data.
  - Self-test: perturbing the future leaves earlier adjusted values unchanged.
  - Validation against BLS's own seasonally adjusted state rates (50 states + DC, 2005-2025). The correlation of monthly changes is 0.947 for my adjustment vs 0.854 for raw NSA, and 0.57 vs 0.35 excluding 2020-21H1. BLS's series is also smoothed, so it is less noisy.
  - Median beat mean, and K=8 years beat K=3 and K=5 (0.598 vs 0.529 and 0.574).
- **Year-over-year statistics.** Some statistics use NSA 12-month (4-quarter) changes instead, which need no adjustment. Those names say "year-over-year" and use `diff`.
- **Transforms.**
  - Rates, shares and ratios: `diff`.
  - Income, wages, home values, index levels: `logdiff`. All logdiff series are strictly positive; the test drops values of 0 or less.
  - Monthly statistics that are already one-month changes (median monthly growth, "share of counties whose X rose this month"): `level`. This departs from the brief's `diff`, which would make the move a change in a change.
  - The same kind of statistic at quarterly or annual frequency uses `diff`. The test's `level` needs at least 10 observations in the trailing 252 trading days, so a quarterly or annual `level` series would never be live.
- **Max/min floors.** Unrestricted county maxima and minima are tiny-county noise, so they use a size floor:
  - LAUS: counties whose trailing 12-month LAUS labor force is at least 50,000 (about 590 counties in 2025; causal).
  - Zillow growth: the same 50,000 floor, using the 12 LAUS months ending 3 months earlier. LAUS month t-3 is public about 25 days before Zillow month t ends, so the membership never uses data that wasn't out yet.
  - BEA and SAIPE income: population at least 100,000 (about 600 counties).

  Medians, p10 and p90 are unweighted over all counties, as asked.
- **Panels.** A fixed membership keeps distribution statistics from jumping when the coverage changes:
  - Zillow: counties with a value in January 2000 (1,194).
  - ZORI: counties with data in January 2016.
  - OECD: 20 countries with monthly survey data.
  - WDI: 40 economies (39 for GNI).
  - WID: 19 economies with good data.
- **Connecticut.** Connecticut replaced its 8 counties with 9 planning regions in 2022. BLS LAUS now publishes only the regions, with history back to 1990. Zillow still uses the counties. BEA has counties to 2023 and regions for 2024; SAIPE has regions from 2022. "Fairfield County, CT" (finance) is represented as follows, never spliced:
  - LAUS: Western Connecticut Planning Region (09190: Greenwich, Stamford, Norwalk, Danbury), id `lau_ur_ct_western_ct`.
  - Zillow: Fairfield County (09001), id `zillow_zhvi_ct_fairfield`.
  - BEA: Fairfield County (09001), ending 2023, id `bea_pcpi_ct_fairfield`.
- **Missing October 2025.** BLS never collected it (2025 shutdown), so it is missing in every LAUS-based series and in the OECD panel (US). It was not filled.

## Sources

### 1. BLS Local Area Unemployment Statistics (LAUS): 117 series, monthly
- **Files.** From `https://download.bls.gov/pub/time.series/la/` (BLS needs a descriptive User-Agent; sent with the contact email set in `C2_CONTACT`):
  - `la.data.64.County` (337 MB, all counties NSA, 1990-01..2026-08);
  - `la.data.3.AllStatesS` (states SA, 1976..);
  - `la.data.2.AllStatesU` (states NSA, only for validating the adjustment);
  - `la.series`, `la.area`, the code tables and `la.txt`.

  All saved in `raw/bls_laus/`. Public domain (US government).
- **Parsing.** Period M13 (annual average) is dropped. "-" values (footnotes N, and X for the Oct-2025 shutdown) become missing. Puerto Rico is excluded, leaving 3,147 counties (50 states + DC).
- **County distribution series (11).** All on the causal-SA rate unless marked NSA:
  - `lau_cty_ur_median`, `_p10`, `_p90`, `_p90_p10_ratio`;
  - `lau_cty_ur_sd`: cross-county standard deviation (dispersion);
  - `lau_cty_ur_max_lf50k`, `_min_lf50k`;
  - `lau_cty_ur_share_rose_mom`: share of counties whose SA rate rose this month (`level`);
  - `lau_cty_ur_share_rose_yoy`: NSA, rate above a year earlier;
  - `lau_cty_emp_yoy_median` and `lau_cty_emp_share_grew_yoy`: household-basis employment, NSA year-over-year.
- **Selected counties (50).** `lau_ur_<state>_<county>`, the causal-SA rate:
  - The 31 largest by 2025 labor force except Queens and Suffolk NY, which repeat the NYC labor market (29 counties).
  - Industry-tied counties: Harris TX (energy), Midland TX (oil), Santa Clara CA and King WA (tech), Wayne and Oakland MI (autos), New York NY and Western CT/Fairfield (finance), Polk IA (farming, insurance), Clark NV (gaming), Orange FL and Honolulu HI (tourism), Mecklenburg NC (banking), Fairfax VA and DC (government), Shelby TN (logistics), Kern CA and Oklahoma County OK (oil), Williams ND (Bakken shale), Elkhart IN (RVs), Lake IN (steel), Snohomish WA (Boeing), Sedgwick KS (aircraft), Benton AR (Walmart), plus Maricopa, Cook, Los Angeles and others.
  - The list with reasons is in `scripts/data-prosperity/counties.py`.
- **States (51 + 5).**
  - `lau_ur_st_<ab>`: BLS SA rate.
  - `lau_st_ur_median`, `_max`, `_min`, `_sd`, `_share_rose_mom`.
- **Timing.** Release dates come from the BLS archive pages `bls.gov/bls/news-release/metro.htm` and `laus.htm`, parsed into `raw/bls_release_archives/*_lags.tsv`.
  - **Counties.** County data ship with the Metropolitan Area release; the release text says county estimates are available on the LAUS site. Over 2002-2025 the lag after month end was median 32 days, max 54 (Jan 2003; January data wait for the annual benchmark: 47-54 days). The September-2013 data were never released separately: they came with October's on 2013-12-05 (66 days). **`known_after_days` 66.**
  - **States.** Median 20 days, max 48 (Jan 2003), Sep-2013 53 (released 2013-11-22). **`known_after_days` 53.**
  - Later than the stock window: Sep-2025 county data appeared 2025-12-17 (78 days) and Jan-2026 on 2026-04-16 (75 days).
- **Revisions.** The prior month is revised with each release, and the previous 5 years each spring (annual benchmark). County data are model-based (they use CES, QCEW, UI claims and CPS controls).

### 2. Philadelphia Fed state coincident indexes: 54 series, monthly
- **Files.** `{ST}PHCI` from FRED's keyless `https://fred.stlouisfed.org/graph/fredgraph.csv?id=..`, saved in `raw/fred_phci/`. The Philly Fed's own `coincident-revised.xls` is binary xls, which the standard library can't read. DC has no index (FRED `DCPHCI` returns 404), so this covers 50 states.
- **Terms.** FRED marks the series "copyrighted: citation required" (Federal Reserve Bank of Philadelphia).
- **Series.**
  - `phci_<ab>`: index 2007=100, SA, `logdiff`, re-dated to month end.
  - `phci_states_mom_median`, `_p10`, `_p90` and `phci_states_share_rising`: cross-state monthly growth and diffusion, `level`.
- **Timing, verified from ALFRED vintages.** ALFRED's `alfredgraph.csv?id=CAPHCI&vintage_date=D` works without a key (script `phci_timing.py`, results in `raw/philfed_coincident_timing/`).
  - Every month from 2008-11 to 2025-06 is in the vintage of month end + 70 days, except January 2015.
  - January data wait for the BLS benchmark. A binary search for the first vintage of each January gives 41-49 days in 2009-2014, then 56-75 days from 2015 (max: Jan-2015 out 2015-04-16 = 75). Sep-2013 (shutdown): 57.
  - ALFRED's release list (`release_dates_109.txt`, from 2005-06-02) bounds 2005-2008 below 75.
  - **`known_after_days` 76**: the verified maximum plus 1 day, because ALFRED's vintage date doesn't give the time of day of the release.
- **Revisions.** The model is re-estimated each month, so all history is revised. First published 2005-06-02; earlier values are back-casts.
- **QC.** April 2020: Nevada -82 log-%, Hawaii -51, Michigan -48. These are genuine (the p10 statistic reaches -41%).

### 3. Zillow: 60 series, monthly
- **Files.** From `https://files.zillowstatic.com/research/public_csvs/`:
  - `zhvi/County_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv`: raw = not smoothed, not SA; mid-tier all homes; 1996-02..2026-09; 3,071 counties.
  - `..._sm_sa_month.csv`: kept only for comparison.
  - `zori/County_zori_uc_sfrcondomfr_sm_month.csv`: rents, 2015-01..

  Saved in `raw/zillow/`.
- **Terms.** zillow.com returns 403 to scripts. Zillow Group's public-data page (zillowgroup.com/developers/api/public-data/real-estate-metrics) says: "All data accessed and downloaded from this page is free for public use by consumers, media, analysts, academics and policymakers, consistent with our published Terms of Use. Proper and clear attribution of all data to Zillow is required." Attribute as "Data provided by Zillow Research".
- **Why raw, not Zillow's SA file.** Raw ZHVI is less smoothed: the lag-1 autocorrelation of its monthly changes is 0.69-0.90, vs 0.87-0.96 for `sm_sa`. Zillow's smoothing and seasonal adjustment may be two-sided (look-ahead). Raw values are adjusted with the causal SA instead.
- **Series.**
  - `zillow_zhvi_<county>`: 50 selected counties, causal-SA ZHVI, `logdiff`. Coverage starts later for Midland TX (2008-08), Williams ND (2008-11), Elkhart IN (2004-08), Lake IN (2003-04) and Sedgwick KS (2001-11). Kings NY is missing 2002-03/04; Williams ND is missing 2011-02.
  - Panel statistics over the 1,194 counties with a value in January 2000:
    - `zillow_cty_zhvi_median` and `_p90_p10_ratio` (levels);
    - `_mom_median`, `_mom_p10`, `_mom_p90`, `_mom_max_lf50k`, `_mom_min_lf50k`, `_share_rose_mom` (monthly growth, `level`);
    - `_yoy_median` (raw 12-month change, `diff`).
  - `zillow_cty_zori_yoy_median`: rents, 2016.. (SHORT).
- **Timing.** Zillow posts around the 16th of the next month: the September 2026 file is dated 2026-10-16 01:51 GMT. Earlier years' schedule was later in the month. **`known_after_days` 35.**
- **Revisions.** Revised every month. The 2019 ZHVI overhaul recomputed all history, and ZORI was launched in 2020 with back-filled history. Treat the history as a reconstruction.

### 4. OECD harmonised monthly unemployment rates: 16 series
- **File.** `https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_LFS@DF_IALFS_UNE_M,1.0/.UNE_LF_M.PT_LF_SUB._Z.Y._T.Y_GE15._Z.M?startPeriod=1990-01`: % of labour force, SA, total, 15+, all countries in one call. Saved as `raw/oecd/oecd_ialfs_une_m.csv`.
- **Terms.** Free under the OECD's open-by-default policy (July 2024): CC BY 4.0 for content from 2024-07-01, OECD terms similar to CC BY before. Cite OECD.
- **Countries (10).** `oecd_ur_<cc>`, the G7 except the US (which data-markets has as UNRATE) plus Korea, Australia, Mexico and Spain. `known_after_days` per national release practice:
  - CA 14, KR 20, AU 25, JP 35, MX 35, DE/FR/IT/ES 40 (Eurostat/ISTAT/Destatis ~30-32 days).
  - GB 85: the ONS figure is a rolling 3-month average that the OECD dates at the middle month. That is public ~75-80 days after the dated month; the OECD's latest GB month is 2026-06 while the others reach 2026-08/09.
- **Panel statistics (6).** `oecd_ur_panel_median`, `_p10`, `_p90`, `_max`, `_min`, `_share_rose_mom`.
  - Panel: AUS AUT BEL CAN CZE DEU DNK ESP FIN FRA IRL ITA JPN KOR MEX NLD POL PRT SWE USA, months where all 20 report (1997-01..2026-08).
  - Left out: 3-month-average or quarterly-based countries (GBR, NOR, CHL, COL, CRI, TUR, HUN, the Baltics, ...), plus CHE and NZL, which have no monthly data. **`known_after_days` 45.**
- **Revisions.** Seasonal factors are re-estimated. Eurostat's monthly figures for quarterly-survey countries (France, Spain, Belgium, Austria, Poland, ...) are model-based and revised when the quarterly survey arrives.

### 5. BLS Quarterly Census of Employment and Wages (QCEW): 6 series, quarterly
- **Download.** Only the `"<year>.q1-q4 10 Total, all industries.csv"` member of each `https://data.bls.gov/cew/data/files/<year>/csv/<year>_qtrly_by_industry.zip` (1990-2026), fetched with HTTP Range requests (`qcew_fetch.py`):
  - read the zip's central directory, fetch the member's compressed bytes, inflate, check the CRC32;
  - saved byte-for-byte in `raw/bls_qcew/<year>_qtrly_industry10_total_all_industries.csv`;
  - byte ranges, CRCs and SHA256 are in `manifest.json`.

  About 4-5 MB per year instead of 70-480 MB per zip; 1990-2026Q1 is 828 MB stored. That total holds every county x ownership for all industries. The full by-industry detail was not stored (it would be about 12 GB).
- **Terms.** Public domain.
- **Rows used.** County total covered, all ownerships (agglvl 70, own 0, industry 10). Statewide-unknown (`xx999`) areas, Puerto Rico and the Virgin Islands are dropped, as are suppressed cells (disclosure N).
- **Connecticut.** QCEW has the 8 counties through 2023 and the 9 planning regions from 2024, never both in one quarter. The regions are missing from the 2024+ year-over-year statistics (no matching 2023 code) and from the wage statistics until they have 3 years of history for the seasonal factors. That is about 9 of 3,140 counties.
- **Series.**
  - `qcew_cty_wage_median`, `_p10`, `_p90` (`logdiff`) and `_p90_p10_ratio` (`diff`): average weekly wage, causal-SA (Q1 carries bonuses).
  - `qcew_cty_emp_yoy_median` and `qcew_cty_emp_share_grew_yoy`: 4-quarter employment change, NSA.
- **Timing.** From `bls.gov/bls/news-release/cewqtr.htm` (`raw/bls_release_archives/cewqtr_lags.tsv`): over 2003-2025 the median is 171 days after quarter end, max 214 (2003Q1). Pre-2003 releases are undocumented. **`known_after_days` 240.**
- **Revisions.** Each quarter is revised in the following releases and finalised with the annual file.

### 6. BEA regional accounts: 66 series
- **Files.** From `https://apps.bea.gov/regional/zip/`: `SQINC.zip` (state quarterly personal income), `CAINC1.zip` (county personal income summary, 1969-2024) and `CAGDP1.zip` (county GDP summary, 2001-2024). Saved in `raw/bea_regional/`.
  - The brief's `SQINC1.zip` no longer exists: it now ships inside `SQINC.zip`.
  - BEA's site showed a "lapse in appropriations" notice (October 2026), but the zips were current: SQINC updated 2026-09-30, CAINC1 and CAGDP1 on 2026-02-05.
- **Terms.** Public domain; cite BEA.
- **SQINC1, quarterly.**
  - `bea_st_pcpi_q_median`, `_p10`, `_p90`, `_min`, `_max` (`logdiff`) and `_p90_p10_ratio` (`diff`): per capita personal income across 50 states + DC. **These start at 2010Q1** because BEA now publishes quarterly state population, and so per capita income, only from 2010.
  - `bea_st_pi_q_growth_median` and `bea_st_pi_q_share_rose`: total personal income growth across states, 1950.., `diff`.
  - Timing: release dates from ALFRED's keyless release-date list (rid 110, `raw/bea_release_timing/release_dates_110.txt`), taking each quarter's first release at least 60 days after its end. Over 2003Q3-2025Q1 the median is 84 days; the max is 119 (2003Q3, released 2004-01-27), with 118 for 2003Q4 and 108 for 2009Q2 (comprehensive revision). **`known_after_days` 120.**
- **CAINC1, annual.**
  - `bea_cty_pcpi_median`, `_p10`, `_p90`, `_min_pop100k`, `_max_pop100k` (`logdiff`) and `_p90_p10_ratio` (`diff`): across about 3,100 county-equivalents. BEA merges some Virginia independent cities into their counties, e.g. Fairfax = 51919.
  - `bea_pcpi_<county>`: the 50 selected counties. Fairfield CT is 09001 through 2023.
  - Timing: until 2012, county income was released in April of t+2 (2010 data on 2012-04-25, 481 days). From November 2012 it moved to mid-November of t+1 (~320 days; ALFRED rid 175 dates 2014-2024), and 2024 data came 2026-02-05 (401 days). **`known_after_days` 510.**
- **CAGDP1, annual.** `bea_cty_rgdp_growth_median` and `bea_cty_rgdp_share_grew`: real GDP growth, chained 2017 $.
  - **BACK-CAST:** county GDP was first published 2019-12-12, all of 2001-2018 at once (ALFRED rid 397; a 2012-2015 prototype appeared in Dec 2018). Values before 2018 were not public at date + `known_after_days`.
  - **`known_after_days` 400** (Dec t+1 = 339-346 days; 2024 data on 2026-02-04).

### 7. Census SAIPE (Small Area Income and Poverty Estimates): 8 series, annual
- **Files.** `https://www2.census.gov/programs-surveys/saipe/datasets/<year>/<year>-state-and-county/est<yy>all.(dat|txt)` for 1989, 1993, 1995 and 1997-2024. Saved in `raw/census_saipe/`. Public domain.
- **Census API not used.** `api.census.gov` now answers every request with "A valid key must be included" (available with a free key; not registered), so the flat files were used.
- **Series.**
  - `saipe_cty_mhi_median`, `_p10`, `_p90`, `_min_pop100k`, `_max_pop100k` (`logdiff`) and `_p90_p10_ratio` (`diff`): county median household income.
  - `saipe_cty_poverty_median` and `_p90`: poverty rate, all ages, `diff`.
- **Timing.**
  - Each file's lines end with its creation date: 2000 data on 2003-10-09; 2001 and 2002 data on 2004-10-29; 2003 data revised 2006-09-21 (the current file); 2004 data 2006-12-06; 2005 data 2007-12-10; 2006-07 data 2008-11-18.
  - From 2008 on, about 10-12 months after the year: file stamps fall in October-December of t+1, and ALFRED rid 346 release dates since 2015 fall between November 30 and December 17 of t+1. 2024 data were released 2026-01-27.
  - **`known_after_days` 1,100**, so the early-2000s lags are covered. Very conservative after 2008.
  - The brief's "about 17 months" understates the 2000-2006 lags.
- **Breaks.** Gaps in 1990-92, 1994 and 1996 (no estimates). The inputs switched from CPS to ACS in 2005.

### 8. World Bank WDI: 8 series, annual
- **Requests.** `https://api.worldbank.org/v2/country/<40 codes>;WLD/indicator/<ind>?format=json&per_page=20000` (keyless). Saved in `raw/worldbank_wdi/`. Terms: CC BY 4.0.
- **The 40 economies.** G20 members (19) plus ESP NLD CHE SWE POL BEL NOR AUT IRL ISR SGP THA MYS PHL VNM EGY NGA PAK BGD CHL COL.
- **Series.**
  - `wb_gdppc_ppp_median`, `_p10`, `_p90` (`logdiff`) and `_p90_p10_ratio` (`diff`): GDP per capita, PPP constant 2021 $, 1990-2025, all 40.
  - `wb_gnipc_atlas_median`: GNI per capita, Atlas, 1992-2025. Nigeria is excluded because its series starts only in 2008.
  - `wb_gini_us`, `wb_bottom20_share_us`: annual 1963-2024, from LIS via the Poverty and Inequality Platform.
  - `wb_poverty_3usd_world`: World headcount at $3.00/day, 1981-2024, missing 2019.
- **Not used as series.** The other countries' Gini and top-10%/bottom-20% shares are sparse survey years, and filling them would be a forward-fill. The raw JSON has them all.
- **Timing.**
  - **`known_after_days` 550** for GDP and GNI (July update of t+1, later for some countries).
  - **`known_after_days` 1,100** for the survey-based series.
- **Revisions.** Every ICP round re-bases the PPP history (2021 PPPs exist only since May 2024).

### 9. World Inequality Database (WID.world): 7 series, annual
- **Download.** Only the `WID_data_<CC>.csv` members for the 40 economies (plus `WID_metadata_US.csv`, `README.md` and `WID_countries.csv`) of the 887 MB bulk zip `https://wid.world/bulk_download/wid_all_data.zip`, fetched by HTTP Range.
  - Stored as gzip of the exact member bytes in `raw/wid/`; `manifest.json` has CRCs and byte ranges.
  - Free bulk download. The terms page could not be reached (404). Cite WID.world and the papers named in the metadata.
- **Variables.** Bulk files name variables concept + population unit + age: `sptincj992` (pre-tax national income share) and `shwealj992` (net personal wealth share), equal-split adults 20+.
- **Points dropped.** Values with data_quality 0, and trailing years that repeat the previous year's value exactly. WID carries its last estimate forward (US income 2022 = 2023 = 2024), so these are not new observations.
- **Series.**
  - `wid_us_top1_income_share`, `_top10_income_share`, `_bottom50_income_share`, `wid_us_top1_wealth_share`, `_top10_wealth_share`, `_bottom50_wealth_share`: 1913-2022/23, percent.
  - `wid_panel_top10_income_share_median`, 1980-2021: median across the 19 economies whose top-10% income share has WID quality 3 or better in at least 30 of 45 years and genuine values through 2021 or later (AU CA CN DE FR GB IN IT US ES NL CH SE PL BE NO AT IE TH).
- **Wealth panel not built.** Wealth shares are quality 0 (imputed) for 23 of the 40 economies.
- **Timing.** **`known_after_days` 1,100.** The latest genuine year trails by 2-3 years. BACK-CAST before the 2016-17 publication of the US distributional national accounts.

## Skipped or not available
- **IRS SOI county AGI** (brief: "if it's easy"). Skipped:
  - annual, and tax year t is published about 2 years later (the TY2022 county file is dated 2025-02-14);
  - 2011 onward come as CSVs, but earlier years are per-state Excel workbooks with changing layouts;
  - the series budget was already above 300.

  If wanted: `https://www.irs.gov/pub/irs-soi/<yy>incyallnoagi.csv`.
- **Census API (SAIPE and other)**: available with a free key; not registered.
- **DC coincident index**: not published.
- **Pre-2010 BEA quarterly state per capita income**: BEA no longer publishes pre-2010 quarterly population. It could be rebuilt by interpolating annual population, but that would put a little look-ahead into the denominator, so it was not done.
- **Philly Fed `coincident-revised.xls`**: binary xls, unreadable with the standard library; the identical FRED copies were used.
- **Zillow's research pages** answer 403 to scripts; the terms were quoted from Zillow Group's public-data page.
- **QCEW industry detail** (a by-industry county panel): not stored (about 12 GB for all zips). The byte-range method fetches any other industry member cheaply.
- **Monthly county employment from CES**: not published for counties (metro areas only).

## Data-quality checks
`qc_data-prosperity.txt` has one line per series. It gives the span, rows, min/max, missing periods, range flags and the first and last 3 rows.
- **Ranges.** Every series was checked against plausible bounds. The only flag, the coincident-index p10 growth in April 2020 (-41%), was confirmed as genuine (Nevada -82, Hawaii -51, Michigan -48 log-%), and the bound was widened.
- **Sentinels and duplicates.** No -9999 or -999 values. No duplicate dates: the writer refuses them, sorts ascending and writes no blanks.
- **Gaps.**
  - 2025-10 (no BLS collection) in 120 series; 3 monthly-change series also miss 2025-11.
  - SAIPE 1990-92/94/96.
  - Zillow panel 2001-01/02 and ZHVI y/y 1998-07/08 (warm-up, before the stock window).
  - Kings NY ZHVI 2002-03/04, Williams ND 2011-02, World poverty 2019.
- **Sanity values.**
  - Median county unemployment (SA): 4.6% (Dec 2006), 9.2% (Oct 2009), 11.9% (Apr 2020), 3.8% (Aug 2026).
  - Share of counties with year-over-year higher unemployment: 98% in Oct 2009.
  - Zillow panel median monthly home-value growth: +1.75% in Jun 2021, and 98.7% of counties rising.
  - SAIPE median county household income: $23.1k (1989), $66.8k (2024).
  - LA County's NSA August seasonal (+0.4-0.6 pp) matches its history.

## Notes for the test
- **Timing is a bound, not the lag.** Most monthly series carry the worst month's lag (January benchmark, or the 2013 shutdown), so short offsets lose power; there is no look-ahead from timing. Typical lags:
  - LAUS county: 32 days (bound 66).
  - States: 20 days (bound 53).
  - Coincident indexes: 22-30 days (bound 76).
  - Zillow: 16 days (bound 35).
- **Revisions and back-casts are the main remaining look-ahead.** All values are current vintage. Several histories were built later:
  - the coincident indexes before 2005;
  - the Zillow history;
  - county GDP before 2019;
  - WID before 2016.

  Results that depend on early-period values of these should be treated with suspicion.
- **Seasonality.** NSA county data were adjusted causally. The adjustment leaves more month-to-month noise than BLS's own smoothed SA (standard deviation of monthly changes 0.27 vs 0.15 pp at the state level). The year-over-year statistics avoid it.
- **Near-duplicates.**
  - `lau_st_ur_median` and `oecd_ur_panel_*` include the US rate, which is close to data-markets' `fred_unrate`.
  - The coincident indexes, state unemployment and county unemployment share inputs: the BLS state data.
- **Annual series (81)** can only enter the full-history period and are nearly untestable.
