# External data, `data-migration` share: population growth and immigration

Written by the `data-migration` agent on 2026-10-18/19 (UTC). Catalog: `catalog_data-migration.csv` (292 series). QC report: `qc_data-migration.txt`. Normalized files: `series/<id>.csv`. Raw downloads: `raw/<source>/`, unmodified. Each folder also holds a `_downloads.tsv` (download time, size, server Last-Modified, URL). Three other files in `raw/` are mine and not original downloads: `raw/cbp/release_lag_evidence.tsv` (HEAD Last-Modified dates), `raw/cbp/nationwide-encounters-page.html` (the page, for its link list), and `raw/statcan_migration/unz/` (the CSV unzipped from the downloaded zip). Derived panels: `derived/data-migration/`. Scripts: `scripts/data-migration/`, standard library plus numpy, run with `python3 -I`. The one exception is `check_refugee_archives.py`, which needs `xlrd` from the data venv.

## What is here

| group | n | frequency | span | transform | ids |
|---|---|---|---|---|---|
| US border (CBP) | 11 | monthly | 1999-10 to 2026-08 (8 USBP series); 2019-10 on (3 total-encounter series) | anomaly_log1p (4), logdiff (7) | `cbp_usbp_{total,swb,northern,rgv,tucson,elpaso,sandiego,delrio}`, `cbp_enc_total`, `cbp_enc_swb`, `cbp_ofo_total` |
| US refugees | 1 | monthly | 2000-10 to 2026-09 | anomaly_log1p | `wraps_refugee_arrivals` |
| US enforcement (OHSS) | 3 | monthly | 2013-10 to 2024-11 (discontinued) | logdiff | `ohss_dhs_removals`, `ohss_ice_arrests`, `ohss_cf_referrals` |
| US legal immigration (OHSS) | 3 | quarterly | FY2016 Q1 to FY2025 Q4 | logdiff | `ohss_lpr_total`, `ohss_lpr_new_arrivals`, `ohss_naturalizations` |
| EU asylum (Eurostat) | 12 | monthly | 2008-01 to 2026-08 (EU27 from 2013, UK to 2020-11) | anomaly_log1p (6), logdiff (6) | `eurostat_asy_{eu27,de,fr,it,es,el,at,se,nl,be,ch,uk}` |
| Canada | 8 | monthly (5), quarterly (3) | IRCC 2015-01 to 2026-07; StatCan 1971 to 2026 Q2 | logdiff / anomaly_log1p / diff | `ircc_pr_{total,on,qc,bc,ab}`, `statcan_immigrants`, `statcan_emigrants`, `statcan_net_npr` |
| Australia (ABS) | 2 | quarterly | 1981 Q2 to 2026 Q1 | diff / anomaly_log1p | `abs_nom`, `abs_overseas_arrivals` |
| UK (Home Office) | 2 | daily (1), quarterly (1) | small boats 2018-01-01 to 2026-10-15; asylum 2001 Q1 to 2026 Q2 | level / anomaly_log1p | `ukho_small_boats_daily`, `ukho_asylum_claims_q` |
| US county distribution (Census PEP) | 18 | annual | 2001 to 2025 | diff (17), logdiff (1) | `popest_dist_{growth,nimr,ndmr}_{median,p10,p90,max,min}`, `popest_dist_share_growing`, `popest_dist_nim_top20_share`, `popest_us_nim` |
| US selected counties (Census PEP) | 120 | annual | 2000/2001 to 2025 | logdiff (pop), diff (rates) | `popest_cty_<county>_{pop,nimr,ndmr}` for 40 counties (list below) |
| US states + DC (Census PEP) | 102 | annual | 2001 to 2025 | diff | `popest_st_<st>_{nimr,ndmr}` for all 50 states and DC |
| International net migration (World Bank) | 10 | annual | 1960 to 2023 | diff | `wbmig_netm_{de,gb,fr,it,es,ca,au,jp,cn,mx}` |

**Testable versus flagged.** 42 series are daily, monthly or quarterly. Under the external test's rules (re-implemented in `build_catalog.py`; see the QC report), each of them should join at least one 5-year correlation period:

- every period: the 8 USBP series, refugees, StatCan, ABS and UK quarterly asylum;
- 2010-14, 2015-19 and 2020-24: Eurostat logdiff countries;
- 2015-19 and 2020-24: Eurostat anomaly countries, EU27 and OHSS monthly;
- 2015-19 only: Eurostat UK;
- 2020-24 only: CBP total encounters, IRCC, the LIAS quarterly series and UK small boats.

The **250 annual series are flagged**. They can join only the full-history period 2005-2025, which holds about 20 release days (see "Notes for the test").

Categories: `migration` (145) for immigration flows and rates; `population` (147) for population levels, growth and domestic migration. `migration` is a category added for this agent, as instructed.

## Conventions

- **Dates.**
  - Monthly: the **1st of the reference month**. Fiscal months are converted, so Oct FY2020 = 2019-10-01.
  - Quarterly: the first day of the quarter (fiscal quarters for OHSS, so FY Q1 = October 1).
  - Annual Census: July 1 of the estimate year. Flows cover July 1 of year t-1 to June 30 of year t, so `NIM 2024` = July 2023 to June 2024.
  - World Bank: July 1.
  - Daily: the calendar day of arrival.
- **`known_after_days` is counted from those dates**, as a conservative bound by the 4 pm New York close. The evidence for each source is below and in the series notes.
- **Transforms.**
  - Counts use `anomaly_log1p` when calendar months (quarters) explain at least 20% of the variance of log changes *and* the series starts early enough to survive the test's 3-year anomaly warm-up. Otherwise they use `logdiff`. Each series' share is in its notes, and the diagnostic is `seasonality.py`.
  - Net flows that can be negative (`abs_nom`, `statcan_net_npr`, `wbmig_*`) and all Census rates and distribution statistics use `diff`.
  - Population levels use `logdiff`, so the move is the growth.
  - UK small boats (daily counts, 66% zero days) use `level`.
- **Zeros are observations.** Refugees have 2 zero months, Eurostat Greece has 1 (April 2020, registration suspended), and UK small boats have zero days. Nothing is forward-filled and nothing is missing inside any series.

## Sources

### 1. Census Bureau Population Estimates Program (`raw/census_popest/`)
Public domain (US government work). Base URL `https://www2.census.gov/programs-surveys/popest/datasets/`.

| file | content | used for |
|---|---|---|
| `2000-2010/intercensal/county/co-est00int-tot.csv` | 2000-2010 intercensal totals, July 2000 to 2010 | growth 2001-2010 |
| `2010-2020/intercensal/county/asrh/cc-est2020int-agesex-all.csv` (released 2025-09-30) | 2010-2020 intercensal. YEAR 1 = April 2010 base, 2..11 = July 2010..2019, 12 = April 2020 Census (checked against LA County) | growth 2011-2020 |
| `2010/2010-eval-estimates/co-est2010-alldata.csv` | Vintage 2010 evaluation estimates with components | components and rates 2001-2010 |
| `2010-2020/counties/totals/co-est2020-alldata.csv` (released 2021-05-04) | Vintage 2020 postcensal with components | components and rates 2011-2020 |
| `2020-2025/counties/totals/co-est2025-alldata.csv` (Last-Modified 2026-03-26) | Vintage 2025 with components | levels 2020-2025, components and rates 2021-2025 |
| `co-est2009-alldata.csv`, `nst-est2020-alldata.csv`, `NST-EST2025-ALLDATA.csv` | downloaded for reference | not used (the county files carry the state rows) |

- **The intercensal files have no components of change**, so components come from the postcensal vintages. Their `R*` rate columns split the years with no overlap: V2010 covers 2001-2010, V2020 2011-2020 and V2025 2021-2025. The partial-year 2000/2010/2020 count columns are never used.
- **Growth is always computed within one file.**
  - 2001-2010: the 2000-2010 intercensal file.
  - 2011-2019: the 2010-2020 intercensal file.
  - 2019→2020: chained through the April 2020 Census, as intercensal (July 2019 → April 2020) × V2025 (April 2020 base → July 2020).
  - 2021-2025: V2025.
- **County population series are chain-linked levels.** They are anchored on V2025 and back-cast with those growth rates, so each year's log change equals that year's within-vintage growth with no vintage seam.
- **Checks.**
  - US NIM summed over states equals the sum over counties in every year.
  - US growth is smooth across the joins: 2009 0.88%, 2010 0.84%, 2011 0.80%, 2019 0.52%, 2020 0.39%, 2021 0.16%.
  - California growth matches data-other's `census_pop_ca` (0.712%, 0.853% and -0.948% in 2005, 2015 and 2021).
  - US NIM: 1.17M (2001), 0.38M (2021), 2.73M (2024), 1.26M (2025, V2025).
- **Distribution statistics.**
  - Growth and rate statistics are over counties with at least 10,000 residents at the start of the year (about 2,400 counties).
  - The NIM top-20 share is over all counties.
  - Extremes have known causes:
    - minimum growth −76.8% in 2006 (St. Bernard Parish, Katrina);
    - maximum growth +42.6% in 2007 (its rebound);
    - Trousdale County TN in 2016 (a new prison);
    - McKenzie County ND in 2014 (Bakken oil boom);
    - minimum NIM rate −58 per 1,000 in 2003 (Chattahoochee GA). Iraq-war deployments sit inside NIM in the V2010 vintage, which counted Armed Forces moves.
    - maximum NIM rate +65 per 1,000 in 2024 (Winn Parish LA and Adams County MS, ICE detention centers whose detainees count as residents);
    - minimum NDM rate −1,287 per 1,000 in 2006 (St. Bernard).
- **Geography changes.** Connecticut's 8 counties became 9 planning regions in V2022+, so the county universe changes in 2021+. No CT county is among the selected ones. Alaska's Valdez-Cordova split in 2019, Shannon SD was renamed Oglala Lakota (46113 → 46102), and Bedford city VA merged in 2013. Every value comes from one file, so none of these breaks a series.
- **Selected counties (40).** Each has population (`_pop`), NIM rate (`_nimr`) and NDM rate (`_ndmr`). The reasons are in `derived/data-migration/census_selected_counties.csv`.
  - Largest counties and gateways: Los Angeles, Cook, Harris, Maricopa, San Diego, Orange CA, Miami-Dade, Dallas, Kings, Riverside, Queens, Clark, King, Tarrant, Bexar, Broward, Santa Clara, Wayne, New York, Middlesex MA, Philadelphia.
  - Fastest growing: Travis, Collin, Denton, Fort Bend, Wake, Mecklenburg, Hillsborough, Orange FL, Lee FL, Pinal, Utah County.
  - Other gateways: Hudson NJ, Fairfax VA.
  - Special cases: Midland TX and Williams ND (oil booms), Orleans LA (Katrina), San Francisco (tech, pandemic outflow), Cuyahoga OH (Rust Belt), Denver.
- **Timing: `known_after_days` = 310 for every PEP series.**
  - Counties are normally released mid/late March of t+1, about 255-270 days after July 1. V2018 came on 2019-04-18 (291 days) and V2020 on 2021-05-04 (307 days, press release).
  - States normally come in December (175-210 days), but V2020 states also came on 2021-05-04.
  - 310 is the maximum, so it is used throughout.

### 2. CBP (`raw/cbp/`)
Public domain. Two sources:

- **FY2000-FY2020:** the PDF table "U.S. Border Patrol Monthly Encounters (FY 2000 - FY 2020) (508)" (`.../assets/documents/2021-Aug/...pdf`), read with `pdftotext -layout`.
  - Checks: for every fiscal year and sector, the months sum to the yearly total; the 9 Southwest sectors sum to "Southwest Border"; and Coastal + Northern + Southwest = "Monthly Total".
- **FY2021 onward:** the Nationwide Encounters CSVs (`https://www.cbp.gov/document/stats/nationwide-encounters`), using the latest file covering each fiscal year: fy20-fy23 (for FY2020), fy21-fy24, fy22-fy25, and fy23-fy26-aug.
  - FY2020 is identical in the PDF and the CSV for the national, Southwest, Northern, RGV, Tucson and San Diego series, so the splice is seamless.
- **Published totals reproduced exactly:**
  - USBP Southwest FY2000 1,643,679; FY2019 851,508; FY2023 2,045,838; FY2024 1,530,523.
  - Total encounters FY2023 3.20M; FY2024 2.90M.
- **Definitions.** From March 2020 USBP counts include Title 42 expulsions; Title 42 ended 2023-05-11. OFO counts include CBP One appointments (2023-24).
- **Timing.** The Last-Modified dates of every monthly CSV on the page are in `release_lag_evidence.tsv`: 2025-26 updates were posted 5-10 days after month end (e.g. Aug 2026 data on 2026-09-10).
  - 2014-2024 updates *typically* came in the second half of the following month. This is unverified: the Internet Archive was offline. Reports put the December 2023 update in late January 2024, about 57 days after Dec 1.
  - `known_after_days` = 65 from the 1st, which covers releases up to about a month after month end.
- **Real-time caveat.** Monthly USBP numbers before about FY2016 were not published monthly in real time; this table was compiled in 2020-21.
- **Revisions.** Later files revise earlier months slightly.
- **Also kept, not used:** the FY2000-FY2019 apprehensions PDF (same numbers), the data dictionary PDF and the page HTML.

### 3. Refugee Processing Center, WRAPS (`raw/wrapsnet/`)
US State Department, public domain.

- The series comes from `Refugee Admissions Report as of September 30, 2026.xlsx`, which has one sheet per fiscal year FY2001-FY2026 with monthly "Grand Totals".
- Checks:
  - every fiscal year's months sum to its total;
  - the archived "Arrivals by Region by Month" files (FY2012-2017 `.xls`, FY2019-2020 `.xlsx`) match month for month (`check_refugee_archives.py`);
  - published totals reproduced: FY2016 84,994, FY2024 100,034.
- The FY2018 and FY2021-26 PDFs were downloaded too; their glyph layout is scrambled and they aren't needed.
- Timing: the 30 Sep 2026 report was posted 2026-10-05, five days after month end. `known_after_days` = 60 from the 1st allows for delays such as the Oct-Nov 2025 shutdown: the FY2025 year-end archive file is dated 2025-11-21.

### 4. DHS Office of Homeland Security Statistics (`raw/ohss/`)
Public domain.

- **Monthly tables.** "Immigration Enforcement and Legal Processes Monthly Tables", final edition (posted 2025-01-16, data Oct 2013 to Nov 2024). The series was discontinued afterwards.
  - Used sheets: DHS Removals, ERO Administrative Arrests, Nationwide Credible Fear referrals (Total columns). Cells are rounded to 10, and fiscal-year totals were checked against the months.
  - Timing: editions came 96-130 days after the 1st of the reference month (e.g. July 2024 data on 2024-11-08), so `known_after_days` = 130. The tables started in January 2024, so the timing of earlier years is notional.
- **LIAS.** The Legal Immigration and Adjustment of Status report, 10 year-end editions FY2016-FY2025 (Table 1A for LPR quarterly totals and new arrivals vs adjustments; Table 3 for naturalizations).
  - Checks: quarters sum to the year, and total = adjustments + new arrivals per quarter.
  - **Timing is uncertain.** Only year-end editions survive. The Wayback index shows FY2024 editions posted 2024-09-16 (Q1), 2024-11-08 (Q3) and 2025-06-24 (Q4). The FY2025 Q1 value first appears 611 days after its quarter start.
  - `known_after_days` = 640 is a deliberate worst case, so these 3 series only test long-horizon leads.

### 5. Eurostat, `migr_asyappctzm` (`raw/eurostat_asylum/`)
Free re-use with attribution (Eurostat copyright notice).

- API: `https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/migr_asyappctzm?...&citizen=TOTAL&sex=T&age=TOTAL&unit=PER&applicant=TOTAL` (JSON-stat; dataset updated 2026-10-16).
- I used `applicant=TOTAL` (first-time + subsequent) because first-time applicants for the EU aggregate start only in 2014. EU27 TOTAL itself starts in 2013.
- Published values reproduced: Germany 2016 745,265; EU27 2023 1.13M.
- Timing: member states report within two months of month end. On 2026-10-16 the latest month was July 2026 for EU27, DE and ES (107 days), so `known_after_days` = 110.
- Values are rounded to 5 and revised. Ukrainians under temporary protection are not counted.

### 6. Canada (`raw/ircc/`, `raw/statcan_migration/`)
- **IRCC** "Permanent Residents – Monthly IRCC Updates" (open.canada.ca dataset f7e5498e-0ad8-4417-85c9-9b8aff9b9eda, Open Government Licence – Canada). Workbook `EN_ODP-PR-ProvImmCat.xlsx`, rows "Total" and "<province> - Total".
  - Counts are rounded to 5 and preliminary. The provinces sum to Canada within 20 per month.
  - 2024 total 483,655, matching the published ~483,600.
  - Timing: the file posted 2026-09-15 runs to July 2026 (76 days from July 1). Updates are irregular and sometimes about two months behind, so `known_after_days` = 100.
- **Statistics Canada** table 17-10-0040-01 (WDS full-table CSV, Statistics Canada Open Licence), quarterly 1946 to 2026 Q2, kept from 1971.
  - Timing: Q2 2026 was released 2026-09-23, 176 days after the quarter start, so `known_after_days` = 185.
  - The NPR estimation method was revised in 2023.

### 7. Australia, ABS (`raw/abs_migration/`)
CC BY 4.0. Data API dataflow `ERP_COMP_Q` (CSV with labels), Australia rows, thousands converted to persons.

- NOM for July 2022 to June 2023 sums to 538,300.
- Timing: the quarterly release comes about 5.5-6 months after quarter end, so `known_after_days` = 275 from the quarter start.
- NOM uses the 12/16-month rule. Preliminary quarters are revised about 18 months later.

### 8. UK Home Office (`raw/uk_homeoffice/`)
Open Government Licence v3.0.

- **Small boats:** `16_October_2026_Small_boats_-_time_series.ods`, table SB_01, one row for every day 2018-01-01 to 2026-10-15. The data are provisional and 9 days are flagged 'Revised'.
  - 2022 total 45,755 (the final published figure is 45,774).
  - Timing: the series is refreshed weekly on Fridays through the previous day, and there has been a daily page since 2022. `known_after_days` = 7.
- **Asylum claims:** `asylum-claims-datasets-jun-2026.xlsx`, Data_Asy_D01, quarterly 2001 Q1 to 2026 Q2, main applicants plus dependants.
  - Timing: Q2 2026 was published 2026-08-27, so `known_after_days` = 155.
- The summary ODS tables (asylum, visas) were downloaded too. They hold rolling year-ending totals, so they weren't used.

### 9. World Bank WDI (`raw/worldbank_migration/`)
CC BY 4.0. `SM.POP.NETM`, `SP.POP.GROW` and `SP.POP.TOTL` for all countries (API, last updated 2026-10-08).

- NETM is UN WPP 2024 modelling: smoothed, sometimes repeated across years (JPN and KOR 2022 = 2023), revised wholesale every two years, and published only as 5-year totals before about 2023.
- I kept estimate years (to 2023); 2024-2025 are projections.
- `known_after_days` = 1100.
- Only 10 economies are catalogued. All 40 are in the derived panel.

## Derived files (`derived/data-migration/`, not in the catalog)
- `census_popest_panel.csv` (83,062 rows): every county, the 50 states, DC and their years 2000-2025, with:
  - the July population and its source file;
  - within-vintage growth and its source;
  - the chain-linked population;
  - births, deaths, natural change, NIM, NDM, net migration and residual;
  - the six rates.
  This is the full county- and state-level data the user asked for; it can be used for cross-sectional work. It also holds the **state population growth for all 51**: the 10 largest states and the 4 regions are catalogued by data-other, and the other 41 were left out of the catalog for the series budget.
- `census_selected_counties.csv`: the 40 counties, their FIPS codes and why each was chosen.
- `worldbank_population_migration_panel.csv`: 40 major economies, 1960-2025, with population growth, net migration and population. A flag column marks WPP projection years.

## Skipped or not available
| item | status |
|---|---|
| State Department monthly IV/NIV issuances (PDFs, 2017-03 on) | travel.state.gov returns a Cloudflare "you have been blocked" page (HTTP 403) to scripted requests, including direct PDF URLs. I didn't try to get around it. The PDFs are available in a browser and in the Internet Archive (which was offline during this run). |
| USCIS quarterly form data (I-485, N-400, I-589) | Free, but each file is a single-quarter snapshot by field office, with names and layouts that change, so a history needs about 50 files. The FY2026 Q3 snapshots are kept in `raw/uscis/`. National green-card and naturalization counts were taken from OHSS LIAS instead. |
| DHS Yearbook LPRs by state | Annual, with one table per yearbook. Skipped as annual near-duplicates under the series budget; the national quarterly LPR series replaces it. |
| ACS foreign-born share by state and county | **The Census API now requires a key** (`X-DataWebAPI-KeyError`, redirect to `missing_key.html`), so it is "available with a free key". The keyless route is the ACS summary files on www2.census.gov, which are heavy to parse for annual data; not done. |
| ABS monthly overseas arrivals (OAD) | These are visitor and resident-return arrivals (tourism), not migration, so not taken. `OMAD_VISA` (quarterly migrant arrivals by visa, 2006 on) overlaps `abs_overseas_arrivals` and wasn't taken either. |
| UK entry-clearance visas (quarterly) | Available (33 MB dataset); skipped under the series budget. |
| OHSS monthly tables, other sheets (book-ins, CBP One, CHNV parole, ...) | Available; not taken (short or overlapping). |
| Real-time (first-release) PEP vintages | The per-vintage directories exist but were not reconstructed (annual data). |
| 41 state population levels, WB population growth, WB NETM for 30 more economies | In the derived panels; not catalogued (budget, near-duplicates). |

## Data-quality checks run on every series
- At write time: values sorted ascending, no duplicate dates, all finite, an explicit range per series, positive values for `logdiff` (Greece's single zero is allowed and noted), and a sentinel scan.
- `build_catalog.py` re-reads every `series/<id>.csv` and checks:
  - the header, ISO dates, strictly increasing dates and no blank values;
  - that `first`, `last` and `rows` match the metadata;
  - no id clash with the other agents' catalogs;
  - no newline in the notes, and none of the harness's single-contract flag phrases in them.
- The QC report records min, max, zeros, negatives and gaps for every series; there are no gaps beyond the native spacing in any series. It also prints the first and last 3 rows of each series.
- The cross-checks against published figures are listed under each source above.

## Notes for whoever runs the test
1. **Annual series are one calendar event.**
   - All 240 Census series become known on the same day each year, July 1 + 310 days, around early May. The 10 World Bank series are likewise known on July 1 + 1100 days.
   - With the test's trailing-252-day standardization, an annual release-day move is about 16 × (this change / last change), so it is clipped to ±5. These series are therefore once-a-year ±5 pulses on a fixed date, and they join only the full-history period.
   - A hit on them is a hit on one early-May (or early-July) calendar dummy, multiplied 240 times. Report them apart from the monthly series, or exclude them from the headline false-discovery count. The harness already lists "full history only" series separately.
2. **Revision look-ahead.**
   - Census levels are intercensal values published in 2012 and 2025; components are final postcensal vintages.
   - CBP, Eurostat, IRCC, StatCan, ABS and the UK data are stored as latest revisions.
   - World Bank NETM is a modelled reconstruction.
   - None of these are first-release values.
3. **Method breaks.**
   - CBP: Title 42 (March 2020 to May 2023).
   - PEP components change vintage at 2011 and 2021, and NIM includes Armed Forces moves (2001-2010) and ICE detainees (2023-25).
   - Canada: the NPR method changed in 2023.
   - Australia: the NOM rule changed in 2006.
   - Eurostat UK ends 2020-11, and the OHSS monthly tables end 2024-11.
4. **Seasonality.** The 24 `logdiff` count series (CBP sectors with under 20% seasonal share, CBP FY20+, IRCC, EU27, 5 Eurostat countries, OHSS) still carry their seasonal swings in their moves. Seasonal stock patterns could therefore produce calendar-driven hits.
5. **Policy shocks dominate several series.** Examples: 2015-16 (Syria) in Eurostat; the 2017 and 2025 refugee suspensions; COVID in April 2020 (refugees 27, Greece 0, Canada PR −84% from February); 2022-24 at the border; and the 2025 border drop (USBP Southwest under 10,000 a month).
