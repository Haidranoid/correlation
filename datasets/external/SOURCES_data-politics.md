# SOURCES: data-politics (category `politics`)

Agent: `data-politics`. Built 2026-10-18. Catalog: `catalog_data-politics.csv` (212 series). Series: `series/<id>.csv`.
Scripts, hand-researched inputs and the full QC table: `scripts_data-politics/` (paths inside the scripts point to the
session scratchpad. Change `SP`/`WORK` in `common.py` to re-run.)

Task: "also include changes in local political conditions", meaning politics below or alongside the US national level
(US states, other countries), plus the national context needed to read them.

## 1. What is in the catalog

| Family (subcategory) | Series | Freq | Coverage | Transform | known_after_days | Geography |
|---|---|---|---|---|---|---|
| State-level EPU, state & local policy terms (`policy_uncertainty_state`) | 51 (50 states + DC) | monthly | 1985-2006 start → 2025-12 | logdiff | 100 | **state** |
| Country EPU (`policy_uncertainty_country`) | 19 countries + UK daily | monthly / daily | 1985-2010 start → 2024-07..2026-07; UK daily 2001-01-01 → 2026-10-17 | logdiff | 200 (monthly), 2 (UK daily) | **country** |
| Global EPU (`policy_uncertainty_global`) | 1 | monthly | 1997-01 → 2026-07 | logdiff | 90 | global |
| US categorical EPU (`policy_uncertainty_us_category`) | 9 | monthly | 1985-01 → 2026-09 | logdiff (7), diff (2) | 21 | national |
| US monthly EPU (`policy_uncertainty_us`) | 2 | monthly | 1985-01 → 2026-09 | logdiff | 21 | national |
| Twitter economic uncertainty (`policy_uncertainty_twitter`) | 2 | daily | 2011-06-01 → 2023-04-21 (SHORT) | logdiff | 1 | US / English-world |
| Caldara-Iacoviello country GPR (`geopolitical_risk_country`) | 44 | monthly | 1985-01 → 2026-09 | logdiff (26), diff (18) | 7 | **country** (USA national) |
| Partisan Conflict Index (`partisan_conflict`) | 1 | monthly | 1981-01 → 2026-09 | logdiff | 30 | national |
| Protests, CCC (`protest`) | 15 | daily | 2017-01-01 → 2026-05-31 | level | 1 | 10 **state** + 5 national |
| Protests and violence, ICEWS (`protest`, `violent_conflict`) | 19 | daily | 1995-01-01 → 2023-04-09 | level | 1 | 16 **country** + 3 world |
| COVID policy stringency, OxCGRT (`covid_policy_stringency`) | 13 | daily | 2020-01-01 → 2022-12-31 (SHORT) | diff | 1 | 12 **state** + federal |
| Presidential approval, Gallup (`presidential_approval`) | 1 | irregular (~weekly) | 1993-01-26 → 2025-12-15 | diff | 7 | national |
| Iowa Electronic Markets (`election_market`) | 4 | daily (trade days) | 2000 → 2024-11 (pres.), 2002 → 2026-10 (Congress) | diff (chained) | 1 | national |
| Federal elections, control changes, standoffs (`election_calendar`, `federal_control_change`, `election_closeness`, `fiscal_standoff`) | 7 | daily | 1990-01-01 → 2026-09-30 (debt limit from 2011) | level | 0 (dated on the day the outcome was public) | national |
| State partisan control (`state_partisan_control`, `election_closeness`) | 15 states + 4 national aggregates | daily | 1995-01-01 → 2026-09-30 | level | 0 (same dating rule) | 15 **state** + 4 national |
| Federal Register activity (`federal_policy_activity`) | 4 | daily (publication days) | 1994/1995/1996 → 2026-10-16 | level | 0 | national |

Counts by geography (the `region` column): **88 state-level** (`US-XX`), **79 other-country** (ISO 3166-1
alpha-2, `IL;PS` for Israel plus the Palestinian territories), 40 US-national (`US`), 5 global (`WORLD`). So 167 of
212 are genuinely local. The 4 national aggregates of state events (`pol_gov_*`, `pol_leg_net_flips`) are sums or
means of local events.

By frequency: 84 daily, 1 weekly-irregular, 127 monthly. By transform: 110 logdiff, 38 diff, 64 level.

## 2. Conventions used

- **Monthly series** are dated at the month's last calendar day. `known_after_days` is the source's release lag
  in steady state, **measured** where possible (see §4). It is not the real-time availability of a backfilled
  research index (see Cautions).
- **Event series** (elections, control changes, shutdowns, DISP, state control) are written for every calendar day
  (0 = no event). Each event is dated on the **first calendar day by whose 4 pm New York close the outcome was
  public**, so `known_after_days = 0`. Examples: a call at 11 pm goes on the next day; 2:30 am goes on that day;
  a Saturday call goes on the Saturday (the test maps it to the next trading day). Sign convention for control
  changes: **+ = toward Republican control, − = toward Democratic control**.
- **Count series** (CCC, ICEWS) are written for every calendar day in coverage (0 = no recorded event).
  **Federal Register** counts are written on publication days only (weekends and holidays are absent, not zero).
- **Zeros in uncertainty indices.** In the EPU family (state EPU, UK daily EPU, Twitter EPU) zeros are archive or
  collection gaps: 139 state-month zeros clustered in January, 50 UK days (mostly 25 December), 5 isolated Twitter
  outage days in 2015. They are dropped as missing, and these series use logdiff. GPR country indices and two US
  categorical sub-indices (financial regulation, sovereign debt) have genuine "no article that month" zeros, so they
  keep them and use `diff`. Each series' notes say which applies.
- **IEM probabilities are chained**: the stored value is the cumulative sum of within-cycle day-to-day changes
  (percentage points). Cross-cycle jumps between election markets are therefore removed, and `diff` returns exactly
  the within-cycle change in the market-implied probability. The levels in each cycle's notes give the actual
  probabilities at its start and end.

## 3. Sources (downloads in `raw/`, unmodified)

| Source | raw folder | Licence / terms | Used for |
|---|---|---|---|
| policyuncertainty.com (Baker, Bloom, Davis et al.): `State_Policy_Uncertainty.xlsx`, `All_Country_Data.xlsx`, `Global_Policy_Uncertainty_Data.xlsx`, `Categorical_EPU_Data.xlsx`, `US_Policy_Uncertainty_Data.xlsx`, `UK_Daily_Policy_Data.csv`, `Twitter_Economic_Uncertainty.xlsx` (Dropbox link on the site), `Debt_Ceiling_Shutdown_Data.xlsx` (not used, §5) + the HTML pages | `raw/policyuncertainty_politics/` | "These data can be used freely with attribution to the authors, the paper, and the website"; site under CC BY 4.0 | `pu_*` |
| Wayback Machine captures of the same files (2019-2026 vintages) | `raw/wayback_vintages/` | Internet Archive | measuring release lags only |
| Caldara & Iacoviello GPR, `data_gpr_export.dta` (+ `.xls`, page, update log), matteoiacoviello.com/gpr.htm | `raw/gpr_country/` | free with citation (Caldara & Iacoviello 2022, AER) | `gpr_c_*` (GPRC_ columns only) |
| Partisan Conflict Index: Philadelphia Fed archive `partisan-conflict.xlsx` (to 2025-06, sunset July 2025) and Richmond Fed continuation `partisan_conflict_index.csv` | `raw/philfed_pci/` | Federal Reserve public data | `pci_partisan_conflict` |
| Crowd Counting Consortium, Harvard Dataverse doi:10.7910/DVN/6OPP7H, 9MMYDI, RI9JFU | `raw/ccc/` (203 MB) | CC0 1.0 | `ccc_*` |
| ICEWS Coded Event Data, Harvard Dataverse doi:10.7910/DVN/28075 (1995 yearly files → weekly files to 2023-04-09) | `raw/icews/` (1.7 GB) | Dataverse page: "made available with limited information on how it can be used" (research use) | `icews_*` |
| Oxford COVID-19 Government Response Tracker, US subnational (`OxCGRT_US_latest.csv`, GitHub OxCGRT/USA-covid-policy) | `raw/oxcgrt_us/` | CC BY 4.0 | `oxcgrt_*` |
| Gallup presidential approval via the UCSB American Presidency Project (Clinton → Trump 2nd term pages) | `raw/ucsb_app_approval/` | APP public pages (Gallup data, attribution) | `gallup_pres_approval` |
| Iowa Electronic Markets price histories: 977 monthly HTML tables (incl. the live 2026 Congress market) via the site's price-history form, plus market info and prospectus pages | `raw/iem/` | University of Iowa research market, public price histories | `iem_*` |
| senate.gov party divisions; Wikipedia articles on shutdowns, debt ceiling, Congress party divisions; CRS R43389 "The Debt Limit Since 2011" and RL31967 via EveryCRSReport; Treasury debt-limit page and letters of 1/17/25 and 1/23/25 (scanned, OCR'd with local tesseract) | `raw/politics_events/` | US government works (public domain); Wikipedia CC BY-SA 4.0 | `pol_us_*` |
| Wikipedia "YYYY United States gubernatorial elections" (1996-2025), "YYYY United States state legislative elections" (2000-2025), close-race articles, popular-vote-margin list, OBBBA, 2021 CA recall (API `action=parse`, wikitext with revid) | `raw/wikipedia_elections/` | CC BY-SA 4.0 | governors, margins, dates |
| Ballotpedia "Historical and potential changes in trifectas" (state tables 1992-2026 of governor, senate and house control), "State government trifectas", copyright page | `raw/ballotpedia/` | GFDL (Ballotpedia:Copyrights); robots.txt allows; crawl-delay respected | legislative chamber control, cross-check of governors |
| Federal Register API daily facets (federalregister.gov, no key), 1994-2026 | `raw/federal_register/` | public domain | `fr_*` |

All downloads were sequential with sleeps: Wikipedia 5 s with back-off after one 429, IEM 1.5 s, Dataverse 3 s.
No API keys were used and nothing was registered. Nothing downloaded was executed. All parsing used the
standard library and numpy. The one exception: the Stata `.dta` was read with pandas from the existing shared venv
(`~/programming/c2-wt/venv-data`, nothing installed).

Coordination: data-other holds the daily global GPR (`gpr_daily*`) and the infectious-disease EMV; data-markets
holds the daily US EPU (`fred_usepuindxd`). I took nothing from `raw/gpr` or `raw/policyuncertainty`. My
downloads are in separate folders. No id collisions with the other catalogs (checked at assembly).

## 4. Timing (`known_after_days`): evidence

| Series | Value | Evidence |
|---|---|---|
| State EPU | 100 | First published 2022-04 (data to 2021-06, lag 278 d). Wayback vintages: 2022-10 file to 2022-06 (97 d); 2026-01 file to 2025-12 (20 d); no update captured between Oct 2022 and Jan 2026. 100 d is a steady-state bound. Real-time availability before 2022 was nil. |
| Country EPU (compiled All_Country file) | 200 | At captures the file lagged 96 d (2022-11), 89 d (2023-03), 193 d (2024-09), and 79 d on 2026-10-18. Single-country files may be fresher. |
| Global EPU | 90 | 31 d and 24 d at the 2022 captures; 79 d today |
| UK daily EPU | 2 | captures 2025-11-24 and 2026-03-14 ended 1-2 days earlier |
| US categorical and monthly EPU | 21 | 18 d (2019-11), 9 d (2024-09), 10 d (2025-07), 17 d (2025-11); 18 d today |
| GPR country | 7 | site: "updated at the beginning of each month"; last update 2026-10-01 with data to 2026-09 |
| PCI | 30 | Philly Fed released June 2025 on 2025-07-09; Richmond monthly |
| Gallup approval | 7 | dated at fieldwork end; Gallup publishes 1-5 days later |
| Twitter EPU, CCC, ICEWS, OxCGRT | 1 | tweets, protests and policies public on the day; the compiled datasets came later and are revised |
| IEM | 1 | "LastPrice refers to the price of the last trade before midnight" (Central) |
| Federal events, state control, Federal Register | 0 | events dated on the day they became public. Federal Register issue is public ~6 am ET |

Researched result dates for late-called races are in `scripts_data-politics/close_race_calls.json`:
- Federal: 2000 (Dec 13), 2020 (Nov 7, Saturday), Jeffords (2001-05-24), Senate 2006 (Nov 9), Georgia runoffs (2021-01-07), House 2022 (Nov 17)
- Governor races decided by under ~2 points: AL/AZ/OK 2002, CT/FL/MN/VT/ME 2010, MA/AK 2014, NC 2016, KY 2019, AZ/NV 2022

Three late-resolved holds are also researched: WA 2004 (recount certified Dec 30), MS 1999 (decided by the House
Jan 4, 2000) and VT 2014 (decided by the legislature Jan 8, 2015). MN 2010 (Emmer conceded Dec 8) and MA 2014
(Coakley conceded Nov 5) come from news reports, not from the downloaded pages. The 11 remaining close races
without a researched date are dated election day + 14. All of them were resolved within two weeks (e.g. FL 2018:
Gillum conceded Nov 17; GA 2018: Abrams ended her campaign Nov 16).

## 5. Skipped or not available, and why

- **GDELT protest/conflict counts: available but skipped, too big.** The 1979-2013 reduced master file is
  1.09 GB, and the 2013-04-01 → 2025-06 daily event files total **41.9 GB** zipped (by year: 2014 3.3 GB,
  2015 4.4 GB, 2016 4.9 GB …). That totals about 43 GB against the ~10 GB cap, and there is no smaller daily
  aggregate (the DOC/GEO APIs start 2017). **ICEWS was used instead** (1.7 GB, same CAMEO coding, 1995 → 2023-04).
- **ICEWS US and US-state protests: not usable.** Recorded US protests drop about tenfold at the 2014 production
  change (≈800-1,300 a year in 2005-2013 vs 110-160 a year in 2014-2022), while US totals and violence don't. CCC
  covers US and state protests from 2017.
- **ACLED: available with a free key** (registration). Not registered. POLECAT (ICEWS successor, 2018 →, PLOVER
  ontology, Dataverse) is available, but it wasn't spliced onto CAMEO-coded ICEWS.
- **FiveThirtyEight approval averages: dead.** projects.fivethirtyeight.com CSVs redirect to ABC News. Gallup via APP
  used instead (Gallup rows only: APP mixes pollsters from Feb 2026).
- **Debt_Ceiling_Shutdown_Data.xlsx** (monthly "debt ceiling"/"shutdown" article shares): skipped because releases
  are irregular (Wayback lags 112-354 d, file ends 2025-09). The exact-date daily indicators `pol_us_shutdown` and
  `pol_us_debtlimit_binding` cover the same events.
- **Debt-limit episodes before 2011** (1995-96, 2002, 2003, 2004, 2006, listed in Treasury's 1/23/25 letter):
  DISP start dates need GAO tables (gao.gov returned 403; CRS gives only bounds). `pol_us_debtlimit_binding`
  starts 2011-01-01.
- **IEM**: no 2014 congressional price history is published. The separate House and Senate markets are
  gain/hold/lose ladders and were downloaded but not used (the combined Congress market is used uniformly). The 2004
  vote-share market is nominee-specific (KERR, DEAN, …) and not used. The 2004 WTA DEM04/REP04 contracts stop on
  2004-09-21.
- **Not normalized (near-duplicates or out of scope):**
  - State EPU National and Composite variants
  - All_Country "US", "SCMP China" and "Mainland China" columns (hybrid China used)
  - GPRHC 1900+ historical country indices
  - TEU-ENG, TEU-WGT and the TMU market-uncertainty series
  - OxCGRT for the other 38 states and DC (short, collinear)
- **State inaugurations** are not built as separate state series: dates vary by state and are fully anticipated.
  Federal inaugurations and new majorities are in `pol_us_power_change`.
- **Available, not used**: Klarner State Partisan Balance (ends 2011), Kaplan "US Governors 1775-2020"
  (ends 2020), NCSL composition (PDF), MIT MEDSL returns, Correlates of State Policy, ParlGov, V-Dem/Polity,
  World Uncertainty Index, UCDP GED. These are annual or quarterly, end too early, or were out of budget.
- Other agents' scope (not collected): PredictIt, Polymarket, Kalshi, Manifold and Metaculus (data-predictions);
  GDELT tone and headline sentiment (data-news); daily GPR, EMV and daily US EPU (data-other, data-markets).

## 6. Data-quality checks and fixes

`scripts_data-politics/final_qc.py` checked all 212 series: header, sorted unique dates, no NaN/blank/sentinel,
per-family plausible ranges, logdiff series strictly positive, catalog first/last/rows equal to the file. Result:
**0 problems.** Per-series n, min, max, non-zero count, gap count and largest gap, plus head/tail rows, are in
`scripts_data-politics/final_qc.tsv`. Gaps are by design for Federal Register (weekends), IEM (no-trade days,
inter-cycle) and Gallup (poll spacing). Elsewhere they are the dropped zeros or imputed months noted per series.

Fixes and decisions:
- All_Country EPU: **343 source-imputed (yellow-highlighted) cells dropped**, detected from the cell fill style.
  Examples: Chile after 2024-07, Pakistan 1997-2010, India 1997-2002, the latest months of Greece, Ireland, Korea
  and China.
- PCI: in the Richmond CSV the row labelled `2024,1` between 2024-12 and 2025-02 was read as 2025-01. It equals
  Philly's 153.74. Otherwise the 534 overlapping months agree within 1 after rounding.
- Gallup: Biden poll end date `2/20/224` read as 2024-02-20; 37 non-Gallup 2026 rows dropped.
- ICEWS: 159,047 duplicate event IDs removed (145,312 inside the 2014 file). Leading-zero-stripped CAMEO codes
  were normalized (27 unparseable). The incomplete last day, 2023-04-10, was dropped.
- CCC: 440 online-only events excluded. Phases joined as published.
- Governors: the parser was cross-checked against each page's own "gain/hold" result text (0 mismatches over 326
  races) and against Ballotpedia's governor rows (0 mismatches). CA 2021 recall, 2023 KY/LA/MS and DE 2024 were
  entered manually from race-page infoboxes (`governor_races_manual.json`). Mid-term party changes were hand-coded:
  - AZ 2008-12-01 (Napolitano nominated, Brewer R succeeds)
  - FL 2010-04-30 (Crist leaves the GOP)
  - RI 2013-05-31 (Chafee joins the Democrats)
  - WV 2017-08-04 (Justice joins the GOP)
- Legislatures: only control changes following a chamber's own regular election are kept (chamber-specific
  calendars: VA, NJ, LA, MS odd years; 4-year cycles in AL, MD, MI Senate, MN Senate). 17 mid-session transitions
  were omitted.
- IEM: the HTML has unclosed `<td>` cells (rows split on cell openings). Congress days are kept only when all named
  outcomes have traded at least once and the last prices sum to 0.85-1.2. Vote-share prints outside 0.35-0.65 are
  dropped. Each cycle's start and end probabilities were checked against the actual outcomes.
- Federal Register: API rule coverage is partial in 1994, so rules start 1995; the significance flag is filled from
  1996.

## 7. Cautions for the test

1. **Retrospective indices.** State EPU (2022), country GPR (2021-22), Twitter EPU and most country EPUs were
   constructed after the fact. `known_after_days` is the steady-state release lag, so a "lead" found in history is
   an information-content result, not a tradable one. Indices are also rescaled and revised across vintages.
2. **Sparse events.** Per-state control series have 2-13 non-zero days in 30 years. The federal flip, margin and
   power-change series have 6-16 events. One aligned event can produce a large correlation, so don't read a single
   hit as a finding. Federal and state election events also cluster on the same November dates and are collinear.
3. **Breaks and non-stationarity in counts.**
   - CCC phase changes (2021-01-01: about twice the coverage and more strikes; 2025-01-01: new guidelines)
   - ICEWS source-coverage shifts (2014, a 2020 dip) and the end on 2023-04-09
   - Small-state EPU newspaper-panel changes (DC moves in ~9-point steps, one article)
   - Gallup diffs across presidencies are changes of president (transition dates are in the notes)
4. **Mild magnitude look-ahead (flagged in notes).**
   - `pol_us_pres_margin` and `pol_gov_swing` use final certified margins; result-day margins differed by up to
     ~2 points.
   - Legislative chamber changes are dated election day + 2, including the rare coalition takeovers that formed
     weeks later (e.g. the WA Senate, Dec 2012).
5. **Seasonality and calendar effects.** Federal Register counts have weekday, fiscal-year-end and
   end-of-administration ("midnight rules") surges. Protests cluster on weekends and around set-piece days (Women's
   March, June 2020, "No Kings" 2025). Shutdowns, DISPs and elections follow the fiscal and electoral calendar.
6. **Short or ending series.**
   - OxCGRT: 2020-2022 only
   - Twitter EPU: to 2023-04
   - CCC: from 2017
   - ICEWS: to 2023-04
   - Chile EPU: to 2024-07
   - Pakistan EPU: from 2010-08
   - IEM: gaps between election cycles; the stock window end (2025-06-27) falls between the 2024 and 2026 markets
7. **Monthly series** (127) are usable only at the 21- and 63-day move lengths.
8. **IEM** is a thin, small-stakes market (many no-trade days, a few erratic prints despite filters). The WTA
   contract pays on the **popular-vote** winner, not the Electoral College.
