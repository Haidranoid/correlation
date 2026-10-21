# SOURCES: data-weather agent

Daily historical weather and climate data that could plausibly lead US stocks. **239 series** are listed in `catalog_data-weather.csv`, with files in `series/<id>.csv` and untouched downloads in `raw/<source>/`. Everything was downloaded on 2026-10-18 (US Central time; 2026-10-19 UTC). Parsers and QA scripts are in `scripts/data-weather/` (standard library plus numpy, run with `python3 -I`). Nothing downloaded was executed.

## What is here

| category | subcategory | series |
|---|---|---|
| climate_index | enso | 3 |
| climate_index | mjo | 3 |
| climate_index | teleconnection | 4 |
| weather | cloud_cover | 1 |
| weather | degree_days | 62 |
| weather | drought | 7 |
| weather | evapotranspiration | 4 |
| weather | precipitation | 37 |
| weather | snowfall | 15 |
| weather | soil_moisture | 6 |
| weather | solar_radiation | 5 |
| weather | sunshine | 1 |
| weather | temperature | 72 |
| weather | tornado | 2 |
| weather | tropical_cyclone | 4 |
| weather | wind | 13 |
| **total** | | **239** |


Date coverage: station series run from 1980-01-01 (or the station's start) to mid-October 2026. CPC degree days run 1981 to 2026-10-16, teleconnections 1980 to 2026-10-16, MJO 1991 to 2026-10-14, HURDAT2 events and SPC tornado counts 1980 to 2025-12-31, USDM 2000-01-04 to 2026-10-13, ENSO weekly 1981-09 to 2026-10-07, ONI 1950 to 2026-09 and Open-Meteo/ERA5 2000-01-01 to 2025-12-31. The only series that starts after the census start (2005-01-03) is `wx_den_snow` (2006-02-01, explained below). All series reach 2025-06-27.

## Timing (`known_after_days`)

| group | frequency | known_after_days | n | ids (examples) |
|---|---|---|---|---|
| cpc_cdd | daily | 2 | 10 | cpc_cdd_conus, cpc_cdd_encentral, cpc_cdd_escentral, ... |
| cpc_enso | weekly | 6 | 2 | cpc_enso_nino12_weekly, cpc_enso_nino34_weekly |
| cpc_enso | monthly | 15 | 1 | cpc_enso_oni |
| cpc_hdd | daily | 2 | 11 | cpc_hdd_conus, cpc_hdd_encentral, cpc_hdd_escentral, ... |
| cpc_tele | daily | 3 | 4 | cpc_tele_aao, cpc_tele_ao, cpc_tele_nao, ... |
| hurdat | daily | 1 | 4 | hurdat_atl_ace, hurdat_atl_hurricanes_active, hurdat_us_hurricane_landfall, ... |
| om | daily | 7 | 27 | om_ar_pampas_prcp, om_ar_pampas_soilm, om_ar_pampas_tmax, ... |
| psl | daily | 6 | 3 | psl_mjo_romi_amp, psl_mjo_romi_pc1, psl_mjo_romi_pc2 |
| spc | daily | 7 | 1 | spc_sig_tornadoes_us |
| spc | daily | 3 | 1 | spc_tornadoes_us |
| usdm | weekly | 2 | 7 | usdm_ca_d1plus, usdm_conus_d1plus, usdm_ia_d1plus, ... |
| wx | daily | 1 | 168 | wx_atl_cdd, wx_atl_hdd, wx_atl_prcp, ... |


The reasons, source by source:

- **GHCN station days = 1.** US first-order (USW) stations use a local-standard-time midnight-to-midnight day, so the day is complete overnight and the NWS daily climate reports publish it that night or the next morning. OBS-TIME was checked for every station: it was 2400 or blank on every row since 1980, with no 7 am cooperative-style days. The foreign stations (London and Frankfurt from ECA&D, Sydney from BoM) follow national climatological-day conventions (readings at 09 UTC, about 06 UTC or 9 am local rather than midnight), so a value dated D can partly cover D-1 or D+1. Either way it is complete by about 09 UTC on D+1, before the next US close, so 1 is safe and may be conservative.
- **CPC degree days = 2.** The file for day D appears at about 04:00 ET on D+2 (Last-Modified 2026-10-18 08:07 UTC with data through 2026-10-16).
- **CPC NAO/AO/PNA/AAO = 3.** These appear about 13:55 UTC on D+2 (Last-Modified 2026-10-18 13:55 UTC with data through 10-16). That is only a few hours before the cutoff, so I added one day of buffer.
- **ROMI (MJO) = 6.** It was observed 4 days behind (Last-Modified 2026-10-18 15:22 UTC with data through 10-14); I added 2 days of buffer.
- **Weekly Nino SST = 6.** Rows are dated at the week's centre Wednesday. CPC releases each week on the following Monday (5 days later); I added one day for Monday holidays. The file is touched daily, so its Last-Modified cannot confirm the row-add time.
- **ONI = 15.** Each row is dated at the last day of its 3-month season. The current table was updated 3 days after JAS 2026 ended (2026-10-03), but historically CPC updated the ONI with the monthly ENSO discussion (second Thursday, up to day 14), so 15 is an upper bound.
- **USDM = 2.** The map is valid Tuesday 8 am ET and is released Thursday 8:30 am ET.
- **HURDAT2 = 1.** UTC days. The day's storms were public in real time through the 6-hourly NHC advisories. The best track itself is a post-season reanalysis (see caveats).
- **Open-Meteo ERA5 = 7.** ERA5T appears about 5 days after real time; I added 2 days of buffer. ERA5 is a reanalysis made after the fact (released 2018-2019 for the full period), but it only describes weather that was observable at the time.

## Sources

### 1. NOAA NCEI GHCN-Daily (station observations)

- **URL:** `https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/<ID>.csv.gz`, plus `ghcnd-stations.txt`, `ghcnd-inventory.txt`, `readme.txt`, `readme-by_station.txt` and `ghcnd-version.txt` (version 3.35-upd-2026101717). The raw files are in `raw/ghcnd/`.
- **Licence:** US Government work, freely available with no restrictions stated in the readme. Citation: Menne et al. 2012, doi:10.1175/JTECH-D-11-00103.1, and dataset doi:10.7289/V5D21VHZ.
- **Stations:** 30 US first-order stations (market centres, agriculture, energy) and 3 foreign centres:

| key | GHCN id | station | role | lat, lon | series | TMAX first..last | TMAX missing days since 2000 |
|---|---|---|---|---|---|---|---|
| nyc | USW00094728 | New York Central Park | market centre | 40.779, -73.969 | tmax, tmin, prcp, snow, awnd, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| chi | USW00094846 | Chicago O'Hare | market centre | 41.960, -87.932 | tmax, tmin, prcp, snow, awnd, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| bos | USW00014739 | Boston Logan | market centre | 42.361, -71.010 | tmax, tmin, prcp, snow, awnd, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| phl | USW00013739 | Philadelphia Intl | market centre | 39.873, -75.227 | tmax, tmin, prcp, snow, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| dca | USW00013743 | Washington Reagan National | market centre | 38.847, -77.034 | tmax, tmin, prcp, snow, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| atl | USW00013874 | Atlanta Hartsfield-Jackson | market centre | 33.630, -84.442 | tmax, tmin, prcp, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| dfw | USW00003927 | Dallas-Fort Worth | market centre | 32.898, -97.022 | tmax, tmin, prcp, awnd, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| hou | USW00012960 | Houston Intercontinental | market/energy centre | 29.984, -95.361 | tmax, tmin, prcp, awnd, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| lax | USW00023174 | Los Angeles Intl | market centre | 33.938, -118.387 | tmax, tmin, prcp, hdd, cdd | 1980-01-01..2026-10-14 | 4 |
| sfo | USW00023234 | San Francisco Intl | market centre | 37.620, -122.366 | tmax, tmin, prcp, hdd, cdd | 1980-01-01..2026-10-14 | 20 |
| sea | USW00024233 | Seattle-Tacoma | market centre | 47.445, -122.314 | tmax, tmin, prcp, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| den | USW00003017 | Denver Intl | market centre | 39.847, -104.656 | tmax, tmin, prcp, snow, hdd, cdd | 1994-07-20..2026-10-15 | 274 |
| phx | USW00023183 | Phoenix Sky Harbor | market centre | 33.428, -112.004 | tmax, tmin, prcp, hdd, cdd | 1980-01-01..2026-10-14 | 0 |
| mia | USW00012839 | Miami Intl | market centre | 25.788, -80.317 | tmax, tmin, prcp, awnd, cdd | 1980-01-01..2026-10-15 | 1 |
| msp | USW00014922 | Minneapolis-St Paul | market centre | 44.885, -93.231 | tmax, tmin, prcp, snow, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| dtw | USW00094847 | Detroit Metro | market centre | 42.231, -83.331 | tmax, tmin, prcp, snow, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| dsm | USW00014933 | Des Moines | agriculture (corn/soy) | 41.534, -93.653 | tmax, tmin, prcp, snow, awnd | 1980-01-01..2026-10-14 | 0 |
| pia | USW00014842 | Peoria IL | agriculture (corn/soy) | 40.667, -89.684 | tmax, tmin, prcp, snow | 1980-01-01..2026-10-15 | 4 |
| oma | USW00014942 | Omaha Eppley | agriculture (corn/cattle) | 41.312, -95.902 | tmax, tmin, prcp, snow | 1980-01-01..2026-10-14 | 0 |
| mci | USW00003947 | Kansas City Intl | agriculture (wheat market) | 39.297, -94.731 | tmax, tmin, prcp, snow | 1980-01-01..2026-10-15 | 0 |
| ict | USW00003928 | Wichita | agriculture (HRW wheat) | 37.654, -97.443 | tmax, tmin, prcp, snow, awnd | 1980-01-01..2026-10-14 | 0 |
| far | USW00014914 | Fargo Hector | agriculture (spring wheat/soy) | 46.924, -96.812 | tmax, tmin, prcp, snow, awnd | 1980-01-01..2026-10-15 | 0 |
| fat | USW00093193 | Fresno Yosemite | agriculture (Central Valley) | 36.780, -119.720 | tmax, tmin, prcp | 1980-01-01..2026-10-15 | 0 |
| mem | USW00013893 | Memphis | agriculture (cotton/soy, river) | 35.056, -89.986 | tmax, tmin, prcp | 1980-01-01..2026-10-15 | 0 |
| lbb | USW00023042 | Lubbock | agriculture (cotton) / wind | 33.654, -101.814 | tmax, tmin, prcp, awnd | 1980-01-01..2026-10-15 | 1 |
| lch | USW00003937 | Lake Charles LA | energy (Henry Hub/LNG/refining) | 30.125, -93.216 | tmax, tmin, prcp, awnd, hdd, cdd | 1980-01-01..2026-10-15 | 2 |
| msy | USW00012916 | New Orleans | energy (Gulf refining/ports) | 29.997, -90.278 | tmax, tmin, prcp, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| okc | USW00013967 | Oklahoma City | energy (Cushing area) / wind | 35.388, -97.600 | tmax, tmin, prcp, awnd, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| maf | USW00023023 | Midland-Odessa TX | energy (Permian) | 31.943, -102.189 | tmax, tmin, prcp, awnd, hdd, cdd | 1980-01-01..2026-10-14 | 0 |
| pit | USW00094823 | Pittsburgh | energy (Appalachian gas) | 40.485, -80.214 | tmax, tmin, prcp, snow, hdd, cdd | 1980-01-01..2026-10-15 | 0 |
| lon | UKE00107650 | London Heathrow | foreign centre | 51.479, -0.449 | tmax, tmin, prcp | 1980-01-01..2026-08-30 | 147 |
| syd | ASN00066037 | Sydney Airport | foreign centre (Asia-Pacific; replaces Tokyo) | -33.947, 151.173 | tmax, tmin, prcp | 1980-01-01..2026-10-08 | 5 |
| fra | GME00122362 | Frankfurt/Main-Westend | foreign centre | 50.128, 8.670 | tmax, tmin, prcp | 1985-11-01..2026-08-31 | 92 |


- **Variables and units:**
  - TMAX and TMIN: tenths of degC divided by 10, giving **degC**.
  - PRCP: tenths of mm divided by 10, giving **mm**.
  - SNOW: already whole **mm** in GHCN (not tenths), so not rescaled.
  - AWND: tenths of m/s divided by 10, giving **m/s**.
  - Trace (M=T) is 0.
  - SNOW is kept only at 15 snowy stations. AWND is kept at 13 stations (wind-power regions, coastal storm exposure, Chicago, New York, Boston).
- **Quality:**
  - Every value with a non-blank QFLAG was dropped. Counts since 1980 are in each catalog note; the largest were London TMAX 270 / TMIN 281 (flag I) and Denver SNOW 277 (flag Z in 1995-97).
  - MFLAG P ("missing presumed zero") was dropped (Lake Charles SNOW 92, DCA SNOW 16).
  - There are no duplicate date-element rows and no -9999 values left.
  - Hard range checks: temperature -50..55 degC, PRCP 0..1000 mm, SNOW 0..1500 mm, AWND 0..40 m/s.
  - One unflagged wind spike was removed: Wichita AWND 1998-08-28 = 29.5 m/s, where the next highest is 14.4. The rule drops values above 2x the station's 99.9th percentile.
  - Two short isolated leading fragments before multi-year holes were dropped: New York AWND 1984-01..07 (then no wind until 1995-11) and Denver SNOW 1995-03.
  - All precipitation and snow extremes were checked against known events: Harvey at IAH 2017-08-27 408 mm; the Miami 2000-10-03 storm 319 mm; Ida in New York 2021-09-01 181 mm; the New York blizzard 2016-01-23 693 mm; Philadelphia 1996-01-07 701 mm.
- **Degree days:**
  - HDD = max(0, 65 - Tavg_F) and CDD = max(0, Tavg_F - 65), with Tavg_F = (TMAX+TMIN)/2 converted to degF. Both TMAX and TMIN must be clean; no day had TMAX < TMIN.
  - Unlike NWS, the values are not rounded to whole degF (the difference is at most 0.5).
  - They are computed only at the 21 market and energy-centre stations; CPC regional degree days cover the agricultural states.
  - Skipped as degenerate: `wx_mia_hdd` (93.4% of days zero).
- **Start date:** 1980 (or the station start), so the test's "earlier years" day-of-year baseline has about 25 years by 2005. Older data were left out to limit the warming-trend and instrument-change bias in the baseline.
- **Notable gaps:**
  - Denver Intl: no TMAX/TMIN/PRCP 2013-04-01..2013-12-30 (absent from GHCN); SNOW none 1997..2006-01 (the airport did not report snowfall; 1995-97 values were Z-flagged), so `wx_den_snow` starts 2006-02-01.
  - Minneapolis SNOW: none 2000-11..2004-06.
  - Omaha: TMAX/TMIN 1995-09..1996-03; SNOW 1997-04..1998-11.
  - Peoria SNOW 1996-10..1997-01.
  - San Francisco TMAX/TMIN 2018-06-22..2018-07-11.
  - Frankfurt 2000-03..2000-04.
  - London: about 270 scattered single days (QFLAG drops).
  - Every gap over 30 days is listed in the series' catalog note, and the appendix lists the longest gap per series.
  - Every daily series with a full calendar year since 2000 under 95% complete, across all sources (only GHCN stations have any; this adds New York AWND 2018-2020, Frankfurt 2008 and Omaha SNOW 2001-02 to the list above):

| series | incomplete years since 2000 (days present / days in year) |
|---|---|
| wx_den_cdd | 2013: 91/365 |
| wx_den_hdd | 2013: 91/365 |
| wx_den_prcp | 2013: 121/365 |
| wx_den_snow | 2006: 334/365, 2013: 304/365 |
| wx_den_tmax | 2013: 91/365 |
| wx_den_tmin | 2013: 91/365 |
| wx_fra_prcp | 2000: 305/366, 2008: 335/366 |
| wx_fra_tmax | 2000: 305/366, 2008: 335/366 |
| wx_fra_tmin | 2000: 305/366, 2008: 335/366 |
| wx_msp_snow | 2000: 305/366, 2001: 0/365, 2002: 0/365, 2003: 0/365, 2004: 184/366 |
| wx_nyc_awnd | 2018: 281/365, 2019: 296/365, 2020: 329/366 |
| wx_oma_snow | 2001: 334/365, 2002: 337/365 |
| wx_sfo_cdd | 2018: 345/365 |
| wx_sfo_hdd | 2018: 345/365 |
| wx_sfo_tmax | 2018: 345/365 |
| wx_sfo_tmin | 2018: 345/365 |

- **Skipped:**
  - **Tokyo** (JA000047662): GSOD-derived (source S), only 6,020 of 9,310 TMAX days in 2000..2025-06, and TMAX ends 2025-04-09. The other Asian GSOD records were also 35-85% complete: Seoul, Osaka, Yokohama, Singapore (TMAX ends 2025-05) and Shanghai (PRCP ends 2024).
  - **Hong Kong:** the Observatory record ends 1992 and King's Park ends 2000.
  - **Frankfurt airport:** not in GHCN, so the Frankfurt/Main-Westend city station (ECA&D) is used.
  - **Sydney Airport** (BoM, complete) was added as the Asia-Pacific centre. That was my choice, not part of the request. Tokyo is covered from ERA5 instead (section 8: `om_tokyo_tmax`, `om_tokyo_tmin`, `om_tokyo_prcp`).
- **Caveats:**
  - Raw station data are not homogenized (instrument changes, siting).
  - The latest ~2 months come from preliminary real-time sources (D, d, H, 1) and are later replaced by quality-controlled values, so they can change.
  - GHCN by_station files are regenerated daily, so the download date matters.
  - The Central Park anemometer is tree-sheltered, so its wind speeds run low.

### 2. NOAA CPC population-weighted degree days (daily)

- **URL:** `https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/<year>/{Population.Heating,Population.Cooling,UtilityGas.Heating}.txt` for 1981-2026; raw files in `raw/cpc/degree_days/`. Public domain (NOAA).
- **Series:** HDD and CDD (base 65F, whole degF-days) for the 9 Census divisions and CONUS (population-weighted), plus the CONUS HDD weighted by homes heated with utility gas (the natural-gas demand HDD). Divisional gas-weighted series were skipped as near-duplicates. CONUS population-weighted and gas-weighted HDD correlate at 0.987 after removing the seasonal cycle.
- **Checks:** every year file has exactly 365/366 date columns starting Jan 1, with no blanks or sentinels.
- **Weights:** fixed, not time-varying. The weights folders are 2010 Census/ACS (dated 2014-02), and the file header changes from `Regions::` to `CPC::Regions::` in 2014, so pre-2014 files came from older code and weights; a small step at 2014-01-01 is possible. None is visible: the yearly mean of CPC Middle Atlantic HDD minus the mean New York/Philadelphia/Pittsburgh station HDD stays at 2.0-3.0 through 2005-2025, and gas-minus-population CONUS HDD stays at 1.2-1.5, with no break at 2014.

### 3. NOAA CPC daily teleconnection indices

- **URL:** `https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.{nao,pna}.cdas.z500.19500101_current.csv`, `norm.daily.ao.cdas.z1000...csv` and `norm.daily.aao.cdas.z700.19790101_current.csv`. The monthly-updated `.ascii` versions were downloaded for comparison and agree exactly on all common days. Raw files in `raw/cpc/teleconnections/`.
- **Series:** NAO, AO, PNA and AAO (the AAO, or Southern Annular Mode, was added as a cheap Southern Hemisphere index). These are daily standardized projections of CDAS height anomalies and are already anomalies, so the transform is `level`.
- **Gaps:** blank days in the source were left out (NAO/PNA 2006-10-26 and 2007-01-26; AO/AAO 2003-04-30).
- **Caveat:** the history uses today's normalization. No real-time vintages exist, so revisions cannot be measured.

### 4. NOAA CPC ENSO

- **ONI** (`https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt`): monthly 3-month running mean of ERSSTv5 Nino-3.4 anomalies. The season year is the year of the middle month, and each row is dated at the end of the season (NDJ 1997 becomes 1998-01-31). Consecutive rows overlap by 2 months.
  - **Revised:** ERSST versions changed (v3b, v4, v5), and CPC uses centered 30-year base periods updated every 5 years, so real-time values can differ from today's by roughly 0.1-0.2 degC.
- **Weekly Nino-3.4 and Nino-1+2 SST anomalies** (`https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for`): OISST weekly means against the 1991-2020 climatology, dated at the week's centre Wednesday, starting 1981-09-02. The 1991-2020 base was applied retroactively, and OISST moved from v2 to v2.1 in 2020. Raw files in `raw/cpc/enso/`.
- **Skipped:** the monthly ERSST Nino table (`ersst5.nino.mth.91-20.ascii`) is stale (last row 2026-06) and ONI covers it; Nino-3 and Nino-4 are near-duplicates; the SOI is another ENSO measure.

### 5. NOAA PSL real-time OLR MJO index (ROMI)

- **URL:** `https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt` (CPC blended OLR). Raw file and the PSL index page are in `raw/psl_mjo/`.
- **Series:** PC1, PC2 and amplitude, daily from 1991-01-01 with no gaps.
- **Method:** ROMI removes the previous 40-day mean (tapered), so it uses no future data; this is unlike the OMI, which uses a 20-96-day two-sided filter and was rejected for look-ahead. The history was computed retroactively with the same real-time method.
- **Rejected:** the classic BoM real-time RMM index (`rmm.74toRealtime.txt`) ends 2024-02-24 and changes method in 2014.

### 6. NOAA NHC HURDAT2 (Atlantic best tracks)

- **URL:** `https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt`, revised 2026-09-23 and covering seasons to 2025. Raw files in `raw/hurdat2/`. Public domain.
- **Series:** daily (UTC), with every day 1980-01-01..2025-12-31 written; zero means no activity.
  - `hurdat_atl_hurricanes_active`: distinct storms at status HU in any record that day (max 4).
  - `hurdat_atl_ace`: daily ACE, the sum over 00/06/12/18 UTC fixes of TS/SS/HU systems with wind of at least 35 kt of v^2/1e4.
  - `hurdat_us_landfall_maxwind_kt`: max wind among CONUS landfall records of at least 34 kt, else 0.
  - `hurdat_us_hurricane_landfall`: 1 if a CONUS landfall that day had wind of at least 64 kt (any status, so post-tropical Sandy 2012-10-29 at 70 kt counts).
- **CONUS landfall:** a record identifier of L inside a heuristic Texas-to-Maine coastal box. It excludes Mexico south of 25.95N and west of -96.5, the Bahamas east of -79.6 below 27.6N, Cuba below 24.4N, and Canada east of -66.98 above 43N; Puerto Rico and the USVI are excluded.
  - Every record near the box edges was printed and checked by hand. All in-box edge cases are real US landfalls (Andrew 1992, Bret 1999, Dolly 2008, Hanna 2020, Irene 2011 NJ/NY, Gordon 2018 FL), and every out-of-box L record of at least 34 kt is Mexico, the Bahamas or Nova Scotia.
  - Spot checks (UTC dates): Katrina 2005-08-25 (FL, 70 kt) and 2005-08-29 (LA, 110 kt); Harvey 2017-08-26 (115); Irma 2017-09-10 (115); Ian 2022-09-28 (130); Helene 2024-09-27 03:10 UTC (120; local date 09-26); Milton 2024-10-10 (100); Ida 2021-08-29 (130); Michael 2018-10-10 (140); Laura 2020-08-27 (130); Beryl 2024-07-08 (80).
  - There are 198 CONUS landfall records since 1980 and 81 hurricane-force landfall days.
- **Caveats:**
  - Best track is a post-season reanalysis, so it is slightly better informed than the real-time advisories (some storms were upgraded or added later).
  - The landfall box is a heuristic, not the official US landfall list.
  - The series are extremely zero-heavy: the landfall indicator is 1 on 0.5% of days.

### 6b. NOAA SPC tornado database (not in the request; added for breadth)

- **URL:** `https://www.spc.noaa.gov/wcm/data/1950-2025_actual_tornadoes.csv`, the final database with one record per tornado. The 2025 database was released 2026-04-24 and runs to 2025-12-28. The changelog and format description (`SPC_severe_database_description.pdf`) are in `raw/spc/`. Public domain.
- **Series:** daily counts of US tornadoes (all ratings) and of significant (E)F2+ tornadoes, on CST days, with every day 1980-01-01..2025-12-31 written (zero = none). Puerto Rico and the USVI are excluded. Only 30 records since 1980 are not CST (pre-1995 MST/unknown); they were left as dated.
- **Checks:** the top days are 2011-04-27 (210, of which 69 were EF2+), 2023-03-31 (127) and 2021-12-15 (125). Yearly totals match SPC's published counts (2011: 1,703; 2024: 1,804).
- **Timing:** all tornadoes = 3, because local storm reports are public the same day (the CST day ends at 01:00 ET) but the final count also includes tornadoes found by damage surveys in the following days. EF2+ = 7, because ratings come from NWS damage surveys, usually within 1-3 days but up to about a week after big outbreaks.
- **Caveats:**
  - The final database cleans the real-time reports (duplicates removed, surveyed paths), so it is better than what was known on the day.
  - All-tornado counts trend upward with better detection (NEXRAD in the 1990s, spotters), so they are non-stationary; EF2+ counts are steadier.
  - The F-scale became the EF-scale on 2007-02-01.
  - The series are zero-heavy (52% of days for all tornadoes, 87% for EF2+).

### 7. US Drought Monitor (NDMC / USDA / NOAA)

- **URL:** `https://usdmdataservices.unl.edu/api/USStatistics/GetDroughtSeverityStatisticsByAreaPercent?aoi=us&startdate=1/1/2000&enddate=10/18/2026&statisticsType=1`, and the same with `StateStatistics` and `aoi=<FIPS>` for IA 19, IL 17, NE 31, KS 20, TX 48 and CA 06. Send `Accept: text/csv`; state queries need FIPS codes (abbreviations return nothing). Raw CSVs in `raw/usdm/`.
- **Terms:** free public data; credit "U.S. Drought Monitor, jointly produced by the National Drought Mitigation Center at the University of Nebraska-Lincoln, USDA and NOAA".
- **Series:** % of area in D1-D4 (moderate drought or worse), weekly from 2000-01-04, with no missing weeks (1,398 maps). With statisticsType=1 the D1 column is already cumulative; every row was checked for None+D0 = 100 and D0 >= D1 >= ... >= D4. CONUS rows only (the US response also carries a "Total" row that includes AK/HI/PR).
- **Caveats:** this is an expert-blended weekly analysis, and maps are not revised after release.

### 8. Open-Meteo historical weather API (ECMWF ERA5)

- **URL:** `https://archive-api.open-meteo.com/v1/archive` with `models=era5` pinned. The default `best_match` blends ERA5-Land and recent IFS analyses over time, which risks a resolution discontinuity in 2017+.
  - Other parameters: `timezone=auto` (local days) and 10 daily variables per request (the most that still cost 1x).
  - Each point was fetched in two halves (2000-2012 and 2013-2025, about 339 weighted calls each), spaced 270-300 s apart to stay under the 600/min, 5,000/h and 10,000/day limits.
  - One request (California, 2013-2025) failed with a server-side `timeoutReached` (not a rate limit). The run stopped as designed, rested 3 minutes and resumed with one retry, plus a split into smaller chunks if the retry also failed.
  - **Weighted calls used:** about 6,447 of the 10,000/day free budget, counting the 3 single-call probes and the failed attempt. No rate-limit response was ever received. I set my own cap at 7,500.
  - Raw JSON per point and half is in `raw/openmeteo/`. All 10 variables are kept there, although only some were turned into series.
  - **Provenance caveat:** the 11 files fetched by the resumed run (California 2013-2025, Lubbock, both Brazil points, the Pampas and Tokyo) were saved through `json.dump` of the parsed response instead of the raw bytes, so their whitespace differs and non-ASCII unit strings are `\u`-escaped.
    - The data values are exactly what the API returned: a parse, `json.dump` and re-parse round-trip of the 7 byte-original files gives identical values.
    - Re-fetching to get byte-original copies would have cost about 3,700 more weighted calls and pushed the day over the 10,000 limit, so I did not.
    - `dl_openmeteo.py` now saves the original bytes.
- **Licence:** free for non-commercial use, no key; data CC BY 4.0 (Open-Meteo; contains modified Copernicus/ECMWF ERA5 data).
- **Points (27 series):** ar_pampas, br_coffee, br_soy, ca_valley, il_corn, ks_wheat, nyc, tokyo, tx_lubbock.
  - New York: sunshine duration, mean cloud cover and shortwave radiation, for the Saunders / Hirshleifer-Shumway sunshine effect, since GHCN has no sunshine or cloud for Central Park.
  - Central Illinois corn (Bloomington-Normal), Kansas wheat (Great Bend) and West Texas (Lubbock): radiation, FAO-56 ET0 and 7-28 cm soil moisture.
  - Central Valley: ET0 and radiation (soil moisture skipped because ERA5 does not model irrigation).
  - Brazil coffee (Sul de Minas): Tmin (frost), Tmax, precipitation and soil moisture.
  - Brazil soy (Sorriso, MT) and Argentina Pampas (Pergamino): precipitation, Tmax and soil moisture.
  - Tokyo: Tmax, Tmin and precipitation (replaces the gappy GHCN record; Hong Kong was not added because a 0.25 deg cell there is mostly sea).
- **Checks:**
  - All 18 raw files are complete (4,749 or 4,748 days) with no nulls.
  - ERA5 against GHCN at the same place peaks at lag 0 (see the cross-checks).
  - Known events are reproduced:
    - Brazil coffee-belt Tmin hits its 2000-2025 minimum on 2021-07-20 (2.2 degC; the frost that spiked arabica). The smoothed grid cell stays above 0 degC although local frost pockets did not.
    - Pampas soil moisture has its lowest yearly mean of 2015-2025 in 2022 (the 2022-23 drought) and its all-period minimum on 2012-01-17 (the 2011-12 drought).
    - Kansas soil moisture is lowest in 2022, 2012, 2002, 2023 and 2011.
    - Illinois July soil moisture is lowest in 2012.
    - Mato Grosso rain averages 0.0-0.1 mm/day in June-August and 7-8 mm/day in December-February.
    - Tokyo's hottest day is 2018-07-23.
- **Download log:**

```
2026-10-18 19:49:05 got nyc 2000-01-01..2012-12-31 HTTP 200 days=4749 grid=(40.75,-74.0) elev=40.0 tz=America/New_York
2026-10-18 19:53:38 got nyc 2013-01-01..2025-12-31 HTTP 200 days=4748 grid=(40.75,-74.0) elev=40.0 tz=America/New_York
2026-10-18 19:53:38 done
2026-10-18 19:53:59 got il_corn 2000-01-01..2012-12-31 HTTP 200 days=4749 grid=(40.5,-89.0) elev=251.0 tz=America/Chicago
2026-10-18 19:58:31 got il_corn 2013-01-01..2025-12-31 HTTP 200 days=4748 grid=(40.5,-89.0) elev=251.0 tz=America/Chicago
2026-10-18 20:03:21 got ks_wheat 2000-01-01..2012-12-31 HTTP 200 days=4749 grid=(38.25,-98.75) elev=563.0 tz=America/Chicago
2026-10-18 20:08:05 got ks_wheat 2013-01-01..2025-12-31 HTTP 200 days=4748 grid=(38.25,-98.75) elev=563.0 tz=America/Chicago
2026-10-18 20:12:41 got ca_valley 2000-01-01..2012-12-31 HTTP 200 days=4749 grid=(36.5,-120.0) elev=58.0 tz=America/Los_Angeles
2026-10-18 20:18:46 non-JSON response for ca_valley 2013-01-01..2025-12-31: b'Unexpected error while streaming data: timeoutReached'; STOPPING
2026-10-18 20:23:20 start run for ['ca_valley', 'tx_lubbock', 'br_coffee', 'br_soy', 'ar_pampas', 'tokyo']; weighted calls used so far today ~2717 (cap 7500)
2026-10-18 20:23:20 have ca_valley 2000-01-01..2012-12-31
2026-10-18 20:23:27 got ca_valley 2013-01-01..2025-12-31 weight=339.1 HTTP 200 days=4748 grid=(36.5,-120.0) elev=58.0 tz=America/Los_Angeles used~3056
2026-10-18 20:28:23 got tx_lubbock 2000-01-01..2012-12-31 weight=339.2 HTTP 200 days=4749 grid=(33.5,-101.75) elev=978.0 tz=America/Chicago used~3395
2026-10-18 20:33:23 got tx_lubbock 2013-01-01..2025-12-31 weight=339.1 HTTP 200 days=4748 grid=(33.5,-101.75) elev=978.0 tz=America/Chicago used~3734
2026-10-18 20:38:24 got br_coffee 2000-01-01..2012-12-31 weight=339.2 HTTP 200 days=4749 grid=(-21.5,-45.5) elev=916.0 tz=America/Sao_Paulo used~4073
2026-10-18 20:43:24 got br_coffee 2013-01-01..2025-12-31 weight=339.1 HTTP 200 days=4748 grid=(-21.5,-45.5) elev=916.0 tz=America/Sao_Paulo used~4412
2026-10-18 20:48:23 got br_soy 2000-01-01..2012-12-31 weight=339.2 HTTP 200 days=4749 grid=(-12.5,-55.75) elev=371.0 tz=America/Cuiaba used~4752
2026-10-18 20:53:24 got br_soy 2013-01-01..2025-12-31 weight=339.1 HTTP 200 days=4748 grid=(-12.5,-55.75) elev=371.0 tz=America/Cuiaba used~5091
2026-10-18 20:58:22 got ar_pampas 2000-01-01..2012-12-31 weight=339.2 HTTP 200 days=4749 grid=(-34.0,-60.5) elev=68.0 tz=America/Argentina/Buenos_Aires used~5430
2026-10-18 21:03:24 got ar_pampas 2013-01-01..2025-12-31 weight=339.1 HTTP 200 days=4748 grid=(-34.0,-60.5) elev=68.0 tz=America/Argentina/Buenos_Aires used~5769
2026-10-18 21:08:22 got tokyo 2000-01-01..2012-12-31 weight=339.2 HTTP 200 days=4749 grid=(35.75,139.75) elev=16.0 tz=Asia/Tokyo used~6108
2026-10-18 21:13:24 got tokyo 2013-01-01..2025-12-31 weight=339.1 HTTP 200 days=4748 grid=(35.75,139.75) elev=16.0 tz=Asia/Tokyo used~6447
2026-10-18 21:13:24 done; weighted calls used today ~6447
```

- **Caveats:**
  - These are reanalysis grid cells (0.25 deg, about 28 km), not stations.
  - Sunshine duration is Open-Meteo's estimate (direct normal irradiance above 120 W/m2), not a sunshine recorder.
  - Southern Hemisphere points have inverted seasons, which the anomaly transform handles.
  - Data end 2025-12-31 so that every value is final ERA5 rather than preliminary ERA5T.

### Skipped or unavailable

- **NOAA Billion-Dollar Disasters** (`https://www.ncei.noaa.gov/access/billions/events-US-1980-2024.csv`, kept in `raw/ncei_billions/`):
  - The list ends with 2024 events; requesting the 1980-2025 file returns the same 1980-2024 file. To my knowledge, NOAA said in 2025 that it would no longer update the product; the landing page does not say so.
  - Which events qualify is decided months later from cost estimates (and is CPI-adjusted), so a dated event series would carry look-ahead unless lagged by months.
  - Drought and heat events span whole years.
- **BoM RMM MJO:** ends 2024-02-24 (ROMI is used instead).
- **GHCN Tokyo and Hong Kong:** records too poor (see section 1).
- **Eastern/Central Pacific HURDAT2:** low relevance to US stocks.
- **CPC climate-division and state degree days:** near-duplicates of the Census divisions.
- **Anything needing an API key** was not used. The NOAA CDO web services (token) would give the same GHCN data, and NASA POWER is an alternative free reanalysis source with no key that was not needed.

## Cross-source checks (dating and units)

The checks are correlations at lags -1/0/+1 days after removing a crude day-of-year mean. A one-day dating error would put the peak at +-1.

```
wx_nyc_hdd vs cpc_hdd_midatlantic                          lag-1=0.616  lag+0=0.931  lag+1=0.771  n=9785  best=+0
wx_chi_hdd vs cpc_hdd_encentral                            lag-1=0.594  lag+0=0.898  lag+1=0.879  n=9785  best=+0
wx_hou_cdd vs cpc_cdd_wscentral                            lag-1=0.607  lag+0=0.799  lag+1=0.684  n=9785  best=+0
wx_dfw_cdd vs cpc_cdd_wscentral                            lag-1=0.546  lag+0=0.805  lag+1=0.793  n=9785  best=+0
wx_sea_hdd vs cpc_hdd_pacific                              lag-1=0.468  lag+0=0.602  lag+1=0.592  n=9784  best=+0
wx_atl_hdd vs cpc_hdd_satlantic                            lag-1=0.535  lag+0=0.831  lag+1=0.838  n=9785  best=+1
wx_den_hdd vs cpc_hdd_mountain                             lag-1=0.574  lag+0=0.794  lag+1=0.700  n=9511  best=+0
wx_msp_hdd vs cpc_hdd_wncentral                            lag-1=0.667  lag+0=0.896  lag+1=0.841  n=9785  best=+0
cpc_hdd_conus vs cpc_hdd_gas_conus                         lag-1=0.876  lag+0=0.987  lag+1=0.819  n=9786  best=+0
wx_nyc_tmax vs wx_phl_tmax                                 lag-1=0.594  lag+0=0.929  lag+1=0.587  n=9785  best=+0
cpc_tele_ao vs cpc_tele_nao                                lag-1=0.526  lag+0=0.561  lag+1=0.544  n=9783  best=+0
usdm_ia_d1plus vs usdm_il_d1plus                           lag-1=nan  lag+0=0.516  lag+1=nan  n=1398  best=-1
ERA5 nyc.temperature_2m_max vs GHCN wx_nyc_tmax            lag-1=0.608  lag+0=0.961  lag+1=0.577  n=9497  best=+0
ERA5 nyc.temperature_2m_min vs GHCN wx_nyc_tmin            lag-1=0.732  lag+0=0.930  lag+1=0.595  n=9497  best=+0
ERA5 nyc.precipitation_sum vs GHCN wx_nyc_prcp             lag-1=0.127  lag+0=0.802  lag+1=0.062  n=9497  best=+0
ERA5 il_corn.temperature_2m_max vs GHCN wx_pia_tmax        lag-1=0.693  lag+0=0.963  lag+1=0.629  n=9495  best=+0
ERA5 ks_wheat.temperature_2m_max vs GHCN wx_ict_tmax       lag-1=0.556  lag+0=0.925  lag+1=0.690  n=9497  best=+0
ERA5 tx_lubbock.temperature_2m_max vs GHCN wx_lbb_tmax     lag-1=0.597  lag+0=0.966  lag+1=0.551  n=9496  best=+0
ERA5 ca_valley.temperature_2m_max vs GHCN wx_fat_tmax      lag-1=0.713  lag+0=0.917  lag+1=0.709  n=9497  best=+0
ERA5 tokyo.temperature_2m_max vs raw GHCN JA000047662 TMAX lag-1=0.513  lag+0=0.944  lag+1=0.477  n=6020  best=+0
ERA5 nyc.cloud_cover_mean vs GHCN wx_nyc_prcp              lag-1=0.094  lag+0=0.532  lag+1=0.260  n=9497  best=+0
```

All pairs peak at lag 0 except Atlanta HDD against South Atlantic division HDD (lag +1 is 0.838 against 0.831 at lag 0). That fits cold fronts reaching Atlanta, at the western edge of the division, before the coastal population. GHCN against ERA5 at the same place peaks sharply at lag 0, so local days line up: New York TMAX 0.96, TMIN 0.93 and PRCP 0.80; TMAX at Peoria 0.96, Wichita 0.93, Lubbock 0.97 and Fresno 0.92; and Tokyo ERA5 against the raw, gappy GHCN Tokyo record (not a series). The USDM pair shows nan at +-1 only because the series is weekly.

## Things the test should be careful about

- **Seasonality.** Temperature, degree days, radiation, ET0, cloud, sunshine, soil moisture, wind and snow are strongly seasonal; use the `anomaly` and `anomaly_log1p` transforms with a day-of-year baseline from earlier years only. Southern Hemisphere points (Sydney, Brazil, Argentina) have inverted seasons. Snow and CDD/HDD are zero for whole seasons, so the anomaly is near zero there.
- **Zero-heavy and event series.** Landfall series are about 99% zeros, EF2+ tornado days 87%, and active hurricanes and ACE are mostly zeros outside June-November; these are rare-event tests with few independent events (81 hurricane-force landfall days in 46 years).
- **Near-duplicates.** TMAX/TMIN/HDD/CDD at one station, nearby stations (Kansas City and Wichita, New Orleans and Lake Charles, New York and Philadelphia; New York and Philadelphia TMAX correlate at 0.93), CPC divisional versus station degree days, and AO versus NAO (0.56). Count the effective number of tests, not the series count.
- **Overlapping or coarse periods.** ONI rows overlap by 2 months (3-month running mean), so monthly changes are smoothed. The weekly Nino series is dated at the week's centre Wednesday and USDM at the Tuesday valid date; the delays above are measured from those dates.
- **Revisions and vintages.** Recent GHCN values are preliminary. ONI and Nino anomalies use base periods chosen after the fact. Teleconnection normalization is today's. HURDAT2 is a post-season reanalysis. ERA5 is a reanalysis. CPC degree-day archives are final values with fixed weights. None of these are real-time vintages. Apart from the anomaly base periods and post-event analysis of the same event (hurricane best tracks, tornado damage surveys), none uses information from well after its date.
- **Inhomogeneities.** Station moves and instrument changes are not corrected (GHCN-D is not homogenized). There is a possible small CPC weighting step at 2014-01-01. OISST moved from v2 to v2.1 in 2020.
- **Day definitions.** US stations use local standard time days; HURDAT2 uses UTC days; Open-Meteo uses local days. A HURDAT2 value for date D covers 20:00 ET on D-1 to 20:00 ET on D (during EDT), so a landfall on the evening of D (ET) can carry date D+1. `known_after_days=1` still holds.

## Reproduction

The scripts are in `scripts/data-weather/`; run each with `python3 -I` from any directory.

1. Download: `dl_ghcnd.sh` (station files listed in `stations.py`), `dl_other.sh` (CPC, PSL, HURDAT2, SPC, USDM, NCEI) and `dl_openmeteo.py` (spaced 270 s apart; stops on the first API error).
2. Parse: `parse_ghcnd.py`, `parse_cpc_dd.py`, `parse_cpc_tele.py`, `parse_cpc_enso.py`, `parse_psl_mjo.py`, `parse_hurdat2.py`, `parse_spc.py`, `parse_usdm.py` and `parse_openmeteo.py`. Each writes `series/*.csv`, a QA JSON per series (`qa/`) and a catalog entry (`catalog_entries.jsonl`).
3. Build: `build_catalog.py` re-validates every file and writes the catalog; `crosscheck.py` runs the lag checks; `make_sources.py` writes this file from `sources_template.md`.

## Appendix: per-series QA (first and last rows, range, gaps)

Every series was checked by the writer (`common.write_series`) and re-checked by `build_catalog.py`: strictly ascending dates (so no duplicates), no blanks, NaN or sentinels, values inside hard plausible ranges, and the catalog's first/last/rows recomputed from the file. Missing days are counted between the first and last date for daily series; the longest gap is between consecutive rows (step in days).

| id | first rows | last rows | min | max | zero % | missing days (daily) | longest gap |
|---|---|---|---|---|---|---|---|
| cpc_cdd_conus | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 4; 2026-10-16 3 | 0 | 17 | 34.0 | 0 | none |
| cpc_cdd_encentral | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 1; 2026-10-16 0 | 0 | 21 | 66.5 | 0 | none |
| cpc_cdd_escentral | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 10; 2026-10-16 4 | 0 | 22 | 47.9 | 0 | none |
| cpc_cdd_midatlantic | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 0; 2026-10-16 0 | 0 | 21 | 68.0 | 0 | none |
| cpc_cdd_mountain | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 0; 2026-10-16 0 | 0 | 20 | 46.1 | 0 | none |
| cpc_cdd_newengland | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 0; 2026-10-16 0 | 0 | 20 | 74.5 | 0 | none |
| cpc_cdd_pacific | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 1; 2026-10-16 4 | 0 | 19 | 60.1 | 0 | none |
| cpc_cdd_satlantic | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 8; 2026-10-16 9 | 0 | 21 | 18.1 | 0 | none |
| cpc_cdd_wncentral | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 1; 2026-10-16 0 | 0 | 21 | 61.9 | 0 | none |
| cpc_cdd_wscentral | 1981-01-01 0; 1981-01-02 0 | 2026-10-15 12; 2026-10-16 3 | 0 | 26 | 33.6 | 0 | none |
| cpc_enso_nino12_weekly | 1981-09-02 -0.1; 1981-09-09 -0.6 | 2026-09-30 5.3; 2026-10-07 5.2 | -2.5 | 5.3 | 3.4 | n/a | none |
| cpc_enso_nino34_weekly | 1981-09-02 -0.2; 1981-09-09 -0.2 | 2026-09-30 3.2; 2026-10-07 3.3 | -2.6 | 3.3 | 4.5 | n/a | none |
| cpc_enso_oni | 1950-02-28 -1.32; 1950-03-31 -1.2 | 2026-08-31 1.8; 2026-09-30 2.16 | -2.04 | 2.59 | 0.5 | n/a | none |
| cpc_hdd_conus | 1981-01-01 28; 1981-01-02 29 | 2026-10-15 4; 2026-10-16 5 | 0 | 51 | 20.2 | 0 | none |
| cpc_hdd_encentral | 1981-01-01 36; 1981-01-02 43 | 2026-10-15 4; 2026-10-16 10 | 0 | 79 | 25.3 | 0 | none |
| cpc_hdd_escentral | 1981-01-01 24; 1981-01-02 29 | 2026-10-15 0; 2026-10-16 0 | 0 | 64 | 40.1 | 0 | none |
| cpc_hdd_gas_conus | 1981-01-01 29; 1981-01-02 32 | 2026-10-15 5; 2026-10-16 7 | 0 | 53 | 19.7 | 0 | none |
| cpc_hdd_midatlantic | 1981-01-01 45; 1981-01-02 42 | 2026-10-15 7; 2026-10-16 5 | 0 | 66 | 25.1 | 0 | none |
| cpc_hdd_mountain | 1981-01-01 22; 1981-01-02 22 | 2026-10-15 13; 2026-10-16 13 | 0 | 57 | 17.3 | 0 | none |
| cpc_hdd_newengland | 1981-01-01 51; 1981-01-02 46 | 2026-10-15 9; 2026-10-16 12 | 0 | 66 | 21.4 | 0 | none |
| cpc_hdd_pacific | 1981-01-01 14; 1981-01-02 16 | 2026-10-15 2; 2026-10-16 1 | 0 | 40 | 16.1 | 0 | none |
| cpc_hdd_satlantic | 1981-01-01 24; 1981-01-02 23 | 2026-10-15 0; 2026-10-16 0 | 0 | 48 | 37.8 | 0 | none |
| cpc_hdd_wncentral | 1981-01-01 33; 1981-01-02 40 | 2026-10-15 10; 2026-10-16 16 | 0 | 79 | 22.2 | 0 | none |
| cpc_hdd_wscentral | 1981-01-01 14; 1981-01-02 17 | 2026-10-15 0; 2026-10-16 3 | 0 | 54 | 47.1 | 0 | none |
| cpc_tele_aao | 1980-01-01 1.038; 1980-01-02 0.897 | 2026-10-15 -0.71; 2026-10-16 -0.339 | -5.125 | 4.635 | 0.0 | 1 | 2003-04-29..2003-05-01 (2d step) |
| cpc_tele_ao | 1980-01-01 -2.546; 1980-01-02 -2.085 | 2026-10-15 1.838; 2026-10-16 1.11 | -6.226 | 6.073 | 0.0 | 1 | 2003-04-29..2003-05-01 (2d step) |
| cpc_tele_nao | 1980-01-01 0.312; 1980-01-02 0.219 | 2026-10-15 0.888; 2026-10-16 0.698 | -3.254 | 2.751 | 0.0 | 2 | 2006-10-25..2006-10-27 (2d step) |
| cpc_tele_pna | 1980-01-01 0.011; 1980-01-02 0.049 | 2026-10-15 0.868; 2026-10-16 1.063 | -3.971 | 3.397 | 0.0 | 2 | 2006-10-25..2006-10-27 (2d step) |
| hurdat_atl_ace | 1980-01-01 0; 1980-01-02 0 | 2025-12-30 0; 2025-12-31 0 | 0 | 16.24 | 83.9 | 0 | none |
| hurdat_atl_hurricanes_active | 1980-01-01 0; 1980-01-02 0 | 2025-12-30 0; 2025-12-31 0 | 0 | 4 | 92.5 | 0 | none |
| hurdat_us_hurricane_landfall | 1980-01-01 0; 1980-01-02 0 | 2025-12-30 0; 2025-12-31 0 | 0 | 1 | 99.5 | 0 | none |
| hurdat_us_landfall_maxwind_kt | 1980-01-01 0; 1980-01-02 0 | 2025-12-30 0; 2025-12-31 0 | 0 | 145 | 99.0 | 0 | none |
| om_ar_pampas_prcp | 2000-01-01 0; 2000-01-02 0 | 2025-12-30 0; 2025-12-31 0 | 0 | 104.9 | 65.1 | 0 | none |
| om_ar_pampas_soilm | 2000-01-01 0.327; 2000-01-02 0.32 | 2025-12-30 0.31; 2025-12-31 0.31 | 0.275 | 0.516 | 0.0 | 0 | none |
| om_ar_pampas_tmax | 2000-01-01 31.9; 2000-01-02 30.7 | 2025-12-30 35.8; 2025-12-31 36.5 | 2.9 | 40.3 | 0.0 | 0 | none |
| om_br_coffee_prcp | 2000-01-01 41; 2000-01-02 30.2 | 2025-12-30 4.2; 2025-12-31 1.1 | 0 | 157.9 | 47.9 | 0 | none |
| om_br_coffee_soilm | 2000-01-01 0.506; 2000-01-02 0.512 | 2025-12-30 0.425; 2025-12-31 0.421 | 0.257 | 0.515 | 0.0 | 0 | none |
| om_br_coffee_tmax | 2000-01-01 22.1; 2000-01-02 21.8 | 2025-12-30 32.6; 2025-12-31 29.9 | 14.9 | 36.8 | 0.0 | 0 | none |
| om_br_coffee_tmin | 2000-01-01 18.6; 2000-01-02 18.6 | 2025-12-30 19.7; 2025-12-31 20.6 | 2.2 | 22.3 | 0.0 | 0 | none |
| om_br_soy_prcp | 2000-01-01 13.7; 2000-01-02 11.4 | 2025-12-30 7.6; 2025-12-31 0.4 | 0 | 93.4 | 40.1 | 0 | none |
| om_br_soy_soilm | 2000-01-01 0.506; 2000-01-02 0.513 | 2025-12-30 0.496; 2025-12-31 0.49 | 0.28 | 0.515 | 0.0 | 0 | none |
| om_br_soy_tmax | 2000-01-01 28; 2000-01-02 26.6 | 2025-12-30 28.7; 2025-12-31 31.5 | 19.6 | 39.2 | 0.0 | 0 | none |
| om_ca_valley_et0 | 2000-01-01 1.5; 2000-01-02 1.78 | 2025-12-30 0.93; 2025-12-31 0.52 | 0.32 | 10.75 | 0.0 | 0 | none |
| om_ca_valley_swrad | 2000-01-01 9.65; 2000-01-02 11 | 2025-12-30 6.2; 2025-12-31 2.56 | 0.69 | 32.44 | 0.0 | 0 | none |
| om_il_corn_et0 | 2000-01-01 1.27; 2000-01-02 1.75 | 2025-12-30 0.64; 2025-12-31 0.67 | 0.18 | 9.41 | 0.0 | 0 | none |
| om_il_corn_soilm | 2000-01-01 0.329; 2000-01-02 0.329 | 2025-12-30 0.371; 2025-12-31 0.366 | 0.138 | 0.437 | 0.0 | 0 | none |
| om_il_corn_swrad | 2000-01-01 7.47; 2000-01-02 7.14 | 2025-12-30 6.43; 2025-12-31 3.62 | 0.44 | 31.48 | 0.0 | 0 | none |
| om_ks_wheat_et0 | 2000-01-01 3.09; 2000-01-02 1.98 | 2025-12-30 1.8; 2025-12-31 2.15 | 0.17 | 12.98 | 0.0 | 0 | none |
| om_ks_wheat_soilm | 2000-01-01 0.195; 2000-01-02 0.194 | 2025-12-30 0.286; 2025-12-31 0.28 | 0.135 | 0.429 | 0.0 | 0 | none |
| om_ks_wheat_swrad | 2000-01-01 9.26; 2000-01-02 10.35 | 2025-12-30 10.39; 2025-12-31 10.34 | 0.72 | 31.36 | 0.0 | 0 | none |
| om_nyc_cloud | 2000-01-01 13; 2000-01-02 92 | 2025-12-30 53; 2025-12-31 64 | 0 | 100 | 1.7 | 0 | none |
| om_nyc_sunshine | 2000-01-01 8.93; 2000-01-02 5 | 2025-12-30 8.94; 2025-12-31 7.92 | 0 | 14.15 | 7.2 | 0 | none |
| om_nyc_swrad | 2000-01-01 8.66; 2000-01-02 5.39 | 2025-12-30 7.92; 2025-12-31 7.73 | 0.45 | 30.98 | 0.0 | 0 | none |
| om_tokyo_prcp | 2000-01-01 0; 2000-01-02 0.1 | 2025-12-30 0; 2025-12-31 0.5 | 0 | 156.2 | 48.3 | 0 | none |
| om_tokyo_tmax | 2000-01-01 11.8; 2000-01-02 8.2 | 2025-12-30 14.7; 2025-12-31 11.2 | 1.4 | 39.2 | 0.0 | 0 | none |
| om_tokyo_tmin | 2000-01-01 2.8; 2000-01-02 1.5 | 2025-12-30 0.1; 2025-12-31 2.1 | -8.2 | 29.2 | 0.4 | 0 | none |
| om_tx_lubbock_et0 | 2000-01-01 4.85; 2000-01-02 3.89 | 2025-12-30 2.22; 2025-12-31 2.52 | 0.2 | 14.26 | 0.0 | 0 | none |
| om_tx_lubbock_soilm | 2000-01-01 0.16; 2000-01-02 0.159 | 2025-12-30 0.166; 2025-12-31 0.166 | 0.131 | 0.424 | 0.0 | 0 | none |
| om_tx_lubbock_swrad | 2000-01-01 12.15; 2000-01-02 13.1 | 2025-12-30 12.75; 2025-12-31 13.03 | 0.78 | 32.03 | 0.0 | 0 | none |
| psl_mjo_romi_amp | 1991-01-01 0.1432; 1991-01-02 0.1918 | 2026-10-13 1.3164; 2026-10-14 1.3212 | 0.00953 | 4.053 | 0.0 | 0 | none |
| psl_mjo_romi_pc1 | 1991-01-01 0.1253; 1991-01-02 0.1854 | 2026-10-13 1.2986; 2026-10-14 1.3119 | -3.441 | 3.169 | 0.0 | 0 | none |
| psl_mjo_romi_pc2 | 1991-01-01 -0.0694; 1991-01-02 -0.0489 | 2026-10-13 -0.2161; 2026-10-14 -0.1564 | -3.662 | 3.708 | 0.0 | 0 | none |
| spc_sig_tornadoes_us | 1980-01-01 0; 1980-01-02 0 | 2025-12-30 0; 2025-12-31 0 | 0 | 69 | 86.7 | 0 | none |
| spc_tornadoes_us | 1980-01-01 0; 1980-01-02 0 | 2025-12-30 0; 2025-12-31 0 | 0 | 210 | 51.7 | 0 | none |
| usdm_ca_d1plus | 2000-01-04 0; 2000-01-11 0 | 2026-10-06 23.26; 2026-10-13 23.26 | 0 | 100 | 15.6 | n/a | none |
| usdm_conus_d1plus | 2000-01-04 23.35; 2000-01-11 24.93 | 2026-10-06 55.21; 2026-10-13 53.73 | 2.28 | 65.45 | 0.0 | n/a | none |
| usdm_ia_d1plus | 2000-01-04 87.66; 2000-01-11 89 | 2026-10-06 0; 2026-10-13 0 | 0 | 100 | 40.6 | n/a | none |
| usdm_il_d1plus | 2000-01-04 73.09; 2000-01-11 80.37 | 2026-10-06 9.59; 2026-10-13 20.9 | 0 | 100 | 47.9 | n/a | none |
| usdm_ks_d1plus | 2000-01-04 7.84; 2000-01-11 9.14 | 2026-10-06 30.85; 2026-10-13 30.85 | 0 | 100 | 16.5 | n/a | none |
| usdm_ne_d1plus | 2000-01-04 56.22; 2000-01-11 63.82 | 2026-10-06 44.29; 2026-10-13 44.29 | 0 | 100 | 22.6 | n/a | none |
| usdm_tx_d1plus | 2000-01-04 73.73; 2000-01-11 74.85 | 2026-10-06 82; 2026-10-13 78.36 | 0 | 100 | 6.0 | n/a | none |
| wx_atl_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 9.6; 2026-10-15 11 | 0 | 27.03 | 49.4 | 0 | none |
| wx_atl_hdd | 1980-01-01 26.5; 1980-01-02 23.5 | 2026-10-14 0; 2026-10-15 0 | 0 | 60 | 50.6 | 0 | none |
| wx_atl_prcp | 1980-01-01 1.8; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 169.7 | 68.5 | 0 | none |
| wx_atl_tmax | 1980-01-01 6.1; 1980-01-02 12.2 | 2026-10-14 30.6; 2026-10-15 31.1 | -8.3 | 41.1 | 0.1 | 0 | none |
| wx_atl_tmin | 1980-01-01 1.1; 1980-01-02 -1.7 | 2026-10-14 16.7; 2026-10-15 17.8 | -22.2 | 27.8 | 1.4 | 0 | none |
| wx_bos_awnd | 1984-01-01 5.7; 1984-01-02 4 | 2026-10-14 4.3; 2026-10-15 4.6 | 0 | 17 | 0.0 | 0 | none |
| wx_bos_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 3; 2026-10-15 0 | 0 | 26.94 | 71.0 | 0 | none |
| wx_bos_hdd | 1980-01-01 33; 1980-01-02 33.5 | 2026-10-14 0; 2026-10-15 8.4 | 0 | 63.51 | 29.0 | 0 | none |
| wx_bos_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0.5; 2026-10-15 0 | 0 | 155.2 | 65.3 | 2 | 2003-08-22..2003-08-25 (3d step) |
| wx_bos_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 599 | 94.1 | 3 | 1982-01-30..1982-02-01 (2d step) |
| wx_bos_tmax | 1980-01-01 3.3; 1980-01-02 2.8 | 2026-10-14 25; 2026-10-15 16.7 | -13.9 | 39.4 | 0.9 | 0 | none |
| wx_bos_tmin | 1980-01-01 -3.3; 1980-01-02 -3.3 | 2026-10-14 15; 2026-10-15 10.6 | -23.2 | 28.3 | 2.1 | 0 | none |
| wx_chi_awnd | 1984-01-01 3.3; 1984-01-02 4 | 2026-10-14 5; 2026-10-15 4.6 | 0.6 | 12.6 | 0.0 | 7 | 1996-11-12..1996-11-15 (3d step) |
| wx_chi_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 27.48 | 69.7 | 0 | none |
| wx_chi_hdd | 1980-01-01 37.5; 1980-01-02 34.4 | 2026-10-14 2.5; 2026-10-15 3.5 | 0 | 83.04 | 30.3 | 0 | none |
| wx_chi_prcp | 1980-01-01 0; 1980-01-02 0.8 | 2026-10-14 2.3; 2026-10-15 23.4 | 0 | 174.2 | 65.5 | 1 | 1997-06-07..1997-06-09 (2d step) |
| wx_chi_snow | 1980-01-01 0; 1980-01-02 8 | 2026-10-14 0; 2026-10-15 0 | 0 | 472 | 92.3 | 190 | 1998-03-31..1998-05-01 (31d step) |
| wx_chi_tmax | 1980-01-01 0; 1980-01-02 0.6 | 2026-10-14 18.9; 2026-10-15 18.9 | -23.9 | 40 | 1.3 | 0 | none |
| wx_chi_tmin | 1980-01-01 -5; 1980-01-02 -2.2 | 2026-10-14 15; 2026-10-15 13.9 | -32.8 | 28.3 | 2.4 | 0 | none |
| wx_dca_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 12.5; 2026-10-15 4 | 0 | 28.56 | 58.5 | 0 | none |
| wx_dca_hdd | 1980-01-01 26.5; 1980-01-02 28 | 2026-10-14 0; 2026-10-15 0 | 0 | 62.97 | 41.5 | 0 | none |
| wx_dca_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 1.5 | 0 | 131.8 | 68.2 | 1 | 2024-05-25..2024-05-27 (2d step) |
| wx_dca_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 417 | 97.8 | 17 | 2010-11-30..2010-12-08 (8d step) |
| wx_dca_tmax | 1980-01-01 8.3; 1980-01-02 6.1 | 2026-10-14 32.2; 2026-10-15 25 | -13.3 | 40.6 | 0.4 | 0 | none |
| wx_dca_tmin | 1980-01-01 -1.1; 1980-01-02 -0.6 | 2026-10-14 18.3; 2026-10-15 16.1 | -20.6 | 28.9 | 2.0 | 0 | none |
| wx_den_cdd | 1994-07-20 8.9; 1994-07-23 13.9 | 2026-10-14 0; 2026-10-15 0 | 0 | 23.07 | 70.8 | 471 | 2013-03-31..2013-12-31 (275d step) |
| wx_den_hdd | 1994-07-20 0; 1994-07-23 0 | 2026-10-14 20; 2026-10-15 20.5 | 0 | 79.8 | 29.2 | 471 | 2013-03-31..2013-12-31 (275d step) |
| wx_den_prcp | 1994-07-20 2; 1994-07-21 0 | 2026-10-14 1; 2026-10-15 0.8 | 0 | 74.2 | 78.4 | 430 | 2013-03-31..2013-11-01 (215d step) |
| wx_den_snow | 2006-02-01 0; 2006-02-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 505 | 93.0 | 62 | 2013-09-30..2013-11-01 (32d step) |
| wx_den_tmax | 1994-07-20 31.6; 1994-07-23 35.5 | 2026-10-14 8.9; 2026-10-15 12.2 | -21 | 40.6 | 0.6 | 471 | 2013-03-31..2013-12-31 (275d step) |
| wx_den_tmin | 1994-07-20 15; 1994-07-21 12.7 | 2026-10-14 5.6; 2026-10-15 1.7 | -31 | 23.3 | 2.3 | 454 | 2013-03-31..2013-12-31 (275d step) |
| wx_dfw_awnd | 1984-01-01 4.8; 1984-01-02 4.4 | 2026-10-14 5.1; 2026-10-15 5.9 | 0.1 | 12.7 | 0.0 | 5 | 1997-04-23..1997-04-27 (4d step) |
| wx_dfw_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 21; 2026-10-15 12 | 0 | 32.52 | 42.9 | 0 | none |
| wx_dfw_hdd | 1980-01-01 17.5; 1980-01-02 14.5 | 2026-10-14 0; 2026-10-15 0 | 0 | 56.85 | 57.1 | 0 | none |
| wx_dfw_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 2.5 | 0 | 148.6 | 78.5 | 2 | 2025-07-18..2025-07-20 (2d step) |
| wx_dfw_tmax | 1980-01-01 17.2; 1980-01-02 15.6 | 2026-10-14 33.9; 2026-10-15 32.2 | -10.6 | 45 | 0.1 | 0 | none |
| wx_dfw_tmin | 1980-01-01 0; 1980-01-02 5 | 2026-10-14 26.1; 2026-10-15 17.8 | -18.8 | 30 | 1.4 | 0 | none |
| wx_dsm_awnd | 1984-01-01 2.6; 1984-01-02 3.1 | 2026-10-13 2.6; 2026-10-14 3.8 | 0.4 | 14.4 | 0.0 | 3 | 1996-08-25..1996-08-28 (3d step) |
| wx_dsm_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0.3; 2026-10-14 0.5 | 0 | 115.1 | 69.4 | 1 | 1997-01-24..1997-01-26 (2d step) |
| wx_dsm_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 0 | 0 | 348 | 93.2 | 0 | none |
| wx_dsm_tmax | 1980-01-01 0; 1980-01-02 -1.1 | 2026-10-13 20; 2026-10-14 16.1 | -24.4 | 42.2 | 1.0 | 1 | 1996-09-14..1996-09-16 (2d step) |
| wx_dsm_tmin | 1980-01-01 -2.2; 1980-01-02 -3.3 | 2026-10-13 12.2; 2026-10-14 10.6 | -32.2 | 27.2 | 2.1 | 1 | 1996-09-14..1996-09-16 (2d step) |
| wx_dtw_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 24.51 | 70.0 | 0 | none |
| wx_dtw_hdd | 1980-01-01 37.5; 1980-01-02 34.4 | 2026-10-14 4.6; 2026-10-15 7 | 0 | 77.01 | 30.0 | 0 | none |
| wx_dtw_prcp | 1980-01-01 0; 1980-01-02 0.3 | 2026-10-14 0; 2026-10-15 3.8 | 0 | 116.1 | 62.8 | 0 | none |
| wx_dtw_snow | 1980-01-01 0; 1980-01-02 5 | 2026-10-14 0; 2026-10-15 0 | 0 | 348 | 89.8 | 13 | 1996-01-31..1996-02-04 (4d step) |
| wx_dtw_tmax | 1980-01-01 -1.1; 1980-01-02 0.6 | 2026-10-14 19.4; 2026-10-15 17.8 | -20 | 40 | 1.1 | 0 | none |
| wx_dtw_tmin | 1980-01-01 -3.9; 1980-01-02 -2.2 | 2026-10-14 12.2; 2026-10-15 11.1 | -29.4 | 26.7 | 2.4 | 0 | none |
| wx_far_awnd | 1984-01-01 1.6; 1984-01-02 5.2 | 2026-10-14 1.9; 2026-10-15 7.4 | 0.3 | 15.1 | 0.0 | 24 | 1997-04-04..1997-04-12 (8d step) |
| wx_far_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 1.8 | 0 | 117.9 | 71.4 | 0 | none |
| wx_far_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 414 | 89.4 | 6 | 2013-09-19..2013-09-21 (2d step) |
| wx_far_tmax | 1980-01-01 -5; 1980-01-02 -10.6 | 2026-10-14 10; 2026-10-15 13.3 | -33.3 | 41.1 | 1.4 | 0 | none |
| wx_far_tmin | 1980-01-01 -10.6; 1980-01-02 -17.2 | 2026-10-14 6.7; 2026-10-15 6.7 | -39.4 | 25.6 | 1.7 | 0 | none |
| wx_fat_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 60.5 | 87.7 | 0 | none |
| wx_fat_tmax | 1980-01-01 15.6; 1980-01-02 13.9 | 2026-10-14 26.1; 2026-10-15 29.4 | 1.1 | 45.6 | 0.0 | 0 | none |
| wx_fat_tmin | 1980-01-01 6.7; 1980-01-02 8.9 | 2026-10-14 10.6; 2026-10-15 11.7 | -7.8 | 32.2 | 1.0 | 0 | none |
| wx_fra_prcp | 1985-11-01 0; 1985-11-02 0.1 | 2026-08-30 8.1; 2026-08-31 4.8 | 0 | 61.5 | 54.1 | 92 | 2000-02-29..2000-05-01 (62d step) |
| wx_fra_tmax | 1985-11-01 7.2; 1985-11-02 8.2 | 2026-08-30 26.8; 2026-08-31 22.9 | -11 | 41.2 | 0.2 | 92 | 2000-02-29..2000-05-01 (62d step) |
| wx_fra_tmin | 1985-11-01 3; 1985-11-02 3.4 | 2026-08-30 17.7; 2026-08-31 15 | -15 | 25 | 0.5 | 92 | 2000-02-29..2000-05-01 (62d step) |
| wx_hou_awnd | 1984-01-01 4.4; 1984-01-02 3 | 2026-10-14 4.2; 2026-10-15 2.9 | 0 | 10.5 | 0.0 | 10 | 1997-06-07..1997-06-13 (6d step) |
| wx_hou_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 17; 2026-10-15 15 | 0 | 30 | 33.6 | 0 | none |
| wx_hou_hdd | 1980-01-01 20.5; 1980-01-02 14 | 2026-10-14 0; 2026-10-15 0 | 0 | 47.49 | 66.4 | 0 | none |
| wx_hou_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0.3; 2026-10-15 0 | 0 | 408.2 | 72.0 | 0 | none |
| wx_hou_tmax | 1980-01-01 16.1; 1980-01-02 21.1 | 2026-10-14 33.3; 2026-10-15 33.3 | -3.8 | 42.8 | 0.0 | 0 | none |
| wx_hou_tmin | 1980-01-01 -2.2; 1980-01-02 0 | 2026-10-14 22.2; 2026-10-15 20 | -13.9 | 28.9 | 0.7 | 0 | none |
| wx_ict_awnd | 1984-01-01 4.5; 1984-01-02 3.1 | 2026-10-13 5.3; 2026-10-14 4.1 | 0 | 14.4 | 0.0 | 93 | 1996-11-17..1996-11-28 (11d step) |
| wx_ict_prcp | 1980-01-01 0; 1980-01-02 0.5 | 2026-10-13 43.2; 2026-10-14 7.1 | 0 | 261.9 | 76.2 | 0 | none |
| wx_ict_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 0 | 0 | 287 | 97.4 | 0 | none |
| wx_ict_tmax | 1980-01-01 7.8; 1980-01-02 3.3 | 2026-10-13 32.8; 2026-10-14 18.9 | -18.9 | 44.4 | 0.6 | 0 | none |
| wx_ict_tmin | 1980-01-01 -3.3; 1980-01-02 -2.2 | 2026-10-13 17.2; 2026-10-14 15 | -29.4 | 28.9 | 1.9 | 0 | none |
| wx_lax_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 4 | 0 | 23.97 | 56.0 | 4 | 2007-05-07..2007-05-09 (2d step) |
| wx_lax_hdd | 1980-01-01 2.5; 1980-01-02 0.5 | 2026-10-13 1; 2026-10-14 0 | 0 | 21.93 | 44.0 | 4 | 2007-05-07..2007-05-09 (2d step) |
| wx_lax_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 0 | 0 | 115.1 | 90.5 | 0 | none |
| wx_lax_tmax | 1980-01-01 21.7; 1980-01-02 26.1 | 2026-10-13 22.8; 2026-10-14 27.2 | 10 | 41.1 | 0.0 | 4 | 2007-05-07..2007-05-09 (2d step) |
| wx_lax_tmin | 1980-01-01 12.2; 1980-01-02 10 | 2026-10-13 12.8; 2026-10-14 13.9 | 0.6 | 25 | 0.0 | 0 | none |
| wx_lbb_awnd | 1984-01-01 4.6; 1984-01-02 4.2 | 2026-10-14 4.3; 2026-10-15 7.9 | 1.1 | 15.1 | 0.0 | 9 | 1996-08-19..1996-08-22 (3d step) |
| wx_lbb_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 16.8; 2026-10-15 32.8 | 0 | 189.5 | 82.7 | 0 | none |
| wx_lbb_tmax | 1980-01-01 17.8; 1980-01-02 11.7 | 2026-10-14 24.4; 2026-10-15 19.4 | -13.3 | 45.6 | 0.3 | 2 | 1997-08-06..1997-08-08 (2d step) |
| wx_lbb_tmin | 1980-01-01 -2.8; 1980-01-02 1.7 | 2026-10-14 17.8; 2026-10-15 12.2 | -18.9 | 30 | 1.9 | 2 | 1996-01-21..1996-01-23 (2d step) |
| wx_lch_awnd | 1984-01-01 3.4; 1984-01-02 2.8 | 2026-10-14 3; 2026-10-15 2.7 | 0 | 12.7 | 0.0 | 46 | 1997-11-06..1997-11-13 (7d step) |
| wx_lch_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 15; 2026-10-15 13 | 0 | 28.56 | 35.5 | 5 | 2007-01-09..2007-01-12 (3d step) |
| wx_lch_hdd | 1980-01-01 18.4; 1980-01-02 16 | 2026-10-14 0; 2026-10-15 0 | 0 | 45.51 | 64.5 | 5 | 2007-01-09..2007-01-12 (3d step) |
| wx_lch_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0.8 | 0 | 398 | 70.4 | 7 | 2025-06-26..2025-06-28 (2d step) |
| wx_lch_tmax | 1980-01-01 15.6; 1980-01-02 17.8 | 2026-10-14 33.3; 2026-10-15 31.1 | -2.2 | 42.8 | 0.0 | 2 | 2007-01-09..2007-01-11 (2d step) |
| wx_lch_tmin | 1980-01-01 0.6; 1980-01-02 1.1 | 2026-10-14 20; 2026-10-15 20 | -14.3 | 28.9 | 0.8 | 4 | 2007-01-10..2007-01-12 (2d step) |
| wx_lon_prcp | 1980-01-01 0; 1980-01-02 0.2 | 2026-08-30 0; 2026-08-31 0 | 0 | 61.8 | 52.6 | 14 | 1997-06-17..1997-07-01 (14d step) |
| wx_lon_tmax | 1980-01-01 2.3; 1980-01-02 10.9 | 2026-08-29 24.6; 2026-08-30 25 | -3.3 | 40.2 | 0.0 | 271 | 2003-11-19..2003-11-23 (4d step) |
| wx_lon_tmin | 1980-01-01 -4.6; 1980-01-02 -6 | 2026-08-30 15.3; 2026-08-31 14.3 | -11.8 | 22.6 | 0.4 | 282 | 1980-12-23..1980-12-26 (3d step) |
| wx_maf_awnd | 1984-01-01 4.4; 1984-01-02 5.4 | 2026-10-13 5.7; 2026-10-14 4.1 | 0.5 | 13 | 0.0 | 10 | 2024-05-30..2024-06-05 (6d step) |
| wx_maf_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 17.5; 2026-10-14 12.5 | 0 | 32.97 | 47.3 | 0 | none |
| wx_maf_hdd | 1980-01-01 18.5; 1980-01-02 18 | 2026-10-13 0; 2026-10-14 0 | 0 | 62.97 | 52.7 | 0 | none |
| wx_maf_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 9.9 | 0 | 118.4 | 86.5 | 0 | none |
| wx_maf_tmax | 1980-01-01 17.8; 1980-01-02 15 | 2026-10-13 33.3; 2026-10-14 30 | -11.1 | 46.7 | 0.2 | 0 | none |
| wx_maf_tmin | 1980-01-01 -1.7; 1980-01-02 1.7 | 2026-10-13 22.8; 2026-10-14 20.6 | -23.9 | 29.4 | 1.8 | 0 | none |
| wx_mci_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0.3; 2026-10-15 1.5 | 0 | 129 | 71.7 | 1 | 2026-03-06..2026-03-08 (2d step) |
| wx_mci_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 279 | 96.2 | 130 | 1996-05-30..1996-10-01 (124d step) |
| wx_mci_tmax | 1980-01-01 0.6; 1980-01-02 -1.7 | 2026-10-14 18.3; 2026-10-15 21.1 | -22.2 | 42.8 | 0.8 | 3 | 1986-03-06..1986-03-08 (2d step) |
| wx_mci_tmin | 1980-01-01 -3.9; 1980-01-02 -3.9 | 2026-10-14 15.6; 2026-10-15 10 | -30.6 | 28.3 | 1.9 | 0 | none |
| wx_mem_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 149.4 | 70.1 | 1 | 1996-06-30..1996-07-02 (2d step) |
| wx_mem_tmax | 1980-01-01 11.1; 1980-01-02 14.4 | 2026-10-14 31.1; 2026-10-15 32.8 | -13.3 | 42.2 | 0.2 | 0 | none |
| wx_mem_tmin | 1980-01-01 0; 1980-01-02 2.2 | 2026-10-14 25; 2026-10-15 22.2 | -20 | 28.9 | 1.4 | 0 | none |
| wx_mia_awnd | 1984-01-01 4.9; 1984-01-02 3.7 | 2026-10-14 3.9; 2026-10-15 4.1 | 0.5 | 17.2 | 0.0 | 9 | 2025-05-31..2025-06-04 (4d step) |
| wx_mia_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 18; 2026-10-15 20 | 0 | 26.04 | 6.6 | 1 | 2026-07-06..2026-07-08 (2d step) |
| wx_mia_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0.3; 2026-10-15 0.5 | 0 | 319 | 62.2 | 0 | none |
| wx_mia_tmax | 1980-01-01 21.1; 1980-01-02 17.8 | 2026-10-14 31.1; 2026-10-15 32.2 | 7.2 | 37.8 | 0.0 | 1 | 2026-07-06..2026-07-08 (2d step) |
| wx_mia_tmin | 1980-01-01 12.2; 1980-01-02 8.9 | 2026-10-14 25.6; 2026-10-15 26.7 | -1.1 | 28.9 | 0.0 | 1 | 2026-07-06..2026-07-08 (2d step) |
| wx_msp_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 25.95 | 71.5 | 0 | none |
| wx_msp_hdd | 1980-01-01 37.5; 1980-01-02 45.5 | 2026-10-14 14.5; 2026-10-15 17 | 0 | 89.52 | 28.5 | 0 | none |
| wx_msp_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 17.5 | 0 | 232.4 | 68.1 | 0 | none |
| wx_msp_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 470 | 89.8 | 1338 | 2000-10-31..2004-07-01 (1339d step) |
| wx_msp_tmax | 1980-01-01 -0.6; 1980-01-02 -3.3 | 2026-10-14 12.8; 2026-10-15 10.6 | -27.2 | 40.6 | 1.5 | 0 | none |
| wx_msp_tmin | 1980-01-01 -4.4; 1980-01-02 -10.6 | 2026-10-14 7.8; 2026-10-15 7.2 | -35.6 | 27.8 | 1.9 | 0 | none |
| wx_msy_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 11; 2026-10-15 12.5 | 0 | 27.57 | 32.7 | 0 | none |
| wx_msy_hdd | 1980-01-01 18.4; 1980-01-02 18 | 2026-10-14 0; 2026-10-15 0 | 0 | 46.05 | 67.3 | 0 | none |
| wx_msy_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 310.9 | 68.6 | 13 | 2026-03-17..2026-03-20 (3d step) |
| wx_msy_tmax | 1980-01-01 15.6; 1980-01-02 17.8 | 2026-10-14 30; 2026-10-15 30.6 | -2.8 | 40.6 | 0.0 | 0 | none |
| wx_msy_tmin | 1980-01-01 0.6; 1980-01-02 -1.1 | 2026-10-14 18.9; 2026-10-15 20 | -11.7 | 28.9 | 0.4 | 0 | none |
| wx_nyc_awnd | 1995-11-01 2.7; 1995-11-02 2.4 | 2026-10-14 2.3; 2026-10-15 2 | 0 | 10.2 | 0.3 | 265 | 2018-10-12..2019-03-11 (150d step) |
| wx_nyc_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 5.5; 2026-10-15 0 | 0 | 29.01 | 64.4 | 0 | none |
| wx_nyc_hdd | 1980-01-01 25.5; 1980-01-02 27 | 2026-10-14 0; 2026-10-15 6 | 0 | 61.53 | 35.6 | 0 | none |
| wx_nyc_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0.5; 2026-10-15 0 | 0 | 192.3 | 66.1 | 0 | none |
| wx_nyc_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 693 | 97.0 | 2 | 1986-02-16..1986-02-18 (2d step) |
| wx_nyc_tmax | 1980-01-01 7.2; 1980-01-02 5.6 | 2026-10-14 25; 2026-10-15 17.8 | -12.8 | 40 | 0.8 | 0 | none |
| wx_nyc_tmin | 1980-01-01 1.1; 1980-01-02 1.1 | 2026-10-14 17.8; 2026-10-15 12.2 | -18.9 | 28.9 | 1.6 | 0 | none |
| wx_okc_awnd | 1984-01-01 4.4; 1984-01-02 3.2 | 2026-10-14 5.1; 2026-10-15 6.9 | 0.8 | 15.7 | 0.0 | 11 | 1996-08-20..1996-08-24 (4d step) |
| wx_okc_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 7; 2026-10-15 0 | 0 | 33.51 | 53.5 | 2 | 1984-03-18..1984-03-20 (2d step) |
| wx_okc_hdd | 1980-01-01 20.9; 1980-01-02 21.9 | 2026-10-14 0; 2026-10-15 2 | 0 | 66.39 | 46.5 | 2 | 1984-03-18..1984-03-20 (2d step) |
| wx_okc_prcp | 1980-01-01 0; 1980-01-02 0.8 | 2026-10-14 142; 2026-10-15 38.4 | 0 | 193.5 | 77.5 | 0 | none |
| wx_okc_tmax | 1980-01-01 15.6; 1980-01-02 11.7 | 2026-10-14 25; 2026-10-15 21.1 | -16.1 | 45 | 0.3 | 2 | 1984-03-18..1984-03-20 (2d step) |
| wx_okc_tmin | 1980-01-01 -2.2; 1980-01-02 0.6 | 2026-10-14 19.4; 2026-10-15 13.3 | -25.5 | 28.9 | 1.8 | 0 | none |
| wx_oma_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 0.3 | 0 | 164.1 | 72.9 | 52 | 1995-10-27..1995-11-03 (7d step) |
| wx_oma_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 0 | 0 | 302 | 94.6 | 1280 | 1997-03-31..1998-12-01 (610d step) |
| wx_oma_tmax | 1980-01-01 -2.2; 1980-01-02 -2.2 | 2026-10-13 15.6; 2026-10-14 16.7 | -24.4 | 42.8 | 1.0 | 153 | 1995-09-29..1996-03-01 (154d step) |
| wx_oma_tmin | 1980-01-01 -4.4; 1980-01-02 -5.6 | 2026-10-13 10; 2026-10-14 7.8 | -30.6 | 28.9 | 2.0 | 153 | 1995-09-29..1996-03-01 (154d step) |
| wx_phl_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 8.5; 2026-10-15 0.5 | 0 | 27.48 | 62.5 | 0 | none |
| wx_phl_hdd | 1980-01-01 30; 1980-01-02 31.5 | 2026-10-14 0; 2026-10-15 0 | 0 | 64.5 | 37.5 | 0 | none |
| wx_phl_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 1; 2026-10-15 0 | 0 | 203.7 | 67.3 | 0 | none |
| wx_phl_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 701 | 96.7 | 0 | none |
| wx_phl_tmax | 1980-01-01 6.1; 1980-01-02 2.8 | 2026-10-14 26.7; 2026-10-15 22.8 | -14.4 | 39.4 | 0.6 | 0 | none |
| wx_phl_tmin | 1980-01-01 -2.8; 1980-01-02 -1.1 | 2026-10-14 19.4; 2026-10-15 14.4 | -21.7 | 28.3 | 2.0 | 0 | none |
| wx_phx_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 7.5; 2026-10-14 3 | 0 | 42.96 | 31.1 | 0 | none |
| wx_phx_hdd | 1980-01-01 8.5; 1980-01-02 9 | 2026-10-13 0; 2026-10-14 0 | 0 | 28.5 | 68.9 | 0 | none |
| wx_phx_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 1.5; 2026-10-14 0 | 0 | 83.8 | 90.5 | 0 | none |
| wx_phx_tmax | 1980-01-01 20; 1980-01-02 21.1 | 2026-10-13 25.6; 2026-10-14 25.6 | 6.1 | 50 | 0.0 | 0 | none |
| wx_phx_tmin | 1980-01-01 7.2; 1980-01-02 5.6 | 2026-10-13 19.4; 2026-10-14 14.4 | -3.3 | 36.1 | 0.1 | 0 | none |
| wx_pia_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 1.5 | 0 | 131.8 | 68.0 | 0 | none |
| wx_pia_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 300 | 94.4 | 296 | 1996-09-30..1997-01-05 (97d step) |
| wx_pia_tmax | 1980-01-01 -1.1; 1980-01-02 -1.7 | 2026-10-14 21.7; 2026-10-15 22.2 | -23.9 | 40.6 | 1.1 | 4 | 2024-10-19..2024-10-22 (3d step) |
| wx_pia_tmin | 1980-01-01 -3.3; 1980-01-02 -2.8 | 2026-10-14 16.7; 2026-10-15 11.7 | -30.6 | 26.1 | 2.3 | 5 | 2026-06-10..2026-06-13 (3d step) |
| wx_pit_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 1.5; 2026-10-15 0 | 0 | 22.53 | 68.5 | 0 | none |
| wx_pit_hdd | 1980-01-01 38; 1980-01-02 38 | 2026-10-14 0; 2026-10-15 6 | 0 | 77.46 | 31.5 | 0 | none |
| wx_pit_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 13.5; 2026-10-15 0 | 0 | 151.1 | 57.9 | 0 | none |
| wx_pit_snow | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 599 | 89.5 | 6 | 2005-01-24..2005-01-26 (2d step) |
| wx_pit_tmax | 1980-01-01 -2.2; 1980-01-02 -2.2 | 2026-10-14 24.4; 2026-10-15 18.3 | -19.4 | 39.4 | 1.2 | 0 | none |
| wx_pit_tmin | 1980-01-01 -3.3; 1980-01-02 -3.3 | 2026-10-14 13.9; 2026-10-15 11.7 | -30 | 25 | 2.7 | 0 | none |
| wx_sea_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-14 0; 2026-10-15 0 | 0 | 23.52 | 85.2 | 1 | 2024-04-24..2024-04-26 (2d step) |
| wx_sea_hdd | 1980-01-01 17; 1980-01-02 23.5 | 2026-10-14 6; 2026-10-15 6 | 0 | 50.55 | 14.8 | 1 | 2024-04-24..2024-04-26 (2d step) |
| wx_sea_prcp | 1980-01-01 5.6; 1980-01-02 4.8 | 2026-10-14 0; 2026-10-15 0 | 0 | 127.5 | 58.3 | 9 | 1998-06-01..1998-06-04 (3d step) |
| wx_sea_tmax | 1980-01-01 11.1; 1980-01-02 7.8 | 2026-10-14 19.4; 2026-10-15 20 | -7.8 | 42.2 | 0.1 | 0 | none |
| wx_sea_tmin | 1980-01-01 6.7; 1980-01-02 2.8 | 2026-10-14 10.6; 2026-10-15 10 | -13.9 | 22.8 | 1.5 | 1 | 2024-04-24..2024-04-26 (2d step) |
| wx_sfo_cdd | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 0 | 0 | 22.98 | 85.0 | 23 | 2018-06-21..2018-07-12 (21d step) |
| wx_sfo_hdd | 1980-01-01 14; 1980-01-02 19.5 | 2026-10-13 2.5; 2026-10-14 1 | 0 | 32.55 | 15.0 | 23 | 2018-06-21..2018-07-12 (21d step) |
| wx_sfo_prcp | 1980-01-01 0; 1980-01-02 0 | 2026-10-13 0; 2026-10-14 0 | 0 | 142 | 82.6 | 0 | none |
| wx_sfo_tmax | 1980-01-01 13.9; 1980-01-02 8.9 | 2026-10-13 23.3; 2026-10-14 24.4 | 3.3 | 40.6 | 0.0 | 22 | 2018-06-21..2018-07-12 (21d step) |
| wx_sfo_tmin | 1980-01-01 7.2; 1980-01-02 6.1 | 2026-10-13 10.6; 2026-10-14 11.1 | -2.8 | 22.2 | 0.0 | 21 | 2018-06-21..2018-07-12 (21d step) |
| wx_syd_prcp | 1980-01-01 0; 1980-01-02 18 | 2026-10-07 0.4; 2026-10-08 0.8 | 0 | 216.2 | 64.8 | 33 | 2025-09-25..2025-09-30 (5d step) |
| wx_syd_tmax | 1980-01-01 25.1; 1980-01-02 20.5 | 2026-10-07 17.4; 2026-10-08 19.2 | 9.7 | 46.4 | 0.0 | 5 | 2003-08-10..2003-08-13 (3d step) |
| wx_syd_tmin | 1980-01-01 21; 1980-01-02 17.9 | 2026-10-07 15.4; 2026-10-08 13 | 0.8 | 27.5 | 0.0 | 10 | 1987-03-19..1987-03-21 (2d step) |
