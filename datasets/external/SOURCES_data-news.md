# SOURCES: data-news (daily news-headline sentiment)

Agent `data-news`, built 2026-10-18/19. Catalog: `catalog_data-news.csv`. Category `news` for every series.
Scripts (stdlib + numpy; VADER from the venv): `scripts/data-news/` (run with `python3 -I`, or
`~/programming/c2-wt/venv-data/bin/python -I` for anything that scores text).

## Summary

135 series (127 daily, 8 weekly), all category `news`, all transform `level`, from eight free headline
archives; 14.46 million distinct day-headlines scored (Reuters 4.75M, Times of India 3.49M, All the News
2.52M, Irish Times 1.48M, ABC 1.24M, Hacker News > 20 points 0.55M, HuffPost 0.21M, Wikipedia 0.10M as of
the next day + 0.12M today's revision):

| source | ids | series | first | last | frequency | known_after_days | default periods it can join (estimate) |
|---|---|---|---|---|---|---|---|
| Wikipedia Current events, page as of D+1 19:30 UTC | `wikice_*` (not `_latest`, `_sec`, `_wk`) | 9 | 2006-08-01 | 2026-10-16 | daily | 1 | 242, 262, 282 (full history borderline, 0.90) |
| Wikipedia, daily section counts | `wikice_sec_*` | 11 | 2010-07-01 | 2026-10-16 | daily | 1 | 262, 282 |
| Wikipedia, weekly section tone | `wikice_wk_*`, `wikice_wk4_*` | 8 | 2010-07-05 | 2026-10-05 | weekly | 7 | 262, 282 |
| Wikipedia, today's revision (**hindsight-edited**) | `wikice_latest_*` | 5 | 2003-01-01 | 2026-10-16 | daily | 1 | all |
| Hacker News, stories > 50 points | `hn_*` | 11 | 2007-02-19 | 2026-10-16 | daily | 2 | 242, 262, 282 |
| Hacker News, stories > 20 points | `hn_p20_*` | 5 | 2007-02-19 | 2026-10-16 | daily | 2 | 242, 262, 282 |
| Reuters US daily archive (Wayback) | `reut_*` | 13 | 2007-01-01 | 2014-12-31 | daily | 1 | 242 |
| ABC News Australia | `abc_*` | 12 | 2003-02-19 | 2021-12-31 | daily | 2 | 222, 242, 262 |
| Times of India | `toi_*` | 13 | 2001-01-02 | 2022-03-31 | daily | 2 | 222, 242, 262 |
| Times of India, business sections | `toi_biz_*` | 5 | 2001-05-22 | 2022-03-31 | daily | 2 | 222, 242, 262 |
| Irish Times | `irish_*` | 12 | 1996-01-02 | 2019-12-31 | daily | 1 | 222, 242, 262 |
| Irish Times, business sections | `irish_biz_*` | 5 | 1996-01-02 | 2019-12-31 | daily | 1 | 222, 242, 262 |
| HuffPost (News Category Dataset v3) | `huff_*` | 8 | 2012-01-28 | 2022-09-22 | daily | 1 | 262 |
| All the News 2.0, 27 US publications | `atn_*` | 13 | 2016-01-01 | 2020-03-27 | daily | 1 | none (4.25 years) |
| All the News 2.0, Reuters + CNBC + Business Insider | `atn_fin_*` | 5 | 2016-01-01 | 2020-03-27 | daily | 1 | none |

Period estimate: share of weekdays in each census correlation period (222 ~2005-2010, 242 ~2010-2015,
262 2015-01-07..2020-01-08, 282 2020-01-09..2025-01-13, 284 full history) on which the source's VADER
series is live (an observation within 10 trading days, after a 126-day standardization burn-in); >= 0.90 is
needed. The test's own `externals.csv` gives the exact answer. Only Wikipedia and Hacker News cover the whole
2020-2025 period (282); ABC, Times of India and HuffPost stop in 2021-2022.

Not obtainable without registration: The Guardian Open Platform (the shared `test` key now returns 401),
the NYT Archive API, and the Kaggle datasets listed under "Not used".

## What each series is

For every source and day D (the source's own calendar day; see timing below):

| suffix | meaning | units |
|---|---|---|
| `_count` | number of distinct headlines that day (identical text within a day counted once) | headlines/day |
| `_vader` | mean VADER compound score of the day's headlines | -1..1 |
| `_lm_negshare` | share of the day's headlines with at least one Loughran-McDonald (LM) negative word | 0..1 |
| `_lm_tone` | LM net tone pooled over the day: (positive words - negative words) / all words | ratio |
| `_crisis_any` | headlines matching any crisis word (below) | headlines/day |
| `_crisis_<word>` | headlines matching one crisis word; a word gets its own series only where it is non-zero on >= 2% of the source's scored days (rarer words count only towards `_crisis_any`; the skipped words are listed per source below). Not made for the business-section subsets (their parent source has them), the HN >20 variant or the Wikipedia latest-revision set | headlines/day |

Days with fewer than 5 headlines get **no value** for `_vader`, `_lm_negshare`, `_lm_tone` and the crisis
counts (rather than a noisy one). `_count` is written for every day the source has data; for Hacker News
and Wikipedia, which were queried for every single day, a zero is a real zero.

Wikipedia also has per-section series (see its section below).

### Scoring method (transparent lexicons, CPU only)

- **VADER** (Hutto & Gilbert 2014), package `vaderSentiment` 3.3.2 installed in
  `~/programming/c2-wt/venv-data`, MIT licence. General-purpose lexicon plus rules (negation, intensifiers,
  capitals, punctuation). Compound score per headline, then the daily mean. VADER's lexicon misses many
  finance words ("Lehman Brothers files for bankruptcy" scores 0.0), which is what LM is for; ABC's headlines
  are lower-case without punctuation, so VADER's capitals/punctuation rules don't fire there.
- **Loughran-McDonald Master Dictionary 1993-2025** (Notre Dame SRAF, updated March 2026), file
  `raw/lexicons/Loughran-McDonald_MasterDictionary_1993-2025.csv`. A word is in the Negative/Positive list
  when its column holds a positive year (a negative year means it was removed): 2,345 negative and 347
  positive words. Tokens are maximal runs of ASCII letters, upper-cased (the dictionary lists inflections,
  so no stemming); no negation handling. Net tone is pooled per day (sum over headlines of positive minus
  negative words, divided by all tokens), which is more robust than averaging per-headline ratios of
  8-word headlines. **Licence:** "The dictionary/sentiment lists are free for use in academic research.
  For commercial licenses, please contact us at loughranmcdonald@gmail.com." Fine for this personal
  research; a commercial use would need their licence.
- **Crisis words** (case-insensitive regexes, word-bounded): recession(s)/recessionary;
  crash/crashes/crashed/crashing; bankrupt/bankruptcy/bankruptcies/bankrupted; layoff(s)/lay-off(s)/
  lay(s) off/laid off/laying off/job cut(s)/redundancy/redundancies; inflation/inflationary;
  war(s)/warfare/wartime except "star wars"; default(s)/defaulted/defaulting; pandemic(s).
  In general news "crash" is mostly vehicle and plane crashes and "war" includes figurative wars (price war,
  war of words). On Hacker News "default" is almost always technical ("on by default"), so it is
  excluded there; "crash" there is often a software crash.
- Scorer: `scripts/data-news/score.py` (one module for every source), about 50,000 headlines/s per core.
  Scorer validation: ABC's daily VADER falls about 1 sd on Australian disaster days (Boxing Day tsunami
  -0.86, Black Saturday fires -1.49, Brisbane floods -1.15, MH17 -0.92, Christchurch shooting -0.90,
  2019-20 bushfires -1.28; z of the 3-day window mean against the daily values of the prior year).

### Transform: `level` for everything (counts included), and why

The test's `level` subtracts a trailing 252-trading-day mean before standardizing (minus the trailing mean,
divided by the trailing sd), which absorbs slow drift such as archive growth. `anomaly` compares a value
with the same calendar date in the previous 3-10 years, which drifts permanently on a growing archive (Hacker
News, Wikipedia, Times of India all grow) and would throw away each source's first 3 years. The data show
almost no annual cycle to remove: the share of variance explained by month of year (after removing a
centred 365-day mean) is <= 0.03 for every series longer than five years (table below, `doy_r2`; All
the News, four years with COVID in one spring, reaches 0.09). What is strong is the **day-of-week**
pattern: weekend volume and tone differ (`dow_r2` up to 0.85 for counts, 0.61 for Reuters' VADER, whose
weekend headlines are a different mix; most other tone series <= 0.2). Neither transform removes it, and with `known_after_days` 1 the test averages the Friday, Saturday
and Sunday values onto Monday's row, so Monday rows of count series are systematically low. Flagged, not
fixed: a day-of-week pattern in B would correlate with it.

### Timing: `known_after_days`

| source | day boundary | known_after_days | why |
|---|---|---|---|
| Wikipedia Current events (`wikice_`) | page date; revision taken as of D+1 19:30 UTC | 1 | 19:30 UTC is 15:30 EDT / 14:30 EST, before the close on D+1 |
| Wikipedia weekly section tone | Monday-Sunday, dated Monday | 7 | Sunday's page is taken as of Monday 19:30 UTC |
| Wikipedia today's revision (`wikice_latest_`) | page date | 1 (nominal) | **contains hindsight edits**, see below |
| Hacker News (`hn_`, `hn_p20_`) | New York calendar day of the post | 2 | `points` is the final score, so the >50-point filter uses later votes; with 2, every story has had >= 40 h of voting by the close it is first used at |
| Reuters US archive (`reut_`) | US Eastern day (archive page date) | 1 | items stamped in EDT/EST; day complete at New York midnight |
| ABC News Australia (`abc_`) | Australian local date | 2 | as instructed for far-off time zones. Verified the dates are Sydney dates (Monday 2008-09-15's Wall St fall is reported as "meldown monday hits wall street" under 2008-09-16), so day D is complete by about 08:00-10:00 New York time on D: 1, even 0, would also be free of look-ahead |
| Times of India (`toi_`) | presumably IST date (not documented) | 2 | far-off time zone; 1 would be safe for an IST or UTC date |
| Irish Times (`irish_`) | GMT date (documented) | 1 | GMT day ends 19:00-20:00 New York time, after the close |
| HuffPost (`huff_`) | US date | 1 | |
| All the News 2.0 (`atn_`) | date part of the timestamp (time zone not stated; US publications) | 1 | |

## Sources used

<!-- SOURCES-DETAIL:begin -->
### Wikipedia "Portal:Current events" (`wikice_*` 28 series as of the next day, `wikice_latest_*` 5 flagged)

- English Wikipedia's daily current-events pages, e.g. `Portal:Current_events/2008_September_15`: one-line
  summaries of the day's (mostly world) news written by volunteer editors, with source links. Titles are
  `Portal:Current events/YYYY Month D` in every year (no zero padding; a handful of old zero-padded titles
  are redirects). A `list=allpages` enumeration (Portal namespace 100, prefix "Current events/",
  non-redirects) found 8,913 daily pages, 2002-01-01..2026-10-19: 2002 is partial (223 days), every day
  from 2003 exists. Text licence CC BY-SA 4.0; only derived statistics are kept.
- MediaWiki API (`en.wikipedia.org/w/api.php`), descriptive User-Agent with the contact email set in `C2_CONTACT`,
  strictly sequential, ~1 request/s, `maxlag=5`. Raw, unmodified API responses:
  `raw/wikipedia_current_events/allpages_portal_current_events_2026-10-19.json` (the page list);
  `latest/YYYY-MM.json` (today's revision of every page, one call per month, 298 calls);
  `asof/YYYY/YYYY-MM-DD.json` (for every day 2006-01-01..2026-10-16, the newest revision made at or before
  D+1 19:30 UTC, one call per day, 7,594 calls).
- **Why the as-of version is the main one:** the daily pages before mid-2006 were created retroactively
  (the 2003 pages were created in September 2017, the 2005 pages in February 2006, from monthly archives),
  and every page keeps being edited for years. Today's revision therefore carries hindsight (which events
  editors later judged important). Taking the revision as it stood at D+1 19:30 UTC removes that and makes
  `known_after_days` 1 exact. 7,406 days have such a revision; the 188 without are 2006-01..2006-07, so the
  main series start 2006-08-01 (the first month from which >= 90% of days have one). Compared with today's
  revision, the as-of pages hold 88-95% of the items (2006-2009: 0.95, 2010-14: 0.95, 2015-19: 0.93,
  2020-24: 0.92, 2025: 0.88) and the daily tone correlates 0.95 (VADER) / 0.94 (LM) with today's. Three days were
  blank at the cutoff although today's page has >= 5 items (2007-02-12, 2012-04-04, 2015-12-19; probably
  vandalism being reverted).
- **Parsing** (`scripts/data-news/wikice_parse.py`): items are the leaf bullet lines. A bullet with deeper
  bullets under it is a topic label (`*[[Russian invasion of Ukraine]]` over `**Russian forces ...`), and
  each item's text is its labels joined with ": " in front of the leaf, which makes the post-2010 nested
  format read like the older inline "Story: sentence" items. Cleanup drops comments, `<ref>`s, templates
  (a few content templates such as `{{convert}}` are expanded), external links with their source label,
  images; `[[a|b]]` becomes b. Leaves under 4 words are dropped. Median items per day: 7-11 (2006-2009),
  20 (2010), 18 (2011), 9-15 (2012-2019), 17-21 (2020-2021), 11-13 (2022-2025).
- **Sections:** topic headers (`;Name` 2010-2020, `'''Name'''` since 2020) exist from 2010-07 (100% of items
  sectioned from 2010H2). 186 header variants are mapped by word-start keywords to ten sections:
  conflict ("Armed conflicts and attacks", 22k items), law_crime, politics ("Politics and elections"),
  disasters ("Disasters and accidents"), health_env ("Health and environment"), intl ("International
  relations"), business ("Business and economy", 5.1k), sports, arts ("Arts and culture"), science_tech;
  156 items under headers like "Current events" or "News" are left unmapped. Daily section counts from
  2010-07-01 (`wikice_sec_<section>_count`, plus `wikice_sec_conflict_disasters_count`). Breaks: health_env
  has 0-0.3 items a day before 2020 and 8-11 a day in 2020-2021 (COVID);
  conflict drops to ~1.7/day in 2017-2019 and rises again from 2022.
- **Section tone** cannot be daily under the 5-item floor (business averages 0.8 items a day, conflict
  3.4), so it is weekly (Monday-Sunday, dated the Monday, `known_after_days` 7) for four groups:
  conflict+disasters, politics+international relations, law and crime (one-week windows, values for 96-100%
  of weeks), and business, pooled over a trailing 4-week window (one-week windows left half the weeks
  empty). VADER and LM tone each.
- Crisis words with own series: crash, bankruptcy, war, pandemic (recession 1.6%, layoffs 1.1%,
  inflation 1.9%, default 0.9% of days are below the 2% rule).
- `wikice_latest_*` (count, VADER, LM negative share, LM tone, crisis_any, 2003-01-01..2026-10-16): today's
  revisions, **hindsight-edited**, kept only because they reach back to 2003 (the as-of set starts 2006-08).

### Hacker News (`hn_*` 11 series for stories with >50 points, `hn_p20_*` 5 for >20 points)

- Algolia HN Search API, `https://hn.algolia.com/api/v1/search_by_date?tags=story&numericFilters=
  created_at_i>=A,created_at_i<B,points>10&hitsPerPage=1000`, one query per New York calendar day
  (A, B = local midnights, DST-aware), 2007-02-19..2026-10-16, 7,180 queries, ~1 request/s (Algolia allows
  10,000/hour). No day came near the 1,000-hit cap (at most 240 stories with >10 points, on 2022-02-27),
  so no window needed splitting (the fetcher would have halved it automatically). Raw responses gzip-compressed (lossless):
  `raw/hn_algolia/YYYY/YYYY-MM-DD.json.gz`. HN content is user-submitted links and titles; only
  aggregate statistics of titles are kept.
- The cached superset (>10 points) is filtered locally: >50 points (351,246 titles; median per day
  3 in 2008, 11 in 2009, 25 in 2010, 36-80 from 2011) and >20 points (551,297; 14.5 a day in 2008). With
  the 5-title floor the >50 set has tone values on 25% of 2008 days and ~100% from 2009; the >20 variant
  reaches 97% in 2008, one year earlier, and is less noisy.
- `points` is the final score, so the membership of a day's well-voted set uses votes cast after the day:
  `known_after_days` 2. Titles are the current (possibly moderator-edited) ones.
- Crisis words: `default` excluded on HN (technical sense); own series for crash, bankruptcy, layoffs,
  inflation, war, pandemic (recession is on 1.6% of days). "crash" is often a software crash; "war" is often
  figurative ("browser wars").

### Reuters US news archive via the Wayback Machine (`reut_*`, 13 series)

- Reuters used to publish one page per US Eastern calendar day listing every headline it put out that day
  (`reuters.com/resources/archive/us/YYYYMMDD.html`, items stamped e.g. "11:58 PM EDT"); the live archive
  is gone. A CDX query found Wayback captures for every day of 2007-2013, 298 days of 2014 and scattered
  days in 2015-2017. Not requested by name; added because it is the only US financial-headline source here
  before 2016 and covers 2008 and 2011.
- Internet Archive CDX API (`web.archive.org/cdx/search/cdx`, one listing per year, saved in
  `raw/reuters_archive_wayback/cdx/`), then one page per day with the `id_` modifier (original bytes, no
  Wayback rewriting), sequential, ~1 request/s plus download time (~3 s/page). 2,885 pages, 0 failures,
  gzip-compressed (lossless) as `raw/reuters_archive_wayback/YYYY/YYYYMMDD_<capture timestamp>.html.gz`.
  Per day the earliest capture taken >= 2 days after the date was used (`capture_plan_original.json`).
  Reuters content is copyrighted; only aggregate statistics of the headlines are kept.
- **Truncated captures:** 21 first-choice captures (mostly a 2008-09 crawl of 2007-2008 pages) held only
  ~20% of the day's items; they were replaced by the largest capture taken >= 2 days after the date
  (`fetch_reuters_fix.py`, `capture_plan.json` now points at the replacements, the old files stay in raw/).
  8 more non-holiday days below 25% of normal volume with no better capture are left missing
  (2007-11-10, 2008-08-02, 2008-09-06, 2008-09-13, 2008-11-15, 2010-11-24, 2010-12-27, 2014-11-10; the two
  2010 ones may be genuine pre-holiday lulls). US market holidays and Christmas/New Year days keep their low
  but genuine counts.
- Items: `<div class="headlineMed">` links. Dropped: press releases (URL `pressRelease`) and
  video stories (~1.6M items together), Briefing.com stubs ("Stock Market Update - ...", "On The Wires"), "Company Profile for",
  formulaic prefixes TABLE/DIARY/PRESS DIGEST/TEXT/ADVISORY/BRIEF/CALENDAR/SCHEDULE (~185k). Version
  prefixes UPDATE n/RPT/REFILE/CORRECTED/WRAPUP n/INSTANT VIEW n are stripped and the day's titles
  de-duplicated (~576k repeats). 4,746,424 headlines on the 2,837 days used.
- 2007-01-01..2014-12-31, 2,837 days, 85 missing (mostly 2014 Q4). Volume steps with the site's structure:
  median ~950 headlines a weekday in 2007-2009, ~2,200 in 2010, ~4,200 in 2011, ~3,800 in 2012, ~2,800-3,000
  in 2013-2014; weekends ~200-300. Ends 2014-12-31: Q4 2014 is sparse and the January 2015 pages behave
  differently (Monday pages hold ~10% of other weekdays), so they are not used.
- All eight crisis words have their own series.

### ABC News Australia: "A Million News Headlines" (`abc_*`, 12 series)

- Harvard Dataverse doi:10.7910/DVN/SYBGZL (Rohit Kulkarni, version 6, 2022-06-11), file id 6329050
  downloaded in original format: `raw/dataverse_abc/abcnews-date-text.csv` (63.8 MB, md5
  5de6fd1235fe029fdc0bfd6b13a1c5e8, matches Dataverse), plus the dataset metadata JSON. **Licence CC0 1.0.**
- 1,244,184 headlines, 2003-02-19..2021-12-31, `publish_date` (yyyyMMdd) + lower-case ASCII text without
  punctuation. 6,882 days, 9 days missing. Median ~190 a day; volume falls from ~250/day (2013) to ~50/day
  (2021) as the archive thins.
- Dates are Australian local dates (checked: Wall Street's Monday 2008-09-15 fall appears under 2008-09-16).
- Crisis words with their own series: all but `default` (non-zero on 1.2% of days).

### Times of India: "Times of India News Headlines" (`toi_*` 13 series, `toi_biz_*` 5)

- Harvard Dataverse doi:10.7910/DVN/DPQMQH (R. Kulkarni, version 3, 2022-04-13), file id 6175512:
  `raw/dataverse_toi/india-news-headlines.csv` (265.8 MB, md5 3b30867a2212eea03a86fa35088ce0be, matches).
  **Licence CC0 1.0.**
- 3,650,970 rows, 2001-01-02..2022-03-31, with `headline_category` (dot-delimited, e.g. `city.mumbai`,
  `business.india-business`). The same headline is often filed under several city sections: 163,126
  same-day duplicates removed. Mostly city news (2.1M rows). Volume ~150/day in 2001-2007, ~700/day
  2011-2018, ~500/day after. 42 days missing (mostly 2001). All eight crisis words have their own series.
- `toi_biz_*`: categories starting `business` (157k unique headlines, ~21 a day), from 2001-05-22.

### The Irish Times: "News Headlines of Ireland" (`irish_*` 12 series, `irish_biz_*` 5)

- Harvard Dataverse doi:10.7910/DVN/0U9Z9F (R. Kulkarni, version 2, 2020-04-06), file id 3790780 in
  original format: `raw/dataverse_irish/ireland-news-headlines.csv` (97.0 MB, md5
  4e8cecdf1fe6db71e4fc8e6140a71068, matches). **Licence CC0 1.0.**
- 1,484,340 rows 1996-01-02..2019-12-31 (news, sport, business, opinion, culture, lifestyle; 1,108 rows in
  category `removed` dropped). `publish_date` documented as the GMT date. ~190 a day; 341 days missing,
  264 of them Sundays (no Sunday paper in the early years). Crisis words: all but `pandemic` (1.0% of days;
  the data end in 2019).
- `irish_biz_*`: categories starting `business` (203k, ~25 a day); most Sundays empty.

### HuffPost: "News Category Dataset" v3 (`huff_*`, 8 series)

- Rishabh Misra (2022), arXiv:2209.11429. The author's page (rishabhmisra.github.io/publications) links
  only to Kaggle (login needed); downloaded the v3 JSON-lines file from the ungated Hugging Face mirror
  `heegyu/news-category-dataset` (`data.json`, 87.3 MB; its card says "Dataset from
  kaggle.com/datasets/rmisra/news-category-dataset"; v3 fields, 209,527 records, matching the paper's
  "around 210k"; not checked byte-for-byte against Kaggle):
  `raw/huffpost_news_category/News_Category_Dataset_v3.json`. **Licence CC BY 4.0.**
- 209,527 headlines with category, 2012-01-28..2022-09-23. ~90-100 a day through 2018-05-26 (where v2 of the
  dataset ended), then only ~5 a day (v3 appended a thin sample). `huff_count` is cut at 2018-05-26; the
  tone series continue to 2022-09-22 under the 5-headline floor but are much noisier after the break.
  Joins no default 1260-day correlation period reliably; mostly a cross-check for 2012-2018. Crisis words
  with their own series: crash, war, pandemic (the others are on < 2% of days).

### All the News 2.0 (`atn_*` 13 series, `atn_fin_*` 5)

- Andrew Thompson / Components (components.one, now components.news), released 2020-04-03, modified
  2022-07-09. The original Dropbox link still serves it: `all-the-news-2-1.zip`, 3,343,521,557 bytes
  (under the ~10 GB limit), `raw/all_the_news_2/`. Parsed by streaming the 8.8 GB CSV out of the zip
  (never extracted). **Licence/terms: "THIS DATASET IS FOR NON-COMMERCIAL, RESEARCH PURPOSES ONLY. IT IS
  VERY MUCH NOT FOR TRAINING COMMERCIAL GENERATIVE MODELS."** Only titles are used, for aggregate statistics.
- 2,688,878 articles from 27 US publications, 2016-01-01..2020-04-02 (Reuters 840k, NYT 252k, CNBC 238k,
  The Hill 208k, People 136k, CNN 128k, ...). 2,524,231 unique day-titles; median ~1,950 a day, 0 days
  missing. The scrape is incomplete from 2020-03-28 (Monday 2020-03-30 has 45% of a normal weekday), so
  every series ends 2020-03-27.
- `atn_fin_*`: the business-focused publications Reuters, CNBC and Business Insider (~650 a day).
- All eight crisis words have their own series (`atn_` only).
- At 4.25 years it cannot reach 90% of any default 1260-day correlation period; usable only with shorter
  periods (`--periods`).
<!-- SOURCES-DETAIL:end -->

## Not used, and why

- **The Guardian Open Platform** (`content.guardianapis.com/search`): the shared `api-key=test` key now
  returns HTTP 401 "Unauthorized" (tested 2026-10-19 UTC on /search with and without filters). Business and
  world headlines back to 1999 are **available with a free developer key** (registration; the free tier
  is rate-limited, reportedly about 500 calls a day, not verified here). Not registered, per the rules.
- **New York Times Archive API** (`api.nytimes.com/svc/archive/v1/{year}/{month}.json`): every NYT headline
  since 1851, one call per month; **available with a free key** (developer.nytimes.com registration).
- **Kaggle** (all need a login; **available with a free account**):
  `aaron7sun/stocknews` "Daily News for Stock Market Prediction" (top 25 r/worldnews headlines per day,
  2008-06-08..2016-07-01, CC BY-NC-SA 4.0, 15 MB);
  `notlucasp/financial-news-headlines` "Financial News Headlines Data" (CNBC, Guardian, Reuters business
  headlines, ~2017-12..2020-07, CC0, 12 MB);
  `miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests` "Daily Financial News for 6000+ Stocks"
  (~4M ticker-tagged headlines 2009-2020, CC0, 0.9 GB; per-stock news, a natural next step);
  `jeet2016/us-financial-news-articles` (306k articles, Jan-May 2018 only, too short);
  `rmisra/news-category-dataset` (HuffPost; used here via its Hugging Face mirror) and `therohk/*` (the ABC,
  Times of India and Irish Times files; used here from Harvard Dataverse instead).
- **GDELT**: the DOC API was judged unsuitable (query-dependent, reprocessed, 2017+). The GDELT 1.0/2.0
  event and GKG files (tone per document from 2013, events back to 1979) are hundreds of GB; skipped.
- **Reddit** (r/worldnews etc.): Pushshift dumps are no longer public and the Reddit API needs OAuth.
- **Reuters RCV1/TRC2, NYT Annotated Corpus, Bloomberg/Reuters (Ding et al. 2014)**: licence agreements or
  paid (LDC), not free downloads.
- **SF Fed Daily News Sentiment, news-based EPU indices (incl. Twitter EPU, UK EPU), Wikipedia page views,
  Google Trends**: covered by other agents (`data-other`, `data-politics`, `data-trends`).

## Data-quality checks

Every series passed: sorted, one row per date, no duplicate dates, values inside their stated range
(tone in [-1, 1], shares in [0, 1], counts >= 0), no sentinels. Range, gaps and the first/last three
rows of every series are in `scripts/data-news/logs/series_checks.txt` (`check_all.py`).

### Coverage and seasonality per series

`missing days` = calendar days inside first..last with no value; `years >= 90%` = calendar years with a
value on >= 90% of days (6-day-a-week papers such as the Irish Times business section can't reach it);
`doy_r2`/`dow_r2` = share of variance explained by month of year / day of week after removing a centred
365-day mean.

<!-- COVERAGE:begin -->
| id | first | last | rows | missing days | years >= 90% covered | doy_r2 | dow_r2 |
|---|---|---|---|---|---|---|---|
| abc_count | 2003-02-19 | 2021-12-31 | 6882 | 9 | 19/19 | 0.015 | 0.644 |
| abc_crisis_any | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.007 | 0.107 |
| abc_crisis_bankruptcy | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.003 | 0.005 |
| abc_crisis_crash | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.003 | 0.090 |
| abc_crisis_inflation | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.013 | 0.029 |
| abc_crisis_layoffs | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.005 | 0.044 |
| abc_crisis_pandemic | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.006 | 0.003 |
| abc_crisis_recession | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.003 | 0.006 |
| abc_crisis_war | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.022 | 0.011 |
| abc_lm_negshare | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.006 | 0.105 |
| abc_lm_tone | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.004 | 0.131 |
| abc_vader | 2003-02-19 | 2021-12-31 | 6880 | 11 | 19/19 | 0.008 | 0.025 |
| atn_count | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.021 | 0.845 |
| atn_crisis_any | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.094 | 0.310 |
| atn_crisis_bankruptcy | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.017 | 0.148 |
| atn_crisis_crash | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.038 | 0.058 |
| atn_crisis_default | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.018 | 0.078 |
| atn_crisis_inflation | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.012 | 0.235 |
| atn_crisis_layoffs | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.048 | 0.102 |
| atn_crisis_pandemic | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.091 | 0.004 |
| atn_crisis_recession | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.070 | 0.049 |
| atn_crisis_war | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.064 | 0.241 |
| atn_fin_count | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.028 | 0.752 |
| atn_fin_crisis_any | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.090 | 0.283 |
| atn_fin_lm_negshare | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.054 | 0.023 |
| atn_fin_lm_tone | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.033 | 0.038 |
| atn_fin_vader | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.056 | 0.214 |
| atn_lm_negshare | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.057 | 0.059 |
| atn_lm_tone | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.046 | 0.044 |
| atn_vader | 2016-01-01 | 2020-03-27 | 1548 | 0 | 5/5 | 0.093 | 0.156 |
| hn_count | 2007-02-19 | 2026-10-16 | 7180 | 0 | 20/20 | 0.011 | 0.347 |
| hn_crisis_any | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.010 | 0.011 |
| hn_crisis_bankruptcy | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.002 | 0.003 |
| hn_crisis_crash | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.002 | 0.003 |
| hn_crisis_inflation | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.002 | 0.001 |
| hn_crisis_layoffs | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.010 | 0.008 |
| hn_crisis_pandemic | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.007 | 0.001 |
| hn_crisis_war | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.005 | 0.002 |
| hn_lm_negshare | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.003 | 0.004 |
| hn_lm_tone | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.003 | 0.005 |
| hn_p20_count | 2007-02-19 | 2026-10-16 | 7180 | 0 | 20/20 | 0.011 | 0.397 |
| hn_p20_crisis_any | 2007-02-19 | 2026-10-16 | 6922 | 258 | 19/20 | 0.015 | 0.014 |
| hn_p20_lm_negshare | 2007-02-19 | 2026-10-16 | 6922 | 258 | 19/20 | 0.003 | 0.006 |
| hn_p20_lm_tone | 2007-02-19 | 2026-10-16 | 6922 | 258 | 19/20 | 0.002 | 0.006 |
| hn_p20_vader | 2007-02-19 | 2026-10-16 | 6922 | 258 | 19/20 | 0.003 | 0.003 |
| hn_vader | 2008-03-12 | 2026-10-16 | 6580 | 213 | 18/19 | 0.004 | 0.002 |
| huff_count | 2012-01-28 | 2018-05-26 | 2309 | 2 | 7/7 | 0.007 | 0.479 |
| huff_crisis_any | 2012-01-28 | 2022-09-22 | 3690 | 201 | 8/11 | 0.009 | 0.005 |
| huff_crisis_crash | 2012-01-28 | 2022-09-22 | 3690 | 201 | 8/11 | 0.006 | 0.005 |
| huff_crisis_pandemic | 2012-01-28 | 2022-09-22 | 3690 | 201 | 8/11 | 0.009 | 0.001 |
| huff_crisis_war | 2012-01-28 | 2022-09-22 | 3690 | 201 | 8/11 | 0.006 | 0.005 |
| huff_lm_negshare | 2012-01-28 | 2022-09-22 | 3690 | 201 | 8/11 | 0.006 | 0.007 |
| huff_lm_tone | 2012-01-28 | 2022-09-22 | 3690 | 201 | 8/11 | 0.009 | 0.008 |
| huff_vader | 2012-01-28 | 2022-09-22 | 3690 | 201 | 8/11 | 0.013 | 0.018 |
| irish_biz_count | 1996-01-02 | 2019-12-31 | 7716 | 1049 | 7/24 | 0.007 | 0.486 |
| irish_biz_crisis_any | 1996-01-02 | 2019-12-30 | 7485 | 1279 | 4/24 | 0.003 | 0.055 |
| irish_biz_lm_negshare | 1996-01-02 | 2019-12-30 | 7485 | 1279 | 4/24 | 0.005 | 0.031 |
| irish_biz_lm_tone | 1996-01-02 | 2019-12-30 | 7485 | 1279 | 4/24 | 0.003 | 0.032 |
| irish_biz_vader | 1996-01-02 | 2019-12-30 | 7485 | 1279 | 4/24 | 0.005 | 0.015 |
| irish_count | 1996-01-02 | 2019-12-31 | 8424 | 341 | 19/24 | 0.021 | 0.718 |
| irish_crisis_any | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.008 | 0.095 |
| irish_crisis_bankruptcy | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.004 | 0.016 |
| irish_crisis_crash | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.011 | 0.007 |
| irish_crisis_default | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.003 | 0.007 |
| irish_crisis_inflation | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.004 | 0.051 |
| irish_crisis_layoffs | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.004 | 0.032 |
| irish_crisis_recession | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.006 | 0.026 |
| irish_crisis_war | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.008 | 0.048 |
| irish_lm_negshare | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.015 | 0.154 |
| irish_lm_tone | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.019 | 0.118 |
| irish_vader | 1996-01-02 | 2019-12-31 | 8422 | 343 | 19/24 | 0.015 | 0.150 |
| reut_count | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.010 | 0.614 |
| reut_crisis_any | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.012 | 0.468 |
| reut_crisis_bankruptcy | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.004 | 0.185 |
| reut_crisis_crash | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.032 | 0.024 |
| reut_crisis_default | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.021 | 0.111 |
| reut_crisis_inflation | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.026 | 0.210 |
| reut_crisis_layoffs | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.025 | 0.105 |
| reut_crisis_pandemic | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.011 | 0.019 |
| reut_crisis_recession | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.020 | 0.083 |
| reut_crisis_war | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.008 | 0.250 |
| reut_lm_negshare | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.008 | 0.375 |
| reut_lm_tone | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.009 | 0.387 |
| reut_vader | 2007-01-01 | 2014-12-31 | 2837 | 85 | 7/8 | 0.007 | 0.608 |
| toi_biz_count | 2001-05-22 | 2022-03-31 | 7492 | 127 | 21/22 | 0.019 | 0.085 |
| toi_biz_crisis_any | 2001-05-22 | 2022-03-31 | 7073 | 546 | 15/22 | 0.009 | 0.021 |
| toi_biz_lm_negshare | 2001-05-22 | 2022-03-31 | 7073 | 546 | 15/22 | 0.004 | 0.001 |
| toi_biz_lm_tone | 2001-05-22 | 2022-03-31 | 7073 | 546 | 15/22 | 0.004 | 0.002 |
| toi_biz_vader | 2001-05-22 | 2022-03-31 | 7073 | 546 | 15/22 | 0.007 | 0.002 |
| toi_count | 2001-01-02 | 2022-03-31 | 7717 | 42 | 21/22 | 0.018 | 0.035 |
| toi_crisis_any | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.007 | 0.001 |
| toi_crisis_bankruptcy | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.002 | 0.001 |
| toi_crisis_crash | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.004 | 0.001 |
| toi_crisis_default | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.002 | 0.004 |
| toi_crisis_inflation | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.005 | 0.017 |
| toi_crisis_layoffs | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.003 | 0.003 |
| toi_crisis_pandemic | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.011 | 0.003 |
| toi_crisis_recession | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.005 | 0.001 |
| toi_crisis_war | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.004 | 0.001 |
| toi_lm_negshare | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.021 | 0.048 |
| toi_lm_tone | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.019 | 0.058 |
| toi_vader | 2001-01-02 | 2022-03-31 | 7695 | 64 | 21/22 | 0.020 | 0.019 |
| wikice_count | 2006-08-01 | 2026-10-16 | 7381 | 1 | 21/21 | 0.015 | 0.037 |
| wikice_crisis_any | 2006-08-01 | 2026-10-16 | 7034 | 348 | 18/21 | 0.022 | 0.007 |
| wikice_crisis_bankruptcy | 2006-08-01 | 2026-10-16 | 7034 | 348 | 18/21 | 0.001 | 0.006 |
| wikice_crisis_crash | 2006-08-01 | 2026-10-16 | 7034 | 348 | 18/21 | 0.004 | 0.001 |
| wikice_crisis_pandemic | 2006-08-01 | 2026-10-16 | 7034 | 348 | 18/21 | 0.026 | 0.006 |
| wikice_crisis_war | 2006-08-01 | 2026-10-16 | 7034 | 348 | 18/21 | 0.014 | 0.002 |
| wikice_latest_count | 2003-01-01 | 2026-10-16 | 8687 | 3 | 24/24 | 0.012 | 0.043 |
| wikice_latest_crisis_any | 2003-01-02 | 2026-10-16 | 8145 | 544 | 19/24 | 0.027 | 0.006 |
| wikice_latest_lm_negshare | 2003-01-02 | 2026-10-16 | 8145 | 544 | 19/24 | 0.004 | 0.007 |
| wikice_latest_lm_tone | 2003-01-02 | 2026-10-16 | 8145 | 544 | 19/24 | 0.003 | 0.026 |
| wikice_latest_vader | 2003-01-02 | 2026-10-16 | 8145 | 544 | 19/24 | 0.012 | 0.007 |
| wikice_lm_negshare | 2006-08-01 | 2026-10-16 | 7034 | 348 | 18/21 | 0.002 | 0.007 |
| wikice_lm_tone | 2006-08-01 | 2026-10-16 | 7034 | 348 | 18/21 | 0.002 | 0.026 |
| wikice_sec_arts_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.007 | 0.002 |
| wikice_sec_business_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.010 | 0.075 |
| wikice_sec_conflict_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.022 | 0.001 |
| wikice_sec_conflict_disasters_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.022 | 0.001 |
| wikice_sec_disasters_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.018 | 0.001 |
| wikice_sec_health_env_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.025 | 0.013 |
| wikice_sec_intl_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.012 | 0.027 |
| wikice_sec_law_crime_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.007 | 0.081 |
| wikice_sec_politics_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.013 | 0.016 |
| wikice_sec_science_tech_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.004 | 0.032 |
| wikice_sec_sports_count | 2010-07-01 | 2026-10-16 | 5952 | 0 | 17/17 | 0.017 | 0.070 |
| wikice_vader | 2006-08-01 | 2026-10-16 | 7034 | 348 | 18/21 | 0.011 | 0.008 |
| wikice_wk4_business_lm_tone | 2010-07-26 | 2026-10-05 | 836 |  |  |  |  |
| wikice_wk4_business_vader | 2010-07-26 | 2026-10-05 | 836 |  |  |  |  |
| wikice_wk_conflict_disasters_lm_tone | 2010-07-05 | 2026-10-05 | 848 |  |  |  |  |
| wikice_wk_conflict_disasters_vader | 2010-07-05 | 2026-10-05 | 848 |  |  |  |  |
| wikice_wk_law_crime_lm_tone | 2010-07-05 | 2026-10-05 | 817 |  |  |  |  |
| wikice_wk_law_crime_vader | 2010-07-05 | 2026-10-05 | 817 |  |  |  |  |
| wikice_wk_politics_intl_lm_tone | 2010-07-05 | 2026-10-05 | 845 |  |  |  |  |
| wikice_wk_politics_intl_vader | 2010-07-05 | 2026-10-05 | 845 |  |  |  |  |
<!-- COVERAGE:end -->

### Share of days with a value, per year (tone series)

First and last years are partial where a series starts or ends mid-year: 2026 runs only to 2026-10-16, so
`.79` there is complete data; `.38` for Wikipedia 2006 reflects the 2006-08-01 start.

<!-- YEARS:begin -->
| series | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wikice_vader |  |  |  |  |  |  | .38 | .88 | .79 | .93 | .99 | .97 | .98 | .96 | .93 | .98 | .98 | .96 | .95 | .91 | .99 | 1.00 | .98 | .97 | .98 | .98 | .79 |
| wikice_latest_vader |  |  |  | .50 | .84 | .93 | .87 | .90 | .82 | .94 | 1.00 | .98 | .99 | .99 | .94 | .98 | .99 | .98 | .97 | .93 | 1.00 | 1.00 | .99 | .98 | .99 | 1.00 | .79 |
| wikice_sec_conflict_count |  |  |  |  |  |  |  |  |  |  | .50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .79 |
| wikice_wk_conflict_disasters_vader |  |  |  |  |  |  |  |  |  |  | .50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .77 |
| wikice_wk4_business_vader |  |  |  |  |  |  |  |  |  |  | .44 | 1.00 | 1.00 | 1.00 | .94 | 1.00 | 1.00 | .98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .92 | .98 | .77 |
| hn_vader |  |  |  |  |  |  |  |  | .25 | .97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .79 |
| hn_p20_vader |  |  |  |  |  |  |  | .19 | .97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .79 |
| reut_vader |  |  |  |  |  |  |  | 1.00 | .99 | 1.00 | .97 | 1.00 | 1.00 | 1.00 | .81 |  |  |  |  |  |  |  |  |  |  |  |  |
| abc_vader |  |  |  | .87 | .99 | 1.00 | .99 | .99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |  |  |  |  |  |
| toi_vader |  | .82 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .25 |  |  |  |  |
| toi_biz_vader |  | .49 | .96 | .94 | .85 | .70 | .80 | .93 | .96 | 1.00 | .99 | .97 | .99 | .99 | .98 | .98 | .96 | .99 | .99 | .94 | .87 | .86 | .20 |  |  |  |  |
| irish_vader | .80 | .99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .99 | 1.00 | 1.00 |  |  |  |  |  |  |  |
| irish_biz_vader | .80 | .84 | .84 | .85 | .85 | .85 | .84 | .84 | .80 | .84 | .85 | .85 | .84 | .86 | .87 | .87 | .92 | .92 | .99 | .95 |  |  |  |  |  |  |  |
| huff_vader |  |  |  |  |  |  |  |  |  |  |  |  | .93 | 1.00 | 1.00 | 1.00 | 1.00 | .99 | .87 | .88 | .90 | .94 | .59 |  |  |  |  |
| atn_vader |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.00 | 1.00 | 1.00 | 1.00 | .24 |  |  |  |  |  |  |
| atn_fin_vader |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  | 1.00 | 1.00 | 1.00 | 1.00 | .24 |  |  |  |  |  |  |
<!-- YEARS:end -->

### Sanity check: tone around known bad days

z-score of the window mean against all baseline windows of the same length that start on the same
weekday, taken from the 250 days ending 30 days before the event (so day-of-week effects cancel; `sanity.py`).
Windows are days relative to the event: [-10,-1] before, [0,2] the event, [3,9] after, [20,40] about a month
later. Expected: VADER and LM tone negative around the event, crisis words up, then recovery; but the
[20,40] window after 2008-09-15 is October 2008, when the crisis deepened, so no recovery is expected there.
Missing rows: Hacker News > 50 points has too few 2008 stories for a value, the Irish Times ends in 2019,
HuffPost starts in 2012, All the News in 2016, and Reuters ends in 2014.

Result: the business sources fall (Reuters VADER -1.75 and LM tone -1.39 on 2008-09-15, LM -3.6 to -5.9 in
the following weeks; Times of India business LM tone -4.08 on 2008-10-10 and -2.33 on 2011-08-08; Irish
Times business -1.74/-2.27 on 2008-09-15 and VADER -2.14 on 2011-08-08; All the News business press
-1.48/-1.98 on 2020-03-16), with exceptions (Reuters only -0.6 on 2011-08-08, Times of India business flat
on 2020-03-16). General news is mixed: the Irish Times (-2.84 on 2011-08-08) and Times of India (-2.73 on
2020-03-16) drop, ABC does not, Wikipedia drops in 2008 (VADER -1.41) but its tone *rises* in March 2020 (topic
mix, see caveat 1), and Hacker News tone does not react while its crisis words jump (+5.8 sd).

<!-- SANITY:begin -->
| source | event | vader -10,-1 | vader 0,2 | vader 3,9 | vader 20,40 | lm_tone -10,-1 | lm_tone 0,2 | lm_tone 3,9 | lm_tone 20,40 | crisis_any 0,2 |
|---|---|---|---|---|---|---|---|---|---|---|
| wikice | 2008-09-15 | -0.45 | -1.41 | -1.12 | -0.36 | +0.92 | -0.50 | +0.89 | -0.96 | -0.33 |
| wikice | 2008-10-10 | -2.61 | -0.62 | +2.36 | -0.31 | -1.72 | -0.49 | +0.53 | -1.32 | -0.10 |
| wikice | 2011-08-08 | -1.47 | -0.98 | +0.11 | +2.02 | +0.86 | +1.06 | +2.11 | +3.19 | -0.64 |
| wikice | 2020-03-16 | +3.31 | +2.51 | +2.93 | +1.46 | +3.26 | +2.73 | +3.48 | +2.35 | +36.89 |
| wikice_latest | 2008-09-15 | -0.33 | -1.26 | -1.19 | -0.60 | +0.65 | -0.86 | +0.84 | -0.88 | +0.23 |
| wikice_latest | 2008-10-10 | -2.57 | -0.59 | +1.90 | +0.18 | -2.62 | -0.37 | +0.75 | -2.22 | -0.07 |
| wikice_latest | 2011-08-08 | -1.33 | -0.90 | +0.23 | +2.75 | +0.64 | +0.99 | +2.22 | +3.37 | -0.98 |
| wikice_latest | 2020-03-16 | +4.52 | +2.86 | +3.26 | +1.84 | +3.87 | +2.72 | +3.59 | +2.57 | +29.76 |
| reut | 2008-09-15 | -0.23 | -1.75 | -1.53 | -0.91 | -1.81 | -1.39 | -3.55 | -5.92 | +0.65 |
| reut | 2008-10-10 | -2.72 | -0.01 | -1.04 | -0.50 | -4.93 | -0.79 | -4.22 | -2.01 | -2.07 |
| reut | 2011-08-08 | -0.85 | -0.61 | +0.06 | -0.10 | -0.76 | -0.46 | +0.55 | +0.07 | -0.08 |
| toi_biz | 2008-09-15 | +0.12 | -1.58 | -0.92 | -4.20 | -0.04 | -1.29 | -2.28 | -6.38 | +1.40 |
| toi_biz | 2008-10-10 | -3.50 | -2.68 | -1.42 | -1.44 | -2.98 | -4.08 | -2.19 | -4.01 | +0.48 |
| toi_biz | 2011-08-08 | -1.32 | -0.80 | -1.07 | -0.08 | -1.18 | -2.33 | -3.04 | -0.17 | -0.49 |
| toi_biz | 2020-03-16 | +1.40 | +1.33 | -0.20 | +0.17 | -1.70 | +0.05 | -0.75 | +0.58 | -0.82 |
| irish_biz | 2008-09-15 | -0.72 | -1.74 | +0.84 | -1.27 | -2.04 | -2.27 | +1.42 | -1.61 | -0.85 |
| irish_biz | 2008-10-10 | -1.58 | -1.57 | -0.50 | +0.10 | -0.84 | -2.19 | -1.63 | -1.77 | +1.34 |
| irish_biz | 2011-08-08 | -1.41 | -2.14 | -0.11 | -1.02 | -2.62 | -1.11 | +0.04 | -0.36 | +1.04 |
| atn_fin | 2020-03-16 | -2.03 | -1.48 | -0.74 |  | -1.82 | -1.98 | -1.31 |  | +4.28 |
| hn | 2011-08-08 | -2.11 | +1.74 | +0.21 | -0.44 | -2.19 | +0.34 | -0.48 | +0.42 | -0.55 |
| hn | 2020-03-16 | +1.92 | -0.65 | +0.48 | +1.28 | +1.74 | -0.72 | -1.28 | +1.03 | +5.76 |
| hn_p20 | 2008-09-15 | -0.47 | +0.89 | -0.55 | +0.43 | -1.71 | +1.41 | +0.23 | +0.25 | -0.45 |
| hn_p20 | 2008-10-10 | -0.42 | +0.66 | +0.18 | -0.22 | -1.34 | -0.12 | -0.07 | -0.55 | -0.41 |
| hn_p20 | 2011-08-08 | -1.86 | +1.45 | +0.05 | -0.63 | -1.64 | +0.01 | -0.44 | +0.43 | -1.04 |
| hn_p20 | 2020-03-16 | +1.83 | -0.62 | +0.31 | +1.66 | +0.88 | -1.20 | -1.26 | +0.48 | +6.47 |
| abc | 2008-09-15 | +1.03 | +1.14 | -0.59 | +0.70 | +2.14 | +1.42 | -0.91 | -1.86 | +0.97 |
| abc | 2008-10-10 | +0.33 | -0.42 | +0.75 | -0.37 | -1.56 | -0.05 | -1.05 | +0.26 | -0.17 |
| abc | 2011-08-08 | +0.96 | +0.05 | -0.75 | +2.74 | +0.02 | +2.17 | +0.20 | +2.21 | +0.71 |
| abc | 2020-03-16 | +1.36 | +0.95 | +0.66 | +3.13 | +0.13 | +1.55 | +1.27 | +2.37 | +1.76 |
| toi | 2008-09-15 | -0.13 | -0.90 | -0.54 | -1.58 | -1.48 | -0.89 | -0.45 | -2.34 | +2.24 |
| toi | 2008-10-10 | -2.54 | -0.96 | -0.19 | -0.91 | -0.93 | -0.84 | -0.45 | -1.91 | +0.86 |
| toi | 2011-08-08 | -0.31 | +0.09 | +1.35 | +0.12 | -0.78 | -1.72 | +0.55 | -2.04 | -0.58 |
| toi | 2020-03-16 | -2.10 | -2.73 | -0.47 | +2.85 | -0.29 | -1.46 | -0.96 | +2.50 | +0.02 |
| irish | 2008-09-15 | +1.90 | +0.22 | +1.11 | +0.72 | +2.77 | -0.46 | +0.31 | -0.81 | +1.60 |
| irish | 2008-10-10 | +0.26 | +0.71 | +0.43 | -0.41 | -0.35 | +0.46 | -0.92 | +0.23 | +0.49 |
| irish | 2011-08-08 | -0.82 | -2.84 | -0.44 | +0.03 | -0.45 | -0.77 | +0.09 | +1.37 | +0.99 |
| huff | 2020-03-16 | -0.06 | +2.39 | +1.43 | +1.72 | +1.00 | +1.14 | +1.48 | +1.74 | +2.35 |
| atn | 2020-03-16 | -0.76 | -0.57 | +0.51 |  | -1.20 | -1.59 | -0.17 |  | +6.04 |
<!-- SANITY:end -->

## Caveats for the test

<!-- CAVEATS:begin -->
1. **General-news tone is mostly topic mix, not market mood.** Business headlines mostly fall on market
   stress days (Reuters, Times of India business, Irish Times business, All the News business press: z of
   -1.3 to -5.9 in the event window or the weeks around it; exceptions in the sanity section). General news
   often does not:
   ABC's tone is slightly *higher* on 2008-09-15, and Wikipedia's tone *rises* on 2020-03-16 because COVID
   policy items (neutral to both lexicons) crowded out the very negative conflict/disaster items, while its
   pandemic count jumped +37 sd. Hacker News tone barely reacts; only its crisis words do (+5.8 sd in March
   2020). Read hits on general-news tone as "the news mix changed", and prefer the business subsets for a
   market-mood reading.
2. **Timing.** Wikipedia as-of values are exactly point-in-time (revision at D+1 19:30 UTC). Hacker News uses
   the final vote count, so `known_after_days` is 2. ABC and Times of India are at 2 as instructed for far-off
   time zones; ABC's dates were verified to be Sydney dates, which end before the New York close of the same
   date, so 1 (even 0) would also be free of look-ahead if more power at the 1-day offset is wanted.
3. **Hindsight.** `wikice_latest_*` are today's pages, which editors wrote or revised later (the 2003-2005
   pages were created in 2006 and 2017); use them only as a long-history robustness check. Their daily tone
   correlates 0.95 with the as-of version, but item selection can reflect later events.
4. **Day of week.** Counts have strong weekly cycles (`dow_r2` up to 0.85: weekends are thin), and so
   does tone in some sources (Reuters VADER 0.61, LM tone 0.39; most others <= 0.2). With
   `known_after_days` 1 the test averages Friday-Sunday onto Monday, so Monday rows differ systematically;
   neither `level` nor `anomaly` removes this. A weekday effect in a stock could pair with it.
5. **Breaks in archive composition** (the trailing 252-day de-meaning absorbs slow drift, not steps):
   Reuters volume steps with its site (~950 headlines a weekday 2007-09, ~2,200 in 2010, ~4,200 in 2011,
   ~2,900 in 2013-14); HuffPost drops from ~95 to ~5 a day on 2018-05-27 (count series cut there, tone kept
   but noisy); ABC thins from ~250 to ~50 a day over 2013-2021; Times of India grows from ~150 to ~700 a day
   (2001-2011); Wikipedia sections start 2010-07 and the health section jumps with COVID in 2020-21.
6. **Near-duplicates; count correlated hits as one finding:** business subsets are parts of their parent
   source; `hn_p20` contains `hn`; `atn_fin` is ~40% of `atn`; `wikice_latest` tracks `wikice` (r = 0.95);
   VADER, LM tone and LM negative share of the same source are strongly related.
7. **Crisis-word counts are zero-heavy event counts.** "crash" is mostly vehicle and plane crashes in
   general news (software crashes on Hacker News), "war" includes figurative wars, and "default" is excluded
   on Hacker News. A word gets its own series only where it appears on >= 2% of the source's days.
8. **Coverage.** All the News (2016-01..2020-03) cannot join any default 1260-day period; Reuters joins only
   242; the Dataverse sets stop in 2019-2022; HuffPost's 2018-2022 tail is thin. Days with fewer than 5
   headlines have no tone value: Wikipedia as-of misses ~5% of days (mostly 2006-2009), Hacker News >50 is
   sparse before 2009.
9. **Licences:** Dataverse sets CC0; HuffPost CC BY 4.0; Wikipedia text CC BY-SA 4.0; LM dictionary free for
   academic research only (commercial licence needed otherwise); All the News non-commercial research only;
   Reuters and Hacker News content is copyrighted, and only aggregate daily statistics are stored outside
   raw/.

## Scripts and reproduction

All in `~/.local/share/correlation2/external/scripts/data-news/` (logs in `logs/`, per-day metrics in
`work/<source>_daily.json`, catalog metadata in `meta/`):
`common.py` (polite fetch with a total-time deadline, series writer and checks), `score.py` (VADER + LM +
crisis words), `emit.py` (standard series set), `wikice_parse.py` (wikitext items and sections);
fetchers `fetch_wikice.py`, `fetch_hn.py`, `fetch_bulk.sh`, `fetch_reuters.py`, `fetch_reuters_fix.py`
(re-points truncated captures; needs a first `build_reuters.py` run); `extract_atn.py work/atn.tsv`
(streams titles out of the zip, ~1 min; the TSV is not kept); builders `build_bulk.py`, `build_hn.py`,
`build_wikice.py`, `build_reuters.py`; `build_catalog.py`; checks `check_all.py`, `coverage.py`, `sanity.py`;
`splice_sources.py` (fills the tables of this file). Text-scoring scripts run with
`~/programming/c2-wt/venv-data/bin/python -I` (vaderSentiment 3.3.2 installed there); everything else with
`python3 -I`.
<!-- CAVEATS:end -->
