"""Polymarket selection: rolling-series components and individual contracts.

Market ids refer to the Gamma API (gamma-api.polymarket.com/markets/<id>); the token used is
the one for `outcome` ("Yes" unless stated). Fed per-meeting events are discovered from the
tag crawl (see fed_events() in polymarket_fetch.py).
"""

# rolling binary series: id -> (name, list of (market_id, outcome, cycle_label))
ROLLING = {
    "pm_us_recession_cur_year": ("Polymarket: probability of a US recession in the current calendar year", [
        ("500741", "Yes", "2024"), ("516710", "Yes", "2025"), ("609655", "Yes", "2026")]),
    "pm_house_gop_cur_cycle": ("Polymarket: probability Republicans control the US House after the next election", [
        ("254600", "Republican", "2024"), ("562803", "Yes", "2026")]),
    "pm_senate_gop_cur_cycle": ("Polymarket: probability Republicans control the US Senate after the next election", [
        ("252298", "Republicans", "2024"), ("562794", "Yes", "2026")]),
}

# individual contracts: (market_id, outcome, series id, short name, subcategory)
INDIVIDUAL = [
    ("253299", "Yes", "pmc_fed_cut_by_2024may01", "Fed rate cut by May 1, 2024?", "fed"),
    ("254247", "Yes", "pmc_fed_cut_by_2024jun12", "Fed rate cut by June 12, 2024?", "fed"),
    ("504069", "Yes", "pmc_fed_2024sep_cut50plus", "Fed cuts 50+ bps at the Sept 2024 meeting?", "fed"),
    ("504465", "Yes", "pmc_fed_emergency_cut_2024", "Fed emergency rate cut in 2024?", "fed"),
    ("501190", "Yes", "pmc_fed_hike_2024", "Fed rate hike in 2024?", "fed"),
    ("526911", "Yes", "pmc_fed_cut_2025", "Fed rate cut in 2025 (market opened Mar 2025)?", "fed"),
    ("516706", "Yes", "pmc_fed_hike_2025", "Fed rate hike in 2025?", "fed"),
    ("516725", "Yes", "pmc_fed_2cuts_2025", "Exactly 2 Fed rate cuts in 2025?", "fed"),
    ("535321", "Yes", "pmc_trump_fire_powell_by_2025jun", "Will Trump try to fire Powell before June 2025?", "fed"),
    ("516708", "Yes", "pmc_ust10y_above5_by_2025jun30", "US 10Y Treasury yield above 5% by June 30, 2025?", "rates"),
    ("508169", "Yes", "pmc_recession_by_2025may", "US recession before May 2025?", "macro"),
    ("526837", "Yes", "pmc_gdp_negative_2025q1", "Negative US GDP growth in Q1 2025?", "macro"),
    ("533849", "Yes", "pmc_gdp_negative_2025q2", "Negative US GDP growth in Q2 2025?", "macro"),
    ("522145", "Yes", "pmc_inflation_above3_2025", "US inflation above 3% in 2025?", "macro"),
    ("517000", "Yes", "pmc_nyse_circuit_breaker_2025", "NYSE market-wide circuit breaker in 2025?", "markets"),
    ("514049", "Yes", "pmc_tariff25_mex_can_by_2025feb", "Trump imposes 25% tariff on Mexico or Canada before Feb 2025?", "trade"),
    ("513304", "Yes", "pmc_large_tariffs_first_6m", "Trump imposes large tariffs in his first 6 months?", "trade"),
    ("513494", "Yes", "pmc_tariff40_china_first_100d", "Trump imposes 40% tariff on China in first 100 days?", "trade"),
    ("521462", "Yes", "pmc_tariff_mexico_up_by_2025may", "Trump increases tariffs on Mexico before May 2025?", "trade"),
    ("535291", "Yes", "pmc_us_china_deal_by_2025jun", "US-China trade deal before June 2025?", "trade"),
    ("537198", "Yes", "pmc_us_eu_deal_by_2025jul", "US-EU trade deal before July 2025?", "trade"),
    ("536531", "Yes", "pmc_courts_block_tariffs_by_2025jun", "Courts block Trump's tariffs before June 2025?", "trade"),
    ("534910", "Yes", "pmc_remove_10pct_tariff_by_2025jul", "Trump removes the 10% blanket tariff before July 2025?", "trade"),
    ("537490", "Yes", "pmc_tariff_revenue_250b_2025", "US tariff revenue above $250b in 2025?", "trade"),
    ("240613", "Republican", "pmc_pres2024_gop_party", "Republican party wins the 2024 US presidential election", "us_election"),
    ("253591", "Yes", "pmc_pres2024_trump", "Donald Trump wins the 2024 US presidential election", "us_election"),
    ("253706", "Yes", "pmc_pres2024_trump_popular_vote", "Donald Trump wins the 2024 popular vote", "us_election"),
    ("253697", "Yes", "pmc_dem2024_biden_nominee", "Joe Biden wins the 2024 Democratic nomination", "us_election"),
    ("252294", "Yes", "pmc_biden_drops_out", "Biden drops out of the 2024 presidential race?", "us_election"),
    ("248596", "Yes", "pmc_turkey2023_erdogan", "Erdogan wins the 2023 Turkish presidential election", "world_election"),
    ("251254", "Yes", "pmc_argentina2023_milei", "Javier Milei wins the 2023 Argentine presidential election", "world_election"),
    ("252836", "Yes", "pmc_taiwan2024_lai", "Lai Ching-te wins the 2024 Taiwan presidential election", "world_election"),
    ("516769", "Yes", "pmc_canada2025_carney_pm", "Mark Carney is PM of Canada after the 2025 election", "world_election"),
    ("516719", "Yes", "pmc_ru_ua_ceasefire_2025", "Russia x Ukraine ceasefire in 2025?", "geopolitics"),
    ("519628", "Yes", "pmc_ru_ua_ceasefire_by_2025jul", "Russia x Ukraine ceasefire before July 2025?", "geopolitics"),
    ("507115", "Yes", "pmc_ru_ua_ceasefire_2024", "Russia x Ukraine ceasefire in 2024?", "geopolitics"),
    ("510319", "Yes", "pmc_trump_ends_ukraine_war_90d", "Trump ends the Ukraine war in his first 90 days?", "geopolitics"),
    ("507792", "Yes", "pmc_isr_hezbollah_ceasefire_2024", "Israel x Hezbollah ceasefire in 2024?", "geopolitics"),
    ("504642", "Yes", "pmc_isr_hamas_ceasefire_2024", "Israel x Hamas ceasefire in 2024?", "geopolitics"),
    ("528977", "Yes", "pmc_isr_hamas_ceasefire_by_2025jul", "Israel x Hamas ceasefire before July 2025?", "geopolitics"),
    ("532741", "Yes", "pmc_us_strike_iran_by_2025jul", "US military action against Iran before July 2025?", "geopolitics"),
    ("532742", "Yes", "pmc_isr_strike_iran_by_2025jul", "Israel military action against Iran before July 2025?", "geopolitics"),
    ("508671", "Yes", "pmc_isr_strike_iran_nuclear_2024", "Israel strike on an Iranian nuclear facility in 2024?", "geopolitics"),
    ("511042", "Yes", "pmc_iran_strike_isr_2024", "Another Iran strike on Israel in 2024?", "geopolitics"),
    ("253889", "Yes", "pmc_china_invade_taiwan_2024", "China invades Taiwan in 2024?", "geopolitics"),
    ("520630", "Yes", "pmc_china_invade_taiwan_2025", "China invades Taiwan in 2025?", "geopolitics"),
    ("519695", "Yes", "pmc_iran_close_hormuz_2025", "Iran closes the Strait of Hormuz in 2025?", "geopolitics"),
    ("507276", "Yes", "pmc_tiktok_ban_by_2025may", "TikTok banned in the US before May 2025?", "policy"),
    ("253254", "Yes", "pmc_btc_spot_etf_by_2024jan15", "Bitcoin spot ETF approved by Jan 15, 2024?", "crypto"),
    ("253750", "Yes", "pmc_eth_etf_by_2024may31", "Ethereum spot ETF approved by May 31, 2024?", "crypto"),
    ("255229", "Yes", "pmc_btc_100k_2024", "Bitcoin hits $100k in 2024?", "crypto"),
    ("254432", "Yes", "pmc_eth_ath_2024", "Ethereum all-time high in 2024?", "crypto"),
    ("512250", "Yes", "pmc_btc_reserve_first_100d", "Trump creates a Bitcoin reserve in his first 100 days?", "crypto"),
    ("516864", "Yes", "pmc_btc_150k_2025", "Bitcoin reaches $150k by Dec 31, 2025?", "crypto"),
    ("252551", "Yes", "pmc_shutdown_by_2023nov19", "US government shutdown by Nov 19, 2023?", "fiscal"),
    ("254128", "Yes", "pmc_shutdown_by_2024mar09", "US government shutdown by Mar 9, 2024?", "fiscal"),
    ("518009", "Yes", "pmc_funding_lapse_2025mar15", "US government funding lapse on Mar 15, 2025?", "fiscal"),
    ("530889", "Yes", "pmc_funding_lapse_2025oct01", "US government funding lapse on Oct 1, 2025?", "fiscal"),
    ("516062", "Yes", "pmc_debt_ceiling_by_inauguration", "Debt ceiling raised or suspended by the Jan 2025 inauguration?", "fiscal"),
]
