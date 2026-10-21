"""Search terms (US, all categories, web search) and window layouts."""
import re
from datetime import date, timedelta

END = '2026-10-17'  # last complete Sun-Sat week is 2026-10-11..17

# (keyword, group, seasonal)
CORE = [
    # Preis, Moat & Stanley 2013 terms
    ('debt', 'preis2013', False), ('stocks', 'preis2013', False), ('portfolio', 'preis2013', False),
    ('inflation', 'preis2013', False), ('housing', 'preis2013', False),
    ('dow jones', 'preis2013', False), ('revenue', 'preis2013', False),
    ('economics', 'preis2013', False), ('credit', 'preis2013', False),
    ('markets', 'preis2013', False), ('unemployment', 'preis2013+fears', False),
    ('money', 'preis2013', False), ('investment', 'preis2013', False), ('bonds', 'preis2013', False),
    ('derivatives', 'preis2013', False), ('profit', 'preis2013', False),
    ('leverage', 'preis2013', False), ('cash', 'preis2013', False),
    ('stock market', 'preis2013', False), ('banking', 'preis2013', False),
    ('crisis', 'preis2013', False), ('nasdaq', 'preis2013', False), ('finance', 'preis2013', False),
    ('sell', 'preis2013', False), ('invest', 'preis2013', False), ('gold', 'preis2013', False),
    ('economy', 'preis2013', False), ('oil', 'preis2013', False), ('bubble', 'preis2013', False),
    ('bankruptcy', 'preis2013+fears', False), ('crash', 'preis2013', False),
    ('forex', 'preis2013', False), ('earnings', 'preis2013', False),
    ('dividend', 'preis2013', False),
    # FEARS (Da, Engelberg & Gao 2015)
    ('recession', 'fears2015', False), ('depression', 'fears2015', False),
    ('gold prices', 'fears2015', False), ('frugal', 'fears2015', False), ('gdp', 'fears2015', False),
    ('social security', 'fears2015', False), ('foreclosure', 'fears2015', False),
    # crypto
    ('bitcoin', 'crypto', False), ('ethereum', 'crypto', False), ('crypto', 'crypto', False),
    ('buy bitcoin', 'crypto', False),
    # jobs and consumer
    ('layoffs', 'jobs_consumer', False), ('file for unemployment', 'jobs_consumer', False),
    ('jobs', 'jobs_consumer', False), ('black friday', 'jobs_consumer', True),
    ('coupons', 'jobs_consumer', True), ('mortgage rates', 'jobs_consumer', False),
    # weather and health
    ('hurricane', 'weather_health', True), ('flu', 'weather_health', True),
    ('heat wave', 'weather_health', True),
    # markets
    ('buy stocks', 'markets', False), ('sell stocks', 'markets', False),
    ('stock market crash', 'markets', False), ('short selling', 'markets', False),
    ('margin call', 'markets', False), ('s&p 500', 'markets', False),
]

TICKERS = ['AAPL', 'TSLA', 'NVDA', 'AMZN', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NFLX', 'GME', 'AMC',
           'SPY', 'QQQ', 'AMD', 'INTC', 'BAC', 'JPM', 'XOM', 'CSCO', 'ORCL']

KEY8 = ['debt', 'recession', 'unemployment', 'stock market crash', 'bitcoin', 'inflation',
        'layoffs', 'buy stocks']

WEEKLY_MR5 = [('2004-01-01', '2008-12-31'), ('2009-01-01', '2013-12-31'),
              ('2014-01-01', '2018-12-31'), ('2019-01-01', '2023-12-31'), ('2024-01-01', END)]
WEEKLY_MR5_SHIFT = [('2004-01-02', '2009-01-01'), ('2009-01-02', '2014-01-01'),
                    ('2014-01-02', '2019-01-01'), ('2019-01-02', '2024-01-01'), ('2024-01-02', END)]
# spec method: overlapping single 5-year windows
WEEKLY_WIN = [('2004-01-01', '2008-12-31'), ('2008-01-01', '2012-12-31'),
              ('2012-01-01', '2016-12-31'), ('2016-01-01', '2020-12-31'),
              ('2020-01-01', '2024-12-31'), ('2021-10-18', END)]


def slug(kw):
    s = re.sub(r'[^a-z0-9]+', '_', kw.lower()).strip('_')
    return s


def items(kw, ranges, geo='US'):
    return [{'keyword': kw, 'time': f'{a} {b}', 'geo': geo} for a, b in ranges]


def daily_blocks(start='2004-01-01', end=END, rng=261, n=5, overlap=84):
    """Blocks of n consecutive daily ranges (each rng days incl.), laid out backwards from end;
    the earliest block is pinned to start (so its overlap with the next may be larger).
    Returns list (most recent first) of lists of (a, b) date strings."""
    e = date.fromisoformat(end)
    s0 = date.fromisoformat(start)
    span = rng * n
    blocks = []
    while True:
        b_start = e - timedelta(days=span - 1)
        if b_start <= s0:
            b_start = s0
        rs = []
        a = b_start
        for i in range(n):
            b = a + timedelta(days=rng - 1)
            rs.append((a.isoformat(), b.isoformat()))
            a = b + timedelta(days=1)
        blocks.append(rs)
        if b_start == s0:
            break
        e = b_start + timedelta(days=overlap - 1)
    return blocks
