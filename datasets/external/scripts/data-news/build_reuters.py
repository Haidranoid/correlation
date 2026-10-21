"""Reuters US daily news archive (Wayback copies, raw/reuters_archive_wayback): one page per
US Eastern calendar day listing every headline Reuters published that day.
Filters: press releases (URL 'pressRelease'), video stories, Briefing.com stubs ('Stock Market
Update', 'On The Wires'), 'Company Profile for', and formulaic prefixes (TABLE, DIARY, PRESS
DIGEST, TEXT, ADVISORY, BRIEF, CALENDAR, SCHEDULE) are dropped; version prefixes (UPDATE n,
RPT, REFILE, CORRECTED, WRAPUP n, INSTANT VIEW n) are stripped; titles de-duplicated per day.
Run with the venv python -I.
"""
import datetime as dt
import glob
import gzip
import html
import json
import os
import re
import statistics
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW
from score import score_many, aggregate
from emit import emit, save_daily

ITEM = re.compile(r"""<div class="headlineMed"><a href=['"]([^'"]+)['"][^>]*>(.*?)</a>""", re.S)
DROP_URL = re.compile(r"pressRelease|/video/|videoStory", re.I)
DROP_TITLE = re.compile(r"^(?:TABLE|DIARY|PRESS DIGEST|TEXT|ADVISORY|BRIEF|CALENDAR|SCHEDULE)\b|"
                        r"^Stock Market Update\b|^On The Wires\b|^Company Profile for\b", re.I)
VERSION = re.compile(r"^(?:UPDATE ?\d*|RPT|REFILE|CORRECTED|CORRECTION|WRAPUP ?\d*|INSTANT VIEW ?\d*)\s*-\s*", re.I)
_WS = re.compile(r"\s+")
END = dt.date(2014, 12, 31)  # Q4 2014 is sparse and Jan 2015 pages look different (Mondays ~10% of other days)


def parse_page(raw):
    s = raw.decode("utf-8", "replace")
    out, seen, dropped = [], set(), Counter()
    for url, title in ITEM.findall(s):
        t = _WS.sub(" ", html.unescape(re.sub(r"<[^>]+>", "", title))).strip()
        if DROP_URL.search(url):
            dropped["url"] += 1
            continue
        if DROP_TITLE.search(t):
            dropped["title"] += 1
            continue
        for _ in range(3):
            t2 = VERSION.sub("", t)
            if t2 == t:
                break
            t = t2
        if len(t.split()) < 3 or t.lower() in seen:
            dropped["dup_or_short"] += 1
            continue
        seen.add(t.lower())
        out.append(t)
    return out, dropped


def us_market_holidays(y):
    """NYSE full-day holidays plus the low-news Christmas/New Year days and the day after Thanksgiving."""
    def nth_weekday(month, weekday, n):
        d = dt.date(y, month, 1)
        d += dt.timedelta((weekday - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)
    def last_weekday(month, weekday):
        d = dt.date(y, month + 1, 1) - dt.timedelta(1)
        return d - dt.timedelta((d.weekday() - weekday) % 7)
    a = y % 19; b, c = divmod(y, 100); d_, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d_ - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; month, day = divmod(h + l - 7 * m + 114, 31)
    easter = dt.date(y, month, day + 1)
    thanks = nth_weekday(11, 3, 4)
    out = {dt.date(y, 1, 1), dt.date(y, 1, 2), nth_weekday(1, 0, 3), nth_weekday(2, 0, 3),
           easter - dt.timedelta(2), last_weekday(5, 0), dt.date(y, 7, 3), dt.date(y, 7, 4), dt.date(y, 7, 5),
           nth_weekday(9, 0, 1), thanks, thanks + dt.timedelta(1),
           dt.date(y, 12, 24), dt.date(y, 12, 25), dt.date(y, 12, 26), dt.date(y, 12, 31)}
    return out


def main():
    plan = json.load(open(os.path.join(RAW, "reuters_archive_wayback", "capture_plan.json")))
    texts, dates = [], []
    early = []
    tot = Counter()
    for dkey, (ts_plan, _) in sorted(plan.items()):
        d = dt.date.fromisoformat(dkey)
        if d > END:
            continue
        fp = os.path.join(RAW, "reuters_archive_wayback", f"{d.year}", f"{d.strftime('%Y%m%d')}_{ts_plan}.html.gz")
        ts = ts_plan
        if ts[:8] < (d + dt.timedelta(days=2)).strftime("%Y%m%d"):
            early.append((d, ts))  # captured before the day was surely complete: skip
            continue
        items, dropped = parse_page(gzip.open(fp).read())
        tot.update(dropped)
        tot["kept"] += len(items)
        for t in items:
            texts.append(t)
            dates.append(d)
    print("pages skipped (captured < 2 days after the date):", len(early), early[:5])
    print("item filter totals:", dict(tot))
    scores = score_many(texts)
    by_day = defaultdict(list)
    for d, s in zip(dates, scores):
        by_day[d].append(s)
    # truncated captures with no complete alternative: < 25% of the +-30-day median of same-type
    # days (weekday vs weekend), unless a US market holiday / Christmas-New Year day -> missing
    cnt = {d: len(v) for d, v in by_day.items()}
    days = sorted(cnt)
    hol = set().union(*(us_market_holidays(y) for y in range(2007, 2016)))
    dropped = []
    for d in days:
        peers = [cnt[e] for e in days if 0 < abs((e - d).days) <= 30 and (e.weekday() >= 5) == (d.weekday() >= 5)]
        if peers and cnt[d] < 0.25 * statistics.median(peers) and d not in hol:
            dropped.append((d, cnt[d], int(statistics.median(peers))))
    for d, _, _ in dropped:
        del by_day[d]
    print(f"days dropped as truncated captures ({len(dropped)}):", dropped)
    m = aggregate(by_day)
    save_daily("reut", m)
    reps = emit("reut", "Reuters US news archive", m, subcategory="headline_sentiment_business", region="US",
                source="Reuters reuters.com/resources/archive/us/YYYYMMDD.html via the Internet Archive Wayback Machine",
                url="https://web.archive.org/web/2014/http://www.reuters.com/resources/archive/us/20080915.html",
                kad=1, words=True,
                notes=("Every headline on Reuters' US daily archive page (US Eastern day; the live archive was removed "
                       "~2019), recovered from Wayback captures taken >= 2 days after the date. Heavy on business, "
                       "markets and company news. Press releases, video, Briefing.com stubs and formulaic items "
                       "(TABLE, DIARY, PRESS DIGEST, BRIEF, ...) dropped; UPDATE/RPT/REFILE versions de-duplicated. "
                       "Coverage: complete 2007-2013, ~80% of 2014 (sparse in Q4); ends 2014-12-31 (Jan 2015 pages "
                       "look structurally different). Truncated captures were replaced by complete ones where the "
                       "Wayback had one (21 days); the rest (non-holiday days under 25% of normal volume) are "
                       "missing. Volume steps with Reuters' site structure (~950 headlines a weekday 2007-2009, "
                       "~2,200 in 2010, ~4,000 in 2011-12, ~2,900 in 2013-14). known_after_days 1."))
    return reps


if __name__ == "__main__":
    main()
