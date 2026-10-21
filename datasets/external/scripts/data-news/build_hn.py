"""Hacker News daily title sentiment from the cached Algolia responses (raw/hn_algolia).
Day = New York calendar day of created_at_i. Main set: stories with points > 50 (final
Algolia score); variant: points > 20. Titles de-duplicated within a day. Run with venv python -I.
"""
import datetime as dt
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW
from score import score_many, aggregate
from emit import emit, save_daily

_WS = re.compile(r"\s+")
NOTE = ("Hacker News (news.ycombinator.com) story titles from the Algolia HN Search API, one query per "
        "New York calendar day (DST-aware midnight to midnight), 2007-02-19 onward. Tech-industry news and "
        "links, not general news. 'points' is the story's final score, so membership of the >{thr}-point set "
        "uses votes cast after day D (stories collect most votes within ~24 h): known_after_days 2 so every "
        "story has had >= 40 h of voting by the US close it is first used at. Moderators sometimes edit "
        "titles later; Algolia holds the current title. Volume grows with the site (2007-2010) then is "
        "roughly flat. 'default' is excluded from the crisis words (on HN it is nearly always technical, "
        "e.g. 'enabled by default'); 'crash' is often a software crash.")


def load_days():
    days = {}
    for fp in sorted(glob.glob(os.path.join(RAW, "hn_algolia", "*", "*.json.gz"))):
        base = os.path.basename(fp)
        d = dt.date.fromisoformat(base[:10])
        hits = json.loads(gzip.open(fp).read())["hits"]
        days.setdefault(d, []).extend(hits)
    return days


def metrics_for(days, thr):
    texts, dates = [], []
    present = set()
    for d, hits in days.items():
        present.add(d)
        seen = set()
        for h in hits:
            if (h.get("points") or 0) <= thr or not h.get("title"):
                continue
            t = _WS.sub(" ", h["title"]).strip()
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            texts.append(t)
            dates.append(d)
    scores = score_many(texts)
    by_day = defaultdict(list)
    for d, s in zip(dates, scores):
        by_day[d].append(s)
    for d in present:  # a queried day with no qualifying story is a real zero
        by_day.setdefault(d, [])
    return aggregate(by_day, exclude_words=("default",)), len(texts)


def main():
    days = load_days()
    print("HN days cached:", len(days), min(days), max(days))
    reps = []
    m, n = metrics_for(days, 50)
    save_daily("hn", m)
    print(f"HN >50: {n} titles")
    reps += emit("hn", "Hacker News stories with >50 points", m, subcategory="headline_sentiment_tech",
                 region="global (US-centred tech)", source="Hacker News via the Algolia HN Search API",
                 url="https://hn.algolia.com/api/v1/search_by_date?tags=story&numericFilters=created_at_i>=A,created_at_i<B,points>10&hitsPerPage=1000",
                 kad=2, words=True, exclude_words=("default",), notes=NOTE.format(thr=50))
    m, n = metrics_for(days, 20)
    save_daily("hn_p20", m)
    print(f"HN >20: {n} titles")
    reps += emit("hn_p20", "Hacker News stories with >20 points", m, subcategory="headline_sentiment_tech",
                 region="global (US-centred tech)", source="Hacker News via the Algolia HN Search API",
                 url="https://hn.algolia.com/api/v1/search_by_date?tags=story&numericFilters=created_at_i>=A,created_at_i<B,points>10&hitsPerPage=1000",
                 kad=2, words=False, exclude_words=("default",),
                 notes=NOTE.format(thr=20) + " Lower-threshold variant: reaches the 5-title floor from 2008, "
                                             "about two years earlier than the >50 set.")
    return reps


if __name__ == "__main__":
    main()
