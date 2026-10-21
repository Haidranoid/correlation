"""Daily headline-sentiment series from the bulk headline datasets:
ABC News Australia, Irish Times, Times of India (Harvard Dataverse, CC0) and HuffPost
(News Category Dataset v3, CC BY 4.0). Run with the venv python -I.
Headlines are de-duplicated within a day (same lower-cased, whitespace-collapsed text).
"""
import csv
import datetime as dt
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, WORK
from score import score_many, aggregate
from emit import emit, save_daily

_WS = re.compile(r"\s+")


def norm(t):
    return _WS.sub(" ", t).strip()


def ymd(s):
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def read_kulkarni(path, has_cat):
    out = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < (3 if has_cat else 2):
                continue
            out.append((ymd(row[0]), row[1] if has_cat else "", norm(row[-1])))
    return out


def read_huffpost(path):
    out = []
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            out.append((dt.date.fromisoformat(o["date"]), o["category"], norm(o["headline"])))
    return out


def daily_metrics(rows, keep=lambda cat: True):
    seen = set()
    texts, days = [], []
    for d, cat, t in rows:
        if not t or not keep(cat):
            continue
        k = (d, t.lower())
        if k in seen:
            continue
        seen.add(k)
        texts.append(t)
        days.append(d)
    scores = score_many(texts)
    by_day = defaultdict(list)
    for d, s in zip(days, scores):
        by_day[d].append(s)
    return aggregate(by_day), len(texts)


def main(which):
    reps = []
    if "abc" in which:
        rows = read_kulkarni(f"{RAW}/dataverse_abc/abcnews-date-text.csv", False)
        m, n = daily_metrics(rows)
        save_daily("abc", m)
        print(f"ABC: {len(rows)} rows, {n} unique day-headlines, {len(m)} days")
        reps += emit("abc", "ABC News Australia", m, subcategory="headline_sentiment_world", region="AU",
                     source="ABC News (Australia) via Harvard Dataverse 'A Million News Headlines' (R. Kulkarni), CC0 1.0",
                     url="https://doi.org/10.7910/DVN/SYBGZL", kad=2, words=True,
                     notes=("ABC News headlines 2003-02-19..2021-12-31, lowercase ASCII without punctuation "
                            "(VADER loses its capitals/punctuation cues). publish_date is the Australian local date: the "
                            "US session of D-1 is reported under D (e.g. 'meldown monday hits wall street' is dated "
                            "2008-09-16), so day D is complete by about 08:00-10:00 New York time on D; "
                            "known_after_days 2 as instructed for far-off time zones (1, even 0, would also be free of "
                            "look-ahead). Volume falls from ~250/day (2013) to ~50/day (2021) as the archive thins."))
    if "irish" in which:
        rows = read_kulkarni(f"{RAW}/dataverse_irish/ireland-news-headlines.csv", True)
        rows = [r for r in rows if not r[1].startswith("removed")]
        base = ("Irish Times headlines 1996-01-02..2019-12-31 (R. Kulkarni, 'News Headlines of Ireland', Harvard "
                "Dataverse, CC0 1.0); category 'removed' dropped. The dataset documents publish_date as the GMT date, "
                "so day D ends 19:00-20:00 New York time, after the close: known_after_days 1.")
        m, n = daily_metrics(rows)
        save_daily("irish", m)
        print(f"Irish: {len(rows)} rows, {n} unique day-headlines, {len(m)} days")
        reps += emit("irish", "Irish Times", m, subcategory="headline_sentiment_world", region="IE",
                     source="The Irish Times via Harvard Dataverse 'News Headlines of Ireland' (R. Kulkarni), CC0 1.0",
                     url="https://doi.org/10.7910/DVN/0U9Z9F", kad=1, words=True, notes=base)
        m, n = daily_metrics(rows, keep=lambda c: c.startswith("business"))
        save_daily("irish_biz", m)
        print(f"Irish business: {n} unique day-headlines, {len(m)} days")
        reps += emit("irish_biz", "Irish Times business section", m, subcategory="headline_sentiment_business",
                     region="IE",
                     source="The Irish Times via Harvard Dataverse 'News Headlines of Ireland' (R. Kulkarni), CC0 1.0",
                     url="https://doi.org/10.7910/DVN/0U9Z9F", kad=1, words=False,
                     notes=base + " Business categories only (headline_category starting 'business'), ~20-25 a day.")
    if "toi" in which:
        rows = read_kulkarni(f"{RAW}/dataverse_toi/india-news-headlines.csv", True)
        base = ("Times of India headlines 2001-01-02..2022-03-31 (R. Kulkarni, 'Times of India News Headlines', "
                "Harvard Dataverse, CC0 1.0). The same headline filed under several city sections is counted once "
                "per day (~163k duplicates). Volume grows from ~150/day (2001-2007) to ~700/day (2011-2018), "
                "~500/day after; mostly city news. publish_date time zone not stated (India, UTC+5:30, presumably); "
                "known_after_days 2 as instructed for far-off time zones (1 would be safe for an IST or UTC date).")
        m, n = daily_metrics(rows)
        save_daily("toi", m)
        print(f"TOI: {len(rows)} rows, {n} unique day-headlines, {len(m)} days")
        reps += emit("toi", "Times of India", m, subcategory="headline_sentiment_world", region="IN",
                     source="Times of India via Harvard Dataverse 'Times of India News Headlines' (R. Kulkarni), CC0 1.0",
                     url="https://doi.org/10.7910/DVN/DPQMQH", kad=2, words=True, notes=base)
        m, n = daily_metrics(rows, keep=lambda c: c.startswith("business"))
        save_daily("toi_biz", m)
        print(f"TOI business: {n} unique day-headlines, {len(m)} days")
        reps += emit("toi_biz", "Times of India business section", m, subcategory="headline_sentiment_business",
                     region="IN",
                     source="Times of India via Harvard Dataverse 'Times of India News Headlines' (R. Kulkarni), CC0 1.0",
                     url="https://doi.org/10.7910/DVN/DPQMQH", kad=2, words=False, count_from=dt.date(2001, 5, 22),
                     notes=base + " Business categories only (headline_category starting 'business'); starts "
                                  "2001-05-22 (one stray item on 2001-01-04 before that).")
    if "huff" in which:
        rows = read_huffpost(f"{RAW}/huffpost_news_category/News_Category_Dataset_v3.json")
        m, n = daily_metrics(rows)
        save_daily("huff", m)
        print(f"HuffPost: {len(rows)} rows, {n} unique day-headlines, {len(m)} days")
        reps += emit("huff", "HuffPost", m, subcategory="headline_sentiment_us", region="US",
                     source="HuffPost via R. Misra 'News Category Dataset' v3 (2022), CC BY 4.0; Hugging Face mirror heegyu/news-category-dataset of the Kaggle file",
                     url="https://huggingface.co/datasets/heegyu/news-category-dataset", kad=1, words=True,
                     count_cut=dt.date(2018, 5, 26),
                     count_note=("Cut at 2018-05-26: the v2 dataset ended there and v3 appended only ~5 headlines a "
                                 "day afterwards, so later counts measure the dataset's sampling, not news flow."),
                     notes=("HuffPost headlines 2012-01-28..2022-09-23 with ~90-100 a day through 2018-05-26 and only "
                            "~5 a day after (sampling break; tone after the break is much noisier). US dates, "
                            "known_after_days 1."))
    if "atn" in which:
        # titles table made by: python3 -I extract_atn.py <WORK>/atn.tsv  (~1 min, ~300 MB; not kept)
        tsv = os.environ.get("ATN_TSV", os.path.join(WORK, "atn.tsv"))
        csv.field_size_limit(sys.maxsize)
        rows = []
        with open(tsv, newline="") as f:
            for d, pub, sec, t in csv.reader(f, delimiter="\t"):
                rows.append((dt.date.fromisoformat(d), pub, norm(t)))
        base = ("All the News 2.0 (Andrew Thompson / Components, 2020; 2,688,878 articles from 27 US publications, "
                "2016-01-01..2020-04-02, scraped from sitemaps/RSS); titles only. Licence: 'for non-commercial, "
                "research purposes only'. Covers only 4.25 years, so it cannot reach 90% of any default 1260-day "
                "correlation period; usable only with shorter periods. Date part of the 'date' field (time zone not "
                "stated; US publications), known_after_days 1. Publication mix shifts over time (Reuters ~31%). Ends "
                "2020-03-27: from 2020-03-28 on the scrape is incomplete (Monday 2020-03-30 has 45% of a normal weekday).")
        ATN_END = dt.date(2020, 3, 27)
        m, n = daily_metrics(rows)
        save_daily("atn", m)
        print(f"ATN: {len(rows)} rows, {n} unique day-headlines, {len(m)} days")
        reps += emit("atn", "All the News 2.0 (27 US publications)", m, subcategory="headline_sentiment_us",
                     region="US", source="All the News 2.0 (Components / A. Thompson), non-commercial research use",
                     url="https://components.one/datasets/all-the-news-2-news-articles-dataset", kad=1, words=True,
                     notes=base, count_cut=ATN_END, tone_cut=ATN_END)
        fin = {"Reuters", "CNBC", "Business Insider"}
        m, n = daily_metrics(rows, keep=lambda pub: pub in fin)
        save_daily("atn_fin", m)
        print(f"ATN fin: {n} unique day-headlines, {len(m)} days")
        reps += emit("atn_fin", "All the News 2.0 business press (Reuters, CNBC, Business Insider)", m,
                     subcategory="headline_sentiment_business", region="US",
                     source="All the News 2.0 (Components / A. Thompson), non-commercial research use",
                     url="https://components.one/datasets/all-the-news-2-news-articles-dataset", kad=1, words=False,
                     count_cut=ATN_END, tone_cut=ATN_END, notes=base + " Subset: the business-focused publications Reuters, CNBC and Business Insider (~730 a day).")

    return reps


if __name__ == "__main__":
    main(sys.argv[1:] or ["abc", "irish", "toi", "huff"])
