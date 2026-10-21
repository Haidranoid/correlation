"""Wikipedia "Portal:Current events" daily series. Run with the venv python -I.

Main set (wikice_*): the page revision as it stood at D+1 19:30 UTC (before the 4 pm New York
close on D+1 in both EDT and EST), so values contain only what was on the page by then:
  daily  count / VADER / LM negative share / LM tone / crisis_any;
  daily  item counts per topic section (sections exist from mid-2010);
  weekly (Mon-Sun, dated Monday, known_after_days 7) VADER and LM tone for section groups,
         because single sections have too few items a day for the 5-item floor.
Flagged set (wikice_latest_*): today's revision of every page, 2003 onward. Pages before
mid-2006 were written retroactively (2003 pages in 2017) and later pages keep being edited, so
these values contain hindsight; kept only for the long history.
"""
import datetime as dt
import glob
import json
import os
import re
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series
from score import score_many, aggregate, MIN_N
from emit import emit, save_daily, LEVEL_NOTE, SCORING
from wikice_parse import parse, SECTIONS

OUT = os.path.join(RAW, "wikipedia_current_events")
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
TITLE = re.compile(r"^Portal:Current events/(\d{4}) ([A-Z][a-z]+) (\d{1,2})$")
END = dt.date.fromisoformat(os.environ.get("NEWS_WIKICE_END", "2026-10-16"))  # last day fetched (its D+1 19:30 UTC cutoff had passed)
GROUPS = {"conflict_disasters": ("conflict", "disasters"), "business": ("business",),
          "politics_intl": ("politics", "intl"), "law_crime": ("law_crime",)}
WEEKLY_SCORING = {
    "vader": "mean VADER compound (vaderSentiment 3.3.2; -1..1) of the items",
    "lm_tone": ("Loughran-McDonald net tone, (positive - negative words) / all word tokens (347 positive, "
                "2,345 negative words; no negation handling)"),
}
POOL_WEEKS = {"business": 4}  # business averages < 1 item/day after 2012: pool a trailing 4-week window
URL = "https://en.wikipedia.org/wiki/Portal:Current_events"
SRC = "English Wikipedia Portal:Current events daily pages via the MediaWiki API (CC BY-SA 4.0)"
PIT_NOTE = ("Leaf bullet items of the daily page, each prefixed by its topic labels (so the post-2010 nested format "
            "matches the older inline 'Story: sentence' style). Page revision as it stood at D+1 19:30 UTC, before the "
            "New York close on D+1, so known_after_days 1 with no hindsight edits; editors add items over 1-2 days, so "
            "this is somewhat thinner than today's page. World news chosen by volunteer editors, not markets.")


def title_date(t):
    m = TITLE.match(t)
    if not m or m.group(2) not in MONTHS:
        return None
    return dt.date(int(m.group(1)), MONTHS.index(m.group(2)) + 1, int(m.group(3)))


def load_latest():
    out = {}
    for fp in sorted(glob.glob(os.path.join(OUT, "latest", "*.json"))):
        for pg in json.load(open(fp))["query"]["pages"]:
            d = title_date(pg["title"])
            if d is None or not pg.get("revisions"):
                continue
            out[d] = parse(pg["revisions"][0]["slots"]["main"]["content"])
    return out


def load_asof():
    out, norev = {}, []
    for fp in sorted(glob.glob(os.path.join(OUT, "asof", "*", "*.json"))):
        d = dt.date.fromisoformat(os.path.basename(fp)[:10])
        pg = json.load(open(fp))["query"]["pages"][0]
        if not pg.get("revisions"):
            norev.append(d)
            continue
        out[d] = parse(pg["revisions"][0]["slots"]["main"]["content"])
    return out, norev


def start_from_coverage(have, first, last, thr=0.9):
    """First month from which every later month has >= thr of its days present."""
    months = sorted({(d.year, d.month) for d in (first + dt.timedelta(n) for n in range((last - first).days + 1))})
    frac = {}
    for (y, mo) in months:
        days = [dt.date(y, mo, k) for k in range(1, 32) if _valid(y, mo, k) and first <= dt.date(y, mo, k) <= last]
        frac[(y, mo)] = sum(1 for d in days if d in have) / len(days)
    start = None
    for ym in reversed(months):
        if frac[ym] >= thr:
            start = ym
        else:
            break
    return dt.date(start[0], start[1], 1), frac


def _valid(y, m, d):
    try:
        dt.date(y, m, d)
        return True
    except ValueError:
        return False


def score_days(days):
    flat, idx = [], []
    for d, items in days.items():
        for t, sec, _ in items:
            flat.append(t)
            idx.append((d, sec))
    scores = score_many(flat)
    by_day = defaultdict(list)
    by_day_sec = defaultdict(lambda: defaultdict(list))
    for (d, sec), s in zip(idx, scores):
        by_day[d].append(s)
        by_day_sec[d][sec].append(s)
    for d in days:
        by_day.setdefault(d, [])
    return by_day, by_day_sec


def main():
    reps = []
    # ---- point-in-time (main) ----
    pit, norev = load_asof()
    pit = {d: v for d, v in pit.items() if d <= END}
    start, frac = start_from_coverage(set(pit), dt.date(2006, 1, 1), END)
    print("as-of revisions:", len(pit), "no revision at cutoff:", len(norev), "main start:", start)
    print("monthly share with an as-of revision, 2006:", {f"{y}-{m:02d}": round(v, 2) for (y, m), v in frac.items() if y == 2006})
    pit = {d: v for d, v in pit.items() if d >= start}
    by_day, by_day_sec = score_days(pit)
    m = aggregate(by_day)
    save_daily("wikice", m)
    reps += emit("wikice", "Wikipedia Current events (as of next day)", m, subcategory="headline_sentiment_world",
                 region="global (English Wikipedia)", source=SRC, url=URL, kad=1, words=True,
                 count_note="Days whose page existed by the cutoff; 0 = page existed but had no items yet.",
                 notes=PIT_NOTE + f" Starts {start}, the first month from which >= 90% of days have a revision by the cutoff.")
    # sections: start when >= 90% of items carry a section, month by month
    sec_share = defaultdict(lambda: [0, 0])
    for d, items in pit.items():
        sec_share[(d.year, d.month)][0] += sum(1 for _, s, _ in items if s)
        sec_share[(d.year, d.month)][1] += len(items)
    sec_start = None
    for ym in sorted(sec_share, reverse=True):
        a, b = sec_share[ym]
        if b and a / b >= 0.9:
            sec_start = ym
        else:
            break
    if sec_start is None:
        raise SystemExit("no month with >= 90% sectioned items (partial download?)")
    sec_start = dt.date(sec_start[0], sec_start[1], 1)
    print("section series start:", sec_start)
    unm = Counter(h for items in pit.values() for _, s, h in items if s == "other")
    print("unmapped section headers (as-of):", unm.most_common(12))
    sec_days = sorted(d for d in pit if d >= sec_start)
    sec_note = (f"Items under the page's topic header, mapped by keyword to 10 canonical sections (186 header variants "
                f"seen; see SOURCES). Sections exist from {sec_start}; days whose page had no items count 0. "
                f"{PIT_NOTE} {LEVEL_NOTE}.")
    labels = {"conflict": "Armed conflicts and attacks", "disasters": "Disasters and accidents",
              "law_crime": "Law and crime", "politics": "Politics and elections", "intl": "International relations",
              "business": "Business and economy", "health_env": "Health and environment",
              "science_tech": "Science and technology", "sports": "Sports", "arts": "Arts and culture"}
    for sec in SECTIONS:
        rows = [(d, len(by_day_sec[d].get(sec, []))) for d in sec_days]
        reps.append(write_series(f"wikice_sec_{sec}_count", rows,
                                 name=f"Wikipedia Current events: items in '{labels[sec]}' (as of next day)",
                                 category="news", subcategory="headline_count_section", region="global (English Wikipedia)",
                                 source=SRC, url=URL, frequency="daily", units="items per day", transform="level",
                                 known_after_days=1, notes=sec_note, value_range=(0, 1000)))
    rows = [(d, len(by_day_sec[d].get("conflict", [])) + len(by_day_sec[d].get("disasters", []))) for d in sec_days]
    reps.append(write_series("wikice_sec_conflict_disasters_count", rows,
                             name="Wikipedia Current events: items in 'Armed conflicts and attacks' + 'Disasters and accidents' (as of next day)",
                             category="news", subcategory="headline_count_section", region="global (English Wikipedia)",
                             source=SRC, url=URL, frequency="daily", units="items per day", transform="level",
                             known_after_days=1, notes=sec_note, value_range=(0, 1000)))
    # weekly group tone
    weeks = defaultdict(lambda: defaultdict(list))
    for d in sec_days:
        wk = d - dt.timedelta(days=d.weekday())
        for g, secs in GROUPS.items():
            for sec in secs:
                weeks[wk][g].extend(by_day_sec[d].get(sec, []))
    full_weeks = [wk for wk in sorted(weeks) if wk >= sec_start and wk + dt.timedelta(days=6) <= END]
    for g, secs in GROUPS.items():
        nm = " + ".join(labels[s] for s in secs)
        for key in ("vader", "lm_tone"):
            rows = []
            span = POOL_WEEKS.get(g, 1)
            for wk in full_weeks:
                if wk - dt.timedelta(weeks=span - 1) < sec_start:
                    continue  # a pooled window must lie inside the section era
                lst = [s for k in range(span) for s in weeks[wk - dt.timedelta(weeks=k)][g]]
                if len(lst) < MIN_N:
                    continue
                if key == "vader":
                    rows.append((wk, sum(s[0] for s in lst) / len(lst)))
                else:
                    w = sum(s[1] for s in lst)
                    rows.append((wk, (sum(s[2] for s in lst) - sum(s[3] for s in lst)) / w if w else 0.0))
            pooled = (f" Pooled over the trailing {span} weeks (the week dated plus the {span - 1} before it) because "
                      f"this section averages under one item a day after 2012; consecutive values overlap."
                      if span > 1 else "")
            reps.append(write_series(
                f"wikice_wk{span if span > 1 else ''}_{g}_{key}", rows,
                name=(f"Wikipedia Current events: weekly {'mean VADER compound' if key == 'vader' else 'LM net tone'} "
                      f"of '{nm}' items" + (f" (trailing {span} weeks)" if span > 1 else "")),
                category="news", subcategory="headline_sentiment_section", region="global (English Wikipedia)",
                source=SRC, url=URL, frequency="weekly",
                units="VADER compound, -1..1" if key == "vader" else "(positive - negative words) / words",
                transform="level", known_after_days=7, value_range=(-1, 1),
                notes=(f"Weekly (Monday-Sunday, dated the Monday) {WEEKLY_SCORING[key]}, pooled over the week's items in "
                       f"these sections; weeks with fewer than {MIN_N} items have no value. Weekly because these "
                       f"sections average about 1-5 items a day, too few for a daily score. Known after 7 days: the "
                       f"Sunday page is taken as of Monday 19:30 UTC.{pooled} {sec_note}")))
    # ---- latest revision (flagged, long history) ----
    lat = {d: v for d, v in load_latest().items() if dt.date(2003, 1, 1) <= d <= END}
    by_day_l, _ = score_days(lat)
    ml = aggregate(by_day_l)
    save_daily("wikice_latest", ml)
    reps += emit("wikice_latest", "Wikipedia Current events (today's revision, hindsight-edited)", ml,
                 subcategory="headline_sentiment_world", region="global (English Wikipedia)", source=SRC, url=URL,
                 kad=1, words=False,
                 notes=("HINDSIGHT-EDITED: today's revision (fetched 2026-10-18) of every daily page 2003-01-01 onward. "
                        "Pages before mid-2006 were written retroactively (the 2003 pages in 2017) and all pages keep "
                        "being edited, so item selection can reflect later events; use the wikice_* as-of series for "
                        "tests and this only for the long history. Parsed as for wikice_*."))
    return reps


if __name__ == "__main__":
    main()
