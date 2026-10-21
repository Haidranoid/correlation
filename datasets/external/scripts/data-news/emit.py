"""Write the standard daily series for one headline source from its per-day metrics.

metrics: dict date -> dict from score.aggregate(): n always; vader, lm_negshare, lm_tone,
cw_<word>, cw_any when n >= 5.
Series written (all daily, transform `level`, category `news`):
  <p>_count            headlines per day (all days with data; real zeros where the source
                       was queried for every day)
  <p>_vader            mean VADER compound
  <p>_lm_negshare      share of headlines containing >= 1 LM negative word
  <p>_lm_tone          LM net tone (sum pos - sum neg) / sum words, pooled over the day
  <p>_crisis_any       headlines matching any crisis word
  <p>_crisis_<word>    per-word counts (words=True), only for words non-zero on >= 2% of scored days
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_series, WORK
from score import CRISIS, CRISIS_ORDER, MIN_N

LEVEL_NOTE = ("transform level: the test subtracts a trailing 252-trading-day mean before "
              "standardizing, which absorbs slow drift in archive volume; `anomaly` would compare "
              "with the same date in earlier years and drift on a growing archive")
SCORING = {
    "vader": "mean VADER compound (vaderSentiment 3.3.2, general-purpose lexicon + rules; -1..1) over the day's headlines",
    "lm_negshare": "share of the day's headlines with at least one Loughran-McDonald negative word (LM Master Dictionary 1993-2025, 2,345 negative words; tokens = runs of letters, upper-cased; no stemming, no negation handling)",
    "lm_tone": "Loughran-McDonald net tone for the day: (positive words - negative words) / all word tokens, pooled over the day's headlines (347 positive, 2,345 negative words; no negation handling)",
}
WORD_NOTE = {
    "recession": "regex recession(s)/recessionary",
    "crash": "regex crash/crashes/crashed/crashing; in general news mostly vehicle and plane crashes, not markets",
    "bankruptcy": "regex bankrupt/bankruptcy/bankruptcies/bankrupted",
    "layoffs": "regex layoff(s)/lay-off(s)/lay(s) off/laid off/laying off/job cut(s)/redundancy(ies)",
    "inflation": "regex inflation/inflationary",
    "war": "regex war(s)/warfare/wartime, excluding 'star wars'; includes figurative wars (price war, war of words)",
    "default": "regex default(s)/defaulted/defaulting",
    "pandemic": "regex pandemic(s)",
}


def save_daily(name, metrics):
    os.makedirs(WORK, exist_ok=True)
    with open(os.path.join(WORK, f"{name}_daily.json"), "w") as f:
        json.dump({d.isoformat(): m for d, m in sorted(metrics.items())}, f)


def emit(prefix, label, metrics, *, subcategory, region, source, url, kad, notes, words=False,
         exclude_words=(), count_cut=None, count_note="", name_suffix="", tone_cut=None,
         count_from=None):
    reps = []
    common = dict(category="news", subcategory=subcategory, region=region, source=source, url=url,
                  frequency="daily", transform="level", known_after_days=kad)
    tail = f"{notes} {LEVEL_NOTE}."
    rows = [(d, m["n"]) for d, m in metrics.items()
            if (count_cut is None or d <= count_cut) and (count_from is None or d >= count_from)]
    reps.append(write_series(f"{prefix}_count", rows, name=f"{label}: headlines per day{name_suffix}",
                             units="headlines per day", value_range=(0, 1e6),
                             notes=f"Number of de-duplicated headlines per day. {count_note} {tail}".replace("  ", " "),
                             **common))
    tone_ok = lambda d: tone_cut is None or d <= tone_cut
    for key, sid, nm, units, rng in [
            ("vader", "vader", "mean VADER compound", "VADER compound score, -1..1", (-1, 1)),
            ("lm_negshare", "lm_negshare", "share of headlines with a Loughran-McDonald negative word", "fraction 0..1", (0, 1)),
            ("lm_tone", "lm_tone", "Loughran-McDonald net tone", "(positive - negative words) / words", (-1, 1))]:
        rows = [(d, m[key]) for d, m in metrics.items() if key in m and tone_ok(d)]
        reps.append(write_series(f"{prefix}_{sid}", rows, name=f"{label}: {nm}{name_suffix}", units=units,
                                 value_range=rng,
                                 notes=f"{SCORING[key]}; days with fewer than {MIN_N} headlines have no value. {tail}",
                                 **common))
    used = [w for w in CRISIS_ORDER if w not in exclude_words]
    rows = [(d, m["cw_any"]) for d, m in metrics.items() if "cw_any" in m and tone_ok(d)]
    reps.append(write_series(f"{prefix}_crisis_any", rows, name=f"{label}: headlines with a crisis word{name_suffix}",
                             units="headlines per day", value_range=(0, 1e6),
                             notes=(f"Headlines matching any of: {', '.join(used)} (case-insensitive word regexes; "
                                    f"see SOURCES_data-news.md). Days with fewer than {MIN_N} headlines have no value. "
                                    f"Mostly war and crash (vehicle crashes) in general news. {tail}"),
                             **common))
    if words:
        scored = [d for d, m in metrics.items() if "cw_any" in m and tone_ok(d)]
        skipped = []
        for w in used:
            share = sum(1 for d in scored if metrics[d]["cw_" + w] > 0) / max(1, len(scored))
            if share < MIN_WORD_SHARE:
                skipped.append(f"{w} ({share:.1%})")
                continue
            rows = [(d, metrics[d]["cw_" + w]) for d in scored]
            reps.append(write_series(f"{prefix}_crisis_{w}", rows,
                                     name=f"{label}: headlines mentioning '{w}'{name_suffix}",
                                     units="headlines per day", value_range=(0, 1e6),
                                     notes=(f"Count of headlines matching {WORD_NOTE[w]}; non-zero on {share:.1%} of "
                                            f"scored days. Days with fewer than {MIN_N} headlines have no value. {tail}"),
                                     **common))
        if skipped:
            print(f"   {prefix}: crisis words below {MIN_WORD_SHARE:.0%} of days, no own series: {', '.join(skipped)}")
            SKIPPED[prefix] = skipped
    return reps


MIN_WORD_SHARE = 0.02  # a word gets its own series where it is non-zero on >= 2% of scored days
SKIPPED = {}
