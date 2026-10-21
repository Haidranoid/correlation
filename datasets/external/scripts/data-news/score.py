"""Headline scoring shared by every source (run with the venv python, -I).

Per headline:
  * VADER compound (vaderSentiment 3.3.2, Hutto & Gilbert 2014; MIT licence), in [-1, 1].
  * Loughran-McDonald (LM) counts: tokens = maximal runs of ASCII letters, upper-cased
    (apostrophes split tokens); n_words = number of tokens; n_pos / n_neg = tokens in the LM
    Positive / Negative lists (Master Dictionary 1993-2025: a word is in a list when its
    column holds a positive year; a negative year means it was removed). The dictionary
    contains inflections, so no stemming. No negation handling.
  * Crisis-word flags (case-insensitive regexes below), one bit per word.
Daily aggregation (aggregate()):
  n            headlines that day
  vader        mean VADER compound
  lm_negshare  share of headlines with >= 1 LM negative word
  lm_tone      (sum pos - sum neg) / sum words, pooled over the day's headlines
  cw_<word>    number of headlines matching the word; cw_any = matching any of the eight
Tone/share/crisis values are written only for days with n >= MIN_N (5).
"""
import csv
import os
import re

MIN_N = 5
LM_CSV = os.path.expanduser(
    "~/.local/share/correlation2/external/raw/lexicons/Loughran-McDonald_MasterDictionary_1993-2025.csv")

CRISIS = {
    "recession": r"\brecessions?\b|\brecessionary\b",
    "crash": r"\bcrash(?:es|ed|ing)?\b",
    "bankruptcy": r"\bbankrupt(?:cy|cies|ed|s)?\b",
    "layoffs": r"\blay-?offs?\b|\blays? off\b|\blaid off\b|\blaying off\b|\bjob cuts?\b|\bredundanc(?:y|ies)\b",
    "inflation": r"\binflation(?:ary)?\b",
    "war": r"(?<!star )\bwars?\b|\bwarfare\b|\bwartime\b",
    "default": r"\bdefault(?:s|ed|ing)?\b",
    "pandemic": r"\bpandemics?\b",
}
CRISIS_ORDER = list(CRISIS)
_CRISIS_RE = [(k, re.compile(v, re.I)) for k, v in CRISIS.items()]
_TOKEN = re.compile(r"[A-Za-z]+")


def load_lm(path=LM_CSV):
    pos, neg = set(), set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        for row in r:
            w = row["Word"].strip().upper()
            if not w:
                continue
            if float(row["Positive"] or 0) > 0:
                pos.add(w)
            if float(row["Negative"] or 0) > 0:
                neg.add(w)
    return pos, neg


_STATE = {}


def init_worker():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _STATE["vader"] = SentimentIntensityAnalyzer()
    _STATE["pos"], _STATE["neg"] = load_lm()


def score_text(text):
    """-> (compound, n_words, n_pos, n_neg, crisis_bits)"""
    if "vader" not in _STATE:
        init_worker()
    comp = _STATE["vader"].polarity_scores(text)["compound"]
    toks = [t.upper() for t in _TOKEN.findall(text)]
    pos, neg = _STATE["pos"], _STATE["neg"]
    npos = sum(1 for t in toks if t in pos)
    nneg = sum(1 for t in toks if t in neg)
    bits = 0
    for i, (_, rx) in enumerate(_CRISIS_RE):
        if rx.search(text):
            bits |= 1 << i
    return comp, len(toks), npos, nneg, bits


def score_chunk(texts):
    return [score_text(t) for t in texts]


def score_many(texts, procs=20, chunk=2000):
    """Score a list of strings in parallel; returns a list of tuples in input order."""
    if len(texts) < 5000:
        return score_chunk(texts)
    import multiprocessing as mp
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    out = []
    with mp.get_context("fork").Pool(procs, initializer=init_worker) as pool:
        for res in pool.imap(score_chunk, chunks, chunksize=1):
            out.extend(res)
    return out


def aggregate(day_scores, exclude_words=()):
    """day_scores: dict date -> list of score tuples. Returns dict date -> metrics dict.
    Tone/share/crisis metrics only when n >= MIN_N; n is always present."""
    excl_mask = 0
    for w in exclude_words:
        excl_mask |= 1 << CRISIS_ORDER.index(w)
    out = {}
    for d, lst in day_scores.items():
        n = len(lst)
        m = {"n": n}
        if n >= MIN_N:
            comp = sum(s[0] for s in lst) / n
            words = sum(s[1] for s in lst)
            pos = sum(s[2] for s in lst)
            neg = sum(s[3] for s in lst)
            m["vader"] = comp
            m["lm_negshare"] = sum(1 for s in lst if s[3] > 0) / n
            m["lm_tone"] = (pos - neg) / words if words else 0.0
            for i, w in enumerate(CRISIS_ORDER):
                m["cw_" + w] = sum(1 for s in lst if s[4] >> i & 1)
            m["cw_any"] = sum(1 for s in lst if s[4] & ~excl_mask)
        out[d] = m
    return out
