"""10-K Item 1 text -> Hoberg-Phillips style word sets, keyed by CIK and fiscal year, for the stocks in the export.

Words: lowercase alphabetic tokens of 3+ letters, plural nouns folded to their singular, kept when they are
"noun-like" (WordNet lists a noun sense, or WordNet doesn't know the word at all, e.g. industry jargon), not a stop
word, not mostly capitalized (proper names, places, brands, acronyms: lowercase in under 25% of the filings that use
them), and used by at least 2 and at most 25% of that fiscal year's filings (all filings in the corpus, not just
mapped ones). No model and no labels, so nothing from after the filing date enters.

Point in time: EDGAR-CORPUS gives no filing date; its `year` is the fiscal year (checked against the "year ended"
dates in the text). A filing counts from the first trading day on or after June 1 of the next year, or 120 days after
the fiscal-year end it mentions most, whichever is later.

    python3 -I text.py        # needs map/cik_to_rows.json from cikmap.py; writes text.npz in the work folder
"""
import gzip
import json
import re
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import EDGAR, WORK, np

STOP = set("""a about above after again against all am an and any are as at be because been before being below between
both but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where which while who whom
why will with would you your yours yourself yourselves also may might must shall within without upon per via among
including include includes included item business company companies inc corporation corp llc ltd year years fiscal
annual report form herein thereof therein hereby pursuant section""".split())
TOKEN = re.compile(r'[A-Za-z]{3,}')
MAX_DF, MIN_DF, MIN_LOWER, MIN_WORDS = 0.25, 2, 0.25, 50


def wordnet():
    z = zipfile.ZipFile(WORK / 'wordnet' / 'wordnet.zip')
    sets = {}
    for pos in ('noun', 'verb', 'adj', 'adv'):
        name = next(n for n in z.namelist() if n.endswith(f'index.{pos}'))
        sets[pos] = {l.split(' ', 1)[0] for l in z.read(name).decode('latin-1').splitlines() if not l.startswith(' ')}
    exc = {}
    name = next(n for n in z.namelist() if n.endswith('noun.exc'))
    for l in z.read(name).decode('latin-1').splitlines():
        f = l.split()
        if len(f) >= 2:
            exc[f[0]] = f[1]
    return sets, exc


def lemma_table(words, sets, exc):
    """word -> its noun lemma, or None when it is not noun-like. Plurals fold to a WordNet noun; words WordNet lists
    only as verbs, adjectives or adverbs, and inflected forms of them (-s, -ed, -ing, -er), are dropped; words
    WordNet doesn't know at all (jargon) are kept."""
    nouns = sets['noun']
    other = sets['verb'] | sets['adj'] | sets['adv']
    known = nouns | other
    infl = (('ied', ('y',)), ('ies', ('y',)), ('ing', ('', 'e')), ('ed', ('', 'e')), ('es', ('', 'e')), ('s', ('',)),
            ('er', ('', 'e')), ('est', ('', 'e')))

    def one(w):
        if w in STOP:
            return None
        if len(w) >= 5:
            if w.endswith('ies') and w[:-3] + 'y' in nouns:
                return w[:-3] + 'y'
            if w.endswith('s') and not w.endswith('ss') and w[:-1] in nouns:
                return w[:-1]
            if w.endswith('es') and w[:-2] in nouns and w[:-2] not in STOP:
                return w[:-2]
        if w in exc and exc[w] in nouns:
            return exc[w]
        if w in nouns:
            return w
        if w in other:
            return None
        for suf, adds in infl:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                stem = w[:-len(suf)]
                cands = [stem + a for a in adds]
                if len(stem) >= 3 and stem[-1] == stem[-2]:
                    cands.append(stem[:-1])
                if any(c in known for c in cands):
                    return None
        return w

    out = {}
    for w in words:
        lw = one(w)
        out[w] = None if lw in STOP else lw
    return out


def fiscal_year(fy, keep_ciks):
    """All filings of one fiscal year: document frequencies over all of them, word sets for mapped CIKs."""
    sets, exc = wordnet()
    docs = []
    df = Counter()
    lower_df = Counter()
    n_docs = 0
    for split in ('train', 'validate', 'test'):
        p = EDGAR / f'{fy}_{split}.jsonl.gz'
        if not p.exists():
            continue
        with gzip.open(p, 'rt') as f:
            for line in f:
                r = json.loads(line)
                toks = TOKEN.findall(r['text'])
                if len(toks) < 50:
                    continue
                n_docs += 1
                low = {t for t in toks if t.islower()}
                allw = {t.lower() for t in toks}
                df.update(allw)
                lower_df.update(low)
                try:
                    cik = int(r['cik'])
                except (TypeError, ValueError):
                    continue
                if cik in keep_ciks:
                    docs.append((cik, r['fye'], allw))
    lem = lemma_table(df.keys(), sets, exc)
    good = {w for w, c in df.items()
            if lem[w] is not None and MIN_DF <= c <= MAX_DF * n_docs and lower_df[w] >= MIN_LOWER * c}
    out = []
    for cik, fye, allw in docs:
        ws = sorted({lem[w] for w in allw if w in good})
        out.append((cik, fye, ws))
    return fy, n_docs, len(good), out


def main():
    m = json.loads((WORK / 'map' / 'cik_to_rows.json').read_text())
    keep = {int(c) for c in m}
    years = sorted({int(p.name[:4]) for p in EDGAR.glob('*_*.jsonl.gz')})
    vocab, rows = {}, []
    with ProcessPoolExecutor(6) as ex:
        for fy, n_docs, n_good, out in ex.map(fiscal_year, years, [keep] * len(years)):
            print(f'FY{fy}: {n_docs} filings, {n_good} words kept, {len(out)} mapped filings', flush=True)
            for cik, fye, ws in out:
                ids = [vocab.setdefault(w, len(vocab)) for w in ws]
                rows.append((cik, fy, fye, ids))
    # one filing per CIK and fiscal year: the one with the most words (amendments are usually shorter)
    best = {}
    for k, r in enumerate(rows):
        key = (r[0], r[1])
        if key not in best or len(r[3]) > len(rows[best[key]][3]):
            best[key] = k
    rows = [rows[k] for k in sorted(best.values()) if len(rows[k][3]) >= MIN_WORDS]
    # availability: June 1 of FY+1, or 120 days after a later fiscal-year end mentioned in the text
    avail = []
    for cik, fy, fye, ids in rows:
        a = np.datetime64(f'{fy + 1}-06-01')
        if fye is not None and fye[0] > fy:
            try:
                a = max(a, np.datetime64(f'{fye[0]:04d}-{fye[1]:02d}-{min(fye[2], 28):02d}') + np.timedelta64(120, 'D'))
            except ValueError:
                pass
        avail.append((a - np.datetime64('1970-01-01')).astype(int))
    indptr = np.cumsum([0] + [len(r[3]) for r in rows])
    indices = np.concatenate([np.array(r[3], dtype=np.int32) for r in rows])
    words = np.array(sorted(vocab, key=vocab.get))
    np.savez_compressed(WORK / 'text.npz', cik=np.array([r[0] for r in rows]), fy=np.array([r[1] for r in rows]),
                        avail=np.array(avail), indptr=indptr, indices=indices, words=words)
    print(f'{len(rows)} filings kept, vocabulary {len(words)}')


if __name__ == '__main__':
    main()
