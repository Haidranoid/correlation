"""Item 1 word sets of every EDGAR-CORPUS 10-K, one file per fiscal year (corpus `year`).

Words: Item 1 (section_1) lower-cased; every run of letters a-z is a word (the lazy prices tokenizer). Per filing:
cik, the fiscal-year end the filing mentions most (from lazy prices' stream), the number of word tokens, and the sorted
unique word ids (a per-year vocabulary). Reads the lazy prices experiment's stream (read only).

    python3 -I text.py [workers]       # writes text/{fy}.npz in the work folder
"""
import gzip
import json
import re
import sys as _sys, os as _os
from concurrent.futures import ProcessPoolExecutor

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import EDGAR, FY_FIRST, FY_LAST, TEXT, log, np  # noqa: E402

TOK = re.compile(r'[a-z]+')


def one_year(fy):
    dst = TEXT / f'{fy}.npz'
    if dst.exists():
        return f'FY{fy}: exists'
    vocab = {}
    cik, fye, ntok, ids, ln, nbad = [], [], [], [], [], 0
    for split in ('train', 'validate', 'test'):
        p = EDGAR / f'{fy}_{split}.jsonl.gz'
        if not p.exists():
            return f'FY{fy}: missing {p.name}'
        with gzip.open(p, 'rt') as f:
            for line in f:
                r = json.loads(line)
                try:
                    c = int(r['cik'])
                except (TypeError, ValueError):
                    nbad += 1
                    continue
                toks = TOK.findall((r.get('section_1') or '').lower())
                u = {vocab.setdefault(w, len(vocab)) for w in set(toks)}
                a = np.fromiter(u, np.int32, len(u))
                a.sort()
                cik.append(c)
                fye.append(r.get('fye') or (-1, -1, -1))
                ntok.append(len(toks))
                ids.append(a)
                ln.append(len(a))
    words = np.array(sorted(vocab, key=vocab.get))
    tmp = TEXT / f'{fy}.tmp.npz'
    np.savez(tmp, cik=np.array(cik, np.int64), fye=np.array(fye, np.int32), ntok=np.array(ntok, np.int64),
             ptr=np.concatenate([[0], np.cumsum(ln)]).astype(np.int64),
             ids=np.concatenate(ids) if ids else np.zeros(0, np.int32), words=words)
    tmp.rename(dst)
    return f'FY{fy}: {len(cik)} filings ({nbad} without a CIK), {int((np.array(ntok) >= 1000).sum())} with >= 1000 ' \
           f'Item 1 words, vocabulary {len(words)}'


if __name__ == '__main__':
    workers = int(_sys.argv[1]) if len(_sys.argv) > 1 else 4
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(one_year, range(FY_FIRST, FY_LAST + 1)):
            log(msg)
