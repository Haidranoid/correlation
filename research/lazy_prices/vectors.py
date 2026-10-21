"""Term-frequency vectors of Items 1, 1A and 7 for every filing of a mapped company, one file per corpus year.

Words: the text lower-cased, then every run of letters a-z is a word (no stop-word list, no stemming; numbers and
punctuation are not words). Each word is identified by a 64-bit hash (blake2b), so vectors from different years line
up without a shared vocabulary. Per filing and section: sorted unique word ids, their counts, and the number of words.

    python3 -I vectors.py [workers]     # reads edgar/*.jsonl.gz, writes vec/{year}.npz
"""
import gzip
import hashlib
import json
import re
import sys as _sys, os as _os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import EDGAR, MAP, SECTIONS, VEC, np  # noqa: E402

TOK = re.compile(r'[a-z]+')
_H = {}


def h(w):
    v = _H.get(w)
    if v is None:
        v = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), 'little')
        _H[w] = v
    return v


def vec(text):
    toks = TOK.findall(text.lower())
    c = Counter(toks)
    ids = np.fromiter((h(w) for w in c), np.uint64, len(c))
    cnt = np.fromiter(c.values(), np.uint32, len(c))
    o = np.argsort(ids)
    return ids[o], cnt[o], len(toks)


def one_year(y, keep):
    dst = VEC / f'{y}.npz'
    if dst.exists():
        return f'{y}: exists'
    meta = {k: [] for k in ('cik', 'year', 'filename', 'fye_y', 'fye_m', 'fye_d', 'split')}
    parts = {s: ([], [], []) for s in SECTIONS}      # ids, counts, lengths (unique words)
    nwords = {s: [] for s in SECTIONS}
    total = 0
    for si, split in enumerate(('train', 'validate', 'test')):
        p = EDGAR / f'{y}_{split}.jsonl.gz'
        if not p.exists():
            return f'{y}: missing {p.name}'
        with gzip.open(p, 'rt') as f:
            for line in f:
                r = json.loads(line)
                total += 1
                try:
                    cik = int(r['cik'])
                except (TypeError, ValueError):
                    continue
                if cik not in keep:
                    continue
                fye = r.get('fye') or (-1, -1, -1)
                meta['cik'].append(cik)
                meta['year'].append(int(r['year']))
                meta['filename'].append(r.get('filename') or '')
                meta['fye_y'].append(fye[0]); meta['fye_m'].append(fye[1]); meta['fye_d'].append(fye[2])
                meta['split'].append(si)
                for s in SECTIONS:
                    ids, cnt, n = vec(r.get(s) or '')
                    parts[s][0].append(ids); parts[s][1].append(cnt); parts[s][2].append(len(ids))
                    nwords[s].append(n)
    out = {'cik': np.array(meta['cik'], np.int64), 'year': np.array(meta['year'], np.int32),
           'filename': np.array(meta['filename']), 'fye': np.array([meta['fye_y'], meta['fye_m'], meta['fye_d']], np.int32).T,
           'split': np.array(meta['split'], np.int8), 'n_corpus': np.array(total)}
    for s in SECTIONS:
        ids, cnt, ln = parts[s]
        out[f'{s}_ids'] = np.concatenate(ids) if ids else np.zeros(0, np.uint64)
        out[f'{s}_cnt'] = np.concatenate(cnt) if cnt else np.zeros(0, np.uint32)
        out[f'{s}_ptr'] = np.concatenate([[0], np.cumsum(ln)]).astype(np.int64)
        out[f'{s}_words'] = np.array(nwords[s], np.int64)
    tmp = VEC / f'{y}.tmp.npz'
    np.savez(tmp, **out)
    tmp.rename(dst)
    return f'{y}: {total} corpus filings, {len(meta["cik"])} of mapped companies'


def main():
    VEC.mkdir(parents=True, exist_ok=True)
    keep = {int(c) for c in json.loads(MAP.read_text())}
    years = sorted({int(p.name[:4]) for p in EDGAR.glob('*_test.jsonl.gz')})
    workers = int(_sys.argv[1]) if len(_sys.argv) > 1 else 4
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(one_year, years, [keep] * len(years)):
            print(msg, flush=True)


if __name__ == '__main__':
    main()
