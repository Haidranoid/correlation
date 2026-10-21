"""Stream EDGAR-CORPUS (eloukas/edgar-corpus on Hugging Face) one year file at a time, keeping only what the lazy
prices experiment needs: cik, year, filename, the fiscal-year end the filing mentions most (the imposters experiment's
rule, used for the filing date), and Items 1, 1A and 7 (section_1, section_1A, section_7). The raw JSONL is never
stored. Output: ~/.local/share/correlation2/experiments/lazy_prices/edgar/{year}_{split}.jsonl.gz

    python3 -I fetch_edgar.py 2003 2020 [workers]   # years inclusive
"""
import gzip
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT = Path.home() / '.local/share/correlation2/experiments/lazy_prices/edgar'
URL = 'https://huggingface.co/datasets/eloukas/edgar-corpus/resolve/main/{y}/{s}.jsonl'
MONTHS = {m: i + 1 for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                           'september', 'october', 'november', 'december'])}
PAT = re.compile(r'years? ended\s+(january|february|march|april|may|june|july|august|september|october|november|'
                 r'december)\s+(\d{1,2}),?\s+(\d{4})', re.I)
KEEP = ('section_1', 'section_1A', 'section_7')


def fye(rec):
    """Copied from research/imposters/fetch_edgar.py: the latest fiscal-year end among the three most mentioned
    'year(s) ended <date>' phrases, as (y, m, d)."""
    txt = ' '.join((rec.get(k) or '') for k in ('section_1', 'section_5', 'section_6', 'section_7', 'section_8'))
    c = Counter((int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))) for m in PAT.finditer(txt))
    if not c:
        return None
    return max(k for k, _ in c.most_common(3))


def one(y, s):
    dst = OUT / f'{y}_{s}.jsonl.gz'
    if dst.exists():
        return f'{dst.name} exists'
    tmp = dst.with_suffix('.tmp')
    p = subprocess.Popen(['curl', '-sL', '--fail', '--retry', '5', URL.format(y=y, s=s)], stdout=subprocess.PIPE)
    n = bad = 0
    with gzip.open(tmp, 'wt', compresslevel=5) as f:
        for line in p.stdout:
            try:
                rec = json.loads(line)
            except Exception:
                bad += 1
                continue
            n += 1
            out = {'cik': rec.get('cik'), 'year': rec.get('year'), 'filename': rec.get('filename'), 'fye': fye(rec)}
            for k in KEEP:
                out[k] = rec.get(k) or ''
            f.write(json.dumps(out) + '\n')
    if p.wait() != 0:
        tmp.unlink(missing_ok=True)
        return f'{y}/{s}: curl failed'
    tmp.rename(dst)
    return f'{y}/{s}: {n} filings, {bad} bad lines'


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    a, b = int(sys.argv[1]), int(sys.argv[2])
    jobs = [(y, s) for y in range(b, a - 1, -1) for s in ('train', 'validate', 'test')]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(one, y, s) for y, s in jobs]
        for fu in futs:
            print(fu.result(), flush=True)
