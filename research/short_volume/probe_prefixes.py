"""Short volume: which FINRA Reg SHO daily-file prefixes exist over time (HEAD requests only).

Three trading days a year (the first trading day of February, June and October, from the new export's calendar),
2009-2026, for each prefix. Polite: 4 parallel requests, a pause between batches, no identifying headers.

    python3 -I probe_prefixes.py   # writes out/prefix_probe.txt
"""
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

BASE = 'https://cdn.finra.org/equity/regsho/daily/'
PREFIXES = ['CNMS', 'FNSQ', 'FNQC', 'FNYX', 'FNRA', 'FORF']
UA = 'Mozilla/5.0 (X11; Linux x86_64) research-script'


def head(url):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status
        except urllib.error.HTTPError as e:
            if e.code == 403 or e.code == 404:
                return e.code
            time.sleep(2 ** attempt * 2)
        except Exception:
            time.sleep(2 ** attempt * 2)
    return -1


def main():
    days = np.load(Path(c2eval.EXPORT_NEW) / 'days.npy')
    dates = [c2eval.iso(d) for d in days]
    pick = []
    for y in range(2009, 2027):
        for m in ('02', '06', '10'):
            c = [d for d in dates if d.startswith(f'{y}-{m}')]
            if c:
                pick.append(c[0])
    jobs = [(d, p) for d in pick for p in PREFIXES]
    res = {}
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(jobs), 4):
            batch = jobs[i:i + 4]
            urls = [f"{BASE}{p}shvol{d.replace('-', '')}.txt" for d, p in batch]
            for (d, p), st in zip(batch, ex.map(head, urls)):
                res[(d, p)] = st
            time.sleep(0.4)
    lines = ['date        ' + ' '.join(f'{p:>5}' for p in PREFIXES)]
    for d in pick:
        lines.append(f'{d}  ' + ' '.join(f'{res[(d, p)]:>5}' for p in PREFIXES))
    out = '\n'.join(lines) + '\n'
    (HERE / 'out' / 'prefix_probe.txt').write_text(out)
    print(out)


if __name__ == '__main__':
    main()
