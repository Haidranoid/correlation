"""Short volume: download FINRA's Reg SHO daily short-sale volume files (cdn.finra.org/equity/regsho/daily/).

Days: every trading day of the old export from 2009-08-03 to 2025-06-27, then the new export's days through
2026-10-19. Files per day (from the prefix probe, out/prefix_probe.txt):
  - FNSQ (Nasdaq TRF Carteret), FNYX (NYSE TRF), FNRA (ADF; often absent) on every day;
  - FNQC (Nasdaq TRF Chicago) and CNMS (consolidated TRF + ADF) from 2018-08-01;
  - FORF (ORF, OTC securities) on the first trading day of each month only, to check that it holds no universe stocks.
403 is what the CDN returns for a file that doesn't exist; it is recorded as missing. Other errors back off and
retry. Polite: 4 parallel requests, a pause between batches; a generic User-Agent, no identifying headers.
Stops (blocked) if 40 consecutive FNSQ/FNYX requests fail.

Writes raw files to ~/.local/share/correlation2/experiments/short_volume/raw/ and a manifest (TSV) next to it.
Resumable: files already in the manifest with status 200 or 403 are skipped.

    python3 -I download.py
"""
import hashlib
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import numpy as np  # noqa: E402
import c2eval  # noqa: E402

BASE = 'https://cdn.finra.org/equity/regsho/daily/'
UA = 'Mozilla/5.0 (X11; Linux x86_64) research-script'
WORKD = c2eval.WORK / 'short_volume'
RAW = WORKD / 'raw'
MANIFEST = WORKD / 'manifest.tsv'


def trading_days():
    old = [c2eval.iso(d) for d in np.load(Path(c2eval.EXPORT_OLD) / 'days.npy')]
    new = [c2eval.iso(d) for d in np.load(Path(c2eval.EXPORT_NEW) / 'days.npy')]
    a = [d for d in old if '2009-08-03' <= d <= '2025-06-27']
    b = [d for d in new if '2025-06-27' < d <= '2026-10-19']
    return a + b


def jobs():
    out = []
    seen_month = set()
    for d in trading_days():
        ymd = d.replace('-', '')
        pre = ['FNSQ', 'FNYX', 'FNRA']
        if d >= '2018-08-01':
            pre += ['FNQC', 'CNMS']
        if d[:7] not in seen_month:
            seen_month.add(d[:7])
            pre.append('FORF')
        out += [(d, p, f'{p}shvol{ymd}.txt') for p in pre]
    return out


def fetch(job):
    d, p, name = job
    url = BASE + name
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    last = ''
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            (RAW / name).write_bytes(body)
            return (d, p, name, 200, len(body), hashlib.sha256(body).hexdigest()[:16], '')
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return (d, p, name, e.code, 0, '', '')
            last = f'HTTP {e.code}'
        except Exception as e:  # network error: back off
            last = type(e).__name__
        time.sleep(min(60, 2 * 2 ** attempt))
    return (d, p, name, -1, 0, '', last)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    done = {}
    if MANIFEST.exists():
        for l in MANIFEST.read_text().splitlines()[1:]:
            f = l.split('\t')
            if f[3] in ('200', '403', '404'):
                done[f[2]] = f
    else:
        MANIFEST.write_text('date\tprefix\tfile\tstatus\tbytes\tsha256_16\tnote\tfetched_unix\n')
    todo = [j for j in jobs() if j[2] not in done]
    print(f'{len(todo)} files to fetch ({len(done)} already in the manifest)', flush=True)
    fails = 0
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex, open(MANIFEST, 'a') as man:
        for i in range(0, len(todo), 4):
            batch = todo[i:i + 4]
            for r in ex.map(fetch, batch):
                man.write('\t'.join(map(str, r)) + f'\t{int(time.time())}\n')
                if r[1] in ('FNSQ', 'FNYX'):
                    fails = 0 if r[3] == 200 else fails + 1
            man.flush()
            if fails >= 40:
                print('BLOCKED: 40 consecutive FNSQ/FNYX requests failed; stopping', flush=True)
                return 2
            if i % 400 == 0:
                print(f'{i}/{len(todo)} {batch[0][0]} {time.time() - t0:.0f}s', flush=True)
            time.sleep(0.25)
    print('done', time.time() - t0, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
