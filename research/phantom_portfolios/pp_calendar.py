"""FOMC press-release days and CPI release days, for the "drop FOMC and CPI days" robustness check.

    python3 -I pp_calendar.py      -> research/phantom_portfolios/fomc_cpi_days.tsv

Sources (fetched without any email address; raw copies in BIG/raw/):
  - federalreserve.gov/monetarypolicy/fomchistorical{2003..2020}.htm and fomccalendars.htm (2021+):
    every monetary-policy press release linked there (statements, unscheduled moves);
  - BLS's CPI release archive (bls.gov/bls/news-release/cpi.htm) via the Wayback Machine (bls.gov refuses
    scripts): one file per release, cpi_MMDDYYYY.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pp_common import BIG, HERE  # noqa: E402

RAW = BIG / 'raw'
UA = 'Mozilla/5.0 (X11; Linux x86_64) research-script'


def fetch(url, path):
    if not path.exists():
        subprocess.run(['curl', '-sL', '--compressed', '-o', str(path), '-A', UA, url], check=True)
    return path.read_text(encoding='utf8', errors='replace')


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    fomc = set()
    for y in range(2003, 2021):
        t = fetch(f'https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm', RAW / f'fomc_hist{y}.htm')
        for m in re.findall(r'monetary/?(\d{8})a?\.htm|press/monetary/\d{4}/(\d{8})', t):
            d = m[0] or m[1]
            if d.startswith(str(y)):
                fomc.add(d)
    t = fetch('https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm', RAW / 'fomc_cal.htm')
    fomc |= set(re.findall(r'monetary(20[2-9]\d{5})a\.htm', t))
    t = fetch('http://web.archive.org/web/2026id_/https://www.bls.gov/bls/news-release/cpi.htm', RAW / 'bls_cpi.htm')
    cpi = set()
    for mmddyyyy in re.findall(r'cpi_(\d{8})\.(?:htm|pdf|txt)', t):
        cpi.add(mmddyyyy[4:] + mmddyyyy[:4])
    for mmddyy in re.findall(r'cpi_(\d{6})\.txt', t):
        yy = int(mmddyy[4:])
        cpi.add(f'{1900 + yy if yy > 50 else 2000 + yy}{mmddyy[:4]}')
    rows = sorted([(d, 'fomc') for d in fomc] + [(d, 'cpi') for d in cpi])
    out = HERE / 'fomc_cpi_days.tsv'
    with open(out, 'w') as f:
        f.write('date\tkind\n')
        for d, k in rows:
            f.write(f'{d[:4]}-{d[4:6]}-{d[6:]}\t{k}\n')
    by = {}
    for d, k in rows:
        by.setdefault((d[:4], k), 0)
        by[(d[:4], k)] += 1
    print(out, len(fomc), 'FOMC days,', len(cpi), 'CPI days')
    print(' '.join(f'{y}:{by.get((y, "fomc"), 0)}/{by.get((y, "cpi"), 0)}' for y in sorted({y for y, _ in by})))


if __name__ == '__main__':
    main()
