"""C6, step 1: monthly alcohol-headline counts and tone from the outside-data catalog's Reuters US archive
(2007-2014) and HuffPost's News Category Dataset (2012-01..2022-09). Reuters pages are chosen and parsed exactly as
the catalog's build_reuters.py does (its parse_page and capture plan, imported read-only); HuffPost headlines as its
build_bulk.py reads them. Alcohol headlines are scored with the catalog's VADER scorer (score.py).

Run with the catalog's VADER environment:
    ~/programming/c2-wt/venv-data/bin/python -I -B news_build.py     # -> WORK/news_monthly.json, news_build.txt
"""
import datetime as dt
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

NEWS = os.path.expanduser('~/.local/share/correlation2/external/scripts/data-news')
sys.path.insert(0, NEWS)
from build_reuters import END, parse_page  # noqa: E402
from common import RAW  # noqa: E402
import score  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = Path.home() / '.local/share/correlation2/experiments/alcohol_demand'

# Fixed in the plan (C6).
WORDS = ['alcohol', 'alcoholic', 'wine', 'wines', 'winery', 'wineries', 'vineyard', 'vineyards', 'beer', 'beers',
         'brewer', 'brewers', 'brewery', 'breweries', 'brewing', 'liquor', 'whiskey', 'whisky', 'vodka', 'tequila',
         'bourbon', 'cocktail', 'cocktails', 'seltzer', 'Budweiser', 'Anheuser', 'Molson', 'Coors', 'Diageo',
         'Heineken', 'Constellation Brands', 'Brown-Forman', 'Boston Beer', 'Dry January']
ALC = re.compile(r'(?<![\w-])(?:' + '|'.join(re.escape(w) for w in WORDS) + r')(?![\w-])', re.I)
_WS = re.compile(r'\s+')


def reuters():
    plan = json.load(open(os.path.join(RAW, 'reuters_archive_wayback', 'capture_plan.json')))
    for dkey, (ts, _) in sorted(plan.items()):
        d = dt.date.fromisoformat(dkey)
        if d > END:
            continue
        if ts[:8] < (d + dt.timedelta(days=2)).strftime('%Y%m%d'):
            continue                                   # captured before the day was surely complete (as the catalog)
        fp = os.path.join(RAW, 'reuters_archive_wayback', f'{d.year}', f"{d.strftime('%Y%m%d')}_{ts}.html.gz")
        items, _ = parse_page(gzip.open(fp).read())
        for t in items:
            yield d, t


def huffpost():
    with open(os.path.join(RAW, 'huffpost_news_category', 'News_Category_Dataset_v3.json')) as f:
        for line in f:
            o = json.loads(line)
            yield dt.date.fromisoformat(o['date']), _WS.sub(' ', o['headline']).strip()


def main():
    score.init_worker()
    out, lines = {}, []
    for name, gen in (('reuters', reuters), ('huffpost', huffpost)):
        tot = defaultdict(int)
        alc = defaultdict(list)
        examples = []
        for d, t in gen():
            m = f'{d.year:04d}-{d.month:02d}'
            tot[m] += 1
            if ALC.search(t):
                alc[m].append(t)
                if len(examples) < 12 and d.month == 6:
                    examples.append(f'{d} {t}')
        mon = {}
        for m in sorted(tot):
            ts = alc.get(m, [])
            sc = [s[0] for s in score.score_chunk(ts)] if ts else []
            mon[m] = {'n': tot[m], 'n_alc': len(ts), 'vader': (sum(sc) / len(sc)) if len(sc) >= 5 else None}
        out[name] = mon
        n_all = sum(v['n'] for v in mon.values())
        n_alc = sum(v['n_alc'] for v in mon.values())
        s = (f'{name}: {len(mon)} months {min(mon)}..{max(mon)}; headlines {n_all}, alcohol headlines {n_alc} '
             f'({n_alc / n_all:.4f}); months with tone (>= 5 alcohol headlines) '
             f'{sum(1 for v in mon.values() if v["vader"] is not None)}')
        print(s, flush=True)
        lines.append(s)
        lines += ['  e.g. ' + e for e in examples[:8]]
    (WORK / 'news_monthly.json').write_text(json.dumps(out, indent=1))
    (HERE / 'news_build.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
