"""Download the outside data for the alcohol-demand test (experiments/untested-luca-alcohol-demand.md).

Google Trends: one monthly series per query, US, 2004-01 to the latest month ('all'), through the outside-data
catalog's polite fetcher (scripts_data-trends/gtfetch.py, imported read-only; its log is pointed at this test's own
folder). Each query's terms are joined with ' + ' (Google's OR), so one query is one combined series.

FRED: final-vintage CSVs from fredgraph.csv (no API key).

    python3 -I -B fetch_data.py      # -> ~/.local/share/correlation2/experiments/alcohol_demand/{trends,fred}/
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WORK = Path.home() / '.local/share/correlation2/experiments/alcohol_demand'
GT = Path.home() / '.local/share/correlation2/external/scripts_data-trends'

TRENDS = {
    'alcohol': 'wine + beer + whiskey + vodka + tequila',
    'moderation': 'non alcoholic + sober + mocktail + dry january',
    'new_markets': 'marijuana + cannabis + hard seltzer + psilocybin',
    'wine': 'wine',
    'hard_seltzer': 'hard seltzer',
    'cannabis': 'marijuana + cannabis',
}
FRED = ('UNRATENSA', 'UMCSENT', 'CPIAUCNS', 'CUUR0000SAF116', 'MRTSSM4453USS', 'MRTSSM44X72USS')


def fetch_fred():
    out = WORK / 'fred'
    for s in FRED:
        f = out / f'{s}.csv'
        if f.exists():
            continue
        url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}'
        data = urllib.request.urlopen(url, timeout=60).read()
        assert data.startswith(b'observation_date') or data.startswith(b'DATE'), (s, data[:80])
        f.write_bytes(data)
        print(f'fred {s}: {len(data.splitlines())} lines', flush=True)
        time.sleep(2)


def fetch_trends():
    sys.path.insert(0, str(GT))
    import gtfetch
    gtfetch.RAW = str(WORK / 'trends')
    gtfetch.LOGF = str(WORK / 'trends' / 'fetch_log.jsonl')
    f = gtfetch.Fetcher(min_gap=13.0, jitter=5.0, tz=300, hl='en-US')
    f.cookie()
    for name, q in TRENDS.items():
        path = str(WORK / 'trends' / f'{name}.json')
        st = f.fetch_to([{'keyword': q, 'geo': 'US', 'time': 'all'}], path, name)
        rows = gtfetch.parse_timeline(path)
        print(f'trends {name}: {st}, {len(rows)} months {rows[0]["date"]}..{rows[-1]["date"]}', flush=True)


if __name__ == '__main__':
    (WORK / 'fetched_utc.json').write_text(json.dumps({'started': datetime.now(timezone.utc).isoformat()}))
    fetch_fred()
    fetch_trends()
    print('DONE')
