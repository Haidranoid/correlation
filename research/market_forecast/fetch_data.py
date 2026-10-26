"""Downloads for the market-forecast test (Luca's third idea): the alcohol market's own monthly data and the inputs
that aren't already in hand. Everything goes to WORK/raw/ with its fetch time and sha256 in WORK/fetched.json.

    BEA      NipaDataM.txt (every monthly NIPA series, underlying detail included) and SeriesRegister.txt
    BLS      (public API v1, no key) CPI for beer, spirits and wine at home and alcohol away from home (NSA), and the
             Consumer Expenditure Survey's alcohol spending by age of the reference person, 1984-2024
    Census   national population by single year of age (2000-2010 intercensal, 2010-2020, 2020-2025) and state totals
    TTB      monthly wine, beer and distilled spirits statistics (CSV)
    FRED     MRTSSM4453USN (beer, wine and liquor stores, not seasonally adjusted)
    MA CCC   Massachusetts adult-use cannabis sales by day and product, and price per gram

The FRED inputs of the earlier alcohol test (UNRATENSA, UMCSENT, CPIAUCNS, MRTSSM4453USS) and its Google Trends
series are read from that test's folder, downloaded the same day.

    python3 -I -B fetch_data.py
"""
import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path

WORK = Path.home() / '.local/share/correlation2/experiments/market_forecast'
RAW = WORK / 'raw'
UA = 'Mozilla/5.0 (X11; Linux x86_64) correlation research script'
FILES = {
    'NipaDataM.txt': 'https://apps.bea.gov/national/Release/TXT/NipaDataM.txt',
    'SeriesRegister.txt': 'https://apps.bea.gov/national/Release/TXT/SeriesRegister.txt',
    'us-est00int-alldata.csv': 'https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/'
                               'national/us-est00int-alldata.csv',
    'nc-est2020-agesex-res.csv': 'https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/national/asrh/'
                                 'nc-est2020-agesex-res.csv',
    'nc-est2025-agesex-res.csv': 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/national/asrh/'
                                 'nc-est2025-agesex-res.csv',
    'NST-EST2025-ALLDATA.csv': 'https://www2.census.gov/programs-surveys/popest/datasets/2020-2025/state/totals/'
                               'NST-EST2025-ALLDATA.csv',
    'Wine_monthly_data_csv.csv': 'https://www.ttb.gov/system/files/2024-08/Wine_monthly_data_csv.csv',
    'Beer_National_Report_Monthly_csv.csv': 'https://www.ttb.gov/system/files/2025-05/'
                                            'Beer_National_Report_Monthly_csv.csv',
    'Distilled_Spirits_monthly_data_csv.csv': 'https://www.ttb.gov/system/files/2024-08/'
                                              'Distilled_Spirits_monthly_data_csv.csv',
    'MRTSSM4453USN.csv': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=MRTSSM4453USN',
    'ma_sales_au_gross.csv': 'https://masscannabiscontrol.com/resource/a_sales_au_gross.csv',
    'ma_sales_au_price_per_gram.csv': 'https://masscannabiscontrol.com/resource/a_sales_au_price_per_gram.csv',
}
BLS_API = 'https://api.bls.gov/publicAPI/v1/timeseries/data/'
BLS_SERIES = (['CUUR0000SEFW01', 'CUUR0000SEFW02', 'CUUR0000SEFW03', 'CUUR0000SEFX']
              + [f'CXUALCBEVGLB04{g:02d}M' for g in range(1, 10)])
BLS_WINDOWS = ((1984, 1993), (1994, 2003), (2004, 2013), (2014, 2023), (2024, 2026))   # v1: 10 years a query


def fetch_bls():
    out = []
    for a, b in BLS_WINDOWS:
        body = json.dumps({'seriesid': BLS_SERIES, 'startyear': str(a), 'endyear': str(b)}).encode()
        req = urllib.request.Request(BLS_API, data=body, headers={'Content-Type': 'application/json'})
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        assert r['status'] == 'REQUEST_SUCCEEDED', (a, b, r.get('message'))
        for s in r['Results']['series']:
            for d in s['data']:
                out.append({'series': s['seriesID'], 'year': d['year'], 'period': d['period'], 'value': d['value']})
        print(f'bls {a}-{b}: {len(r["Results"]["series"])} series', flush=True)
        time.sleep(2)
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    log = {}
    for name, url in FILES.items():
        if (RAW / name).exists():      # a rerun after a timeout keeps what already arrived
            data = (RAW / name).read_bytes()
            when = dt.datetime.fromtimestamp((RAW / name).stat().st_mtime, dt.timezone.utc)
            log[name] = {'url': url, 'utc': when.isoformat(timespec='seconds'), 'bytes': len(data),
                         'sha256': hashlib.sha256(data).hexdigest()}
            continue
        hdr = {} if 'stlouisfed' in url else {'User-Agent': UA}    # FRED stalls on this agent; its default works
        data = urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=600).read()
        assert len(data) > 1000 and not data.lstrip()[:15].lower().startswith((b'<!doctype', b'<html')), name
        (RAW / name).write_bytes(data)
        log[name] = {'url': url, 'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                     'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        print(f'{name}: {len(data):,} bytes', flush=True)
    rows = fetch_bls()
    data = json.dumps(rows).encode()
    (RAW / 'bls_api.json').write_bytes(data)
    log['bls_api.json'] = {'url': BLS_API, 'series': BLS_SERIES, 'windows': BLS_WINDOWS,
                           'utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
                           'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    (WORK / 'fetched.json').write_text(json.dumps(log, indent=1))


if __name__ == '__main__':
    main()
