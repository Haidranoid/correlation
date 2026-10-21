"""CIK -> export rows, from a public mirror of SEC's company_tickers_exchange.json (SEC itself refuses requests whose
User-Agent has no email address). Source: Hugging Face DenyTranDFW/SEC_Company_Tickers_Exchange (GPL, updated
2026-10-18), read through the Hugging Face datasets-server JSON API. Current tickers only: Stooq's data holds only
stocks still listed, so a current map covers most of the export.

    python3 -I cikmap.py      # writes map/sec_tickers.json and map/cik_to_rows.json in the work folder
"""
import json
import sys as _sys, os as _os
import time
import urllib.error
import urllib.request

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from common import WORK, c2eval, np  # noqa: E402

API = ('https://datasets-server.huggingface.co/rows?dataset=DenyTranDFW/SEC_Company_Tickers_Exchange'
       '&config=default&split=train&offset={o}&length=100')


def fetch():
    out = WORK / 'map' / 'sec_tickers.json'
    if out.exists():
        return json.loads(out.read_text())
    rows, o, total = [], 0, None
    while total is None or o < total:
        for wait in (2, 5, 15, 30, 60, 120, 240):
            try:
                with urllib.request.urlopen(API.format(o=o), timeout=60) as r:
                    d = json.load(r)
                break
            except urllib.error.HTTPError as err:
                if err.code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(wait)
        else:
            raise RuntimeError('rate-limited too long')
        time.sleep(1.5)
        total = d['num_rows_total']
        rows += [x['row'] for x in d['rows']]
        o += 100
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    return rows


def main():
    rows = fetch()
    e = c2eval.load_export(c2eval.EXPORT_OLD, arrays=())
    row_of = {t: j for j, t in enumerate(e['tickers'])}
    cik_rows = {}
    hit = 0
    for r in rows:
        t = r['ticker'].upper().replace('.', '-').replace('/', '-')
        if t in row_of:
            hit += 1
            cik_rows.setdefault(str(int(r['cik'])), []).append(int(row_of[t]))
    (WORK / 'map' / 'cik_to_rows.json').write_text(json.dumps(cik_rows))
    mapped = {j for v in cik_rows.values() for j in v}
    print(f'{len(rows)} SEC tickers, {hit} found in the export; {len(cik_rows)} CIKs cover {len(mapped)} of '
          f'{len(e["tickers"])} export stocks')


if __name__ == '__main__':
    main()
