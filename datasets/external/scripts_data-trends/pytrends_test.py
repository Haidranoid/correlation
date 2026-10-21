"""One polite pytrends 4.9.2 test call (3 requests: cookie GET, explore POST, multiline GET),
15 s apart, no retries. Records the outcome in qa/extra.json -> pytrends_test. Run only after
the runner has exited (one request at a time overall)."""
import json
import os
import sys
import time
import traceback

QA = sys.argv[1]
res = {}
t0 = time.time()
try:
    import pandas as pd  # noqa: F401
    from pytrends.request import TrendReq
    pt = TrendReq(hl='en-US', tz=300, retries=0, backoff_factor=0, timeout=(10, 30))  # GET cookies
    time.sleep(15)
    pt.build_payload(['debt'], timeframe='2024-01-01 2026-10-17', geo='US')  # POST /api/explore
    time.sleep(15)
    df = pt.interest_over_time()  # GET widgetdata/multiline
    res = {'ok': True, 'rows': int(len(df)), 'first': str(df.index[0].date()) if len(df) else None,
           'last': str(df.index[-1].date()) if len(df) else None,
           'max': float(df['debt'].max()) if len(df) else None}
except Exception as e:
    res = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:300]}', 'tb': traceback.format_exc()[-800:]}
res['seconds'] = round(time.time() - t0, 1)
p = os.path.join(QA, 'extra.json')
d = json.load(open(p)) if os.path.exists(p) else {}
if res.get('ok'):
    d['pytrends_test'] = (f"worked (debt, 2024-01-01..2026-10-17, {res['rows']} weekly rows, "
                          f"{res['first']}..{res['last']}).")
else:
    d['pytrends_test'] = f"failed: {res['error']}"
d['pytrends_raw'] = res
json.dump(d, open(p, 'w'), indent=1)
print(json.dumps(res)[:1500])
