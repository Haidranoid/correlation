"""Polite, resumable Google Trends fetcher (own code; mirrors the trendspy 0.1.6 route).

Route: GET trends.google.com/trends/embed/explore/TIMESERIES?req=...  (HTML page that embeds
the widget JSON with its signed token), then GET /trends/api/widgetdata/multiline (or
multirange) with that token. Every multiline/multirange response is saved unmodified under
raw/google_trends/ the moment it arrives; the widget token JSON is saved next to it.

One request at a time, MIN_GAP + jitter seconds between any two requests, exponential
backoff 60 s -> 900 s on 429/302/5xx/network errors.
"""
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

import requests

RAW = os.path.expanduser('~/.local/share/correlation2/external/raw/google_trends')
LOGF = os.path.join(RAW, 'fetch_log.jsonl')
EMBED_TS = 'https://trends.google.com/trends/embed/explore/TIMESERIES'
WIDGET = {
    'fe_line_chart': 'https://trends.google.com/trends/api/widgetdata/multiline',
    'fe_multi_range_chart': 'https://trends.google.com/trends/api/widgetdata/multirange',
}
HOME = 'https://trends.google.com/trends/explore?geo=US'
UA = ('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/124.0.0.0 Safari/537.36')

_HEX = {r'\x7b': '{', r'\x7d': '}', r'\x22': '"', r'\x5d': ']', r'\x5b': '[', '\\\\': '\\'}


def now_iso():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(**kw):
    kw['t'] = now_iso()
    with open(LOGF, 'a') as f:
        f.write(json.dumps(kw) + '\n')
    print(json.dumps(kw), flush=True)


def decode_escape_text(text):
    for k, v in _HEX.items():
        text = text.replace(k, v)
    if r'\x' in text:
        text = re.sub(r'\\x[0-9a-fA-F]{2}', lambda m: chr(int(m.group(0)[2:], 16)), text)
    return text


class Fetcher:
    def __init__(self, min_gap=13.0, jitter=6.0, tz=300, hl='en-US'):
        self.min_gap, self.jitter, self.tz, self.hl = min_gap, jitter, tz, hl
        self.s = requests.Session()
        self.s.headers.update({'User-Agent': UA, 'Accept-Language': 'en-US,en;q=0.9'})
        self.last = 0.0
        self.n_requests = 0
        self.n_429 = 0
        self.have_cookie = False

    def _pace(self):
        wait = self.min_gap + random.uniform(0, self.jitter) - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)

    def get(self, url, params=None, headers=None, what=''):
        backoff = 60
        tries = 0
        while True:
            self._pace()
            tries += 1
            try:
                r = self.s.get(url, params=params, headers=headers, timeout=60,
                               allow_redirects=False)
                code = r.status_code
            except requests.RequestException as e:
                r, code = None, f'EXC {type(e).__name__}: {e}'[:300]
            self.last = time.time()
            self.n_requests += 1
            if code == 200:
                return r
            if code == 429:
                self.n_429 += 1
            log(ev='http_fail', what=what, code=code, tries=tries, sleep=backoff)
            if isinstance(code, int) and code in (400, 401, 403, 404):
                if tries >= 2:
                    return r
            time.sleep(backoff)
            backoff = min(backoff * 2, 900)
            if code in (429, 302) and tries % 3 == 0:
                # refresh the session cookie every few failures
                self.cookie(force=True)

    def cookie(self, force=False):
        if self.have_cookie and not force:
            return
        try:
            self._pace()
            r = self.s.get(HOME, timeout=60)
            self.last = time.time()
            self.n_requests += 1
            log(ev='cookie', code=r.status_code, cookies=sorted(self.s.cookies.keys()))
            self.have_cookie = True
        except requests.RequestException as e:
            self.last = time.time()
            log(ev='cookie_fail', err=str(e)[:200])

    def token(self, items, what=''):
        """items: list of dicts {keyword,time,geo}. Returns the widget dict (with token)."""
        req = json.dumps({'comparisonItem': items, 'category': 0, 'property': ''})
        params = {'req': req, 'hl': self.hl, 'tz': self.tz}
        r = self.get(EMBED_TS, params=params, what='embed ' + what)
        if r is None or r.status_code != 200:
            raise RuntimeError(f'embed failed: {what}: {getattr(r, "status_code", None)}')
        m = re.findall(r"JSON\.parse\('([^']+)'\)", r.text)
        if not m:
            raise RuntimeError(f'no widget JSON in embed page: {what}: {r.text[:300]!r}')
        return json.loads(decode_escape_text(m[0]))

    def widget_data(self, widget, what=''):
        url = WIDGET[widget['type']]
        params = {'req': json.dumps(widget['request']), 'token': widget['token'],
                  'hl': self.hl, 'tz': self.tz}
        r = self.get(url, params=params, what='data ' + what)
        if r is None or r.status_code != 200:
            raise RuntimeError(f'widget data failed: {what}: {getattr(r, "status_code", None)}')
        return r

    def fetch_to(self, items, out_path, what=''):
        """Fetch one comparison (token + data) and save raw response at out_path. Resumable."""
        if os.path.exists(out_path):
            return 'cached'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        w = self.token(items, what)
        r = self.widget_data(w, what)
        txt = r.text
        body = txt.split('\n', 1)[1] if txt.startswith(")]}'") else txt
        json.loads(body)  # validate before saving
        tmp = out_path + '.part'
        with open(tmp, 'w') as f:
            f.write(txt)
        os.replace(tmp, out_path)
        meta = {'fetched_utc': now_iso(), 'items': items, 'tz': self.tz, 'hl': self.hl,
                'widget_type': w['type'], 'widget_request': w['request'],
                'endpoint': WIDGET[w['type']], 'token_page': EMBED_TS}
        with open(out_path[:-4] + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=1)
        log(ev='saved', what=what, path=out_path, bytes=len(txt))
        return 'fetched'


def parse_timeline(path):
    """Return list of rows: dict(time=int, date=YYYY-MM-DD (UTC), values=[...], partial=bool, ft=str)."""
    txt = open(path).read()
    body = txt.split('\n', 1)[1] if txt.startswith(")]}'") else txt
    d = json.loads(body)
    rows = []
    for p in d['default']['timelineData']:
        if 'columnData' in p:  # multirange
            vals, times, parts = [], [], []
            for c in p['columnData']:
                vals.append(c.get('value'))
                times.append(int(c['time']) if c.get('time') else None)
                parts.append(bool(c.get('isPartial', False)))
            rows.append({'times': times, 'values': vals, 'partials': parts,
                         'ft': [c.get('formattedTime') for c in p['columnData']]})
        else:
            t = int(p['time'])
            rows.append({'time': t,
                         'date': datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%d'),
                         'values': p['value'], 'hasData': p.get('hasData'),
                         'partial': bool(p.get('isPartial', False)),
                         'ft': p.get('formattedTime')})
    return rows
