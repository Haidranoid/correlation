"""Forward confirmation: each later download through the frozen code (the .md, "Holdouts").

    python3 -I -B forward.py EXPORT_DIR STORE_DIR

A 500 chosen at the close of 2026-10-19, bought at the next close, then the year-end rule. P1 and P2's daily net log
differences are kept in a ledger by date with first-seen values: a later download never overwrites a date already
recorded. Downloads are expected monthly; the trading days of a calendar month with no download of its own (a month
strictly between two consecutive downloads' months) are logged as a gap and left out. No div_fix panel exists for
later downloads, so dividends after 2022 are not added (both portfolios alike). Report at 12 months (about 250
trading days). Neither primary passed in development, so the forward test is reported only.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eg_common as C  # noqa: E402
import run as RUN  # noqa: E402

np = C.np
FWD = C.WORK / 'forward'
START = '2026-10-19'


def months_between(a, b):
    """Calendar months strictly between 'YYYY-MM' a and b."""
    y, m = map(int, a.split('-'))
    out = []
    while True:
        m += 1
        if m > 12:
            y, m = y + 1, 1
        if f'{y:04d}-{m:02d}' >= b:
            return out
        out.append(f'{y:04d}-{m:02d}')


def main(export_dir, store_dir, start=START, fwd=None):
    global FWD
    FWD = Path(fwd) if fwd else FWD
    FWD.mkdir(parents=True, exist_ok=True)
    frozen = (C.HERE / 'frozen_sha256.txt').read_text().splitlines()
    for line in frozen:
        if line.strip() and not line.startswith('#'):
            h, name = line.split()
            assert hashlib.sha256((C.HERE / name).read_bytes()).hexdigest() == h, f'{name} changed since the freeze'
    res = RUN.main('fwd', forward={'export': export_dir, 'store': store_dir, 'start': start})
    S = np.load(C.WORK / 'series_fwd.npz')
    iso = S['iso']
    d = {}
    for cname, a in (('P1', 'R1d'), ('P2', 'R1m')):
        ra = C.net(S[f'{a}_gross'], S[f'{a}_turn'], C.COST)
        rb = C.net(S['B1_gross'], S['B1_turn'], C.COST)
        d[cname] = np.log1p(ra) - np.log1p(rb)
    led_p = FWD / 'ledger.json'
    led = json.loads(led_p.read_text()) if led_p.exists() else {'days': {}, 'downloads': [], 'gaps': []}
    last_seen = max(led['days']) if led['days'] else start
    this_month = iso[-1][:7]
    prev_month = led['downloads'][-1]['last'][:7] if led['downloads'] else start[:7]
    missed = months_between(prev_month, this_month)
    added = 0
    for k, day in enumerate(iso):
        if day <= last_seen:
            continue
        if day[:7] in missed:
            continue
        led['days'][day] = {'P1': float(d['P1'][k]), 'P2': float(d['P2'][k])}
        added += 1
    if missed:
        led['gaps'] += missed
    led['downloads'].append({'export': str(export_dir), 'last': iso[-1], 'added': added, 'missed_months': missed})
    led_p.write_text(json.dumps(led, indent=1))
    days = sorted(led['days'])
    L = [f'Forward ledger: {len(days)} days ({days[0] if days else "-"}..{days[-1] if days else "-"}); this download added '
         f'{added}; gaps (months left out): {led["gaps"]}']
    for cname in ('P1', 'P2'):
        x = np.array([led['days'][k][cname] for k in days])
        if len(x) > 50:
            t = C.test_series(x)
            L.append(f"{cname}: {100 * t['per_year']:+.3f}% a year, NW t {t['t_nw21']:+.2f}, p {t['p']:.4f}")
        else:
            L.append(f'{cname}: {len(x)} days, too few to test')
    L.append('Report at 12 months (about 250 trading days): ' + ('due' if len(days) >= 250 else f'{len(days)} of 250 days'))
    txt = '\n'.join(L) + '\n'
    (FWD / 'forward.txt').write_text(txt)
    print(txt)


if __name__ == '__main__':
    main(*sys.argv[1:3])
