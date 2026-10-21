"""Scan judge JSONs under the given folders: change over random at 0 bps, years, first year."""
import sys, json, glob, os
sys.path.insert(0, '/home/bhoult/programming/correlation/research/lib')
import c2eval
for folder in sys.argv[1:]:
    for p in sorted(glob.glob(os.path.join(folder, '**', '*.json'), recursive=True)):
        try:
            js = json.load(open(p))
        except Exception:
            continue
        if not isinstance(js, dict):
            continue
        try:
            if js.get('years'):
                c = c2eval.change_over_random(js)
                ys = js['years']
                print(f"{c['change']:+7.2f}  n={c['years']:2d} {ys[0]['first_day']}..{ys[-1]['last_day']}  {p}")
            elif 'report' in js and 'strategy' in js['report']:
                c = c2eval.change_over_random(js)
                print(f"{c['change']:+7.2f}  single {js['report'].get('first_day')}..{js['report'].get('last_day')}  {p}")
        except Exception as e:
            pass
