import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terms import *
Q = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queue')
os.makedirs(Q, exist_ok=True)
def wj(kw):
    return {'label': f'weekly_joint {kw}', 'out': f'weekly_joint/{slug(kw)}.txt', 'items': items(kw, WEEKLY_MR5)}
def dump(name, tasks):
    json.dump(tasks, open(os.path.join(Q, name), 'w'), indent=1)
    print(name, len(tasks))
first = ['debt', 'bitcoin', 'stock market crash']
dump('010_weekly_first.json', [wj(k) for k in first])
blk = daily_blocks()
print('daily blocks', len(blk), blk[0][0], blk[0][-1], blk[-1][0], blk[-1][-1])
dump('015_daily_probe.json', [{'label': f'daily_block {kw} {b[0][0]}', 'out': f'daily_blocks/{slug(kw)}/{b[0][0]}_{b[-1][1]}.txt', 'items': items(kw, b)} for kw in ['debt', 'stock market crash'] for b in blk[:1]])
key_rest = [k for k in KEY8 if k not in first]
t = [wj(k) for k in key_rest]
t.append({'label': 'weekly_joint_repeat_exact debt', 'out': 'weekly_joint_repeat/debt__exact.txt', 'items': items('debt', WEEKLY_MR5)})
dump('020_weekly_key.json', t)
dump('025_dup_shift.json', [{'label': f'weekly_joint_repeat_shift1d {k}', 'out': f'weekly_joint_repeat/{slug(k)}__shift1d.txt', 'items': items(k, WEEKLY_MR5_SHIFT)} for k in KEY8])
rest = [kw for kw, g, s in CORE if kw not in KEY8]
dump('030_weekly_rest.json', [wj(k) for k in rest])
dump('040_weekly_tickers.json', [wj(k) for k in TICKERS])
print('core', len(CORE), 'rest', len(rest), 'tickers', len(TICKERS))
for b in blk: print(b[0][0], '->', b[-1][1])
