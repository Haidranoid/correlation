"""Daily sampling noise from overlapping items inside one multirange request (same scale,
separate samples): per term, sd of log(b/a) over shared days (both >= 5), median |diff| relative,
and corr of 1-day logdiffs across the two samples. Also weekly: identical? (boundary weeks)."""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtparse import parse_any
RAW = os.path.expanduser('~/.local/share/correlation2/external/raw/google_trends')
QA = os.path.join(os.environ.get('GT_OUT', os.path.dirname(os.path.abspath(__file__))), 'qa')
res = {}
for d in sorted(glob.glob(os.path.join(RAW, 'daily_req', '*'))):
    term = os.path.basename(d)
    lr, da, db, nsh, nid, scale = [], [], [], 0, 0, []
    for p in sorted(glob.glob(os.path.join(d, '*.txt'))):
        merged, cols, conf = parse_any(p)
        for i in range(len(cols) - 1):
            sh = sorted(set(cols[i]) & set(cols[i + 1]))
            a = np.array([cols[i][k][0] for k in sh]); b = np.array([cols[i + 1][k][0] for k in sh])
            nsh += len(sh); nid += int((a == b).sum())
            m = (a >= 5) & (b >= 5)
            lr.extend(np.log(b[m] / a[m]).tolist())
            if m.sum() >= 8:
                scale.append(float(np.median(b[m] / a[m])))
            la, lb = np.log(np.maximum(a, 0.5)), np.log(np.maximum(b, 0.5))
            da.extend(np.diff(la).tolist()); db.extend(np.diff(lb).tolist())
    if nsh:
        res[term] = {'shared_days': nsh, 'identical_frac': nid / nsh,
                     'sd_log_ratio': float(np.std(lr)) if lr else None,
                     'scale_ratio_median_per_pair': float(np.median(scale)) if scale else None,
                     'corr_1day_logdiff': float(np.corrcoef(da, db)[0, 1]) if len(da) > 10 else None}
for k, v in res.items():
    print(f"{k:22s} shared {v['shared_days']:5d} identical {100*v['identical_frac']:4.0f}%  sd log(b/a) {100*(v['sd_log_ratio'] or 0):5.1f}%  scale {v['scale_ratio_median_per_pair']}  corr 1-day logdiff {v['corr_1day_logdiff']:.3f}")
os.makedirs(QA, exist_ok=True)
json.dump(res, open(os.path.join(QA, 'daily_noise.json'), 'w'), indent=1)
