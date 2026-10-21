"""Binary-search the first ALFRED vintage containing a given reference month of CAPHCI.
Usage: python3 -I phci_bsearch.py YYYY-MM [YYYY-MM ...]"""
import datetime as dt, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib, phci_timing as pt
out = open(os.path.join(pt.OUT, "alfred_first_vintage.tsv"), "a")
for arg in sys.argv[1:]:
    y, m = map(int, arg.split("-"))
    ref, end = dt.date(y, m, 1), plib.month_end(y, m)
    lo, hi = end + dt.timedelta(days=15), end + dt.timedelta(days=130)
    assert pt.last_obs(lo) < ref and pt.last_obs(hi) >= ref, (arg, pt.last_obs(lo), pt.last_obs(hi))
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) // 2
        if pt.last_obs(mid) >= ref:
            hi = mid
        else:
            lo = mid
    print(arg, "first vintage", hi, "lag", (hi - end).days, flush=True)
    out.write(f"{arg}\t{hi}\t{(hi - end).days}\n")
