"""Compact QC table for one or more parts: python3 -I qcsum.py etf stooq_world ..."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmlib  # noqa: E402

names = sys.argv[1:] or sorted(f[:-11] for f in os.listdir(dmlib.PARTS) if f.endswith(".summ.jsonl"))
for name in names:
    print(f"#### {name}")
    with open(os.path.join(dmlib.PARTS, name + ".summ.jsonl")) as f:
        for line in f:
            s = json.loads(line)
            flag = []
            if s.get("gaps"):
                flag.append(f"gaps={s['gaps']}(max {s['max_gap']}d)")
            if s.get("jumps"):
                flag.append(f"jumps={s['jumps']}")
            if s.get("zero_change_frac", 0) > 0.05:
                flag.append(f"zero={s['zero_change_frac']:.2f}")
            if s.get("longest_flat", 0) >= 5:
                flag.append(f"flat={s['longest_flat']}")
            if s.get("sentinels"):
                flag.append(f"sentinel={s['sentinels']}")
            print(f"{s['id']:<28} n={s['n']:>6} {s['first']}..{s['last']} min={s['min']:<12.6g} max={s['max']:<12.6g} {' '.join(flag)}")
