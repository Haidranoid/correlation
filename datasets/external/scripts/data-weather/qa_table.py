"""Print a markdown QA table (first/last rows, range, gaps) for every catalogued series.

Usage: python3 -I qa_table.py > qa_table.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENTRIES, QA_DIR  # noqa: E402


def main():
    with open(ENTRIES) as f:
        ids = sorted(json.loads(l)["id"] for l in f if l.strip())
    print("| id | first rows | last rows | min | max | zero % | missing days (daily) | longest gap |")
    print("|---|---|---|---|---|---|---|---|")
    for sid in ids:
        q = json.load(open(os.path.join(QA_DIR, sid + ".json")))
        head = "; ".join(f"{d} {v}" for d, v in q["head"][:2])
        tail = "; ".join(f"{d} {v}" for d, v in q["tail"][-2:])
        g = q["longest_gaps"][0] if q["longest_gaps"] else None
        gtxt = f"{g[0]}..{g[1]} ({g[2]}d step)" if g else "none"
        miss = q["missing_days"] if q["missing_days"] is not None else "n/a"
        print(f"| {sid} | {head} | {tail} | {q['min']:.4g} | {q['max']:.4g} | {100*q['zero_frac']:.1f} | {miss} | {gtxt} |")


if __name__ == "__main__":
    main()
