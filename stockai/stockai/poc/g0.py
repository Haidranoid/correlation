"""G0, the pipeline check (PLAN.md M0): the judge grades a scores file exactly as it grades its own reversal family.

    python -m stockai.poc.g0 scores PY.npy DUMP.npy        Python scores vs the judge's --dump-scores, bit for bit
    python -m stockai.poc.g0 json A.json B.json             two judge --json outputs, every field but "options"
    python -m stockai.poc.g0 tables REF.txt JUDGE.txt       a reference `c2 census holdout --walk-forward` table:
                                                            each line an exact prefix of the judge's line
Each exits 1 on any difference, and `--record FILE` merges the result into a JSON record of the checks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


def total_order_key(x: np.ndarray) -> np.ndarray:
    """int64 keys that order float64 values as Rust's `f64::total_cmp` (the judge's ranking order): -0.0 sits one
    step below +0.0, and neighbouring floats differ by 1."""
    b = np.ascontiguousarray(x, dtype=np.float64).view(np.int64)
    return np.where(b < 0, b ^ np.int64(0x7FFFFFFFFFFFFFFF), b)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Steps between a and b in the total order, as uint64 (no overflow for any pair)."""
    ka, kb = total_order_key(a), total_order_key(b)
    ua, ub = ka.view(np.uint64), kb.view(np.uint64)
    return np.where(ka >= kb, ua - ub, ub - ua)


def compare_scores(py_path: str, dump_path: str) -> dict:
    """Where the judge computed a score (finite in the dump), the Python score must have the same bits, or be
    within 1 ulp (reported, with any +0.0 / -0.0 mismatch, which the judge's total_cmp ranking can see)."""
    py = np.load(py_path, mmap_mode="r")
    du = np.load(dump_path, mmap_mode="r")
    if py.shape != du.shape:
        raise SystemExit(f"shapes differ: {py.shape} vs {du.shape}")
    m = np.isfinite(du)
    a = np.asarray(py, dtype=np.float64)[m]
    b = np.asarray(du, dtype=np.float64)[m]
    same = a.view(np.int64) == b.view(np.int64)
    finite = np.isfinite(a)
    ulp = ulp_distance(a[finite], b[finite])
    cols = np.flatnonzero(m.any(axis=0))
    res = {
        "cells_compared": int(m.sum()),
        "rebalance_days": int(cols.size),
        "bitwise_equal": int(same.sum()),
        "max_ulp": int(ulp.max()) if ulp.size else 0,
        "cells_1_ulp": int((ulp == 1).sum()),
        "signed_zero_mismatches": int(((a == 0) & (b == 0) & ~same).sum()),
        "py_dtype": str(py.dtype),
        "dump_dtype": str(du.dtype),
        "py_nan_where_judge_scored": int((~finite).sum()),
    }
    res["passed"] = bool(res["cells_compared"] > 0 and res["max_ulp"] <= 1 and res["py_nan_where_judge_scored"] == 0)
    return res


def _walk(a, b, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'A' if k in a else 'B'}")
            else:
                _walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: lengths {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", out)
    else:
        nan = isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)
        if not (a == b or nan) or type(a) is not type(b):
            out.append(f"{path}: {a!r} vs {b!r}")


def _leaves(x) -> int:
    if isinstance(x, dict):
        return sum(_leaves(v) for v in x.values())
    if isinstance(x, list):
        return sum(_leaves(v) for v in x)
    return 1


def compare_json(a_path: str, b_path: str) -> dict:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    for d in (a, b):
        d.pop("options", None)
    diffs: list[str] = []
    _walk(a, b, "$", diffs)
    return {"fields_compared": _leaves(a), "years": len(a.get("years", [])), "differences": len(diffs),
            "first_differences": diffs[:20], "passed": not diffs}


def compare_tables(ref_path: str, judge_path: str) -> dict:
    ref = Path(ref_path).read_text().splitlines()
    new = Path(judge_path).read_text().splitlines()
    timing = re.compile(r"\s+\[\d+s\]$")
    bad = []
    for i, r in enumerate(ref):
        n = new[i] if i < len(new) else ""
        if timing.search(r):
            r, n = timing.sub("", r), timing.sub("", n)
        if not n.startswith(r):
            bad.append(f"line {i + 1}: ref {ref[i]!r} | judge {new[i] if i < len(new) else ''!r}")
    rows = sum(1 for r in ref if re.match(r"\d{4}-\d\d-\d\d\.\.", r))
    return {"reference_lines": len(ref), "year_rows": rows, "differences": len(bad), "first_differences": bad[:10],
            "passed": not bad and rows > 0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.poc.g0", description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=("scores", "json", "tables"))
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--record", default=None, help="JSON file to merge this check's result into")
    ap.add_argument("--name", default=None, help="key in the record (default: the check kind)")
    a = ap.parse_args(argv)
    res = {"scores": compare_scores, "json": compare_json, "tables": compare_tables}[a.what](a.a, a.b)
    res.update(a=a.a, b=a.b)
    if a.what == "scores":
        msg = (f"{res['bitwise_equal']:,} of {res['cells_compared']:,} cells bitwise equal "
               f"({res['rebalance_days']} rebalance days), max difference {res['max_ulp']} ulp")
    elif a.what == "json":
        msg = f"{res['fields_compared']} fields compared ({res['years']} years): {res['differences']} differ"
    else:
        msg = f"{res['reference_lines']} reference lines ({res['year_rows']} year rows): {res['differences']} differ"
    print(f"G0 {a.what}: {'PASS' if res['passed'] else 'FAIL'}: {msg}")
    for d in res.get("first_differences", []):
        print("  " + d)
    if a.record:
        p = Path(a.record)
        rec = json.loads(p.read_text()) if p.exists() else {}
        rec[a.name or a.what] = res
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=2) + "\n")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
