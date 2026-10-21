"""Shared helpers for the data-weather agent: write normalized series, QA them,
and record catalog entries (one JSON line per series) for the final catalog build."""
import datetime as dt
import json
import math
import os

EXT = os.path.expanduser("~/.local/share/correlation2/external")
SERIES_DIR = os.path.join(EXT, "series")
RAW = os.path.join(EXT, "raw")
SCR = os.path.dirname(os.path.abspath(__file__))
ENTRIES = os.path.join(SCR, "catalog_entries.jsonl")
QA_DIR = os.path.join(SCR, "qa")

CATALOG_COLS = ["id", "name", "category", "subcategory", "region", "source", "url",
                "frequency", "units", "transform", "known_after_days", "first", "last",
                "rows", "notes"]
VALID_CATEGORIES = {"weather", "climate_index", "energy", "commodity", "rates", "credit", "fx",
                    "world_equity", "crypto", "volatility", "macro", "population",
                    "space_weather", "geophysical", "astronomical", "attention", "health", "other"}
VALID_FREQ = {"daily", "weekly", "monthly", "quarterly", "annual"}
VALID_TRANSFORM = {"logdiff", "diff", "anomaly", "anomaly_log1p", "level"}
EXPECTED_STEP = {"daily": 1, "weekly": 7, "monthly": 31, "quarterly": 92, "annual": 366}


def fmt(v, nd):
    """Plain-number formatting with nd decimals, trailing zeros stripped."""
    s = f"{v:.{nd}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "0"
    return s


def write_series(sid, rows, meta, lo, hi, nd=2, min_year_last=2025):
    """rows: iterable of (date_obj, float). Writes series/<sid>.csv, runs QA checks,
    appends a catalog entry. Raises on duplicates, NaN or out-of-range values."""
    import re
    assert re.fullmatch(r"[a-z0-9_]+", sid), sid
    assert meta["category"] in VALID_CATEGORIES, meta["category"]
    assert meta["frequency"] in VALID_FREQ, meta["frequency"]
    assert meta["transform"] in VALID_TRANSFORM, meta["transform"]
    rows = sorted(rows, key=lambda r: r[0])
    dates = [r[0] for r in rows]
    if len(set(dates)) != len(dates):
        raise ValueError(f"{sid}: duplicate dates")
    bad = [(d, v) for d, v in rows if v is None or (isinstance(v, float) and math.isnan(v))]
    if bad:
        raise ValueError(f"{sid}: NaN/None values {bad[:3]}")
    oor = [(d.isoformat(), v) for d, v in rows if not (lo <= v <= hi)]
    if oor:
        raise ValueError(f"{sid}: {len(oor)} values outside [{lo},{hi}]: {oor[:5]}")
    if not rows:
        raise ValueError(f"{sid}: empty")
    path = os.path.join(SERIES_DIR, sid + ".csv")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("date,value\n")
        for d, v in rows:
            f.write(f"{d.isoformat()},{fmt(v, nd)}\n")
    os.replace(tmp, path)

    # QA: gaps relative to the expected step
    freq = meta["frequency"]
    step = EXPECTED_STEP[freq]
    gaps = []
    for a, b in zip(dates, dates[1:]):
        delta = (b - a).days
        if delta > step * (1.5 if freq != "daily" else 1):
            gaps.append((a.isoformat(), b.isoformat(), delta))
    if freq == "daily":
        span = (dates[-1] - dates[0]).days + 1
        missing = span - len(dates)
    else:
        span, missing = None, None
    vals = [v for _, v in rows]
    nz = sum(1 for v in vals if v != 0)
    qa = {
        "id": sid, "rows": len(rows), "first": dates[0].isoformat(), "last": dates[-1].isoformat(),
        "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals),
        "zero_frac": 1 - nz / len(vals), "n_gaps": len(gaps),
        "missing_days": missing, "span_days": span,
        "longest_gaps": sorted(gaps, key=lambda g: -g[2])[:5],
        "head": [(d.isoformat(), fmt(v, nd)) for d, v in rows[:3]],
        "tail": [(d.isoformat(), fmt(v, nd)) for d, v in rows[-3:]],
    }
    if dates[-1].year < min_year_last or (dates[-1] < dt.date(2025, 6, 27) and min_year_last >= 2025):
        qa["warn_short_end"] = True
    os.makedirs(QA_DIR, exist_ok=True)
    with open(os.path.join(QA_DIR, sid + ".json"), "w") as f:
        json.dump(qa, f, indent=1)
    entry = dict(meta)
    entry.update(id=sid, first=qa["first"], last=qa["last"], rows=qa["rows"])
    _upsert_entry(entry)
    return qa


def _upsert_entry(entry):
    entries = []
    if os.path.exists(ENTRIES):
        with open(ENTRIES) as f:
            entries = [json.loads(l) for l in f if l.strip()]
    entries = [e for e in entries if e["id"] != entry["id"]] + [entry]
    with open(ENTRIES + ".tmp", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    os.replace(ENTRIES + ".tmp", ENTRIES)


def drop_entries(prefix):
    """Remove catalog entries (and series files) whose id starts with prefix (for reruns)."""
    if not os.path.exists(ENTRIES):
        return
    with open(ENTRIES) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    keep = [e for e in entries if not e["id"].startswith(prefix)]
    with open(ENTRIES + ".tmp", "w") as f:
        for e in keep:
            f.write(json.dumps(e) + "\n")
    os.replace(ENTRIES + ".tmp", ENTRIES)


def qa_line(qa):
    g = qa["longest_gaps"][0] if qa["longest_gaps"] else None
    miss = f" missing_days={qa['missing_days']}" if qa["missing_days"] is not None else ""
    return (f"{qa['id']:34} {qa['first']}..{qa['last']} rows={qa['rows']:6}{miss} gaps={qa['n_gaps']}"
            f" longest={g} min={qa['min']:.4g} max={qa['max']:.4g} zero%={100*qa['zero_frac']:.1f}"
            + (" SHORT-END" if qa.get("warn_short_end") else ""))
