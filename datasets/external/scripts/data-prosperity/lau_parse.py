"""Parse BLS LAUS county (la.data.64.County, NSA) and state SA (la.data.3.AllStatesS) flat files
into numpy arrays cached under work/. Monthly periods M01-M12 only (M13 = annual average is dropped);
'-' values (footnotes N/X: not available, Oct-2025 shutdown) are NaN.

Output: work/lau_county.npz with
  fips[str], name[str], months[int: y*12+m-1], ur, unemp, emp, lf  (county x month, NaN missing)
and work/lau_state_sa.npz with fips, name, months, ur, lf (SA).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402

RAWD = os.path.join(plib.RAW, "bls_laus")
WORK = os.path.join(plib.HERE, "work")
os.makedirs(WORK, exist_ok=True)
MEAS = {"03": "ur", "04": "unemp", "05": "emp", "06": "lf"}


def series_meta():
    meta = {}
    with open(os.path.join(RAWD, "la.series"), encoding="latin-1") as f:
        hdr = [c.strip() for c in next(f).rstrip("\n").split("\t")]
        for line in f:
            r = dict(zip(hdr, [c.strip() for c in line.rstrip("\n").split("\t")]))
            meta[r["series_id"]] = r
    return meta


def parse(datafile, want_type, want_seasonal, out):
    meta = series_meta()
    vals = {}  # (area_code, measure) -> {month_index: value}
    nbad = 0
    with open(os.path.join(RAWD, datafile), encoding="latin-1") as f:
        next(f)
        for line in f:
            p = line.split("\t")
            sid = p[0].strip()
            per = p[2].strip()
            if per == "M13":
                continue
            m = meta.get(sid)
            if m is None or m["area_type_code"] != want_type or m["seasonal"] != want_seasonal:
                continue
            meas = MEAS.get(m["measure_code"])
            if meas is None:
                continue
            v = p[3].strip()
            try:
                x = float(v)
            except ValueError:
                nbad += 1
                continue
            mi = int(p[1]) * 12 + int(per[1:]) - 1
            vals.setdefault((m["area_code"], meas), {})[mi] = x
    areas = sorted({a for a, _ in vals})
    allm = sorted({mi for d in vals.values() for mi in d})
    m0, m1 = allm[0], allm[-1]
    months = np.arange(m0, m1 + 1)
    arr = {k: np.full((len(areas), len(months)), np.nan) for k in MEAS.values()}
    aidx = {a: i for i, a in enumerate(areas)}
    for (a, meas), d in vals.items():
        row = arr[meas][aidx[a]]
        for mi, x in d.items():
            row[mi - m0] = x
    # names from the unemployment-rate series titles
    names = {}
    for sid, m in meta.items():
        if m["area_type_code"] == want_type and m["measure_code"] == "03" and m["seasonal"] == want_seasonal:
            t = m["series_title"]
            names[m["area_code"]] = t.split(": ", 1)[1].rsplit(" (", 1)[0] if ": " in t else t
    # area_code: 2-letter type prefix + 2-digit state fips + 3-digit county fips + zeros (CN0100100000000)
    fips = [a[2:7] for a in areas]
    np.savez_compressed(os.path.join(WORK, out), fips=np.array(fips), area=np.array(areas),
                        name=np.array([names.get(a, "") for a in areas]), months=months, **arr)
    print(f"{datafile}: {len(areas)} areas, months {m0 // 12}-{m0 % 12 + 1:02d}..{m1 // 12}-{m1 % 12 + 1:02d}, "
          f"non-numeric values skipped {nbad}")


if __name__ == "__main__":
    parse("la.data.64.County", "F", "U", "lau_county.npz")
    parse("la.data.3.AllStatesS", "A", "S", "lau_state_sa.npz")
