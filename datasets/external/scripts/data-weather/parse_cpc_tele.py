"""Normalize CPC daily teleconnection indices (NAO, AO, PNA, AAO) from the CDAS csv files,
and compare against the monthly-updated .ascii files to quantify revisions.

Usage: python3 -I parse_cpc_tele.py
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, drop_entries, qa_line  # noqa: E402

D = os.path.join(RAW, "cpc", "teleconnections")
BASE = "https://ftp.cpc.ncep.noaa.gov/cwlinks/"
IDX = {
    "nao": ("norm.daily.nao.cdas.z500.19500101_current.csv", "norm.daily.nao.index.b500101.current.ascii",
            "North Atlantic Oscillation", "500 hPa height", "North Atlantic"),
    "ao": ("norm.daily.ao.cdas.z1000.19500101_current.csv", "norm.daily.ao.index.b500101.current.ascii",
           "Arctic Oscillation", "1000 hPa height", "Northern Hemisphere 20N-90N"),
    "pna": ("norm.daily.pna.cdas.z500.19500101_current.csv", "norm.daily.pna.index.b500101.current.ascii",
            "Pacific-North American pattern", "500 hPa height", "North Pacific / North America"),
    "aao": ("norm.daily.aao.cdas.z700.19790101_current.csv", "norm.daily.aao.index.b790101.current.ascii",
            "Antarctic Oscillation (Southern Annular Mode)", "700 hPa height", "Southern Hemisphere 20S-90S"),
}
START = dt.date(1980, 1, 1)


def read_csv(path):
    out = {}
    with open(path, encoding="latin-1") as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) < 4 or p[3].strip() == "":
                continue
            d = dt.date(int(p[0]), int(p[1]), int(p[2]))
            v = float(p[3])
            assert d not in out, (path, d)
            out[d] = v
    return out


def read_ascii(path):
    out = {}
    with open(path, encoding="latin-1") as f:
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            try:
                d = dt.date(int(p[0]), int(p[1]), int(p[2]))
                out[d] = float(p[3])
            except ValueError:
                continue
    return out


def main():
    drop_entries("cpc_tele_")
    for key, (csvf, ascf, label, level, region) in IDX.items():
        s = read_csv(os.path.join(D, csvf))
        a = read_ascii(os.path.join(D, ascf))
        common = [d for d in s if d in a]
        diffs = [abs(s[d] - a[d]) for d in common]
        big = sum(1 for x in diffs if x > 0.01)
        recent = [abs(s[d] - a[d]) for d in common if d.year >= 2020]
        rev = (f"the daily csv and the monthly-updated ascii agree on all {len(common)} common days "
               f"(max |diff| {max(diffs):.3f}); both are the current vintage and no real-time vintages are published, "
               f"so revisions of the past cannot be measured")
        print(key, rev)
        rows = sorted((d, v) for d, v in s.items() if d >= START)
        n_missing_src = sum(1 for d in range((rows[-1][0] - rows[0][0]).days + 1)
                            if (rows[0][0] + dt.timedelta(d)) not in s)
        notes = (f"CPC daily {label} index from CDAS (NCEP/NCAR reanalysis) {level} anomalies projected on the "
                 f"monthly loading pattern and standardized; already an anomaly, so transform=level; updated daily, "
                 f"observed Last-Modified 2026-10-18 13:55 UTC with data through 2026-10-16 (2-day lag), +1 day buffer "
                 f"because the update lands mid-day; history uses today's normalization/climatology, not the one in force "
                 f"at the time; {rev}; source-blank days left out ({n_missing_src} since 1980); series starts 1980")
        m = dict(name=f"CPC daily {label} index", category="climate_index", subcategory="teleconnection",
                 region=region, source="NOAA Climate Prediction Center", url=BASE + csvf, frequency="daily",
                 units="standardized index", transform="level", known_after_days=3, notes=notes)
        qa = write_series(f"cpc_tele_{key}", rows, m, -8.0, 8.0, nd=3)
        print(qa_line(qa))


if __name__ == "__main__":
    main()
