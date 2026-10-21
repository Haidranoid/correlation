"""Normalize NOAA PSL real-time OLR MJO index (ROMI, CPC blended OLR).

Usage: python3 -I parse_psl_mjo.py
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, drop_entries, qa_line  # noqa: E402

F = os.path.join(RAW, "psl_mjo", "romi.cpcolr.1x.txt")
URL = "https://psl.noaa.gov/mjo/mjoindex/romi.cpcolr.1x.txt"
NOTE = ("Real-time OLR MJO Index (Kiladis et al. 2014) from NOAA PSL using CPC blended OLR: OLR anomalies minus the "
        "previous 40-day mean (no future data), projected on the OMI EOFs (1979-2012); history computed with the same "
        "real-time method but published retroactively (PSL began ROMI in the 2010s); observed Last-Modified "
        "2026-10-18 15:22 UTC with data through 2026-10-14 (4-day lag), +2 days buffer; already a standardized "
        "anomaly -> level; BoM RMM (the classic real-time MJO index) stops 2024-02-24 so ROMI is used instead")


def main():
    drop_entries("psl_mjo_")
    pc1, pc2, amp = [], [], []
    seen = set()
    with open(F, encoding="latin-1") as f:
        for line in f:
            p = line.split()
            if len(p) != 7:
                continue
            d = dt.date(int(p[0]), int(p[1]), int(p[2]))
            assert d not in seen, d
            seen.add(d)
            a, b, c = float(p[4]), float(p[5]), float(p[6])
            if max(abs(a), abs(b), abs(c)) > 900:
                continue
            pc1.append((d, a)); pc2.append((d, b)); amp.append((d, c))
    for sid, rows, label, lo, hi in (("psl_mjo_romi_pc1", pc1, "PC1", -6, 6), ("psl_mjo_romi_pc2", pc2, "PC2", -6, 6),
                                     ("psl_mjo_romi_amp", amp, "amplitude sqrt(PC1^2+PC2^2)", 0, 8)):
        m = dict(name=f"MJO real-time OLR index (ROMI) {label}", category="climate_index", subcategory="mjo",
                 region="Tropics (Indian Ocean / western Pacific convection)", source="NOAA Physical Sciences Laboratory",
                 url=URL, frequency="daily", units="standardized index", transform="level", known_after_days=6,
                 notes=NOTE)
        qa = write_series(sid, rows, m, lo, hi, nd=4)
        print(qa_line(qa))


if __name__ == "__main__":
    main()
