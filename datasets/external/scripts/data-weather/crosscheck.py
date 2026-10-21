"""Cross-source sanity checks: correlations at lags -1/0/+1 between related series.

Usage: python3 -I crosscheck.py
A dating error of one day would show the peak correlation at lag +-1 instead of 0.
"""
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, SERIES_DIR  # noqa: E402


def load(sid):
    out = {}
    with open(os.path.join(SERIES_DIR, sid + ".csv")) as f:
        next(f)
        for ln in f:
            d, v = ln.strip().split(",")
            out[dt.date.fromisoformat(d)] = float(v)
    return out


def load_om(point, var):
    from parse_openmeteo import load as load_point
    days, _ = load_point(point)
    if days is None:
        return None
    return {d: v[var] for d, v in days.items() if v.get(var) is not None}


def deseason(s):
    """subtract day-of-year mean (crude, full-sample) so seasonality does not dominate the correlation"""
    by = {}
    for d, v in s.items():
        by.setdefault(d.timetuple().tm_yday, []).append(v)
    mean = {k: sum(v) / len(v) for k, v in by.items()}
    return {d: v - mean[d.timetuple().tm_yday] for d, v in s.items()}


def lagcorr(a, b, lag, start=dt.date(2000, 1, 1)):
    xs, ys = [], []
    for d, v in a.items():
        if d < start:
            continue
        d2 = d + dt.timedelta(lag)
        if d2 in b:
            xs.append(v); ys.append(b[d2])
    return np.corrcoef(xs, ys)[0, 1], len(xs)


def report(label, a, b):
    a, b = deseason(a), deseason(b)
    cs = {lag: lagcorr(a, b, lag) for lag in (-1, 0, 1)}
    best = max(cs, key=lambda k: cs[k][0])
    print(f"{label:58} " + "  ".join(f"lag{l:+d}={c:.3f}" for l, (c, n) in cs.items()) + f"  n={cs[0][1]}  best={best:+d}")


def main():
    pairs = [("wx_nyc_hdd", "cpc_hdd_midatlantic"), ("wx_chi_hdd", "cpc_hdd_encentral"),
             ("wx_hou_cdd", "cpc_cdd_wscentral"), ("wx_dfw_cdd", "cpc_cdd_wscentral"),
             ("wx_sea_hdd", "cpc_hdd_pacific"), ("wx_atl_hdd", "cpc_hdd_satlantic"),
             ("wx_den_hdd", "cpc_hdd_mountain"), ("wx_msp_hdd", "cpc_hdd_wncentral"),
             ("cpc_hdd_conus", "cpc_hdd_gas_conus"), ("wx_nyc_tmax", "wx_phl_tmax"),
             ("cpc_tele_ao", "cpc_tele_nao"), ("usdm_ia_d1plus", "usdm_il_d1plus")]
    for a, b in pairs:
        report(f"{a} vs {b}", load(a), load(b))
    for point, var, sid in (("nyc", "temperature_2m_max", "wx_nyc_tmax"), ("nyc", "temperature_2m_min", "wx_nyc_tmin"),
                            ("nyc", "precipitation_sum", "wx_nyc_prcp"), ("il_corn", "temperature_2m_max", "wx_pia_tmax"),
                            ("ks_wheat", "temperature_2m_max", "wx_ict_tmax"), ("tx_lubbock", "temperature_2m_max", "wx_lbb_tmax"),
                            ("ca_valley", "temperature_2m_max", "wx_fat_tmax")):
        om = load_om(point, var)
        if om is None:
            print(f"(no raw Open-Meteo for {point})")
            continue
        report(f"ERA5 {point}.{var} vs GHCN {sid}", om, load(sid))
    # Tokyo: ERA5 vs the raw (gappy, GSOD-derived) GHCN Tokyo record, which is not a series
    import gzip
    raw = {}
    with gzip.open(os.path.join(RAW, "ghcnd", "by_station", "JA000047662.csv.gz"), "rt") as f:
        for ln in f:
            p = ln.split(",")
            if p[2] == "TMAX" and p[5] == "" and p[1] >= "20000101":
                raw[dt.date(int(p[1][:4]), int(p[1][4:6]), int(p[1][6:8]))] = int(p[3]) / 10
    om = load_om("tokyo", "temperature_2m_max")
    if om:
        report("ERA5 tokyo.temperature_2m_max vs raw GHCN JA000047662 TMAX", om, raw)
    # ERA5 cloud vs GHCN NYC precipitation occurrence (sanity: positive)
    om = load_om("nyc", "cloud_cover_mean")
    pr = load("wx_nyc_prcp")
    report("ERA5 nyc.cloud_cover_mean vs GHCN wx_nyc_prcp", om, {d: float(v > 0) for d, v in pr.items()})


if __name__ == "__main__":
    main()
