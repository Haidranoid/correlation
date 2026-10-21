"""Normalize CPC population-weighted (and gas-weighted) daily degree days.

Usage: python3 -I parse_cpc_dd.py
Reads raw/cpc/degree_days/<year>/{Population.Heating,Population.Cooling,UtilityGas.Heating}.txt
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, drop_entries, qa_line  # noqa: E402

D = os.path.join(RAW, "cpc", "degree_days")
URL = "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{year}/{file}"
DIVS = {"1": ("newengland", "New England"), "2": ("midatlantic", "Middle Atlantic"),
        "3": ("encentral", "East North Central"), "4": ("wncentral", "West North Central"),
        "5": ("satlantic", "South Atlantic"), "6": ("escentral", "East South Central"),
        "7": ("wscentral", "West South Central"), "8": ("mountain", "Mountain"),
        "9": ("pacific", "Pacific"), "CONUS": ("conus", "CONUS")}


def parse_file(path, year):
    out = {}
    with open(path, encoding="latin-1") as f:
        lines = [l.rstrip("\r\n") for l in f]
    hdr = [l for l in lines if l.startswith("Region|")]
    assert len(hdr) == 1, path
    dates = [dt.datetime.strptime(x, "%Y%m%d").date() for x in hdr[0].split("|")[1:] if x]
    assert all(d.year == year for d in dates), path
    assert all((b - a).days == 1 for a, b in zip(dates, dates[1:])), path
    assert dates[0] == dt.date(year, 1, 1), path
    for l in lines:
        if "|" not in l or l.startswith("Region|"):
            continue
        parts = l.split("|")
        reg, vals = parts[0], [p for p in parts[1:]]
        # trailing empty field possible
        if len(vals) == len(dates) + 1 and vals[-1] == "":
            vals = vals[:-1]
        assert len(vals) == len(dates), (path, reg, len(vals), len(dates))
        series = {}
        for d, v in zip(dates, vals):
            v = v.strip()
            if v == "" or v.startswith("-9"):
                continue
            series[d] = float(v)
        out[reg] = series
    return out


def main():
    drop_entries("cpc_hdd_")
    drop_entries("cpc_cdd_")
    years = sorted(int(y) for y in os.listdir(D) if y.isdigit())
    data = {}  # (file, region) -> {date: value}
    for y in years:
        for fn in ("Population.Heating.txt", "Population.Cooling.txt", "UtilityGas.Heating.txt"):
            for reg, s in parse_file(os.path.join(D, str(y), fn), y).items():
                data.setdefault((fn, reg), {}).update(s)
    common_notes = ("CPC daily degree days (base 65F, whole degF-days) from daily max/min at first-order stations, "
                    "aggregated to Census divisions/CONUS; weights are fixed (not time-varying); the file header changes "
                    "from 'Regions::' to 'CPC::Regions::' in 2014, when CPC moved to 2010 Census/ACS weights, so a small "
                    "step at 2014-01-01 is possible (none visible in yearly comparisons with GHCN station HDD or between gas and population weights); values published about 04:00 ET two days after the date (observed "
                    "Last-Modified 2026-10-18 08:07 UTC with data through 2026-10-16); archived values are final, real-time "
                    "values could differ slightly")
    for (fn, reg), s in sorted(data.items()):
        kind = "hdd" if "Heating" in fn else "cdd"
        gas = fn.startswith("UtilityGas")
        if gas and reg != "CONUS":
            continue  # keep only the national gas-weighted HDD (divisions would be near-duplicates)
        slug, label = DIVS[reg]
        sid = f"cpc_{kind}_{'gas_' if gas else ''}{slug}"
        weight = "utility-gas-heated-home" if gas else "population"
        name = (f"CPC {weight}-weighted daily {'heating' if kind == 'hdd' else 'cooling'} degree days, "
                f"{label}{' Census division' if reg != 'CONUS' else ''}")
        files = "UtilityGas.Heating.txt" if gas else fn
        m = dict(name=name, category="weather", subcategory="degree_days",
                 region=f"US {label}" if reg != "CONUS" else "US CONUS",
                 source="NOAA Climate Prediction Center",
                 url="https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/", frequency="daily",
                 units="degF-days",
                 transform="anomaly", known_after_days=2,
                 notes=f"files <year>/{files} for 1981-2026 under that directory; " + common_notes
                 + ("; weights = homes heated with utility gas (the natural gas demand HDD)" if gas else ""))
        rows = sorted(s.items())
        qa = write_series(sid, rows, m, 0.0, 120.0, nd=0)
        print(qa_line(qa))


if __name__ == "__main__":
    main()
