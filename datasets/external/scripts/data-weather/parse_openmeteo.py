"""Normalize Open-Meteo ERA5 daily point data into om_<point>_<var> series.

Usage: python3 -I parse_openmeteo.py
Uses whichever raw/openmeteo/<point>_<start>_<end>.json halves exist (a point needs both halves).
"""
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, write_series, drop_entries, qa_line  # noqa: E402
from dl_openmeteo import POINTS, HALVES  # noqa: E402

D = os.path.join(RAW, "openmeteo")

# variable -> (suffix, label, subcategory, units, scale, lo, hi, decimals, transform)
VARS = {
    "sunshine_duration": ("sunshine", "daily sunshine duration", "sunshine", "hours", 1 / 3600, 0, 24, 2, "anomaly"),
    "cloud_cover_mean": ("cloud", "daily mean total cloud cover", "cloud_cover", "percent", 1, 0, 100, 0, "anomaly"),
    "shortwave_radiation_sum": ("swrad", "daily shortwave (solar) radiation sum", "solar_radiation", "MJ/m2", 1, 0, 45, 2,
                                "anomaly"),
    "et0_fao_evapotranspiration": ("et0", "daily FAO-56 reference evapotranspiration", "evapotranspiration", "mm", 1, 0,
                                   20, 2, "anomaly"),
    "soil_moisture_7_to_28cm_mean": ("soilm", "daily mean soil moisture 7-28 cm", "soil_moisture", "m3/m3", 1, 0, 0.8, 3,
                                     "anomaly"),
    "temperature_2m_max": ("tmax", "daily maximum 2 m temperature", "temperature", "degC", 1, -50, 55, 1, "anomaly"),
    "temperature_2m_min": ("tmin", "daily minimum 2 m temperature", "temperature", "degC", 1, -50, 55, 1, "anomaly"),
    "precipitation_sum": ("prcp", "daily precipitation", "precipitation", "mm", 1, 0, 600, 1, "anomaly_log1p"),
}
SELECT = {
    "nyc": ("New York City", "US-NY New York City", ["sunshine_duration", "cloud_cover_mean", "shortwave_radiation_sum"],
            "the 'Wall Street weather' sunshine/cloud hypothesis (Saunders 1993, Hirshleifer & Shumway 2003); GHCN has no sunshine/cloud for Central Park"),
    "il_corn": ("Central Illinois Corn Belt (Bloomington-Normal)", "US-IL McLean County",
                ["shortwave_radiation_sum", "et0_fao_evapotranspiration", "soil_moisture_7_to_28cm_mean"],
                "corn/soy growing conditions; temperature/precipitation come from GHCN Peoria/Des Moines"),
    "ks_wheat": ("Central Kansas wheat belt (Great Bend)", "US-KS Barton County",
                 ["et0_fao_evapotranspiration", "soil_moisture_7_to_28cm_mean", "shortwave_radiation_sum"],
                 "hard red winter wheat conditions; temperature/precipitation come from GHCN Wichita"),
    "ca_valley": ("California Central Valley (west of Fresno)", "US-CA San Joaquin Valley",
                  ["et0_fao_evapotranspiration", "shortwave_radiation_sum"],
                  "irrigated crops (almonds, tomatoes, etc.): water demand; soil moisture skipped because ERA5 does not model irrigation"),
    "tx_lubbock": ("West Texas (Lubbock)", "US-TX South Plains",
                   ["et0_fao_evapotranspiration", "soil_moisture_7_to_28cm_mean", "shortwave_radiation_sum"],
                   "cotton and West Texas solar generation"),
    "br_coffee": ("Brazil coffee belt (Sul de Minas, Varginha)", "BR-MG Sul de Minas",
                  ["temperature_2m_min", "temperature_2m_max", "precipitation_sum", "soil_moisture_7_to_28cm_mean"],
                  "arabica coffee: frost (low Tmin, e.g. July 2021) and drought; Southern Hemisphere seasons"),
    "br_soy": ("Brazil soy belt (Sorriso, Mato Grosso)", "BR-MT Mato Grosso",
               ["precipitation_sum", "temperature_2m_max", "soil_moisture_7_to_28cm_mean"],
               "soybeans/safrinha corn: wet-season onset and dry spells; Southern Hemisphere seasons"),
    "ar_pampas": ("Argentina Pampas (Pergamino)", "AR-B Buenos Aires province",
                  ["precipitation_sum", "temperature_2m_max", "soil_moisture_7_to_28cm_mean"],
                  "soybeans/corn/wheat: drought (e.g. 2022-23); Southern Hemisphere seasons"),
    "tokyo": ("Tokyo", "JP Tokyo", ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
              "foreign financial centre; replaces the gappy GHCN Tokyo record (GSOD-derived, ends 2025-04)"),
}


def load(point):
    """Merge every saved chunk for a point; require full 2000-01-01..2025-12-31 coverage, no duplicate days."""
    import re
    days, meta = {}, None
    files = sorted(fn for fn in os.listdir(D)
                   if re.fullmatch(rf"{re.escape(point)}_\d{{4}}-\d\d-\d\d_\d{{4}}-\d\d-\d\d\.json", fn))
    for fn in files:
        js = json.load(open(os.path.join(D, fn)))
        meta = meta or js
        dl = js["daily"]
        for i, t in enumerate(dl["time"]):
            d = dt.date.fromisoformat(t)
            assert d not in days, (point, d)
            days[d] = {k: dl[k][i] for k in dl if k != "time"}
    first, last = dt.date.fromisoformat(HALVES[0][0]), dt.date.fromisoformat(HALVES[-1][1])
    if not days or len(days) != (last - first).days + 1 or min(days) != first or max(days) != last:
        return None, None
    return days, meta


def main():
    drop_entries("om_")
    for point, (label, region, vars_, why) in SELECT.items():
        days, meta = load(point)
        if days is None:
            print(f"{point}: raw data missing, skipped")
            continue
        lat, lon = POINTS[point]
        for var in vars_:
            suffix, vlabel, subcat, units, scale, lo, hi, nd, transform = VARS[var]
            rows, nulls = [], 0
            for d in sorted(days):
                v = days[d].get(var)
                if v is None:
                    nulls += 1
                    continue
                rows.append((d, v * scale))
            notes = (f"Open-Meteo historical weather API, ECMWF ERA5 reanalysis only (models=era5; 0.25 deg grid cell at "
                     f"({meta['latitude']},{meta['longitude']}), model elevation {meta['elevation']} m, requested "
                     f"({lat},{lon})); daily aggregates over local days (timezone {meta['timezone']}); why: {why}; "
                     f"a model reanalysis, not an observation, produced retrospectively (ERA5T appears ~5 days after real "
                     f"time; 2 days buffer added); values through 2025-12-31 are final ERA5 (fetched 2026-10-18); "
                     f"{nulls} null days left out; CC BY 4.0 (Open-Meteo / Copernicus ECMWF)")
            if var == "sunshine_duration":
                notes += "; sunshine = Open-Meteo estimate from direct normal irradiance > 120 W/m2 (WMO threshold), not a recorder"
            m = dict(name=f"{label} {vlabel} (ERA5)", category="weather", subcategory=subcat, region=region,
                     source="Open-Meteo archive (ECMWF ERA5)", url=("https://archive-api.open-meteo.com/v1/archive?"
                     f"latitude={lat}&longitude={lon}&start_date=2000-01-01&end_date=2025-12-31&daily={var}&models=era5&timezone=auto"),
                     frequency="daily",
                     units=units, transform=transform, known_after_days=7, notes=notes)
            qa = write_series(f"om_{point}_{suffix}", rows, m, lo, hi, nd=nd)
            print(qa_line(qa))


if __name__ == "__main__":
    main()
