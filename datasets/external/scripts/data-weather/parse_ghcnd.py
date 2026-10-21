"""Normalize GHCN-Daily by_station files into series/wx_<key>_<var>.csv.

Usage: python3 -I parse_ghcnd.py
Reads raw/ghcnd/by_station/<ID>.csv.gz (data only, never executed).
"""
import collections
import datetime as dt
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RAW, SCR, write_series, drop_entries, qa_line  # noqa: E402
from stations import STATIONS  # noqa: E402

START = "19800101"
BYST = os.path.join(RAW, "ghcnd", "by_station")
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{}.csv.gz"
DD_ZERO_MAX = 0.90  # skip a degree-day series if more than 90% of its days are zero

# element -> (divisor, hard lower, hard upper, units, decimals)
ELEM = {
    "TMAX": (10.0, -50.0, 55.0, "degC", 1),
    "TMIN": (10.0, -50.0, 55.0, "degC", 1),
    "PRCP": (10.0, 0.0, 1000.0, "mm", 1),
    "SNOW": (1.0, 0.0, 1500.0, "mm", 0),
    "AWND": (10.0, 0.0, 40.0, "m/s", 1),
}
UNIT_NOTE = {
    "TMAX": "GHCN tenths of degC divided by 10 -> degC",
    "TMIN": "GHCN tenths of degC divided by 10 -> degC",
    "PRCP": "GHCN tenths of mm divided by 10 -> mm; trace (M=T) = 0",
    "SNOW": "GHCN SNOW is already whole mm (not tenths); trace (M=T) = 0",
    "AWND": "GHCN tenths of m/s divided by 10 -> m/s (average daily wind speed)",
}
NAMES = {"TMAX": "daily maximum temperature", "TMIN": "daily minimum temperature",
         "PRCP": "daily precipitation", "SNOW": "daily snowfall", "AWND": "average daily wind speed"}
SUBCAT = {"TMAX": "temperature", "TMIN": "temperature", "PRCP": "precipitation",
          "SNOW": "snowfall", "AWND": "wind"}
TRANSFORM = {"TMAX": "anomaly", "TMIN": "anomaly", "PRCP": "anomaly_log1p",
             "SNOW": "anomaly_log1p", "AWND": "anomaly"}


def _gaps(rows):
    out = []
    for (a, _), (b, _) in zip(rows, rows[1:]):
        if (b - a).days > 1:
            out.append((a.isoformat(), b.isoformat(), (b - a).days - 1))
    return sorted(out, key=lambda g: -g[2])


def load_meta():
    meta = {}
    with open(os.path.join(RAW, "ghcnd", "ghcnd-stations.txt"), encoding="latin-1") as f:
        for line in f:
            meta[line[0:11]] = (float(line[12:20]), float(line[21:30]), line[38:40].strip(),
                                line[41:71].strip())
    return meta


def parse_station(sid):
    data = collections.defaultdict(dict)   # elem -> {date: value}
    stats = collections.Counter()
    obstimes = collections.defaultdict(collections.Counter)
    sources = collections.defaultdict(collections.Counter)
    dup = 0
    with gzip.open(os.path.join(BYST, sid + ".csv.gz"), "rt", encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 8:
                stats["short_line"] += 1
                continue
            _, ymd, el, val, mfl, qfl, sfl, obst = p[:8]
            if el not in ELEM or ymd < START:
                continue
            stats[f"{el}_raw"] += 1
            obstimes[el][obst or "blank"] += 1
            v = int(val)
            if v == -9999:
                stats[f"{el}_missing9999"] += 1
                continue
            if qfl.strip():
                stats[f"{el}_qflag_{qfl}"] += 1
                stats[f"{el}_qflag_total"] += 1
                continue
            if mfl == "P":
                stats[f"{el}_mflag_P"] += 1
                continue
            d = dt.date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
            if d in data[el]:
                dup += 1
                continue
            sources[el][sfl or "blank"] += 1
            data[el][d] = v / ELEM[el][0]
    stats["duplicates"] = dup
    return data, stats, obstimes, sources


def main():
    meta = load_meta()
    drop_entries("wx_")
    report = {}
    skipped = []
    for key, sid, disp, role, use_snow, use_wind, use_dd in STATIONS:
        lat, lon, st, gname = meta[sid]
        if sid == "UKE00107650":
            lon = -abs(lon)  # station file has +0.449; Heathrow is west of Greenwich
        country = sid[:2]
        region = (f"US-{st} {disp} ({lat:.3f},{lon:.3f})" if country == "US"
                  else f"{ {'UK': 'GB', 'AS': 'AU', 'GM': 'DE'}[country] } {disp} ({lat:.3f},{lon:.3f})".strip())
        data, stats, obstimes, sources = parse_station(sid)
        rep = {"stats": dict(stats), "obstimes": {k: dict(v) for k, v in obstimes.items()},
               "sources": {k: dict(v) for k, v in sources.items()}, "series": {}}
        # OBS-TIME sanity: midnight-to-midnight days only (2400 or blank)
        for el, c in obstimes.items():
            odd = {k: n for k, n in c.items() if k not in ("2400", "blank")}
            if odd:
                rep.setdefault("odd_obstime", {})[el] = odd
        elems = ["TMAX", "TMIN", "PRCP"] + (["SNOW"] if use_snow else []) + (["AWND"] if use_wind else [])
        for el in elems:
            div, lo, hi, units, nd = ELEM[el]
            rows = sorted(data[el].items())
            if not rows:
                skipped.append((f"wx_{key}_{el.lower()}", "no data"))
                continue
            extra = []
            # Unflagged wind spikes: drop values above 2x the station's 99.9th percentile.
            if el == "AWND":
                vs = sorted(v for _, v in rows)
                p999 = vs[int(0.999 * (len(vs) - 1))]
                spikes = [(d, v) for d, v in rows if v > 2 * p999]
                if spikes:
                    rows = [(d, v) for d, v in rows if v <= 2 * p999]
                    extra.append(f"dropped unflagged spikes above 2x p99.9 ({2*p999:.1f} m/s): "
                                 + ";".join(f"{d.isoformat()}={v}" for d, v in spikes))
                    rep.setdefault("spikes", {})[el] = [(d.isoformat(), v) for d, v in spikes]
            # Short isolated leading fragment (<2 years) before a gap of more than a year: drop it.
            for i in range(1, len(rows)):
                gap = (rows[i][0] - rows[i - 1][0]).days
                if gap > 365:
                    if (rows[i - 1][0] - rows[0][0]).days < 730:
                        extra.append(f"dropped isolated leading fragment {rows[0][0]}..{rows[i-1][0]} "
                                     f"before a {gap}-day gap")
                        rows = rows[i:]
                    break
            nflag = stats.get(f"{el}_qflag_total", 0)
            srcs = ",".join(f"{k}:{n}" for k, n in sorted(sources[el].items(), key=lambda x: -x[1])[:4])
            notes = (f"GHCN-D station {sid} ({gname}); {UNIT_NOTE[el]}; values with any QFLAG dropped (n={nflag} since 1980)"
                     f"; MFLAG P (missing presumed zero) dropped; series starts 1980 (or station start) to give the anomaly a baseline"
                     f"; raw station data are not homogenized (instrument/siting changes possible); latest ~2 months come from preliminary"
                     f" real-time sources and may be revised; "
                     + ("day = local standard time midnight-midnight" if country == "US" else
                        "day follows the national convention (09 UTC, about 06 UTC or 9 am local climatological day, not midnight),"
                        " so a value dated D may partly cover D-1 or D+1; it is complete by about 09 UTC on D+1, before the next US close")
                     + f"; top sources {srcs}")
            if extra:
                notes += "; " + "; ".join(extra)
            big = [g for g in _gaps(rows) if g[2] > 30]
            if big:
                notes += "; gaps >30 days: " + ", ".join(f"{a}..{b} ({n}d)" for a, b, n in big[:4])
            if key == "nyc" and el == "AWND":
                notes += "; Central Park anemometer is tree-sheltered, so speeds run low"
            if sid.startswith(("UK", "GM")):
                notes += "; European ECA&D-sourced record (source E)"
            if sid.startswith("AS"):
                notes += "; Australian BoM-sourced record (source a); Southern Hemisphere seasons"
            m = dict(name=f"{disp} {NAMES[el]}", category="weather", subcategory=SUBCAT[el],
                     region=region, source="NOAA NCEI GHCN-Daily", url=URL.format(sid), frequency="daily",
                     units=units, transform=TRANSFORM[el], known_after_days=1, notes=notes)
            qa = write_series(f"wx_{key}_{el.lower()}", rows, m, lo, hi, nd=nd)
            rep["series"][qa["id"]] = qa
            print(qa_line(qa))
        if use_dd:
            tx, tn = data["TMAX"], data["TMIN"]
            hdd, cdd, inverted = [], [], 0
            for d in sorted(set(tx) & set(tn)):
                if tx[d] < tn[d]:
                    inverted += 1
                    continue
                tavg_f = ((tx[d] + tn[d]) / 2.0) * 9.0 / 5.0 + 32.0
                hdd.append((d, max(0.0, 65.0 - tavg_f)))
                cdd.append((d, max(0.0, tavg_f - 65.0)))
            rep["dd_inverted_days"] = inverted
            for kind, rows in (("hdd", hdd), ("cdd", cdd)):
                zf = sum(1 for _, v in rows if v == 0) / len(rows)
                if zf > DD_ZERO_MAX:
                    skipped.append((f"wx_{key}_{kind}", f"degenerate: {100*zf:.1f}% of days are zero"))
                    continue
                label = "heating" if kind == "hdd" else "cooling"
                notes = (f"Derived from GHCN-D {sid} TMAX/TMIN (QFLAG-clean, both present): {kind.upper()} = "
                         f"max(0, {'65 - Tavg' if kind == 'hdd' else 'Tavg - 65'}) with Tavg = (TMAX+TMIN)/2 converted to degF, base 65F"
                         f"; not rounded to whole degF as NWS does (differences <=0.5); {inverted} days with TMAX<TMIN skipped"
                         f"; {100*zf:.0f}% of days are zero; series starts 1980 (or station start)")
                big = [g for g in _gaps(rows) if g[2] > 30]
                if big:
                    notes += "; gaps >30 days: " + ", ".join(f"{a}..{b} ({n}d)" for a, b, n in big[:4])
                m = dict(name=f"{disp} daily {label} degree days (base 65F)", category="weather",
                         subcategory="degree_days", region=region, source="NOAA NCEI GHCN-Daily (derived)",
                         url=URL.format(sid), frequency="daily", units="degF-days", transform="anomaly",
                         known_after_days=1, notes=notes)
                qa = write_series(f"wx_{key}_{kind}", rows, m, 0.0, 120.0, nd=1)
                rep["series"][qa["id"]] = qa
                print(qa_line(qa))
        report[key] = rep
    with open(os.path.join(SCR, "qa_ghcnd_report.json"), "w") as f:
        json.dump({"stations": report, "skipped": skipped}, f, indent=1, default=str)
    print("\nSKIPPED:")
    for s in skipped:
        print("  ", s)
    print("\nODD OBSTIMES / FLAG DROPS / DUPLICATES:")
    for key, rep in report.items():
        st = rep["stats"]
        flags = {k: v for k, v in st.items() if "qflag_total" in k or "mflag" in k}
        print(f"  {key}: odd_obstime={rep.get('odd_obstime')} dups={st.get('duplicates')} "
              f"inverted={rep.get('dd_inverted_days')} drops={flags}")


if __name__ == "__main__":
    main()
