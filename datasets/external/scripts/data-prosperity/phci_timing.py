"""Verify a known_after_days bound for the Philadelphia Fed state coincident indexes from ALFRED
vintages (keyless alfredgraph.csv?id=CAPHCI&vintage_date=D returns the series as FRED had it on D).

For each reference month m (2008-11 .. 2025-06), we need: m was in the FRED vintage of m_end + K.
A vintage on date D whose last observation is L(D) proves every month <= L(D) was out by D, so many
months are covered by one query. Writes raw/philfed_coincident_timing/alfred_checks.tsv.
Usage: python3 -I phci_timing.py K
"""
import datetime as dt
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402

OUT = os.path.join(plib.RAW, "philfed_coincident_timing")
CACHE = {}


def last_obs(vdate):
    if vdate in CACHE:
        return CACHE[vdate]
    url = f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CAPHCI&vintage_date={vdate.isoformat()}"
    req = urllib.request.Request(url, headers={"User-Agent": plib.UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                lines = r.read().decode().strip().splitlines()
            break
        except Exception:  # noqa: BLE001
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(url)
    time.sleep(1.5)
    last = dt.date.fromisoformat(lines[-1].split(",")[0])
    CACHE[vdate] = last
    return last


def main():
    K = int(sys.argv[1])
    rows = []
    y, m = 2008, 11
    known_through = dt.date(1900, 1, 1)
    fails = []
    while (y, m) <= (2025, 6):
        ref = dt.date(y, m, 1)
        end = plib.month_end(y, m)
        if ref > known_through:
            d = end + dt.timedelta(days=K)
            last = last_obs(d)
            rows.append((ref, d, last))
            if last >= ref:
                known_through = max(known_through, last)
            else:
                fails.append((ref, d, last))
            # the check for this month is the vintage at m_end+K; a later vintage can't be used
        m += 1
        if m == 13:
            y, m = y + 1, 1
    with open(os.path.join(OUT, f"alfred_checks_K{K}.tsv"), "w") as f:
        f.write("ref_month\tvintage_date\tlast_obs_in_vintage\n")
        for r in rows:
            f.write("\t".join(x.isoformat() for x in r) + "\n")
    print(f"K={K}: {len(rows)} queries; failures {len(fails)}")
    for r in fails:
        print("  not yet out:", *[x.isoformat() for x in r])


if __name__ == "__main__":
    main()
