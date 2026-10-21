"""Fetch the per-country members WID_data_<ISO2>.csv (+ README, WID_countries.csv) of the WID bulk zip
https://wid.world/bulk_download/wid_all_data.zip with HTTP Range requests, CRC-check them, and store
them gzip-compressed (lossless) under raw/wid/. Usage: python3 -I wid_fetch.py
"""
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402
import qcew_fetch as zr  # noqa: E402  (zip range helpers)

URL = "https://wid.world/bulk_download/wid_all_data.zip"
OUT = os.path.join(plib.RAW, "wid")
PANEL = ["AR", "AU", "BR", "CA", "CN", "DE", "FR", "GB", "ID", "IN", "IT", "JP", "KR", "MX", "RU", "SA", "TR", "US",
         "ZA", "ES", "NL", "CH", "SE", "PL", "BE", "NO", "AT", "IE", "IL", "SG", "TH", "MY", "PH", "VN", "EG", "NG",
         "PK", "BD", "CL", "CO"]


def main():
    os.makedirs(OUT, exist_ok=True)
    entries = {e["name"]: e for e in zr.central_directory(URL)}
    want = ["README.md", "WID_countries.csv"] + [f"WID_data_{c}.csv" for c in PANEL] + [f"WID_metadata_{c}.csv" for c in ["US"]]
    mpath = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for n in want:
        dest = os.path.join(OUT, n + ".gz")
        if n in manifest and os.path.exists(dest):
            continue
        e = entries[n]
        data, rng = zr.fetch_member(URL, e)
        with gzip.open(dest, "wb", compresslevel=6) as f:
            f.write(data)
        manifest[n] = dict(url=URL, member=n, crc32=f"{e['crc']:08x}", compressed_bytes=e["csize"],
                           uncompressed_bytes=e["usize"], byte_range=rng, saved_as=n + ".gz (gzip of the exact member bytes)")
        json.dump(manifest, open(mpath, "w"), indent=1)
        print(n, e["csize"], "->", e["usize"], flush=True)


if __name__ == "__main__":
    main()
