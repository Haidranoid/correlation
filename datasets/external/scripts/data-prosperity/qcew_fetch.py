"""Fetch only the industry-10 ("Total, all industries") member of each year's QCEW
<year>_qtrly_by_industry.zip with HTTP Range requests (the full zips are 70-480 MB per year;
the member is a few MB). The member is decompressed in memory, CRC-checked against the zip's
central directory and saved unmodified under raw/bls_qcew/, with a manifest of byte ranges.

Usage: python3 -I qcew_fetch.py 1990 2026
"""
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plib  # noqa: E402

OUT = os.path.join(plib.RAW, "bls_qcew")
os.makedirs(OUT, exist_ok=True)


def get_range(url, start, end=None):
    rng = f"bytes={start}-" if end is None else f"bytes={start}-{end}"
    if start < 0:
        rng = f"bytes={start}"
    req = urllib.request.Request(url, headers={"User-Agent": plib.UA, "Range": rng})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status != 206:
                    raise RuntimeError(f"expected 206, got {r.status}")
                data = r.read()
            time.sleep(1.0)
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"{url} {rng}: {last}")


def central_directory(url):
    tail = get_range(url, -65536)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no EOCD")
    (_, _, _, _, n_entries, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    if cd_off == 0xFFFFFFFF or n_entries == 0xFFFF:
        j = tail.rfind(b"PK\x06\x07")
        (_, _, z64_off, _) = struct.unpack("<IIQI", tail[j:j + 20])
        z = get_range(url, z64_off, z64_off + 55)
        (_, _, _, _, _, _, n_entries, _, cd_size, cd_off) = struct.unpack("<IQHHIIQQQQ", z[:56])
    cd = get_range(url, cd_off, cd_off + cd_size - 1)
    entries, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (_, _, _, _, method, _, _, crc, csize, usize, nlen, xlen, clen, _, _, _, lho) = struct.unpack(
            "<IHHHHHHIIIHHHHHII", cd[p:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("cp437")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        # zip64 extra field for large sizes/offsets
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                vals = list(struct.unpack("<" + "Q" * (hlen // 8), extra[q + 4:q + 4 + hlen]))
                if usize == 0xFFFFFFFF:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = vals.pop(0)
                if lho == 0xFFFFFFFF:
                    lho = vals.pop(0)
            q += 4 + hlen
        entries.append(dict(name=name, method=method, crc=crc, csize=csize, usize=usize, offset=lho))
        p += 46 + nlen + xlen + clen
    return entries


def fetch_member(url, e):
    hdr = get_range(url, e["offset"], e["offset"] + 29)
    assert hdr[:4] == b"PK\x03\x04", "bad local header"
    nlen, xlen = struct.unpack("<HH", hdr[26:30])
    start = e["offset"] + 30 + nlen + xlen
    comp = get_range(url, start, start + e["csize"] - 1)
    if e["method"] == 8:
        data = zlib.decompressobj(-15).decompress(comp)
    elif e["method"] == 0:
        data = comp
    else:
        raise RuntimeError(f"unsupported compression {e['method']}")
    assert len(data) == e["usize"], (len(data), e["usize"])
    assert (zlib.crc32(data) & 0xFFFFFFFF) == e["crc"], "CRC mismatch"
    return data, (start, start + e["csize"] - 1)


def main():
    y0, y1 = int(sys.argv[1]), int(sys.argv[2])
    manifest_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    for y in range(y0, y1 + 1):
        if str(y) in manifest and os.path.exists(os.path.join(OUT, manifest[str(y)]["saved_as"])):
            continue
        url = f"https://data.bls.gov/cew/data/files/{y}/csv/{y}_qtrly_by_industry.zip"
        try:
            entries = central_directory(url)
        except Exception as e:  # noqa: BLE001
            print(y, "no zip:", e)
            continue
        cands = [e for e in entries if " 10 " in e["name"] and "Total, all industries" in e["name"]]
        if len(cands) != 1:
            print(y, "member not unique:", [e["name"] for e in cands][:5], len(entries))
            continue
        e = cands[0]
        data, rng = fetch_member(url, e)
        saved = f"{y}_qtrly_industry10_total_all_industries.csv"
        with open(os.path.join(OUT, saved), "wb") as f:
            f.write(data)
        manifest[str(y)] = dict(url=url, member=e["name"], crc32=f"{e['crc']:08x}", compressed_bytes=e["csize"],
                                uncompressed_bytes=e["usize"], byte_range=rng, saved_as=saved,
                                sha256=plib.sha256(os.path.join(OUT, saved)), zip_entries=len(entries))
        json.dump(manifest, open(manifest_path, "w"), indent=1)
        print(y, e["name"], e["csize"], "->", e["usize"], "bytes", flush=True)


if __name__ == "__main__":
    main()
