"""Minimal reader for Stata .dta format 117/118/119 (numeric and fixed-width string columns).
Parses bytes only; never executes anything from the file."""
import struct

TYPES = {65526: ("d", 8), 65527: ("f", 4), 65528: ("l", 4), 65529: ("h", 2), 65530: ("b", 1)}
MISSING_ABOVE = {"d": 8.988465674311579e307, "f": 1.7014117e38, "l": 2147483620, "h": 32740, "b": 100}

def read_dta(path):
    raw = open(path, "rb").read()
    assert raw.startswith(b"<stata_dta><header><release>"), "not a dta 117+ file"
    rel = int(raw[28:31])
    assert rel in (117, 118, 119), rel
    bo = raw.find(b"<byteorder>") + len(b"<byteorder>")
    end = "<" if raw[bo:bo + 3] == b"LSF" else ">"
    p = raw.find(b"<K>") + 3
    K = struct.unpack(end + ("I" if rel == 119 else "H"), raw[p:p + (4 if rel == 119 else 2)])[0]
    p = raw.find(b"<N>") + 3
    N = struct.unpack(end + ("Q" if rel >= 118 else "I"), raw[p:p + (8 if rel >= 118 else 4)])[0]
    m = raw.find(b"<map>") + 5
    offs = struct.unpack(end + "14Q", raw[m:m + 112])
    vt = offs[2] + len(b"<variable_types>")
    types = struct.unpack(end + f"{K}H", raw[vt:vt + 2 * K])
    vn = offs[3] + len(b"<varnames>")
    namelen = 129 if rel >= 118 else 33
    names = [raw[vn + i * namelen: vn + (i + 1) * namelen].split(b"\0")[0].decode("utf-8")
             for i in range(K)]
    fmt_parts, widths = [], []
    for t in types:
        if t in TYPES:
            fmt_parts.append(TYPES[t][0]); widths.append(TYPES[t][1])
        elif 1 <= t <= 2045:
            fmt_parts.append(f"{t}s"); widths.append(t)
        else:
            raise ValueError(f"unsupported Stata type {t} (strL?)")
    rowfmt = end + "".join(fmt_parts)
    rowlen = struct.calcsize(rowfmt)
    assert rowlen == sum(widths)
    d0 = offs[9] + len(b"<data>")
    assert raw[offs[9]:d0] == b"<data>"
    rows = []
    for i in range(N):
        vals = list(struct.unpack(rowfmt, raw[d0 + i * rowlen: d0 + (i + 1) * rowlen]))
        for j, (v, f) in enumerate(zip(vals, fmt_parts)):
            if isinstance(v, bytes):
                vals[j] = v.split(b"\0")[0].decode("utf-8", "replace")
            elif v > MISSING_ABOVE[f]:
                vals[j] = None
        rows.append(dict(zip(names, vals)))
    assert raw[d0 + N * rowlen: d0 + N * rowlen + 7] == b"</data>", "row layout mismatch"
    return names, types, rows
