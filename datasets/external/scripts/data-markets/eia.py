"""EIA weekly crude-oil stocks (ex-SPR) and Lower-48 natural-gas storage, parsed from EIA history pages saved
unmodified in raw/eia/dm_*.htm(l). Gas storage before 2010 (EIA page starts 2010-01-01) is prepended from the
Stooq macro archive (nginus.m), after checking the two agree on the overlap.
"""
import datetime as dt
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmlib  # noqa: E402

RAW = os.path.join(dmlib.RAW, "eia")
MON = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
STOOQ_MACRO = os.path.join(dmlib.RAW, "stooq", "d_macro_txt.zip")


def parse_leaf(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for m in re.finditer(r"<tr>\s*<td class='B6'>(?:&nbsp;)*\s*(\d{4})-(\w{3})</td>(.*?)</tr>", t, re.S):
        year, mon = int(m.group(1)), MON[m.group(2)]
        cells = re.findall(r"<td class='B5'>(.*?)</td>\s*<td class='B3'>(.*?)</td>", m.group(3), re.S)
        for d, v in cells:
            d = d.replace("&nbsp;", "").strip()
            v = v.replace("&nbsp;", "").replace(",", "").strip()
            if not d or not v:
                continue
            mm, dd = map(int, d.split("/"))
            y = year
            if mm == 12 and mon == 1:  # a January row can hold the last week of December
                y -= 1
            assert mm in (mon, mon - 1 if mon > 1 else 12, mon % 12 + 1), (year, mon, d)
            out.append((dt.date(y, mm, dd).isoformat(), float(v)))
    return out


def stooq_gas():
    z = zipfile.ZipFile(STOOQ_MACRO)
    lines = z.read("data/daily/macro/us/nginus.m.txt").decode().splitlines()[1:]
    return [(dmlib.ymd(p.split(",")[2]), float(p.split(",")[7])) for p in lines]


def main():
    part = dmlib.Part("eia")
    crude = parse_leaf(os.path.join(RAW, "dm_WCESTUS1_leaf.html"))
    part.add(dict(id="eia_crude_stocks_wk", name="US commercial crude oil ending stocks excluding SPR (weekly)",
                  category="energy", subcategory="inventories", region="US", source="EIA Weekly Petroleum Status Report",
                  url="https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=WCESTUS1&f=W", frequency="weekly",
                  units="thousand barrels", transform="logdiff", known_after_days=6,
                  notes="week ending Friday; WPSR released Wednesday 10:30 New York (Thursday in holiday weeks), so 6; "
                        "values are EIA's current history (weekly estimates later benchmarked to monthly data); mild seasonality; "
                        "replaces Stooq clinus.m, which has missing weeks (e.g. 2017-04..07)"),
             crude, jump=0.1)
    gas = parse_leaf(os.path.join(RAW, "dm_nw2_epg0_swo_r48_bcfw.htm"))
    st = dict(stooq_gas())
    ov = [(d, v, st[d]) for d, v in gas if d in st]
    same = sum(1 for _, a, b in ov if abs(a - b) <= 1)
    print(f"gas overlap with Stooq: {len(ov)} dates, |diff|<=1 Bcf on {same}; first overlaps: {ov[:5]}")
    first = gas[0][0]
    pre = [(d, v) for d, v in sorted(st.items()) if d < first]
    part.add(dict(id="eia_ng_storage_wk", name="US Lower-48 working natural gas in underground storage (weekly)",
                  category="energy", subcategory="inventories", region="US", source="EIA Weekly Natural Gas Storage Report (+ Stooq macro before 2010)",
                  url="https://www.eia.gov/dnav/ng/hist/nw2_epg0_swo_r48_bcfw.htm", frequency="weekly",
                  units="billion cubic feet", transform="anomaly", known_after_days=7,
                  notes=f"strongly seasonal, hence anomaly; week ending Friday; released Thursday 10:30 New York (Wednesday or Friday in holiday weeks), so 7; "
                        f"EIA series from {first}; 1993-12-31..{pre[-1][0]} taken from Stooq nginus.m (EIA's earlier estimates; "
                        f"Stooq and EIA agree within 1 Bcf on {same} of {len(ov)} overlapping weeks); EIA changed regions/methodology in 2015"),
             pre + gas, jump=None)
    part.save()
    print("catalog rows:", dmlib.merge_catalog())


if __name__ == "__main__":
    main()
