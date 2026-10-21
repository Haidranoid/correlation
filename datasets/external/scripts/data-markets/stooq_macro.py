"""Stooq d_macro_txt.zip (June 29 2025 snapshot): selected US and big-economy headline macro series.

Stooq dates monthly values on the last business day of the reference month (weekly: the week-ending day),
not on the release date, so known_after_days is set from each release schedule (conservative bounds).
"""
import datetime as dt
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dmlib  # noqa: E402

USB = "/run/media/bhoult/My Book/backups/popos-cosmic-2025/bhoult/programming/correlation"
ZIP = os.path.join(USB, "d_macro_txt.zip")
RAWDIR = os.path.join(dmlib.RAW, "stooq")
URL = "https://stooq.com/db/h/ (bulk daily macro archive d_macro_txt.zip, local copy dated 2025-06-29)"
M = ("monthly; date = last business day of the reference month (Stooq convention); values as of the 2025-06-29 "
     "snapshot (older months revised, latest months first prints)")
W = "weekly; date = end of the reference week (Stooq convention)"

# stem (country dir/file) -> (id, name, category, subcategory, region, frequency, units, transform, kad, note)
SEL = [
    ("us/ismnus", "stooq_ism_mfg_us", "ISM manufacturing PMI", "macro", "survey", "US", "monthly", "index (50 = no change)", "diff", 6,
     "released 10:00 New York on the first business day of the next month (Jan: 2nd-3rd), so <=5 days after the last business day; not on FRED (ISM licensing); starts 2010"),
    ("us/isnfus", "stooq_ism_services_us", "ISM services PMI", "macro", "survey", "US", "monthly", "index (50 = no change)", "diff", 8,
     "released on the third business day of the next month, so 8; starts 2010"),
    ("us/pmchus", "stooq_chicago_pmi_us", "Chicago PMI (MNI Chicago Business Barometer)", "macro", "survey", "US", "monthly", "index (50 = no change)", "diff", 1,
     "released 09:45 Chicago on the last business day of the reference month (= the date), so 1 as a cushion; starts 2001"),
    ("us/fdphus", "stooq_philly_fed_us", "Philadelphia Fed manufacturing index (general activity)", "macro", "survey", "US", "monthly", "diffusion index (% up minus % down)", "diff", 0,
     "released the third Thursday of the reference month, before the date; starts 1968"),
    ("us/esmius", "stooq_empire_state_us", "NY Fed Empire State manufacturing index", "macro", "survey", "US", "monthly", "diffusion index (% up minus % down)", "diff", 0,
     "released ~15th of the reference month, before the date; starts 2010"),
    ("us/fdrhus", "stooq_richmond_fed_us", "Richmond Fed manufacturing index", "macro", "survey", "US", "monthly", "diffusion index", "diff", 0,
     "released the fourth Tuesday of the reference month, on or before the date; starts 2010"),
    ("us/cncius", "stooq_conf_board_confidence_us", "Conference Board consumer confidence", "macro", "sentiment", "US", "monthly", "index 1985=100", "diff", 0,
     "released 10:00 New York on the last Tuesday of the reference month, on or before the date; starts 2010"),
    ("us/nahbus", "stooq_nahb_us", "NAHB/Wells Fargo housing market index", "macro", "housing", "US", "monthly", "index (50 = neutral)", "diff", 0,
     "released mid-month (~15th-18th) of the reference month; starts 1985"),
    ("us/ldiius", "stooq_lei_us", "Conference Board leading economic index, % change m/m", "macro", "activity", "US", "monthly", "percent change m/m", "level", 30,
     "already a monthly change, hence level; released ~3rd-4th week of the next month, so 30; starts 2010"),
    ("us/adpeus", "stooq_adp_us", "ADP national employment change", "macro", "labor", "US", "monthly", "thousands of jobs (monthly change)", "level", 10,
     "already a monthly change, hence level; released the Wednesday before the jobs report (<=8 days after month end), so 10; ADP rebuilt its methodology and history in 2022, so vintages are mixed; starts 2001"),
    ("us/hoesus", "stooq_existing_home_sales_us", "NAR existing home sales (SAAR)", "macro", "housing", "US", "monthly", "millions of units (SAAR)", "logdiff", 30,
     "released ~3rd-4th week of the next month, so 30; FRED carries only the last 13 months (NAR licensing); starts 1999"),
    ("us/honsus", "stooq_new_home_sales_us", "New single-family home sales (SAAR)", "macro", "housing", "US", "monthly", "thousands of units (SAAR)", "logdiff", 78,
     "usual release 17-26 days after month end; bound 78 covers the shutdown catch-up releases (Sept-2013 data released 2013-12-04; Nov/Dec-2018 data released only in early 2019, Dec-2018 on 2019-03-04) with margin; heavily revised; starts 1963"),
    ("cn/cpiycn", "stooq_cpi_yoy_cn", "China CPI, % y/y", "macro", "inflation", "CN", "monthly", "percent y/y", "diff", 20,
     "NBS release ~9th-16th of the next month (later around Chinese New Year), so 20; starts 2010"),
    ("cn/inpycn", "stooq_indprod_yoy_cn", "China industrial production, % y/y", "macro", "activity", "CN", "monthly", "percent y/y", "diff", 20,
     "NBS release ~15th-17th of the next month, so 20; January is not published separately (Jan-Feb combined), hence yearly gaps; starts 2010"),
    ("de/ifbcde", "stooq_ifo_de", "Ifo business climate (Germany)", "macro", "survey", "DE", "monthly", "index 2015=100", "diff", 0,
     "released ~22nd-26th of the reference month, before the date; starts 2010"),
    ("de/zwesde", "stooq_zew_sentiment_de", "ZEW economic sentiment (Germany)", "macro", "survey", "DE", "monthly", "balance (% optimists minus % pessimists)", "diff", 0,
     "released mid-month (Tuesday ~13th-20th) of the reference month, before the date; starts 2010"),
    ("eu/cpiyeu", "stooq_hicp_yoy_ea", "Euro area HICP inflation, % y/y", "macro", "inflation", "EA", "monthly", "percent y/y", "diff", 20,
     "flash at the end of the month / first days of the next, final ~17th-19th of the next month; 20 covers the final; starts 2010"),
    ("eu/pmmneu", "stooq_pmi_mfg_ea", "Euro area manufacturing PMI (HCOB/S&P Global)", "macro", "survey", "EA", "monthly", "index (50 = no change)", "diff", 5,
     "flash ~3rd-4th week of the reference month, final on the first business day of the next month; history holds finals, so 5; starts 2010"),
    ("jp/inpyjp", "stooq_indprod_yoy_jp", "Japan industrial production (preliminary), % y/y", "macro", "activity", "JP", "monthly", "percent y/y", "diff", 33,
     "METI preliminary release ~28th-31st of the next month, so 33; starts 2010"),
    ("uk/cpiyuk", "stooq_cpi_yoy_gb", "UK CPI inflation, % y/y", "macro", "inflation", "GB", "monthly", "percent y/y", "diff", 23,
     "ONS release mid-month (Wednesday ~14th-20th) of the next month, so 23; starts 2010"),
]


def main():
    os.makedirs(RAWDIR, exist_ok=True)
    shutil.copy2(ZIP, os.path.join(RAWDIR, "d_macro_txt.zip"))  # small (0.9 MB): keep the whole archive
    z = zipfile.ZipFile(ZIP)
    part = dmlib.Part("stooq_macro")
    for stem, sid, name, cat, sub, region, freq, units, tr, kad, note in SEL:
        inner = f"data/daily/macro/{stem}.m.txt"
        data = z.read(inner).decode()
        lines = data.splitlines()
        assert lines[0].startswith("<TICKER>,<PER>,<DATE>"), lines[0]
        rows = []
        wd = {}
        moved = []
        for ln in lines[1:]:
            p = ln.split(",")
            d = dmlib.ymd(p[2])
            if freq == "monthly":
                e = dt.date.fromisoformat(dmlib.month_end(d))
                while e.weekday() >= 5:
                    e -= dt.timedelta(days=1)
                if e.isoformat() != d:
                    assert e.isoformat() > d, (sid, d, e)
                    moved.append(f"{d}->{e.isoformat()}")
                    d = e.isoformat()
            rows.append((d, float(p[7])))
            k = dt.date.fromisoformat(d).weekday()
            wd[k] = wd.get(k, 0) + 1
        conv = M if freq == "monthly" else W
        if moved:
            conv += f"; {len(moved)} date(s) moved later to the month's last weekday: " + ", ".join(moved[:3])
        part.add(dict(id=sid, name=name, category=cat, subcategory=sub, region=region, source="Stooq bulk d_macro_txt",
                      url=URL + f" member {inner}", frequency=freq, units=units, transform=tr, known_after_days=kad,
                      notes=conv + "; " + note), rows, jump=0.3 if tr == "logdiff" else None, min_rows=40)
        print(f"   weekday counts (Mon=0) for {sid}: {dict(sorted(wd.items()))}")
    part.save()
    print("catalog rows:", dmlib.merge_catalog())


if __name__ == "__main__":
    main()
