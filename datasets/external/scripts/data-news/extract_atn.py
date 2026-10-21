"""Stream all-the-news-2-1.csv out of its zip (never extracted) and keep date, publication,
section, title as a TSV for the series builder. Usage: python3 -I extract_atn.py OUT.tsv"""
import csv, io, os, sys, zipfile
src = os.path.expanduser("~/.local/share/correlation2/external/raw/all_the_news_2/all-the-news-2-1.zip")
csv.field_size_limit(sys.maxsize)
n = bad = 0
with zipfile.ZipFile(src) as z, z.open("all-the-news-2-1.csv") as f, open(sys.argv[1], "w", newline="") as out:
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    r = csv.reader(io.TextIOWrapper(f, encoding="utf-8", errors="replace", newline=""))
    hdr = next(r)
    i_date, i_title, i_sec, i_pub = (hdr.index(k) for k in ("date", "title", "section", "publication"))
    for row in r:
        if len(row) != len(hdr):
            bad += 1
            continue
        n += 1
        w.writerow([row[i_date][:10], row[i_pub], row[i_sec].replace("\t", " ").replace("\n", " "),
                    " ".join(row[i_title].split())])
print("rows", n, "malformed", bad)
