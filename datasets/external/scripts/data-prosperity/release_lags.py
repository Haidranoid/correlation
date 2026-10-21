"""Parse BLS news-release archive index pages into (reference period, release date, lag days).

Usage: python3 -I release_lags.py <archive.htm> <prefix> [out.tsv]
The release date comes from the archive file name (<prefix>_MMDDYYYY.htm); the reference
period from the link text ("August 2026", "First Quarter 2007", ...). Lag = release date minus
the last day of the reference period.
"""
import calendar, datetime as dt, re, sys

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
QTRS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}

def period_end(text):
    t = re.sub(r"\s+", " ", text).strip().lower()
    m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s*,?\s*(\d{4})", t)
    if m:
        y, mo = int(m.group(2)), MONTHS[m.group(1)]
        return dt.date(y, mo, calendar.monthrange(y, mo)[1]), f"{y}-{mo:02d}"
    m = re.search(r"(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s*,?\s*(\d{4})", t)
    if m:
        y, q = int(m.group(2)), QTRS[m.group(1)]
        mo = 3 * q
        return dt.date(y, mo, calendar.monthrange(y, mo)[1]), f"{y}Q{q}"
    return None, None

def main():
    path, prefix = sys.argv[1], sys.argv[2]
    html = open(path, encoding="utf-8", errors="replace").read()
    out = []
    for li in re.finditer(r"<li[^>]*>(.*?)</li>", html, re.S | re.I):
        body = li.group(1)
        m = re.search(r'href="[^"]*archives/' + re.escape(prefix) + r'_(\d{8})\.(?:htm|pdf)"', body, re.I)
        if not m:
            continue
        d = m.group(1)
        rel = dt.date(int(d[4:]), int(d[:2]), int(d[2:4]))
        text = re.sub(r"<[^>]+>", " ", body)
        end, label = period_end(text)
        if end is None:
            continue
        out.append((label, end.isoformat(), rel.isoformat(), (rel - end).days, re.sub(r"\s+", " ", text).strip()[:90]))
    seen, rows = set(), []
    for r in out:
        if (r[0], r[2]) in seen:
            continue
        seen.add((r[0], r[2]))
        rows.append(r)
    rows.sort()
    dst = open(sys.argv[3], "w") if len(sys.argv) > 3 else sys.stdout
    dst.write("period\tperiod_end\trelease\tlag_days\tlink_text\n")
    for r in rows:
        dst.write("\t".join(map(str, r)) + "\n")

main()
