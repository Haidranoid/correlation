"""Announcement days for experiments/-0.6-announcement-beta.md: scheduled FOMC statements, CPI releases and
Employment Situation releases, 2006 .. 2026-10-19 (calendar dates; common.calendar() maps them to trading days).

    python3 -I -B calendar.py      -> calendar.tsv (date, fomc, cpi, jobs flags) and calendar.txt (checks)

Inputs (read-only):
  - research/phantom_portfolios/fomc_cpi_days.tsv: its FOMC and CPI days (built by pp_calendar.py);
  - phantom portfolios' cached Fed pages (BIG/raw/fomc_hist{2006..2020}.htm, fomc_cal.htm): the meeting calendars
    that separate scheduled statements from unscheduled moves;
  - this experiment's own downloads in RAW (fetched by hand with curl through the Wayback Machine, browser-like
    User-Agent, no email address): BLS's Employment Situation archive page (snapshot 2026-10-08), the current CPI
    and Employment Situation releases (snapshots 2026-10-16 and 2026-10-17), and BLS's release schedules
    (snapshots 2026-10-13 and 2026-10-12).
"""
import datetime as dt
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PP = HERE.parent / 'phantom_portfolios'
PP_RAW = Path.home() / '.local/share/correlation2/experiments/phantom_portfolios/raw'
RAW = Path.home() / '.local/share/correlation2/experiments/announcement_beta/raw'
START, END = '2006-01-01', '2026-10-19'
MONTHS = {m: i + 1 for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                          'September', 'October', 'November', 'December'])}
MONTHS |= {m[:3]: i for m, i in list(MONTHS.items())} | {'Sept': 9}
WD = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def read(p):
    return Path(p).read_text(encoding='utf8', errors='replace')


def no_comments(t):
    return re.sub(r'<!--.*?-->', '', t, flags=re.S)


def iso8(d):  # 'YYYYMMDD' -> 'YYYY-MM-DD'
    return f'{d[:4]}-{d[4:6]}-{d[6:]}'


def heading_last_day(h, year):
    """'January 30-31 Meeting - 2007' -> 2007-01-31; 'April/May 30-1 Meeting - 2013' -> 2013-05-01;
    'March 18 Meeting - 2008' -> 2008-03-18."""
    m = re.match(r'([A-Za-z]+) (\d+)-([A-Za-z]+) (\d+)', h)          # 'July 31-August 1'
    if m:
        return f'{year}-{MONTHS[m.group(3)]:02d}-{int(m.group(4)):02d}'
    m = re.match(r'([A-Za-z]+)(?:/([A-Za-z]+))?\s+(\d+)(?:-(\d+))?', h)
    mon1, mon2, d1, d2 = m.groups()
    mon = MONTHS[mon2 or mon1]
    day = int(d2 or d1)
    return f'{year}-{mon:02d}-{day:02d}'


def fomc_scheduled():
    """Scheduled FOMC meetings 2006..2026 from the Fed's meeting calendars. Returns list of dicts with the heading,
    the statement date (from the meeting's own press-release link, else the meeting's last day) and the link dates;
    and the unscheduled / cancelled sections for the record."""
    sched, other = [], []
    for y in range(2006, 2021):
        t = read(PP_RAW / f'fomc_hist{y}.htm')
        parts = re.split(r'(<h[1-6][^>]*>.*?</h[1-6]>)', t, flags=re.S)
        for i in range(1, len(parts), 2):
            h = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', parts[i])).strip()
            if not re.search(r'(Meeting|Conference Call|unscheduled)\s*-\s*\d{4}$', h):
                continue
            body = parts[i + 1] if i + 1 < len(parts) else ''
            links = sorted({a or b for a, b in re.findall(r'monetary/?(\d{8})[a-z]?\d?\.htm|press/monetary/\d{4}/(\d{8})', body)})
            links = [iso8(d) for d in links if d.startswith(str(y))]
            rec = {'heading': h, 'year': y, 'links': links}
            if 'Meeting' in h and 'unscheduled' not in h and 'cancelled' not in h:
                last = heading_last_day(h, y)
                rec['last_day'] = last
                rec['date'] = last
                sched.append(rec)
            else:
                other.append(rec)
    # 2021+: the current calendars page, one block per year, one row per meeting.
    t = read(PP_RAW / 'fomc_cal.htm')
    blocks = re.split(r'(\d{4}) FOMC Meetings', t)
    for i in range(1, len(blocks), 2):
        y = int(blocks[i])
        if y < 2021:
            continue
        rows = re.split(r'<div class="(?:fomc-meeting--shaded )?row fomc-meeting"', blocks[i + 1])[1:]
        for r in rows:
            mon = re.search(r'fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>', r)
            day = re.search(r'fomc-meeting__date[^>]*>([^<]+)<', r)
            if not mon or not day:
                continue
            h = f'{mon.group(1).strip()} {day.group(1).strip()} - {y}'
            stm = sorted(set(re.findall(r'pressreleases/monetary(\d{8})a\.htm', r)))
            rec = {'heading': h, 'year': y, 'links': [iso8(d) for d in stm]}
            if re.search(r'unscheduled|cancel|notation vote', r, re.I) or 'Statement:' not in r and stm == []:
                other.append(rec)
                continue
            last = heading_last_day(re.sub(r'[*]', '', h), y)
            rec['last_day'] = last
            rec['date'] = last
            sched.append(rec)
    return sched, other


def bls_dates(text, prefix):
    """Release dates (ISO) from a BLS archive page: links prefix_MMDDYYYY.(htm|pdf|txt), comments stripped."""
    t = no_comments(text)
    out = {f'{x[4:]}-{x[:2]}-{x[2:4]}' for x in re.findall(prefix + r'_(\d{8})\.(?:htm|pdf|txt)', t)}
    return out


def embargo_date(text):
    m = re.search(r'embargoed until\s+.*?(\d{1,2}:\d{2} [ap]\.m\. \(ET\))\s+([A-Za-z]+),\s+([A-Za-z]+) (\d{1,2}), (\d{4})', text, re.S)
    if not m:
        return None
    _, wd, mon, d, y = m.groups()
    return f'{y}-{MONTHS[mon]:02d}-{int(d):02d}', m.group(1), wd


def main():
    lines = []
    say = lambda *a: lines.append(' '.join(str(x) for x in a))  # noqa: E731

    # --- the tsv
    tsv = [l.rstrip('\n').split('\t') for l in open(PP / 'fomc_cpi_days.tsv')][1:]
    tsv_fomc = {d for d, k in tsv if k == 'fomc' and START <= d <= END}
    tsv_cpi = {d for d, k in tsv if k == 'cpi' and START <= d <= END}

    # --- FOMC: scheduled statements from the meeting calendars
    sched, other = fomc_scheduled()
    sched = [s for s in sched if START <= s['date'] <= END]
    say('FOMC scheduled meetings', START, '..', END, ':', len(sched))
    bad = [s for s in sched if s['links'] and s['last_day'] not in s['links']]
    say('scheduled meetings whose press-release link date differs from the last meeting day:', len(bad))
    for s in bad:
        say('   ', s['heading'], s['links'])
    nolink = [s for s in sched if not s['links']]
    say('scheduled meetings with no statement link on the calendar page (date = last meeting day):', len(nolink))
    for s in nolink:
        say('   ', s['heading'], '->', s['date'])
    sched_days = {s['date'] for s in sched}
    dropped = sorted(tsv_fomc - sched_days)
    added = sorted(sched_days - tsv_fomc)
    say('tsv FOMC days', len(tsv_fomc), '; not a scheduled statement (dropped):', len(dropped))
    sec_of = {}
    for o in other:
        for d in o['links']:
            sec_of[d] = o['heading']
    for d in dropped:
        say('   ', d, dt.date.fromisoformat(d).strftime('%a'), '|', sec_of.get(d, '(not under any meeting heading: other press release)'))
    say('scheduled statement days missing from the tsv (added):', len(added))
    for d in added:
        s = [x for x in sched if x['date'] == d][0]
        say('   ', d, '|', s['heading'], '| links', s['links'])
    say('unscheduled / cancelled / conference-call sections on the calendars:')
    for o in other:
        if START[:4] <= str(o['year']) <= END[:4]:
            say('   ', o['heading'], o['links'])

    # --- CPI: the tsv, plus the release after its archive snapshot (2026-10-02)
    cpi = set(tsv_cpi)
    e = embargo_date(read(RAW / 'cpi_nr0_20261016.htm'))
    say('CPI: tsv days', len(tsv_cpi), '; current release in the 2026-10-16 snapshot:', e)
    if e and START <= e[0] <= END:
        if e[0] not in cpi:
            say('    added', e[0])
        cpi.add(e[0])

    # --- Employment Situation: BLS archive via Wayback (snapshot 2026-10-08), plus the current-release check
    jobs = {d for d in bls_dates(read(RAW / 'bls_empsit.htm'), 'empsit') if START <= d <= END}
    e = embargo_date(read(RAW / 'empsit_nr0_20261017.htm'))
    say('Jobs: archive days', len(jobs), '; current release in the 2026-10-17 snapshot:', e)
    if e and START <= e[0] <= END:
        if e[0] not in jobs:
            say('    added', e[0])
        jobs.add(e[0])

    # --- schedule pages: the next releases after the snapshots (to show nothing else falls before END)
    for name, f in (('CPI', 'sched_cpi_20261013.htm'), ('Jobs', 'sched_empsit_20261012.htm')):
        s = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', no_comments(read(RAW / f))))
        m = re.search(r'Reference Month Release Date Release Time (.{0,120})', s)
        say(f'{name} schedule page ({f}): next rows:', m.group(1) if m else None)

    fomc = set(sched_days)
    allA = sorted(fomc | cpi | jobs)
    with open(HERE / 'calendar.tsv', 'w') as f:
        f.write('date\tfomc\tcpi\tjobs\n')
        for d in allA:
            f.write(f'{d}\t{int(d in fomc)}\t{int(d in cpi)}\t{int(d in jobs)}\n')

    say()
    say('Per year (calendar dates, before dropping non-trading days): FOMC / CPI / jobs / union')
    for y in range(int(START[:4]), int(END[:4]) + 1):
        ys = str(y)
        n = [sum(d.startswith(ys) for d in S) for S in (fomc, cpi, jobs, set(allA))]
        say(f'  {y}: {n[0]:2d} / {n[1]:2d} / {n[2]:2d} / {n[3]:2d}')
    say()
    say('Weekday distribution (calendar dates):')
    for name, S in (('FOMC scheduled', fomc), ('CPI', cpi), ('Jobs', jobs)):
        c = Counter(dt.date.fromisoformat(d).weekday() for d in S)
        say(f'  {name:15s}', ' '.join(f'{WD[k]} {c.get(k, 0)}' for k in range(7)), f'(n = {len(S)})')
    for name, S in (('FOMC scheduled', fomc), ('Jobs', jobs)):
        for a, b in (('2006-01-01', '2010-12-31'), ('2011-01-01', '2026-10-19')):
            c = Counter(dt.date.fromisoformat(d).weekday() for d in S if a <= d <= b)
            say(f'  {name:15s} {a[:4]}-{b[:4]}', ' '.join(f'{WD[k]} {c.get(k, 0)}' for k in range(7)))
    odd = sorted(d for d in jobs if dt.date.fromisoformat(d).weekday() != 4)
    say('  jobs days not on a Friday:', ', '.join(f'{d} {dt.date.fromisoformat(d).strftime("%a")}' for d in odd))
    odd = sorted(d for d in fomc if dt.date.fromisoformat(d).weekday() != 2)
    say('  FOMC days not on a Wednesday:', ', '.join(f'{d} {dt.date.fromisoformat(d).strftime("%a")}' for d in odd))
    both = [d for d in allA if (d in fomc) + (d in cpi) + (d in jobs) > 1]
    say('  days with two releases:', ', '.join(f'{d} ({"+".join(k for k, S in (("fomc", fomc), ("cpi", cpi), ("jobs", jobs)) if d in S)})' for d in both))
    say('  last of each through', END, ':', 'FOMC', max(fomc), 'CPI', max(cpi), 'jobs', max(jobs))
    (HERE / 'calendar.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
