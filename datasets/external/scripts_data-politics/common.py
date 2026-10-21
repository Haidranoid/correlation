"""Shared helpers for the data-politics normalizers (stdlib + numpy only).

- read_xlsx(path): minimal .xlsx reader (zipfile + xml.etree), returns {sheet: [[cell,...],...]}
- write_series(sid, rows): write series/<sid>.csv (date,value), sorted, no dup dates, no NaN
- catalog(sid, **fields): record one catalog row in work/catalog_parts/<group>.jsonl
- qc(sid, ...): range / duplicate / gap checks and head/tail print
"""
import csv
import datetime as dt
import json
import math
import os
import re
import zipfile
import xml.etree.ElementTree as ET

EXT = os.path.expanduser('~/.local/share/correlation2/external')
RAW = os.path.join(EXT, 'raw')
SERIES = os.path.join(EXT, 'series')
SP = '/tmp/claude-1000/-home-bhoult-programming/0233028d-5555-4d00-84a1-6db185f7ac67/scratchpad/politics'
WORK = os.path.join(SP, 'work')
PARTS = os.path.join(WORK, 'catalog_parts')
os.makedirs(PARTS, exist_ok=True)
os.makedirs(SERIES, exist_ok=True)

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
RNS = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'


def _col_index(ref):
    letters = re.match(r'([A-Z]+)', ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path, sheets=None):
    """Return {sheet_name: rows}; each row a list of str/float/None (dense, by column)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        shared = []
        if 'xl/sharedStrings.xml' in names:
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('m:si', NS):
                shared.append(''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])))
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        relmap = {r.get('Id'): r.get('Target') for r in rels}
        for sh in wb.find('m:sheets', NS):
            name = sh.get('name')
            if sheets and name not in sheets:
                continue
            target = relmap[sh.get(RNS + 'id')]
            target = target.lstrip('/')
            if not target.startswith('xl/'):
                target = 'xl/' + target
            root = ET.fromstring(z.read(target))
            rows = []
            for r in root.iter('{%s}row' % NS['m']):
                cells = {}
                for c in r.findall('m:c', NS):
                    ref = c.get('r')
                    t = c.get('t')
                    v = c.find('m:v', NS)
                    val = None
                    if t == 's' and v is not None:
                        val = shared[int(v.text)]
                    elif t == 'inlineStr':
                        is_ = c.find('m:is', NS)
                        val = ''.join(x.text or '' for x in is_.iter('{%s}t' % NS['m'])) if is_ is not None else ''
                    elif t == 'str':
                        val = v.text if v is not None else None
                    elif t == 'b':
                        val = float(v.text) if v is not None else None
                    elif t == 'e':
                        val = None
                    elif v is not None and v.text is not None:
                        try:
                            val = float(v.text)
                        except ValueError:
                            val = v.text
                    cells[_col_index(ref)] = val
                if cells:
                    width = max(cells) + 1
                    row = [cells.get(i) for i in range(width)]
                else:
                    row = []
                rn = r.get('r')
                if rn is not None:
                    while len(rows) < int(rn) - 1:   # keep list index == sheet row number - 1
                        rows.append([])
                rows.append(row)
            out[name] = rows
    return out


def excel_date(serial):
    """Excel 1900-system serial -> date."""
    return dt.date(1899, 12, 30) + dt.timedelta(days=int(round(float(serial))))


def month_end(y, m):
    if m == 12:
        return dt.date(y, 12, 31)
    return dt.date(y, m + 1, 1) - dt.timedelta(days=1)


def fmt(v):
    if isinstance(v, bool):
        return '1' if v else '0'
    if isinstance(v, int):
        return str(v)
    s = format(float(v), '.10g')
    if 'e' in s or 'E' in s:
        s = repr(float(v))
    return s


def write_series(sid, rows, allow_dups=False):
    """rows: iterable of (date|str, value). Returns (first, last, n)."""
    assert re.fullmatch(r'[a-z0-9_]+', sid), sid
    clean = []
    for d, v in rows:
        if v is None:
            continue
        v = float(v)
        if math.isnan(v) or math.isinf(v):
            continue
        if isinstance(d, (dt.date, dt.datetime)):
            d = d.strftime('%Y-%m-%d')
        assert re.fullmatch(r'\d{4}-\d{2}-\d{2}', d), (sid, d)
        clean.append((d, v))
    clean.sort(key=lambda x: x[0])
    dates = [d for d, _ in clean]
    dups = len(dates) - len(set(dates))
    if dups and not allow_dups:
        raise ValueError(f'{sid}: {dups} duplicate dates')
    path = os.path.join(SERIES, sid + '.csv')
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as f:
        f.write('date,value\n')
        for d, v in clean:
            f.write(f'{d},{fmt(v)}\n')
    os.replace(tmp, path)
    if not clean:
        raise ValueError(f'{sid}: empty series')
    return clean[0][0], clean[-1][0], len(clean)


CAT_COLS = ['id', 'name', 'category', 'subcategory', 'region', 'source', 'url', 'frequency', 'units',
            'transform', 'known_after_days', 'first', 'last', 'rows', 'notes']


def catalog(group, sid, **f):
    """Append a catalog row (first/last/rows are filled in at assembly from the series file)."""
    f['id'] = sid
    f.setdefault('category', 'politics')
    missing = [c for c in CAT_COLS if c not in f and c not in ('first', 'last', 'rows')]
    if missing:
        raise ValueError(f'{sid}: missing catalog fields {missing}')
    assert f['transform'] in ('logdiff', 'diff', 'anomaly', 'anomaly_log1p', 'level'), f['transform']
    assert f['frequency'] in ('daily', 'weekly', 'monthly', 'quarterly', 'annual'), f['frequency']
    path = os.path.join(PARTS, group + '.jsonl')
    with open(path, 'a') as fh:
        fh.write(json.dumps(f) + '\n')


def reset_group(group):
    path = os.path.join(PARTS, group + '.jsonl')
    if os.path.exists(path):
        os.remove(path)


def read_series(sid):
    with open(os.path.join(SERIES, sid + '.csv')) as f:
        r = csv.reader(f)
        next(r)
        return [(d, float(v)) for d, v in r]


def qc(sid, freq, lo=None, hi=None, show=3, quiet=False):
    """Checks: range, duplicates, gaps. Returns dict summary; prints a short report."""
    rows = read_series(sid)
    dates = [dt.date.fromisoformat(d) for d, _ in rows]
    vals = [v for _, v in rows]
    rep = {'id': sid, 'n': len(rows), 'first': rows[0][0], 'last': rows[-1][0],
           'min': min(vals), 'max': max(vals)}
    assert len(set(dates)) == len(dates), f'{sid} duplicate dates'
    assert dates == sorted(dates), f'{sid} not sorted'
    bad = []
    if lo is not None and rep['min'] < lo:
        bad.append(f'min {rep["min"]} < {lo}')
    if hi is not None and rep['max'] > hi:
        bad.append(f'max {rep["max"]} > {hi}')
    if any(v in (-9999, -999, -99, 9999, 99999) for v in vals):
        bad.append('sentinel-like value present')
    # gaps
    gaps = []
    if freq == 'daily':
        for a, b in zip(dates, dates[1:]):
            if (b - a).days > 1:
                gaps.append((a, b, (b - a).days - 1))
    elif freq == 'monthly':
        for a, b in zip(dates, dates[1:]):
            ma = a.year * 12 + a.month
            mb = b.year * 12 + b.month
            if mb - ma > 1:
                gaps.append((a, b, mb - ma - 1))
    elif freq == 'weekly':
        for a, b in zip(dates, dates[1:]):
            if (b - a).days > 7:
                gaps.append((a, b, (b - a).days // 7 - 1))
    rep['gaps'] = len(gaps)
    rep['gap_units'] = sum(g[2] for g in gaps)
    rep['largest_gap'] = max(gaps, key=lambda g: g[2]) if gaps else None
    rep['bad'] = bad
    rep['zeros'] = sum(1 for v in vals if v == 0)
    if not quiet:
        lg = rep['largest_gap']
        lgs = f'{lg[0]}..{lg[1]} ({lg[2]})' if lg else '-'
        print(f'{sid}: n={rep["n"]} {rep["first"]}..{rep["last"]} min={rep["min"]:.6g} max={rep["max"]:.6g} '
              f'zeros={rep["zeros"]} gaps={rep["gaps"]}/{rep["gap_units"]} largest={lgs} {"BAD:" + ";".join(bad) if bad else "ok"}')
        if show:
            print('   head', rows[:show], ' tail', rows[-show:])
    return rep


def xlsx_highlighted(path):
    """Return {sheet_name: set((row_idx0, col_idx0))} of cells with a solid non-white fill.
    Row/col are 0-based to match read_xlsx output (rows as listed in the sheet XML, assuming no
    fully empty rows are skipped before data -- callers should verify with the 'r' attribute)."""
    out = {}
    with zipfile.ZipFile(path) as z:
        st = ET.fromstring(z.read('xl/styles.xml'))
        fills = st.find('m:fills', NS).findall('m:fill', NS)
        solid = []
        for f in fills:
            pf = f.find('m:patternFill', NS)
            ok = False
            if pf is not None and pf.get('patternType') == 'solid':
                fg = pf.find('m:fgColor', NS)
                rgb = fg.get('rgb') if fg is not None else None
                ok = not (rgb and rgb.upper().endswith('FFFFFF'))
            solid.append(ok)
        xfs = st.find('m:cellXfs', NS).findall('m:xf', NS)
        xf_solid = [solid[int(x.get('fillId', 0))] for x in xfs]
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        relmap = {r.get('Id'): r.get('Target') for r in rels}
        for sh in wb.find('m:sheets', NS):
            target = relmap[sh.get(RNS + 'id')].lstrip('/')
            if not target.startswith('xl/'):
                target = 'xl/' + target
            root = ET.fromstring(z.read(target))
            cells = set()
            for r in root.iter('{%s}row' % NS['m']):
                rn = int(r.get('r')) - 1
                for c in r.findall('m:c', NS):
                    s = int(c.get('s', 0))
                    if s < len(xf_solid) and xf_solid[s] and c.find('m:v', NS) is not None:
                        cells.add((rn, _col_index(c.get('r'))))
            out[sh.get('name')] = cells
    return out


def xlsx_rows_by_number(path, sheet):
    """Like read_xlsx but returns {row_number0: row} keyed by the sheet's own row index."""
    with zipfile.ZipFile(path) as z:
        pass
    data = read_xlsx(path, sheets=[sheet])[sheet]
    return data
