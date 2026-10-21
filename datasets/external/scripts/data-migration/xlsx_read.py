"""Minimal .xlsx and .ods readers (stdlib zipfile + ElementTree). They parse XML only.

xlsx_sheets(path) -> list of sheet names
xlsx_rows(path, sheet) -> list of rows, each a list of cell values (str or float), dense by column
ods_sheets(path), ods_rows(path, sheet): the same for OpenDocument spreadsheets
"""
import re
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _shared_strings(z):
    try:
        data = z.open("xl/sharedStrings.xml")
    except KeyError:
        return []
    out = []
    for ev, el in ET.iterparse(data, events=("end",)):
        if el.tag == NS + "si":
            out.append("".join(t.text or "" for t in el.iter(NS + "t")))
            el.clear()
    return out


def _sheet_paths(z):
    wb = z.read("xl/workbook.xml").decode("utf-8")
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    rmap = {}
    for m in re.finditer(r"<Relationship\b[^>]*>", rels):
        tag = m.group(0)
        i = re.search(r'Id="([^"]+)"', tag)
        t = re.search(r'Target="([^"]+)"', tag)
        if i and t:
            rmap[i.group(1)] = t.group(1)
    out = []
    for m in re.finditer(r"<sheet\b[^>]*>", wb):
        tag = m.group(0)
        name = re.search(r'name="([^"]+)"', tag).group(1)
        rid = re.search(r'r:id="([^"]+)"', tag).group(1)
        t = rmap[rid]
        t = t.lstrip("/")
        if not t.startswith("xl/"):
            t = "xl/" + t
        name = (name.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
                .replace("&quot;", '"').replace("&apos;", "'"))
        out.append((name, t))
    return out


def xlsx_sheets(path):
    with zipfile.ZipFile(path) as z:
        return [n for n, _ in _sheet_paths(z)]


def _col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def xlsx_rows(path, sheet):
    with zipfile.ZipFile(path) as z:
        ss = _shared_strings(z)
        sp = dict(_sheet_paths(z))[sheet]
        rows = []
        for ev, el in ET.iterparse(z.open(sp), events=("end",)):
            if el.tag != NS + "row":
                continue
            r = int(el.get("r")) - 1
            cells = {}
            for c in el.iter(NS + "c"):
                ref = c.get("r")
                ci = _col_index(ref)
                t = c.get("t")
                v = c.find(NS + "v")
                if t == "s" and v is not None:
                    val = ss[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(NS + "t"))
                elif t in ("str", "e") and v is not None:
                    val = v.text
                elif t == "b" and v is not None:
                    val = v.text
                elif v is not None and v.text is not None:
                    try:
                        val = float(v.text)
                    except ValueError:
                        val = v.text
                else:
                    val = None
                cells[ci] = val
            while len(rows) < r:
                rows.append([])
            width = max(cells) + 1 if cells else 0
            rows.append([cells.get(i) for i in range(width)])
            el.clear()
        return rows


# ---- OpenDocument (.ods) ----
ONS = {
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
}
T = "{%s}" % ONS["table"]
O = "{%s}" % ONS["office"]
X = "{%s}" % ONS["text"]


def ods_sheets(path):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    return [t.get(T + "name") for t in root.iter(T + "table")]


def ods_rows(path, sheet, max_repeat=1000):
    with zipfile.ZipFile(path) as z:
        root = ET.fromstring(z.read("content.xml"))
    for t in root.iter(T + "table"):
        if t.get(T + "name") != sheet:
            continue
        rows = []
        for row in t.iter(T + "table-row"):
            rrep = int(row.get(T + "number-rows-repeated", "1"))
            vals = []
            for c in row:
                if c.tag not in (T + "table-cell", T + "covered-table-cell"):
                    continue
                rep = int(c.get(T + "number-columns-repeated", "1"))
                vt = c.get(O + "value-type")
                if vt in ("float", "percentage", "currency"):
                    v = float(c.get(O + "value"))
                elif vt == "date":
                    v = c.get(O + "date-value")
                elif vt is None:
                    v = None
                else:
                    v = "".join("".join(p.itertext()) for p in c.iter(X + "p")) or None
                vals.extend([v] * min(rep, max_repeat))
            while vals and vals[-1] is None:
                vals.pop()
            for _ in range(min(rrep, 5 if not vals else max_repeat)):
                rows.append(list(vals))
        return rows
    raise KeyError(sheet)
