"""Minimal streaming .xlsx reader (stdlib zipfile + iterparse). Yields rows as {col_letter: value}.
Values: shared strings resolved, inline strings, numbers as float. Parses XML only."""
import re, zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

def shared_strings(z):
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

def sheet_paths(z):
    wb = z.read("xl/workbook.xml").decode()
    rels = z.read("xl/_rels/workbook.xml.rels").decode()
    rmap = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))
    rmap.update({b: a for a, b in re.findall(r'Target="([^"]+)"[^>]*Id="(rId\d+)"', rels)})
    out = {}
    for name, rid in re.findall(r'<sheet name="([^"]+)"[^>]*r:id="(rId\d+)"', wb):
        t = rmap[rid]
        out[name] = t if t.startswith("xl/") else "xl/" + t.lstrip("/")
    return out

def iter_rows(path, sheet_name, max_rows=None):
    z = zipfile.ZipFile(path)
    ss = shared_strings(z)
    sp = sheet_paths(z)[sheet_name]
    n = 0
    for ev, el in ET.iterparse(z.open(sp), events=("end",)):
        if el.tag != NS + "row":
            continue
        row = {}
        for c in el.iter(NS + "c"):
            ref = c.get("r")
            col = re.match(r"[A-Z]+", ref).group(0)
            t = c.get("t")
            v = c.find(NS + "v")
            if t == "s" and v is not None:
                row[col] = ss[int(v.text)]
            elif t == "inlineStr":
                row[col] = "".join(x.text or "" for x in c.iter(NS + "t"))
            elif t in ("str", "e") and v is not None:
                row[col] = v.text
            elif v is not None and v.text is not None:
                try:
                    row[col] = float(v.text)
                except ValueError:
                    row[col] = v.text
        yield int(el.get("r")), row
        el.clear()
        n += 1
        if max_rows and n >= max_rows:
            return
