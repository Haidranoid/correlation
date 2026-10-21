"""Parse a Wikipedia "Portal:Current events/YYYY Month D" page (wikitext) into headline items.

Items are bullet lines (`*`, `**`, ...). A bullet followed by deeper bullets is a topic label
(e.g. `*[[Russian invasion of Ukraine]]` over `**Russian forces shell ...`); only leaf bullets
are items, and each item's text is its labels' text joined with ": " in front of the leaf
text, which makes the post-2010 nested format equivalent to the older inline
"[[Story]]: sentence" format. Topic sections are header lines `;Name` (2010-2020) or
`'''Name'''` (2020-), mapped to canonical names by keyword (SECTION_RULES).
Wikitext cleanup: HTML comments, <ref>..</ref>, templates (a few content templates are
expanded), external links with their source label, images/categories removed; internal links
[[a|b]] -> b, [[a]] -> a; bold/italic quotes and HTML tags stripped. Leaves with fewer than 4
words after cleanup are dropped.
"""
import html
import re

SECTION_RULES = [  # first match wins; applied to the lower-cased header
    ("conflict", ("armed", "conflict", "attack", "military", "terror", "war")),
    ("disasters", ("disaster", "accident", "incident", "weather")),
    ("law_crime", ("law", "crime", "justice", "legal")),
    ("politics", ("politic", "election", "government")),
    ("intl", ("international", "diplomac", "foreign")),
    ("business", ("business", "econom", "financ", "market", "trade")),
    ("health_env", ("health", "medic", "environment", "disease")),
    ("science_tech", ("science", "technolog", "space", "transport")),
    ("sports", ("sport",)),
    ("arts", ("art", "culture", "entertainment", "media", "television", "music", "film", "religion")),
]
SECTIONS = [s for s, _ in SECTION_RULES]

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_TEMPLATE = re.compile(r"\{\{([^{}]*)\}\}")
_EXTLINK = re.compile(r"\[(?:https?:)?//[^\s\]]*(?:\s[^\]]*)?\]")
_BAREURL = re.compile(r"https?://\S+")
_FILELINK = re.compile(r"\[\[(?:File|Image|Category|Media):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
_WIKILINK = re.compile(r"\[\[([^\[\]|]*)(?:\|([^\[\]]*))?\]\]")
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_HDR_SEMI = re.compile(r"^;\s*(.+?)\s*:?\s*$")
_HDR_BOLD = re.compile(r"^'''\s*([^']+?)\s*'''\s*:?\s*$")
_HDR_EQ = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")
_WS = re.compile(r"\s+")
_KEEP_FIRST = {"nowrap", "nobr", "small", "lang", "ill", "interlanguage link", "tooltip", "abbr",
               "nowrap begin", "noitalic", "nobold", "illm", "ill-wd", "transl", "lang-en"}


def _template(m):
    parts = [p.strip() for p in m.group(1).split("|")]
    name = parts[0].lower().replace("_", " ")
    args = [p for p in parts[1:] if "=" not in p]
    if name in ("convert", "cvt") and len(args) >= 2:
        return f"{args[0]} {args[1]}"
    if name == "lang" and len(args) >= 2:
        return args[1]
    if name in _KEEP_FIRST and args:
        return args[0]
    if name in ("us$", "usd") and args:
        return f"US${args[0]}"
    if name == "sortname" and len(args) >= 2:
        return f"{args[0]} {args[1]}"
    return ""


def clean(s):
    s = _REF.sub("", s)
    for _ in range(6):  # innermost templates first
        s2 = _TEMPLATE.sub(_template, s)
        if s2 == s:
            break
        s = s2
    s = s.replace("{{", "").replace("}}", "")
    s = _EXTLINK.sub("", s)
    s = _BAREURL.sub("", s)
    s = _FILELINK.sub("", s)
    s = _WIKILINK.sub(lambda m: (m.group(2) if m.group(2) is not None else m.group(1)).strip(), s)
    s = s.replace("'''", "").replace("''", "")
    s = _TAG.sub("", s)
    s = html.unescape(s).replace("\xa0", " ")
    return _WS.sub(" ", s).strip(" :;,")


_SECTION_RX = [(name, re.compile(r"\b(?:" + "|".join(keys) + ")")) for name, keys in SECTION_RULES]


def section_of(header):
    """Canonical section for a header; keywords must start a word ('war' matches 'warfare',
    not 'awards' or 'software')."""
    h = clean(header).lower()
    for name, rx in _SECTION_RX:
        if rx.search(h):
            return name
    return "other"


def parse(wikitext):
    """-> list of (text, section_or_None, raw_header_or_None)."""
    lines = _COMMENT.sub("", wikitext).splitlines()
    rows = []  # (depth, text, section, header)
    section = header = None
    for line in lines:
        s = line.strip()
        if not s or s.startswith("{{Current events") or s in ("}}", "|}"):
            continue
        if s.startswith("*"):
            depth = len(s) - len(s.lstrip("*"))
            body = s.lstrip("*").strip()
            if body.endswith("}}") and body.count("{{") < body.count("}}"):
                body = body[:-2]
            rows.append((depth, clean(body), section, header))
            continue
        m = _HDR_SEMI.match(s) or _HDR_BOLD.match(s)
        if m:
            header = m.group(1)
            section = section_of(header)
            continue
        m = _HDR_EQ.match(s)
        if m:
            header = m.group(2)
            section = section_of(header)
    items = []
    stack = []  # labels by depth
    for i, (depth, text, sec, hdr) in enumerate(rows):
        stack = stack[:depth - 1]
        while len(stack) < depth - 1:
            stack.append("")
        nxt = rows[i + 1][0] if i + 1 < len(rows) else 0
        if nxt > depth:  # has children: a label
            stack.append(text)
            continue
        if len(text.split()) < 4:
            continue
        labels = [t for t in stack if t]
        items.append((": ".join(labels + [text]), sec, hdr))
    return items
