"""Replace a <!-- MARKER --> line in SOURCES_data-news.md with the contents of a file.
Usage: python3 -I splice_sources.py MARKER file.md   (the marker line is replaced; re-running
needs the marker, so the inserted block is wrapped as <!-- MARKER:begin --> ... <!-- MARKER:end -->)."""
import os
import re
import sys

path = os.path.expanduser("~/.local/share/correlation2/external/SOURCES_data-news.md")
marker, src = sys.argv[1], sys.argv[2]
text = open(path).read()
block = open(src).read().rstrip("\n")
wrapped = f"<!-- {marker}:begin -->\n{block}\n<!-- {marker}:end -->"
pat = re.compile(rf"<!-- {marker}:begin -->.*?<!-- {marker}:end -->", re.S)
if pat.search(text):
    text = pat.sub(lambda m: wrapped, text)
elif f"<!-- {marker} -->" in text:
    text = text.replace(f"<!-- {marker} -->", wrapped)
else:
    raise SystemExit(f"marker {marker} not found")
open(path + ".part", "w").write(text)
os.replace(path + ".part", path)
print(f"spliced {src} into {marker}")
