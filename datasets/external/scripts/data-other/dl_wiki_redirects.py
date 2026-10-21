"""For every canonical article: list its redirects (namespace 0, with continuation) and
fetch each redirect title's daily user views, so the parser can sum article + redirects.
That captures pre-move history for pages renamed during the window."""
import json, os, sys, time, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fetch, RAW
import urllib.error

OUT = os.path.join(RAW, "wikimedia")
START, END = "20150701", "20261017"
canon = json.load(open(os.path.join(OUT, "title_map.json")))

def safe(t):
    return "".join(c if c.isalnum() else "_" for c in t)

def views_url(title):
    t = urllib.parse.quote(title.replace(" ", "_"), safe="")
    return ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
            f"all-access/user/{t}/daily/{START}/{END}")

for title in sorted(set(canon.values())):
    reds, cont, k = [], None, 0
    while True:
        p = dict(action="query", titles=title, prop="redirects", rdlimit="max", rdnamespace=0,
                 format="json", formatversion=2)
        if cont:
            p["rdcontinue"] = cont
        dest = os.path.join(OUT, f"redirects_{safe(title)}{'' if k == 0 else '_' + str(k)}.json")
        fetch("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(p), dest,
              sleep=1, overwrite=True)
        j = json.load(open(dest))
        reds += [r["title"] for pg in j["query"]["pages"] for r in pg.get("redirects", [])]
        cont = j.get("continue", {}).get("rdcontinue")
        k += 1
        if not cont:
            break
    n404 = 0
    for r in reds:
        d2 = os.path.join(OUT, f"rviews_{safe(title)}__{safe(r)}.json")
        if os.path.exists(d2) or os.path.exists(d2 + ".404"):
            continue
        try:
            fetch(views_url(r), d2, sleep=0.4)
        except urllib.error.HTTPError as e:
            if e.code == 404:  # no views recorded for that title in the window
                open(d2 + ".404", "w").close()
                n404 += 1
                time.sleep(0.4)
            else:
                raise
    print(f"{title}: {len(reds)} redirects ({n404} with no views)", flush=True)
print("done", flush=True)
