"""Re-read every series in catalog_data-news.csv and log range, gaps, first/last rows
(common.check_series) plus a duplicate/order/range re-check, to logs/series_checks.txt."""
import contextlib, csv, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXT, LOGS, check_series
rows = list(csv.DictReader(open(os.path.join(EXT, "catalog_data-news.csv"))))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for r in rows:
        rep = check_series(r["id"], r["frequency"])
        assert rep["rows"] == int(r["rows"]) and rep["first"] == r["first"] and rep["last"] == r["last"], r["id"]
        if "_count" in r["id"] or "_crisis_" in r["id"]:
            assert rep["min"] >= 0, r["id"]
        else:
            assert -1 <= rep["min"] and rep["max"] <= 1, r["id"]
open(os.path.join(LOGS, "series_checks.txt"), "w").write(buf.getvalue())
print(f"{len(rows)} series checked -> {os.path.join(LOGS, 'series_checks.txt')}")
