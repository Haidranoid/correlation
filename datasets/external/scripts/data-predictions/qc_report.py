"""QC for the data-predictions series: per-series stats and cross-venue consistency checks.

Prints a markdown table (for SOURCES_data-predictions.md) and the cross-checks:
- same question on different venues should agree in level and co-move day to day;
- a probability series must stay in [0, 1]; flat stretches and roll jumps are counted.
"""
import csv
import datetime as dt
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXT, SERIES  # noqa: E402


def load(sid):
    with open(os.path.join(SERIES, f"{sid}.csv")) as f:
        r = csv.reader(f)
        next(r)
        return {dt.date.fromisoformat(a): float(b) for a, b in r}


def stats(sid, freq_days=1):
    s = load(sid)
    ds = sorted(s)
    gaps = [(ds[i - 1], ds[i]) for i in range(1, len(ds)) if (ds[i] - ds[i - 1]).days > freq_days]
    miss = sum((b - a).days - 1 for a, b in gaps)
    vals = np.array([s[d] for d in ds])
    dif = np.diff(vals)
    zero = float(np.mean(dif == 0)) if len(dif) else float("nan")
    in_window = sum(1 for d in ds if dt.date(2020, 1, 1) <= d <= dt.date(2025, 6, 27))
    in_panel = sum(1 for d in ds if dt.date(2005, 1, 3) <= d <= dt.date(2025, 6, 27))
    return dict(first=ds[0], last=ds[-1], rows=len(ds), gaps=len(gaps), missing=miss, min=vals.min(),
                max=vals.max(), zero_moves=zero, in_stock_window=in_window, in_panel=in_panel,
                big_moves=int(np.sum(np.abs(dif) >= 0.25)) if vals.max() <= 1 else None)


FOMC_DAYS = {dt.date.fromisoformat(x) for x in (
    "2021-07-28 2021-09-22 2021-11-03 2021-12-15 2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 "
    "2022-09-21 2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14 2023-07-26 2023-09-20 "
    "2023-11-01 2023-12-13 2024-01-31 2024-03-20 2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 "
    "2024-12-18 2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17 2025-10-29 2025-12-10 "
    "2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09").split()}
ROLL_EXCLUDE = FOMC_DAYS | {d + dt.timedelta(days=1) for d in FOMC_DAYS} | \
    {dt.date(y, 1, 1) for y in range(2021, 2028)} | {dt.date(y, 1, 2) for y in range(2021, 2028)}


def compare(a, b, label, scale=1.0):
    """Roll days (FOMC days, the day after, Jan 1-2) are excluded: the two venues roll on different days."""
    sa, sb = load(a), load(b)
    common = sorted((set(sa) & set(sb)) - ROLL_EXCLUDE)
    if len(common) < 20:
        print(f"- {label}: only {len(common)} common days")
        return
    x = np.array([sa[d] for d in common])
    y = np.array([sb[d] for d in common]) * scale
    lv = np.corrcoef(x, y)[0, 1]
    # day-to-day changes on consecutive common days
    pairs = [(i - 1, i) for i in range(1, len(common)) if (common[i] - common[i - 1]).days == 1]
    dx = np.array([x[j] - x[i] for i, j in pairs])
    dy = np.array([y[j] - y[i] for i, j in pairs])
    dc = np.corrcoef(dx, dy)[0, 1] if len(pairs) > 10 and dx.std() > 0 and dy.std() > 0 else float("nan")
    print(f"- {label}: {len(common)} common days {common[0]}..{common[-1]}; level corr {lv:.3f}, "
          f"mean abs diff {np.mean(np.abs(x - y)):.3f}, daily-change corr {dc:.3f} (n={len(pairs)})")


def forecast_check(series, sid, grid, scale=1.0, label="", unit=0.1):
    """Expected print on the last day before each release vs the realized print implied by the
    settled 'above K' ladder (X > K_yes_max and X <= K_no_min)."""
    from common import ny16
    from kalshi_lib import close_ts, events_of, load_markets, strike_of
    s = load(sid)
    errs, pairs = [], []
    for e, lst in events_of(load_markets(series)).items():
        ks = [(strike_of(m), m.get("result")) for m in lst if strike_of(m) is not None]
        yes = [k for k, r in ks if r == "yes"]
        no = [k for k, r in ks if r == "no"]
        if not yes or not no:
            continue
        lo, hi = max(yes), min(no)
        if hi - lo > grid + 1e-9:
            continue  # realized value not pinned down by the ladder
        real = ((lo + unit) + hi) / 2 * scale  # bucket centre on the grid (= hi when strikes are contiguous)
        c = max(close_ts(m) for m in lst)
        d = dt.datetime.fromtimestamp(c, dt.timezone.utc).date()
        while ny16(d) >= c:
            d -= dt.timedelta(days=1)
        if d in s:
            pairs.append((s[d], real))
    if len(pairs) < 5:
        print(f"- {label}: too few pinned releases ({len(pairs)})")
        return
    x = np.array([p for p, _ in pairs])
    y = np.array([r for _, r in pairs])
    print(f"- {label}: {len(pairs)} releases; corr(expected, realized) {np.corrcoef(x, y)[0, 1]:.3f}; "
          f"mean error (realized-expected) {np.mean(y - x):+.3f}; mean abs error {np.mean(np.abs(y - x)):.3f}")


def main():
    cat = list(csv.DictReader(open(os.path.join(EXT, "catalog_data-predictions.csv"))))
    print("| id | first | last | rows | gaps (missing days) | rows in census panel 2005-01-03..2025-06-27 | of which 2020-01-01 on | min | max | share of zero day-to-day moves |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c in cat:
        s = stats(c["id"])
        print(f"| {c['id']} | {s['first']} | {s['last']} | {s['rows']} | {s['gaps']} ({s['missing']}) | "
              f"{s['in_panel']} | {s['in_stock_window']} | {s['min']:.4g} | {s['max']:.4g} | {s['zero_moves']:.2f} |")
    print()
    print("Cross-venue checks:")
    compare("ks_fed_next_pcut", "pm_fed_next_pcut", "Fed cut at next meeting, Kalshi ladder vs Polymarket")
    compare("ks_fed_next_phike", "pm_fed_next_phike", "Fed hike at next meeting, Kalshi vs Polymarket")
    compare("ks_fed_next_expchg_bp", "pm_fed_next_expchg_bp", "expected change at next meeting (bp), Kalshi vs Polymarket")
    compare("ks_recession_cur_year", "pm_us_recession_cur_year", "US recession this year, Kalshi vs Polymarket")
    compare("ksc_recession_2023", "mf_us_recession_by_end_2023", "recession in 2023, Kalshi vs Manifold (play money)")
    compare("ksc_recession_by_end_2024", "mf_us_recession_before_end_2024", "recession by end of 2024, Kalshi vs Manifold")
    compare("pmc_pres2024_gop_party", "pmc_pres2024_trump", "2024: Republican party vs Trump (Polymarket)")
    compare("ksc_fed_cut_by_2024jun12", "pmc_fed_cut_by_2024jun12", "Fed cut by Jun 12 2024, Kalshi vs Polymarket")
    compare("ks_shutdown_next_deadline", "pmc_funding_lapse_2025oct01", "shutdown Oct 1 2025, Kalshi rolling vs Polymarket")
    print()
    print("Forecast vs outcome (value on the last day before each release vs the settled print):")
    forecast_check("KXCPI", "ks_cpi_mom_next_exp", 0.1, label="CPI MoM (%)")
    forecast_check("KXCPIYOY", "ks_cpi_yoy_next_exp", 0.1, label="CPI YoY (%)")
    forecast_check("KXCPICORE", "ks_core_cpi_mom_next_exp", 0.1, label="core CPI MoM (%)")
    forecast_check("KXU3", "ks_u3_next_exp", 0.1, label="unemployment rate (%)")
    forecast_check("KXGDP", "ks_gdp_next_exp", 0.5, label="GDP advance (%), realized = centre of its <=0.5-wide strike bucket")
    forecast_check("KXPAYROLLS", "ks_payrolls_next_exp", 50000, scale=0.001, unit=0.0,
                   label="payrolls (k), realized = centre of its <=50k-wide strike bucket")


if __name__ == "__main__":
    main()
