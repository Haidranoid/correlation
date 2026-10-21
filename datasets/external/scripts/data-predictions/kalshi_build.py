"""Build normalized series from the cached Kalshi candles.

Daily value = the as-of value at 16:00 America/New_York (all calendar days; Kalshi trades
around the clock). Hourly candles are sparse (a candle exists only when the book or trades
changed), so the state at 16:00 is the last candle that ENDS at or before 16:00 New York;
its close quotes are the book at the end of that hour. Price rule (Polymarket's display rule):
midpoint if the yes bid-ask spread is <= $0.10, else the last trade price, else unknown.

Rolling rule ("nearest unresolved"): on date D use the nearest event whose markets are still
trading at 16:00 New York on D. Kalshi closes its macro and Fed markets minutes before the
release/announcement (CPI/jobs/GDP 08:25-08:29 ET, FOMC 13:55 ET), so on a release or
meeting day the series already uses the following event. If that event's markets are not
listed/priced yet, the day is missing (roll gap, not filled).
"""
import bisect
import datetime as dt
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, daterange, ladder_expectation, ny16, pav_decreasing, write_series  # noqa: E402
from kalshi_lib import (SPREAD_MAX, cached_candles, close_ts, events_of, load_markets, now_ts,  # noqa: E402
                        open_ts, strike_of)

SRC = "Kalshi (CFTC-regulated exchange) public market-data API, no key"
API_URL = "https://api.elections.kalshi.com/trade-api/v2/historical/markets/{ticker}/candlesticks"
TIMING = ("value = as-of 16:00 New York from hourly candles (last candle ending <= 16:00 ET; "
          "midpoint if spread <= $0.10 else last trade), so known_after_days 0")
QC = {}  # per-series diagnostics, written to meta_qc/kalshi_qc.json

# FOMC decision days and the upper bound of the target range AFTER each meeting (percent)
FOMC_UPPER = [
    ("2021-01-27", 0.25), ("2021-03-17", 0.25), ("2021-04-28", 0.25), ("2021-06-16", 0.25),
    ("2021-07-28", 0.25), ("2021-09-22", 0.25), ("2021-11-03", 0.25), ("2021-12-15", 0.25),
    ("2022-01-26", 0.25), ("2022-03-16", 0.50), ("2022-05-04", 1.00), ("2022-06-15", 1.75),
    ("2022-07-27", 2.50), ("2022-09-21", 3.25), ("2022-11-02", 4.00), ("2022-12-14", 4.50),
    ("2023-02-01", 4.75), ("2023-03-22", 5.00), ("2023-05-03", 5.25), ("2023-06-14", 5.25),
    ("2023-07-26", 5.50), ("2023-09-20", 5.50), ("2023-11-01", 5.50), ("2023-12-13", 5.50),
    ("2024-01-31", 5.50), ("2024-03-20", 5.50), ("2024-05-01", 5.50), ("2024-06-12", 5.50),
    ("2024-07-31", 5.50), ("2024-09-18", 5.00), ("2024-11-07", 4.75), ("2024-12-18", 4.50),
    ("2025-01-29", 4.50), ("2025-03-19", 4.50), ("2025-05-07", 4.50), ("2025-06-18", 4.50),
    ("2025-07-30", 4.50), ("2025-09-17", 4.25), ("2025-10-29", 4.00), ("2025-12-10", 3.75),
    ("2026-01-28", None), ("2026-03-18", None), ("2026-04-29", None), ("2026-06-17", None),
    ("2026-07-29", None), ("2026-09-16", None), ("2026-10-28", None), ("2026-12-09", None),
    ("2027-01-27", None), ("2027-03-17", None), ("2027-04-28", None), ("2027-06-09", None),
]


STALE_TRADE = 7 * 86400  # a last-trade price older than this is not used
NOW = now_ts()  # no value for a date whose 16:00 New York lies in the future


class Mkt:
    """A market's as-of price function.

    Each candle gives the book at the end of its hour: mid if the spread is <= $0.10, else
    the last trade price, used only if that trade happened within STALE_TRADE of the query
    time (the time of the trade is the end of the latest candle with volume > 0)."""

    def __init__(self, m):
        self.m = m
        self.ticker = m["ticker"]
        self.open = open_ts(m)
        self.close = close_ts(m)
        self.ts, self.mid, self.last, self.last_t = [], [], [], []
        lt = None
        for t, bid, ask, last, vol in cached_candles(m):
            if vol:
                lt = t
            b = 0.0 if bid is None else bid
            a = 1.0 if ask is None else ask
            self.ts.append(t)
            self.mid.append((a + b) / 2 if a - b <= SPREAD_MAX + 1e-9 else None)
            self.last.append(last)
            self.last_t.append(lt)

    def trading(self, cut):
        return self.open is not None and self.close is not None and self.open <= cut < self.close

    def at(self, cut):
        """as-of price at timestamp cut, or None (not trading / no usable data / in the future)."""
        if cut > NOW or not self.trading(cut):
            return None
        i = bisect.bisect_right(self.ts, cut)
        if not i:
            return None
        i -= 1
        if self.mid[i] is not None:
            return self.mid[i]
        if self.last[i] is not None and self.last_t[i] is not None and cut - self.last_t[i] <= STALE_TRADE:
            return self.last[i]
        return None

    def has_data(self):
        return bool(self.ts)


def ev_close(ms):
    return max(close_ts(m) for m in ms)


def first_last_dates():
    return dt.date(2021, 6, 1), dt.datetime.fromtimestamp(now_ts(), dt.timezone.utc).date() - dt.timedelta(days=1)


def ladder_at(mkts, cut, round_to=None):
    ks, ps = [], []
    for k, mk in mkts:
        p = mk.at(cut)
        if p is None:
            continue
        ks.append(round(k, round_to) if round_to is not None else k)
        ps.append(min(1.0, max(0.0, p)))
    if not ks:
        return [], []
    order = sorted(range(len(ks)), key=lambda i: ks[i])
    ks = [ks[i] for i in order]
    ps = [ps[i] for i in order]
    # duplicate strikes (re-listed tickers) -> average
    uk, up = [], []
    for k, p in zip(ks, ps):
        if uk and abs(uk[-1] - k) < 1e-9:
            up[-1] = (up[-1] + p) / 2
        else:
            uk.append(k)
            up.append(p)
    return uk, pav_decreasing(uk, up)


# ------------------------------------------------------------------------ next-event ladders
NEXT_CFG = {
    # id: (series, grid, scale, name, units, round_to, extra notes)
    "ks_cpi_mom_next_exp": ("KXCPI", 0.1, 1.0, "Kalshi: expected next US CPI month-over-month change (headline, SA)", "percent (MoM)", 1,
                            "BLS CPI-U seasonally adjusted MoM change, 1 decimal; strikes 'above X%'"),
    "ks_cpi_yoy_next_exp": ("KXCPIYOY", 0.1, 1.0, "Kalshi: expected next US CPI year-over-year inflation print (headline)", "percent (YoY)", 1,
                            "BLS CPI-U YoY, 1 decimal; strikes 'above X%'"),
    "ks_core_cpi_mom_next_exp": ("KXCPICORE", 0.1, 1.0, "Kalshi: expected next US core CPI month-over-month change", "percent (MoM)", 1,
                                 "BLS core CPI (ex food and energy) SA MoM, 1 decimal"),
    "ks_payrolls_next_exp": ("KXPAYROLLS", None, 0.001, "Kalshi: expected next US nonfarm payrolls change (first print)", "thousands of jobs", None,
                             "BLS Employment Situation headline NFP change as first reported; strikes 'above X jobs'; treated as continuous (bucket midpoints)"),
    "ks_u3_next_exp": ("KXU3", 0.1, 1.0, "Kalshi: expected next US unemployment rate (U-3) print", "percent", 1,
                       "BLS U-3 SA, 1 decimal; strikes 'above X%'"),
    "ks_gdp_next_exp": ("KXGDP", 0.1, 1.0, "Kalshi: expected next US real GDP growth print (advance estimate, SAAR)", "percent (q/q annualized)", 1,
                        "BEA advance estimate of real GDP growth, annualized; strikes 'above X%' typically 0.5 apart (grid-aware buckets)"),
    "ks_gas_month_exp": ("KXAAAGASM", None, 1.0, "Kalshi: expected AAA US average regular gasoline price at the end of the current month", "USD per gallon", 3,
                         "AAA national average regular gas price on the month's last day; strikes 'strictly greater than $X'; continuous buckets. Mechanically tied to the daily AAA price (a slow-moving level), so most of the move is the spot gas price itself"),
}
MIN_STRIKES = 3
TAIL_MAX = 0.30


def build_next(sid, cfg):
    series, grid, scale, name, units, rnd, extra = cfg
    ms = load_markets(series)
    evs = events_of(ms)
    events = []
    for e, lst in evs.items():
        mk = [(strike_of(m), Mkt(m)) for m in lst if strike_of(m) is not None and close_ts(m)]
        mk = [(k, x) for k, x in mk if x.has_data()]
        if not mk:
            continue
        events.append((ev_close(lst), e, mk))
    events.sort()
    closes = [c for c, _, _ in events]
    d0, d1 = first_last_dates()
    rows, used, n_gap, n_thin, n_tail = [], {}, 0, 0, 0
    tails = []
    for d in daterange(d0, d1):
        cut = ny16(d)
        i = bisect.bisect_right(closes, cut)  # first event closing after 16:00
        if i >= len(events):
            continue
        c, e, mk = events[i]
        ks, ss = ladder_at(mk, cut, rnd)
        if not ks:
            n_gap += 1
            continue
        if len(ks) < MIN_STRIKES:
            n_thin += 1
            continue
        ev_, lo, hi = ladder_expectation(ks, ss, grid=grid)
        if lo > TAIL_MAX or hi > TAIL_MAX:
            n_tail += 1
            continue
        tails.append(max(lo, hi))
        rows.append((d, ev_ * scale))
        used[e] = used.get(e, 0) + 1
    # leading days before the first priced event are not "gaps"
    QC[sid] = dict(events_used=len(used), days=len(rows), thin_days=n_thin, tail_days=n_tail,
                   unpriced_days=n_gap, median_max_tail=sorted(tails)[len(tails) // 2] if tails else None)
    notes = (f"ROLLING continuous series (successive contracts stitched): on each date, the nearest unresolved {series} event "
             f"(the next release; Kalshi closes it minutes before the 08:30 ET release, so the release day itself already "
             f"uses the following event). Expected value from the 'above K' strike ladder: probabilities made monotone "
             f"(isotonic), buckets between strikes valued at their mean grid point, tails lumped at the outermost grid point "
             f"(truncation); days with < {MIN_STRIKES} priced strikes or > {int(TAIL_MAX * 100)}% probability in either "
             f"tail are left out. {extra}. Level jumps at each roll (new reference month). {TIMING}.")
    return write_series(sid, rows, name=name, category="prediction_market", subcategory="macro_expectation",
                        region="US", source=SRC, url=API_URL.format(ticker=f"<{series} market>"),
                        frequency="daily", units=units, transform="diff", known_after_days=0, notes=notes)


# ------------------------------------------------------------------------ Fed
def fed_events():
    ms = load_markets("KXFED")
    out = {}
    for e, lst in events_of(ms).items():
        mk = [(strike_of(m), Mkt(m), m) for m in lst if strike_of(m) is not None and close_ts(m)]
        if not mk:
            continue
        c = ev_close(lst)
        cd = dt.datetime.fromtimestamp(c, dt.timezone.utc).date()
        near = [f for f, _ in FOMC_UPPER if abs((dt.date.fromisoformat(f) - cd).days) <= 3]
        if not near:
            print("  fed event not near FOMC date:", e, cd)
            continue
        out[near[0]] = (e, c, [(k, x) for k, x, _ in mk if x.has_data()], [(k, m) for k, _, m in mk])
    return out


def check_fed_results(fev):
    """Cross-check the hard-coded upper bounds against Kalshi settlements."""
    bad = []
    derived = {}
    for f, (e, c, mk, raw) in sorted(fev.items()):
        yes = [k for k, m in raw if m.get("result") == "yes"]
        no = [k for k, m in raw if m.get("result") == "no"]
        if not yes and not no:
            continue
        lo = max(yes) + 0.25 if yes else None  # U >= lo
        hi = min(no) if no else None           # U <= hi
        derived[f] = (lo, hi)
        u = dict(FOMC_UPPER)[f]
        if u is None:
            if lo is not None and hi is not None and abs(lo - hi) < 1e-9:
                dict_set(f, lo)
            continue
        if (lo is not None and u < lo - 1e-9) or (hi is not None and u > hi + 1e-9):
            bad.append((f, u, lo, hi))
    return bad, derived


def dict_set(f, u):
    for i, (g, _) in enumerate(FOMC_UPPER):
        if g == f:
            FOMC_UPPER[i] = (g, u)


def build_fed():
    fev = fed_events()
    bad, derived = check_fed_results(fev)
    print("  FOMC path check vs Kalshi settlements: mismatches", bad)
    meetings = [dt.date.fromisoformat(f) for f, _ in FOMC_UPPER]
    upper = [u for _, u in FOMC_UPPER]
    d0, d1 = first_last_dates()
    phike, pcut, expchg, six = [], [], [], []
    roll_dates = []
    prev_key = None
    for d in daterange(d0, d1):
        cut = ny16(d)
        # next meeting: first meeting still unresolved at 16:00 on d
        j = None
        for i, md in enumerate(meetings):
            f = md.isoformat()
            if f in fev:
                if fev[f][1] > cut:
                    j = i
                    break
            elif md > d:
                j = i
                break
        if j is None or j == 0:
            continue
        f = meetings[j].isoformat()
        u_cur = upper[j - 1]
        if f != prev_key:
            roll_dates.append(d.isoformat())
            prev_key = f
        if f in fev and u_cur is not None:
            ks, ss = ladder_at(fev[f][2], cut, 2)
            if ks:
                sk = dict(zip(ks, ss))
                if u_cur in sk:
                    phike.append((d, sk[u_cur]))
                if round(u_cur - 0.25, 2) in sk:
                    pcut.append((d, 1 - sk[round(u_cur - 0.25, 2)]))
                if ks[0] <= u_cur + 1e-9 and u_cur in sk:
                    e, lo, hi = ladder_expectation(ks, ss, grid=0.25)
                    expchg.append((d, round(100 * (e - u_cur), 4)))
        # ~6 months ahead: the meeting nearest to d + 182 days that is still trading
        target = d + dt.timedelta(days=182)
        cands = sorted((abs((md - target).days), md) for md in meetings
                       if md.isoformat() in fev and fev[md.isoformat()][1] > cut and abs((md - target).days) <= 45)
        for _, md in cands:  # nearest meeting with a usable ladder
            ks, ss = ladder_at(fev[md.isoformat()][2], cut, 2)
            if len(ks) >= MIN_STRIKES:
                e, lo, hi = ladder_expectation(ks, ss, grid=0.25)
                if lo <= TAIL_MAX and hi <= TAIL_MAX:
                    six.append((d, e))
                    break
    QC["fed"] = dict(mismatches=bad, roll_dates=len(roll_dates))
    common_notes = ("ROLLING continuous series (successive contracts stitched): Kalshi KXFED/FED 'upper bound of the target range above K% "
                    "after the meeting' strike ladder for the next scheduled FOMC meeting (the meeting's markets close at "
                    "~13:55 ET, before the 14:00 announcement, so the meeting day itself already uses the following meeting); "
                    "no Kalshi market existed for some 2021 meetings (Nov 2021), which are gaps. U_cur = upper bound before the "
                    "meeting (FOMC record, cross-checked against Kalshi settlements). Level jumps at each roll; roll days are "
                    "FOMC days (an FOMC-calendar effect in stocks is well known, so treat roll-day moves with care). " + TIMING)
    reps = []
    reps.append(write_series("ks_fed_next_phike", phike, name="Kalshi: probability of a Fed rate hike at the next FOMC meeting",
                             category="prediction_market", subcategory="fed", region="US", source=SRC,
                             url=API_URL.format(ticker="FED-<yyMMM>-T<U_cur>"), frequency="daily", units="probability (0-1)",
                             transform="diff", known_after_days=0, value_range=(0, 1),
                             notes="P(upper bound after meeting > U_cur) = price of the strike at U_cur. " + common_notes))
    reps.append(write_series("ks_fed_next_pcut", pcut, name="Kalshi: probability of a Fed rate cut at the next FOMC meeting",
                             category="prediction_market", subcategory="fed", region="US", source=SRC,
                             url=API_URL.format(ticker="FED-<yyMMM>-T<U_cur-0.25>"), frequency="daily", units="probability (0-1)",
                             transform="diff", known_after_days=0, value_range=(0, 1),
                             notes="P(upper bound after meeting < U_cur) = 1 - price of the strike at U_cur-0.25; missing while "
                                   "no such strike was listed (2021 to early 2022, hiking cycle). " + common_notes))
    reps.append(write_series("ks_fed_next_expchg_bp", expchg, name="Kalshi: expected change in the fed funds target at the next FOMC meeting",
                             category="prediction_market", subcategory="fed", region="US", source=SRC,
                             url=API_URL.format(ticker="FED-<yyMMM>-T*"), frequency="daily", units="basis points",
                             transform="diff", known_after_days=0, value_range=(-150, 150),
                             notes="100*(E[upper bound after meeting] - U_cur), E from the strike ladder (0.25 grid, tails lumped "
                                   "at the outermost strikes); requires the lowest priced strike <= U_cur (in 2021-22 only "
                                   "strikes >= U_cur existed, i.e. cuts were assumed impossible). " + common_notes))
    reps.append(write_series("ks_fed_6m_exp_upper", six, name="Kalshi: expected fed funds upper bound after the FOMC meeting ~6 months ahead",
                             category="prediction_market", subcategory="fed", region="US", source=SRC,
                             url=API_URL.format(ticker="FED-<yyMMM>-T*"), frequency="daily", units="percent (target-range upper bound)",
                             transform="diff", known_after_days=0, value_range=(0, 8),
                             notes="E[upper bound] after the scheduled FOMC meeting closest to date+182 days (within 45 days), from "
                                   "its strike ladder (>=3 priced strikes, <=30% in either truncated tail). The target meeting "
                                   "steps forward every ~6 weeks (horizon varies ~4.5-7.5 months), so level jumps at those rolls. "
                                   "Starts when Kalshi listed meetings that far ahead (late 2021). " + TIMING))
    return reps


# ------------------------------------------------------------------------ S&P 500 year-end
INXY_EVENTS = {2022: "INXY-22DEC30", 2023: "INXY-23DEC29", 2024: "INXD-24DEC31", 2025: "KXINXY-25DEC31",
               2026: "KXINXY-26DEC31H1600"}


def bucket_bounds(m):
    st = m.get("strike_type")
    f, c = m.get("floor_strike"), m.get("cap_strike")
    if st == "between" and f is not None and c is not None:
        return float(f), float(c)
    if st == "greater" and f is not None:
        return float(f), None
    if st == "less" and c is not None:
        return None, float(c)
    sub = (m.get("yes_sub_title") or "").replace(",", "")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", sub)]
    if "or higher" in sub or "or above" in sub:
        return nums[0], None
    if "or lower" in sub or "or below" in sub:
        return None, nums[0]
    if len(nums) == 2:
        return nums[0], nums[1]
    return None


def build_inxy():
    ms = load_markets("KXINXY")
    evs = events_of(ms)
    d0, d1 = first_last_dates()
    rows = []
    bad = 0
    for year, e in sorted(INXY_EVENTS.items()):
        lst = evs.get(e) or []
        bks = []
        for m in lst:
            b = bucket_bounds(m)
            if b is None:
                continue
            bks.append((b, Mkt(m)))
        if not bks:
            continue
        widths = sorted((hi - lo) for (lo, hi), _ in bks if lo is not None and hi is not None)
        w = widths[len(widths) // 2] if widths else 200.0
        for d in daterange(max(d0, dt.date(year - 1, 6, 1)), min(d1, dt.date(year, 12, 31))):
            if d.year != year:
                continue  # current-year contract only
            cut = ny16(d)
            ps, mids = [], []
            for (lo, hi), mk in bks:
                p = mk.at(cut)
                if p is None:
                    continue
                mid = (lo + hi) / 2 if lo is not None and hi is not None else (lo + w / 2 if lo is not None else hi - w / 2)
                ps.append(p)
                mids.append(mid)
            tot = sum(ps)
            if len(ps) < 0.8 * len(bks) or not (0.8 <= tot <= 1.25):
                if ps:
                    bad += 1
                continue
            rows.append((d, sum(p * x for p, x in zip(ps, mids)) / tot))
    QC["ks_spx_yearend_exp"] = dict(rejected_days=bad)
    return write_series("ks_spx_yearend_exp", rows, name="Kalshi: expected S&P 500 close on the last trading day of the current year",
                        category="prediction_market", subcategory="equity_index_expectation", region="US", source=SRC,
                        url=API_URL.format(ticker="KXINXY-<yy>DEC31-B*"), frequency="daily", units="index points",
                        transform="logdiff", known_after_days=0,
                        notes="ROLLING by calendar year (the current year's year-end range market: INXY-22DEC30, INXY-23DEC29, "
                              "INXD-24DEC31, KXINXY-25DEC31, KXINXY-26DEC31H1600; 2022 starts in July when Kalshi listed it). "
                              "Probability-weighted mean of the 200-point range buckets (tails at edge +/- half a bucket), "
                              "normalized by the sum of bucket prices (days with <80% of buckets priced or a sum outside "
                              "0.8-1.25 left out). WARNING: mechanically tied to the S&P 500 level itself (it is mostly the "
                              "index plus a drift), so any lead-lag with stocks is largely the index; jumps at each Jan 1 roll. "
                              + TIMING)


# ------------------------------------------------------------------------ rolling binaries
def build_recession():
    ms = load_markets("KXRECSSNBER")
    by_year = {}
    for m in ms.values():
        y = re.search(r"-(\d\d)$", m["event_ticker"])
        if y:
            by_year[2000 + int(y.group(1))] = Mkt(m)
    d0, d1 = first_last_dates()
    rows = []
    for d in daterange(d0, d1):
        mk = by_year.get(d.year)
        if mk is None:
            continue
        p = mk.at(ny16(d))
        if p is not None:
            rows.append((d, p))
    return write_series("ks_recession_cur_year", rows, name="Kalshi: probability of a US recession in the current calendar year",
                        category="prediction_market", subcategory="recession", region="US", source=SRC,
                        url=API_URL.format(ticker="RECSSNBER-<yy>"), frequency="daily", units="probability (0-1)",
                        transform="diff", known_after_days=0, value_range=(0, 1),
                        notes="ROLLING by calendar year: during year Y the RECSSNBER-<YY> contract (2022 only Jul 1-28, when the "
                              "2022 contract traded; then 2023, 2024, 2025, 2026 contracts from Jan 1). Despite the ticker, "
                              "the 2024+ rules resolve on two consecutive quarters of negative real GDP growth (BEA) starting "
                              "in the year, not an NBER call; the 2023 contract had NBER/BEA wording; check per-year rules. "
                              "Jumps at each Jan 1 roll. " + TIMING)


def build_shutdown():
    mks = []
    for s in ("KXGOVSHUT", "KXGOVTSHUTDOWN"):
        for m in load_markets(s).values():
            if close_ts(m):
                mks.append((close_ts(m), Mkt(m)))
    mks.sort(key=lambda x: x[0])
    closes = [c for c, _ in mks]
    d0, d1 = first_last_dates()
    rows = []
    for d in daterange(d0, d1):
        cut = ny16(d)
        i = bisect.bisect_right(closes, cut)
        # nearest still-trading deadline market that is listed
        while i < len(mks) and not mks[i][1].trading(cut):
            if mks[i][1].open is not None and mks[i][1].open > cut:
                break  # nearest deadline not listed yet -> gap
            i += 1
        if i >= len(mks):
            continue
        p = mks[i][1].at(cut)
        if p is not None:
            rows.append((d, p))
    return write_series("ks_shutdown_next_deadline", rows, name="Kalshi: probability the US government is shut down at the next funding deadline",
                        category="prediction_market", subcategory="fiscal", region="US", source=SRC,
                        url=API_URL.format(ticker="GOVSHUT-<yyMMMdd>"), frequency="daily", units="probability (0-1)",
                        transform="diff", known_after_days=0, value_range=(0, 1),
                        notes="ROLLING: the nearest still-trading 'Will the government be shut down on <date>?' contract "
                              "(GOVSHUT/KXGOVSHUT/KXGOVTSHUTDOWN, Sep 2021 - 2026). Long gaps when no deadline was listed "
                              "(e.g. Mar-Aug 2022, Mar 2024 - Sep 2025); a few deadlines had several alternative-date "
                              "contracts (the nearest is used). " + TIMING)


INDIVIDUAL = [
    ("KXRECSSNBER", "RECSSNBER-23", "ksc_recession_2023", "Kalshi: will there be a US recession in 2023?"),
    ("KXRECSSNBER", "RECSSNBER-24", "ksc_recession_by_end_2024", "Kalshi: will there be a US recession by the end of 2024?"),
    ("KXRECSSNBER", "RECSSNBER-25", "ksc_recession_2025", "Kalshi: will there be a US recession in 2025?"),
    ("KXRECSSNBER", "KXRECSSNBER-26", "ksc_recession_2026", "Kalshi: will there be a US recession in 2026?"),
    ("KXRATECUT", "RATECUT-23DEC31", "ksc_fed_cut_by_2023dec31", "Kalshi: will the Fed cut rates by Dec 31, 2023?"),
    ("KXRATECUT", "RATECUT-24MAR20", "ksc_fed_cut_by_2024mar20", "Kalshi: will the Fed cut rates by Mar 20, 2024?"),
    ("KXRATECUT", "RATECUT-24MAY01", "ksc_fed_cut_by_2024may01", "Kalshi: will the Fed cut rates by May 1, 2024?"),
    ("KXRATECUT", "RATECUT-24JUN12", "ksc_fed_cut_by_2024jun12", "Kalshi: will the Fed cut rates by Jun 12, 2024?"),
    ("KXRATECUT", "RATECUT-24JUL31", "ksc_fed_cut_by_2024jul31", "Kalshi: will the Fed cut rates by Jul 31, 2024?"),
    ("KXRATECUT", "RATECUT-24SEP18", "ksc_fed_cut_by_2024sep18", "Kalshi: will the Fed cut rates before Sep 19, 2024?"),
    ("KXSHUTDOWNBY", "SHUTDOWNBY-24", "ksc_shutdown_in_2024", "Kalshi: government shutdown in 2024?"),
    ("KXSHUTDOWNBY", "KXSHUTDOWNBY-25", "ksc_shutdown_in_2025", "Kalshi: government shutdown in 2025?"),
    ("KXGOVSHUT", "GOVSHUT-23OCT02", "ksc_shutdown_on_2023oct02", "Kalshi: will the government be shut down on Oct 2, 2023?"),
    ("KXGOVSHUT", "GOVSHUT-23NOV17", "ksc_shutdown_on_2023nov17", "Kalshi: will the government be shut down on Nov 17, 2023?"),
    ("KXGOVSHUT", "KXGOVSHUT-26JAN31", "ksc_shutdown_on_2026jan31", "Kalshi: will the government be shut down on Jan 31, 2026?"),
    ("KXDCEIL", "DCEIL-23JUN1", "ksc_debt_ceiling_by_2023jun01", "Kalshi: debt ceiling raised or suspended by Jun 1, 2023?"),
    ("KXDCEIL", "DCEIL-23JUN9", "ksc_debt_ceiling_by_2023jun09", "Kalshi: debt ceiling raised or suspended by Jun 9, 2023?"),
    ("KXDCEIL", "DCEIL-23DEC31", "ksc_debt_ceiling_by_2023dec31", "Kalshi: debt ceiling raised or suspended by Dec 31, 2023?"),
    ("KXDCEIL", "DCEIL-21DEC15", "ksc_debt_ceiling_by_2021dec15", "Kalshi: debt ceiling raised or suspended before Dec 15, 2021?"),
    ("KXDCEIL", "DCEIL-21OCT14", "ksc_debt_ceiling_by_2021oct14", "Kalshi: debt ceiling raised or suspended before Oct 14, 2021?"),
]


def build_individual():
    reps = []
    for series, ticker, sid, name in INDIVIDUAL:
        ms = load_markets(series)
        m = ms.get(ticker)
        if m is None:
            print("  missing", ticker)
            continue
        mk = Mkt(m)
        if not mk.has_data():
            print("  no candles", ticker)
            continue
        d0 = dt.datetime.fromtimestamp(mk.open, dt.timezone.utc).date()
        d1 = dt.datetime.fromtimestamp(min(mk.close, now_ts()), dt.timezone.utc).date()
        rows = []
        for d in daterange(d0, d1):
            p = mk.at(ny16(d))
            if p is not None:
                rows.append((d, p))
        if len(rows) < 30:
            print("  too short", ticker, len(rows))
            continue
        res = m.get("result") or "unresolved"
        rules = (m.get("rules_primary") or "").replace("\n", " ")[:220]
        reps.append(write_series(sid, rows, name=name, category="prediction_market", subcategory="contract",
                                 region="US", source=SRC, url=API_URL.format(ticker=ticker), frequency="daily",
                                 units="probability (0-1), price of the Yes contract", transform="diff",
                                 known_after_days=0, value_range=(0, 1),
                                 notes=f"finite-life single contract {ticker} (open {m.get('open_time', '')[:10]}, close "
                                       f"{m.get('close_time', '')[:10]}, settled: {res}); values end when trading stops and "
                                       f"converge to 0/1 near resolution. Rule: {rules}. {TIMING}"))
    return reps


def main():
    reps = []
    for sid, cfg in NEXT_CFG.items():
        reps.append(build_next(sid, cfg))
    reps += build_fed()
    reps.append(build_inxy())
    reps.append(build_recession())
    reps.append(build_shutdown())
    reps += build_individual()
    os.makedirs(os.path.join(HERE, "meta_qc"), exist_ok=True)
    with open(os.path.join(HERE, "meta_qc", "kalshi_qc.json"), "w") as f:
        json.dump(dict(qc=QC, reports=reps), f, indent=1, default=str)
    print(json.dumps(QC, indent=1, default=str))


if __name__ == "__main__":
    main()
