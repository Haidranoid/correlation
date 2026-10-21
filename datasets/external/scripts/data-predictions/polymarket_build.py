"""Build normalized series from cached Polymarket price histories.

Daily value = the last CLOB prices-history sample at or before 16:00 America/New_York (all
calendar days). Samples come every 10 minutes (fidelity=10, at hh:m0:02 UTC), so the value is
the 15:50:02 New York sample; it must be no older than 6 hours, else the day is missing. The
sampled price is what Polymarket charts (order-book midpoint, or the last trade when the
spread is wide). A date is included only while the market was still open at 16:00 New York.
"""
import bisect
import datetime as dt
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HERE, daterange, iso_to_ts, ny16, write_series  # noqa: E402
from pm_select import INDIVIDUAL, ROLLING  # noqa: E402
from polymarket_fetch import fed_events, load_prices, market_meta, token_for  # noqa: E402

SRC = "Polymarket (Gamma API metadata + CLOB prices-history), no key"
URL = "https://clob.polymarket.com/prices-history?market=<token>&startTs=..&endTs=..&fidelity=10 (market {mid})"
STALE = 6 * 3600
NOW = int(time.time())  # no value for a date whose 16:00 New York lies in the future
TIMING = ("value = Polymarket price sample at 15:50 New York (last 10-minute sample <= 16:00 ET), "
          "so known_after_days 0")
QC = {}


class PMkt:
    def __init__(self, mid, outcome="Yes"):
        self.mid = mid
        self.m = market_meta(mid)
        self.token = token_for(self.m, outcome)
        pts = load_prices(mid, self.token)
        self.ts = [t for t, _ in pts]
        self.ps = [p for _, p in pts]
        self.close = iso_to_ts(self.m.get("closedTime")) if self.m.get("closed") else None
        self.start = iso_to_ts(self.m.get("startDate") or self.m.get("createdAt"))

    def at(self, cut):
        if cut > NOW or (self.close is not None and cut >= self.close):
            return None
        i = bisect.bisect_right(self.ts, cut)
        if not i or cut - self.ts[i - 1] > STALE:
            return None
        return self.ps[i - 1]

    def first_date(self):
        return dt.datetime.fromtimestamp(self.ts[0], dt.timezone.utc).date() if self.ts else None

    def last_date(self):
        end = self.close if self.close is not None else int(time.time())
        return dt.datetime.fromtimestamp(min(end, self.ts[-1] + 86400), dt.timezone.utc).date() if self.ts else None


def bp_of(label):
    s = (label or "").strip().lower()
    if s.startswith("no change"):
        return 0
    mt = re.match(r"(\d+)\+?\s*bps?\s*(increase|decrease)", s)
    if mt:
        v = int(mt.group(1))
        return v if mt.group(2) == "increase" else -v
    return None  # 'Other'


def build_fed():
    evs = fed_events()
    legs = []
    for meet, prev, eid, rows in evs:
        outs = []
        for r in rows:
            b = bp_of(r.get("group_item"))
            if b is None:
                continue
            outs.append((b, PMkt(r["market_id"], "Yes")))
        legs.append((dt.date.fromisoformat(prev), dt.date.fromisoformat(meet), eid, outs))
    pc, ph, ex = [], [], []
    rej = 0
    for prev, meet, eid, outs in legs:
        for d in daterange(prev + dt.timedelta(days=1), meet):
            cut = ny16(d)
            vals = [(b, mk.at(cut)) for b, mk in outs]
            if any(v is None for _, v in vals):
                continue
            tot = sum(v for _, v in vals)
            if not (0.8 <= tot <= 1.2):
                rej += 1
                continue
            pc.append((d, sum(v for b, v in vals if b < 0) / tot))
            ph.append((d, sum(v for b, v in vals if b > 0) / tot))
            ex.append((d, sum(v * b for b, v in vals) / tot))
    QC["pm_fed"] = dict(meetings=len(legs), rejected_days_sum=rej)
    notes = ("ROLLING continuous series (successive contracts stitched): Polymarket per-FOMC-meeting decision events ('Fed Interest Rates: "
             "<Month> 2024', 'Fed decision in <Month>?', first one the Mar 2024 meeting) for the next scheduled meeting, "
             "from the day after the previous meeting through the meeting day. Polymarket keeps these markets open until "
             "resolution (~17:00-19:00 ET), so the meeting-day 16:00 value already reflects the 14:00 announcement (near "
             "0/1); the series rolls to the following meeting the next day. Outcome prices (No change, 25 bps, 50+ bps, ... "
             "increase/decrease; 'Other' excluded) normalized to sum to 1 (days with a raw sum outside 0.8-1.2 or a missing "
             "outcome are left out). Roll days are FOMC days. " + TIMING)
    reps = [
        write_series("pm_fed_next_pcut", pc, name="Polymarket: probability of a Fed rate cut at the next FOMC meeting",
                     category="prediction_market", subcategory="fed", region="US", source=SRC,
                     url="https://polymarket.com/event/fed-decision-in-<month>", frequency="daily",
                     units="probability (0-1)", transform="diff", known_after_days=0, value_range=(0, 1),
                     notes="Sum of the decrease outcomes. " + notes),
        write_series("pm_fed_next_phike", ph, name="Polymarket: probability of a Fed rate hike at the next FOMC meeting",
                     category="prediction_market", subcategory="fed", region="US", source=SRC,
                     url="https://polymarket.com/event/fed-decision-in-<month>", frequency="daily",
                     units="probability (0-1)", transform="diff", known_after_days=0, value_range=(0, 1),
                     notes="Sum of the increase outcomes. " + notes),
        write_series("pm_fed_next_expchg_bp", ex, name="Polymarket: expected change in the fed funds target at the next FOMC meeting",
                     category="prediction_market", subcategory="fed", region="US", source=SRC,
                     url="https://polymarket.com/event/fed-decision-in-<month>", frequency="daily",
                     units="basis points", transform="diff", known_after_days=0, value_range=(-100, 100),
                     notes="Probability-weighted change; '50+ bps' counted as 50 and '75+' as 75 (truncation). " + notes),
    ]
    return reps


ROLL_RANGES = {
    "pm_us_recession_cur_year": {"2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
                                 "2026": ("2026-01-01", "2026-12-31")},
    "pm_house_gop_cur_cycle": {"2024": ("2023-01-01", "2024-11-05"), "2026": ("2024-11-06", "2026-11-03")},
    "pm_senate_gop_cur_cycle": {"2024": ("2023-01-01", "2024-11-05"), "2026": ("2024-11-06", "2026-11-03")},
}
ROLL_NOTES = {
    "pm_us_recession_cur_year": ("ROLLING by calendar year (rules differ by leg): 2024 = 'U.S. Recession in 2024?' (listed "
                                 "2024-08-05; Yes only on two consecutive negative BEA real GDP quarters within 2024); 2025 = "
                                 "'US recession in 2025?' (listed 2025-01-08; two negative quarters Q4 2024-Q4 2025 incl. "
                                 "advance estimates, or an NBER announcement during 2025); 2026 = 'US recession by end of "
                                 "2026?' (two negative quarters Q2 2025-Q4 2026 or NBER). Jumps at each Jan 1 roll; gaps "
                                 "before each listing (Jan-Aug 2024, Jan 1-7 2025)."),
    "pm_house_gop_cur_cycle": ("ROLLING by election cycle: 'House control after 2024 election?' (Republican outcome) through "
                               "election day 2024-11-05, then 'Which party will win the House in 2026?' Republican market "
                               "(listed 2025-07-11). Gap between cycles; the post-election count period is not used."),
    "pm_senate_gop_cur_cycle": ("ROLLING by election cycle: 'Senate control after 2024 election?' (Republicans outcome) "
                                "through election day 2024-11-05, then 'Which party will win the Senate in 2026?' "
                                "Republican market (listed 2025-07-11). Gap between cycles."),
}


def build_rolling():
    reps = []
    for sid, (name, legs) in ROLLING.items():
        rows = []
        for mid, outcome, label in legs:
            mk = PMkt(mid, outcome)
            a, b = ROLL_RANGES[sid][label]
            for d in daterange(dt.date.fromisoformat(a), min(dt.date.fromisoformat(b), dt.date.today())):
                p = mk.at(ny16(d))
                if p is not None:
                    rows.append((d, p))
        sub = "recession" if "recession" in sid else "us_election"
        reps.append(write_series(sid, rows, name=name, category="prediction_market", subcategory=sub, region="US",
                                 source=SRC, url=URL.format(mid="/".join(m for m, _, _ in legs)), frequency="daily",
                                 units="probability (0-1)", transform="diff", known_after_days=0, value_range=(0, 1),
                                 notes=ROLL_NOTES[sid] + " " + TIMING))
    return reps


REGION = {"pmc_turkey2023_erdogan": "TR", "pmc_argentina2023_milei": "AR", "pmc_taiwan2024_lai": "TW",
          "pmc_canada2025_carney_pm": "CA", "pmc_china_invade_taiwan_2024": "CN/TW", "pmc_china_invade_taiwan_2025": "CN/TW",
          "pmc_ru_ua_ceasefire_2025": "RU/UA", "pmc_ru_ua_ceasefire_by_2025jul": "RU/UA", "pmc_ru_ua_ceasefire_2024": "RU/UA",
          "pmc_trump_ends_ukraine_war_90d": "RU/UA", "pmc_isr_hezbollah_ceasefire_2024": "Middle East",
          "pmc_isr_hamas_ceasefire_2024": "Middle East", "pmc_isr_hamas_ceasefire_by_2025jul": "Middle East",
          "pmc_us_strike_iran_by_2025jul": "Middle East", "pmc_isr_strike_iran_by_2025jul": "Middle East",
          "pmc_isr_strike_iran_nuclear_2024": "Middle East", "pmc_iran_strike_isr_2024": "Middle East",
          "pmc_iran_close_hormuz_2025": "Middle East", "pmc_btc_spot_etf_by_2024jan15": "US", "pmc_btc_100k_2024": "global",
          "pmc_eth_ath_2024": "global", "pmc_btc_150k_2025": "global"}


def build_individual():
    reps = []
    for mid, outcome, sid, name, sub in INDIVIDUAL:
        mk = PMkt(mid, outcome)
        if not mk.ts:
            print("  no data", mid, sid)
            continue
        rows = []
        for d in daterange(mk.first_date(), mk.last_date()):
            p = mk.at(ny16(d))
            if p is not None:
                rows.append((d, p))
        if len(rows) < 30:
            print("  too short", mid, sid, len(rows))
            continue
        q = (mk.m.get("question") or "").strip()
        outs = mk.m.get("outcomes")
        prices = mk.m.get("outcomePrices")
        res = f"final outcome prices {prices} for {outs}" if mk.m.get("closed") else "still open"
        reps.append(write_series(sid, rows, name=f"Polymarket: {name}", category="prediction_market",
                                 subcategory="contract", region=REGION.get(sid, "US" if sub not in ("world_election", "geopolitics") else "global"),
                                 source=SRC, url=f"https://polymarket.com/market/{mk.m.get('slug')} (Gamma market id {mid})",
                                 frequency="daily", units=f"probability (0-1), price of the '{outcome}' token",
                                 transform="diff", known_after_days=0, value_range=(0, 1),
                                 notes=f"finite-life single contract ({sub}): \"{q}\" (Polymarket start "
                                       f"{(mk.m.get('startDate') or '')[:10]}, closed {(mk.m.get('closedTime') or '')[:10]}; {res}). "
                                       f"History starts with the CLOB order book. Converges to 0/1 near resolution. {TIMING}"))
    return reps


def main():
    reps = build_fed() + build_rolling() + build_individual()
    os.makedirs(os.path.join(HERE, "meta_qc"), exist_ok=True)
    with open(os.path.join(HERE, "meta_qc", "polymarket_qc.json"), "w") as f:
        json.dump(dict(qc=QC, reports=reps), f, indent=1, default=str)
    print(json.dumps(QC, indent=1))


if __name__ == "__main__":
    main()
