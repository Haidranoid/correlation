"""What stockai-30d predicts and who is eligible: H-day forward returns, the grader's universe, the decision schedule.

- Forward return of stock j decided at the close of day d: close[d+H] / close[d] - 1. If close[d+H] is missing, the
  last close in the min(FILL_DAYS, H - 1) days before it stands in (still inside the window and after d); otherwise
  NaN. Computed on the development columns only (stockai.baselines.common.dev_cutoff), so every window that would
  reach the locked year (2024-06-28 on) is NaN: such decisions are never graded.
- A stock delisted inside the window (its last close L < d + H; Sharadar exports only, docs/sharadar.md) gets an
  outcome instead of none, by DELIST_RULES: under "main" a buyout counts at its last price and a failure or an
  unknown delisting as a total loss; "last-price" and "total-loss" are the bounds. Dropping such stocks, as a missing
  end close would, hides exactly the failures a survivorship-free test is for.
- The universe of a test year is the grader's (judge/src/holdout.rs `liquid_universe`, window 756): stocks with a
  census value on at least 90% of the 756 days before the year's first day h0, and a median close of at least $5 and a
  median dollar volume of at least $5M over the 252 days before h0. Fixed for the year, from data before it. With
  `price_close` (as-traded closes) the $5 test uses prices as they traded, so later reverse splits can't lift a stock
  over it.
- Decision days are the grader's rebalance days at --rebalance H: h0-1, h0-1+H, ... while inside the test period. A
  day is graded only when its whole window lies in the development columns, and (non_overlapping) only when its window
  ends by the next graded day. The test years start on fixed dates, so a year's last rebalance can fall fewer than H
  days before the next year's first, and its month then overlaps the next one: by a day or two when it falls 19-20
  days before (five of the years 2010-2023), almost entirely when it falls one day before (two of them).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import data, splits

H = 21  # trading days: one calendar month
FILL_DAYS = 5
UNIVERSE_WINDOW = 756
LOOKBACK = 252
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 5e6
DELIST_RULES = ("main", "last-price", "total-loss")
BUYOUT = 1  # stockai.data.DELIST_KINDS


def forward_returns(close: np.ndarray, h: int = H, fill: int = FILL_DAYS, delist_day: np.ndarray | None = None,
                    delist_kind: np.ndarray | None = None, rule: str = "main") -> np.ndarray:
    """float64 [N, C]: close[d+h] / close[d] - 1 (missing end close: the last close within min(fill, h - 1) days before
    it); NaN where close[d] is missing or not positive, where no end close is found, and for d + h >= C. A stock
    delisted inside the window (delist_day L with d <= L < d + h) gets its outcome from `rule` (module docstring)."""
    if rule not in DELIST_RULES:
        raise ValueError(f"unknown delisting rule {rule!r}")
    c = np.asarray(close, dtype=np.float64)
    end = c.copy()
    for k in range(1, min(fill, h - 1) + 1):  # nearest earlier close within the fill days, never close[d] itself
        gap = ~np.isfinite(end[:, k:])
        end[:, k:][gap] = c[:, :-k][gap]
    out = np.full(c.shape, np.nan)
    if c.shape[1] > h:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, :-h] = end[:, h:] / c[:, :-h] - 1.0
    if delist_day is not None:
        last_ok = c.shape[1] - h  # decisions whose whole window lies inside the columns
        for j in np.flatnonzero((delist_day >= 0) & (delist_day < c.shape[1])):
            last = int(delist_day[j])
            ds = np.arange(max(last - h + 1, 0), min(last + 1, last_ok))
            if not ds.size:
                continue
            at_last = rule == "last-price" or (rule == "main" and int(delist_kind[j]) == BUYOUT)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[j, ds] = c[j, last] / c[j, ds] - 1.0 if at_last else -1.0
    out[~(np.isfinite(c) & (c > 0))] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def judge_universe(v: np.ndarray, close: np.ndarray, dollar_volume: np.ndarray, h0: int,
                   window: int = UNIVERSE_WINDOW, price_close: np.ndarray | None = None) -> np.ndarray:
    """Column indices of the grader's candidates for a holdout starting at day h0 (holdout.rs `liquid_universe`).
    `price_close`, if given, replaces `close` in the $5 median-price test (as-traded closes)."""
    w0 = max(h0 - window, 0)
    n_days = h0 - w0
    if n_days <= 0:
        return np.empty(0, dtype=np.int64)
    tail = min(n_days, LOOKBACK)
    vv = np.asarray(v[:, w0:h0], dtype=np.float32)
    enough = np.isfinite(vv).sum(axis=1) * 10 >= n_days * 9
    cc = np.asarray((close if price_close is None else price_close)[:, h0 - tail:h0], dtype=np.float32)
    dv = np.asarray(dollar_volume[:, h0 - tail:h0], dtype=np.float32)

    def upper_median(x: np.ndarray) -> np.ndarray:
        """Per row: the (k // 2)-th smallest of the k finite values (select_nth at len / 2), 0 when none."""
        k = np.isfinite(x).sum(axis=1)
        s = np.sort(np.where(np.isfinite(x), x, np.inf), axis=1)
        idx = np.minimum(k // 2, x.shape[1] - 1)
        m = np.take_along_axis(s, idx[:, None], axis=1)[:, 0]
        return np.where(k > 0, m, np.float32(0.0))

    ok = enough & (upper_median(cc) >= np.float32(MIN_PRICE)) & (upper_median(dv) >= np.float32(MIN_DOLLAR_VOLUME))
    return np.flatnonzero(ok)


@dataclass(frozen=True)
class YearSpec:
    year: int
    h0: int  # first day of the test period
    t: int  # first day after it
    universe: np.ndarray  # column indices (stocks)
    decisions: np.ndarray  # graded decision days: whole H-day window inside the development columns
    dropped: np.ndarray  # ungraded rebalance days: window leaves the development columns or overlaps the next graded one


def non_overlapping(years: list[YearSpec], h: int = H) -> list[YearSpec]:
    """Drops graded days whose H-day window runs past the next graded day (moved to `dropped`)."""
    out = []
    for k, ys in enumerate(years):
        nxt = years[k + 1].decisions[0] if k + 1 < len(years) and years[k + 1].decisions.size else None
        keep = ys.decisions if nxt is None else ys.decisions[ys.decisions + h <= nxt]
        gone = np.setdiff1d(ys.decisions, keep)
        out.append(YearSpec(year=ys.year, h0=ys.h0, t=ys.t, universe=ys.universe, decisions=keep,
                            dropped=np.sort(np.r_[ys.dropped, gone]).astype(ys.decisions.dtype)))
    return out


def year_spec(days: np.ndarray, cutoff: int, v: np.ndarray, close: np.ndarray, dollar_volume: np.ndarray, year: int,
              h: int = H, price_close: np.ndarray | None = None) -> YearSpec:
    """The test year starting `year`-06-28, as the grader runs it at --rebalance h (`price_close`: judge_universe)."""
    if year >= splits.LOCKED_YEAR:
        raise ValueError(f"{year} is the locked year or later")
    h0 = int(np.searchsorted(days, data.str_to_day(f"{year}-{splits.PERIOD_START}"), side="left"))
    t = int(np.searchsorted(days, data.str_to_day(f"{year + 1}-{splits.PERIOD_START}"), side="left"))
    reb = np.arange(h0 - 1, t - 1, h)  # the grader's decision days d = h0-1 .. t-2, rebalancing every h
    graded = reb + h <= cutoff - 1  # the window's last close must be a development column
    uni = judge_universe(v, close, dollar_volume, h0, price_close=price_close)
    return YearSpec(year=year, h0=h0, t=t, universe=uni, decisions=reb[graded], dropped=reb[~graded])
