"""Walk-forward splits shared by every model, so all of them are tested on the same days with the same gaps.

correlation2's walk-forward treats each one-year period [Y-06-28, (Y+1)-06-28) as the future in turn. Its holdout
decides at the close of day d and holds over day d+1, for decision days h0-1 .. t-2: h0 is the first trading day of
the period and t the first trading day after it. So the model for year Y must score decision days h0-1 .. t-2.

A sample with decision day d uses inputs through day d and an outcome over days d+1 .. d+H (H = horizon).
Outcomes used for training and validation must be known before the first test decision, with `gap` spare days:
- validation decisions end so that their outcomes finish by day h0-2-gap;
- training decisions end so that their outcomes finish `gap` days before the first validation decision.
All ranges are indices into the export's `days` array.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import str_to_day

DEV_YEARS = tuple(range(2010, 2024))  # test periods 2010-06-28 .. 2024-06-27
LOCKED_YEAR = 2024  # 2024-06-28 .. 2025-06-27: one run at the end (PLAN.md §4.4)
PERIOD_START = "06-28"
GAP = 5
VAL_DAYS = 252
TRAIN_START = "2005-01-03"


@dataclass(frozen=True)
class Split:
    year: int
    horizon: int
    gap: int
    train_decisions: range
    val_decisions: range
    test_decisions: range

    def describe(self, days: np.ndarray) -> str:
        from .data import day_to_str

        def r(x: range) -> str:
            return f"{day_to_str(days[x.start])}..{day_to_str(days[x.stop - 1])} ({len(x)} days)"

        return f"year {self.year}, horizon {self.horizon}: train {r(self.train_decisions)}, val {r(self.val_decisions)}, test {r(self.test_decisions)}"


def _first_index(days: np.ndarray, date: str) -> int:
    return int(np.searchsorted(days, str_to_day(date), side="left"))


def split(days: np.ndarray, year: int, horizon: int = 1, gap: int | None = None, val_days: int = VAL_DAYS,
          train_start: str = TRAIN_START) -> Split:
    """Training, validation and test decision days for the test period starting `year`-06-28."""
    if year == LOCKED_YEAR or year > LOCKED_YEAR:
        raise ValueError(f"{year} is the locked year or later; it is run once at the end, not during development")
    return _split(days, year, horizon, gap, val_days, train_start)


def locked_split(days: np.ndarray, horizon: int = 1, gap: int | None = None, val_days: int = VAL_DAYS,
                 train_start: str = TRAIN_START) -> Split:
    """The locked year's split. Only for the single final run (PLAN.md M7)."""
    return _split(days, LOCKED_YEAR, horizon, gap, val_days, train_start)


def _split(days, year, horizon, gap, val_days, train_start) -> Split:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    gap = max(GAP, horizon) if gap is None else gap
    h0 = _first_index(days, f"{year}-{PERIOD_START}")
    t = _first_index(days, f"{year + 1}-{PERIOD_START}")
    if h0 < 1 or h0 >= len(days):
        raise ValueError(f"test period {year} is outside the data")
    test = range(h0 - 1, min(t, len(days)) - 1)
    # Last validation decision d satisfies d + horizon <= h0 - 2 - gap.
    val_stop = h0 - 1 - gap - horizon
    val = range(val_stop - val_days, val_stop)
    # Last training decision d satisfies d + horizon <= val.start - 1 - gap.
    train_stop = val.start - gap - horizon
    train = range(_first_index(days, train_start), train_stop)
    if len(train) <= 0 or val.start < 0:
        raise ValueError(f"not enough history before test period {year}")
    return Split(year=year, horizon=horizon, gap=gap, train_decisions=train, val_decisions=val, test_decisions=test)


def dev_splits(days: np.ndarray, horizon: int = 1, years=DEV_YEARS) -> list[Split]:
    out = [split(days, y, horizon) for y in years]
    for a, b in zip(out, out[1:]):
        assert a.test_decisions.stop == b.test_decisions.start, "test periods must tile without gaps or overlap"
    return out
