"""What every baseline shares: the locked-year cutoff, a loader that never reads past it, and CLI helpers.

The locked year's first decision is the close of the last trading day before 2024-06-28 (`splits` scores decision
days h0-1 .. t-2 for a test period starting at h0). So development may read and score day columns [0, cutoff) with
cutoff = h0 - 1: the 2023 test period's last decision is cutoff - 1, and nothing at or after the cutoff is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import data, splits


def dev_cutoff(days: np.ndarray) -> int:
    """Number of leading day columns development code may read and score (see the module docstring)."""
    h0 = int(np.searchsorted(days, data.str_to_day(f"{splits.LOCKED_YEAR}-{splits.PERIOD_START}"), side="left"))
    return max(h0 - 1, 0)


@dataclass
class DevData:
    """The export's arrays cut to columns [0, cutoff). Arrays are views of the memory-mapped files."""

    export: data.Export
    cutoff: int
    v: np.ndarray  # float32 [N, cutoff]
    close: np.ndarray  # float32 [N, cutoff]
    dollar_volume: np.ndarray  # float32 [N, cutoff]
    market: np.ndarray  # float64 [cutoff]
    close_unadj: np.ndarray | None = None  # float32 [N, cutoff]: as-traded closes (Sharadar exports only)

    @property
    def days(self) -> np.ndarray:
        """Trading days of the cut columns."""
        return self.export.days[: self.cutoff]

    @property
    def delist_day(self) -> np.ndarray | None:
        """Column of each delisted stock's last close (-1: not delisted). A day at or after the cutoff is never in a
        development window, so it changes nothing there."""
        return self.export.delist_day

    @property
    def delist_kind(self) -> np.ndarray | None:
        return self.export.delist_kind

    def price_close(self, test: str) -> np.ndarray:
        """The closes the $5 price tests use: as traded ('traded') or adjusted as of the download ('adjusted')."""
        if test == "adjusted":
            return self.close
        if test == "traded":
            if self.close_unadj is None:
                raise ValueError("--price-test traded needs an export with as-traded closes (close_unadj.npy)")
            return self.close_unadj
        raise ValueError(f"unknown price test {test!r}")

    def default_price_test(self) -> str:
        return "traded" if self.close_unadj is not None else "adjusted"


def load_dev(path: str | Path | None = None) -> DevData:
    ex = data.load(path)
    c = dev_cutoff(ex.days)
    return DevData(export=ex, cutoff=c, v=ex.v[:, :c], close=ex.close[:, :c], dollar_volume=ex.dollar_volume[:, :c],
                   market=ex.market[:c], close_unadj=None if ex.close_unadj is None else ex.close_unadj[:, :c])


def full_width(part: np.ndarray, t: int, dtype=None) -> np.ndarray:
    """Pads an [N, cutoff] score array with NaN columns up to the export's T (the scores-file shape)."""
    out = np.full((part.shape[0], t), np.nan, dtype=dtype or part.dtype)
    out[:, : part.shape[1]] = part
    return out


def parse_years(spec: str | None) -> list[int]:
    """'2010-2023', '2015', '2010,2012' or '2010-2012,2015' -> sorted years. None means every development year."""
    if not spec:
        return list(splits.DEV_YEARS)
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            years.update(range(a, b + 1))
        elif part:
            years.add(int(part))
    bad = sorted(y for y in years if y not in splits.DEV_YEARS)
    if bad:
        raise ValueError(f"years {bad} are not development test years {splits.DEV_YEARS[0]}..{splits.DEV_YEARS[-1]} "
                         f"({splits.LOCKED_YEAR} is the locked year and is never run during development)")
    return sorted(years)


def describe_scores(scores: np.ndarray, days: np.ndarray) -> str:
    finite = np.isfinite(scores)
    cols = np.flatnonzero(finite.any(axis=0))
    if cols.size == 0:
        return "no finite scores"
    return (f"{int(finite.sum()):,} finite scores of {scores.size:,}, decision days "
            f"{data.day_to_str(days[cols[0]])}..{data.day_to_str(days[cols[-1]])}")
