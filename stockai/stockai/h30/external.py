"""correlation2's outside series (weather, politics, news, rates, ...) as time-only LightGBM inputs (test-results-external.md).

`stockai-judge export-external` writes each series as correlation2's census-style standardized daily move on the
export's trading days (docs/external.md): ext_v.npy float32 [K, T], NaN where the series is unavailable, and
ext_meta.tsv with each series' category and status. This module turns them into features:

- For each category of the headline series (status `headline`: not annual, single-contract or noise), the mean of the
  day's standardized moves over its series with a value, or NaN when fewer than MIN_LIVE series have one.
- Summed over the last h = 1, 5, 21 trading days and divided by sqrt(h), a missing day counting as 0, as in
  correlation2's census sums; NaN when every day of the window is missing.

They are the same for every stock on a day, so they enter the model only through interactions with stock features. A
feature at day d uses ext_v columns through d only, and ext_v itself is point in time (publication lags, trailing
standardization through d-1). Everything is cut to the development columns before anything is computed.

`shift` (the control in test-results-external.md) rolls every series together by that many trading days, circularly
within the development columns, before the features are built: the same inputs, no longer aligned with the stocks.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..baselines.features import Features

WINDOWS = (1, 5, 21)
MIN_LIVE = 3
REVISED = ("macro", "prosperity", "migration", "population")  # official figures revised after release


def default_dir() -> Path:
    return Path.home() / ".local/share/stockai/external"


@dataclass
class External:
    path: Path
    v: np.ndarray  # float32 [K, T] on the export's days (memory-mapped)
    meta: list[dict]  # one row of ext_meta.tsv per series
    info: dict  # meta.json


def load(path: str | Path | None, days: np.ndarray, fingerprint: str | None = None) -> External:
    """`fingerprint`: the export's (meta.json); the outside data must have been built on the same days."""
    p = Path(path) if path else default_dir()
    info = json.loads((p / "meta.json").read_text())
    if fingerprint and info.get("fingerprint") and info["fingerprint"] != fingerprint:
        raise ValueError(f"{p}: built for export {info['fingerprint']}, not {fingerprint}")
    with open(p / "ext_meta.tsv", newline="") as f:
        meta = list(csv.DictReader(f, delimiter="\t"))
    v = np.load(p / "ext_v.npy", mmap_mode="r")
    if v.shape != (len(meta), len(days)):
        raise ValueError(f"{p}/ext_v.npy: shape {v.shape}, expected ({len(meta)}, {len(days)}): built for another export?")
    first, last = info.get("first_day"), info.get("last_day")
    from .. import data
    if first and last and (first, last) != (data.day_to_str(int(days[0])), data.day_to_str(int(days[-1]))):
        raise ValueError(f"{p}: days {first}..{last} don't match the export's")
    return External(path=p, v=v, meta=meta, info=info)


def categories(ext: External, exclude: tuple[str, ...] = ()) -> list[str]:
    return sorted({m["category"] for m in ext.meta if m["status"] == "headline" and m["category"] not in exclude})


def feature_names(cats: list[str]) -> tuple[str, ...]:
    return tuple(f"ext_{c}_{h}" for c in cats for h in WINDOWS)


def features(ext: External, cutoff: int, shift: int = 0, exclude: tuple[str, ...] = ()) -> tuple[tuple[str, ...], np.ndarray]:
    """(names, float32 [len(names), cutoff]): the category features on the development columns."""
    cats = categories(ext, exclude)
    v = np.asarray(ext.v[:, :cutoff], dtype=np.float32)
    if shift:
        v = np.roll(v, shift, axis=1)
    out = np.full((len(cats) * len(WINDOWS), cutoff), np.nan, dtype=np.float32)
    for ci, c in enumerate(cats):
        rows = [k for k, m in enumerate(ext.meta) if m["category"] == c and m["status"] == "headline"]
        x = v[rows]
        live = np.isfinite(x)
        n = live.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n >= MIN_LIVE, np.where(live, x, 0).sum(axis=0) / np.maximum(n, 1), np.nan)
        have = np.isfinite(mean)
        filled = np.where(have, mean, 0.0)
        for wi, h in enumerate(WINDOWS):
            s = np.convolve(filled, np.ones(h), mode="full")[:cutoff]  # s[d] = filled[d-h+1] + ... + filled[d]
            k = np.convolve(have.astype(np.float64), np.ones(h), mode="full")[:cutoff]
            out[ci * len(WINDOWS) + wi] = np.where(k > 0, s / np.sqrt(h), np.nan)
    return feature_names(cats), out


@dataclass
class WithExternal:
    """Stock features plus time-only outside features, served to LightGBM like `Features`."""

    base: Features
    ext: np.ndarray  # float32 [F, C]
    ext_names: tuple[str, ...]

    @property
    def complete(self) -> np.ndarray:
        return self.base.complete  # outside inputs may be missing; LightGBM handles that

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.base.names) + self.ext_names

    def rows(self, j: np.ndarray, d: np.ndarray) -> np.ndarray:
        x = self.base.rows(j, d)
        return np.concatenate([x, self.ext[:, np.asarray(d)].T], axis=1)
