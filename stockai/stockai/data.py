"""Loads correlation2's export (`stockai-judge export`, see docs/c2-export.md) as memory-mapped arrays.

Every per-stock array is float32 [N, T]: row j = stock j, column d = trading day d (`days`).
Values at day d use data through the close of day d, with two exceptions (docs/c2-export.md, "Point in time"):
correlation2's bad-print repair decides whether close[d] exists by looking at close[d+1], and closes are Stooq's
split- and dividend-adjusted closes as of the download, so price levels (unlike returns) reflect later splits and
dividends, and dollar volume reflects later dividends.

Exports built from Sharadar (docs/sharadar.md) also carry as-traded closes and each stock's delisting, loaded as the
optional `close_unadj`, `delist_day` and `delist_kind` (None for a Stooq export).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT = "c2-export-1"
DELIST_KINDS = {0: "listed", 1: "buyout", 2: "failure", 3: "unknown"}  # delist_kind.npy codes (docs/sharadar.md)


def default_export_dir() -> Path:
    base = os.environ.get("C2_DATA_DIR")
    root = Path(base) if base else Path.home() / ".local/share/correlation2"
    return root / "export" / "full"


def str_to_day(s: str) -> int:
    """Days since 1970-01-01 for YYYY-MM-DD."""
    return int((np.datetime64(s, "D") - np.datetime64("1970-01-01", "D")).astype(np.int64))


def day_to_str(day: int) -> str:
    return str(np.datetime64("1970-01-01", "D") + np.timedelta64(int(day), "D"))


@dataclass
class Export:
    path: Path
    meta: dict
    days: np.ndarray  # int32 [T]
    store_ids: np.ndarray  # int64 [N]
    tickers: list[str]  # [N]
    v: np.ndarray  # float32 [N, T]; NaN = unavailable
    close: np.ndarray  # float32 [N, T]; NaN = missing
    dollar_volume: np.ndarray  # float32 [N, T]
    market: np.ndarray  # float64 [T]
    close_unadj: np.ndarray | None = None  # float32 [N, T]: as-traded closes (Sharadar exports only)
    delist_day: np.ndarray | None = None  # int32 [N]: column of a delisted stock's last close, else -1
    delist_kind: np.ndarray | None = None  # int8 [N]: DELIST_KINDS

    @property
    def n(self) -> int:
        return self.v.shape[0]

    @property
    def t(self) -> int:
        return self.v.shape[1]

    def index_of(self, date: str) -> int:
        """Index of the first trading day on or after `date` (len(days) if none)."""
        return int(np.searchsorted(self.days, str_to_day(date), side="left"))

    def date(self, idx: int) -> str:
        return day_to_str(int(self.days[idx]))

    def column(self, ticker: str) -> int:
        return self.tickers.index(ticker)


def load(path: str | Path | None = None) -> Export:
    p = Path(path) if path else default_export_dir()
    meta = json.loads((p / "meta.json").read_text())
    if meta.get("format") != FORMAT:
        raise ValueError(f"{p}: format {meta.get('format')!r}, expected {FORMAT!r}")
    days = np.load(p / "days.npy")
    store_ids, tickers = [], []
    with open(p / "symbols.tsv") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["col", "id", "ticker"]:
            raise ValueError(f"{p}/symbols.tsv: unexpected header {header}")
        for k, line in enumerate(f):
            col, sid, ticker = line.rstrip("\n").split("\t")
            if int(col) != k:
                raise ValueError(f"{p}/symbols.tsv: column {col} on line {k + 2}")
            store_ids.append(int(sid))
            tickers.append(ticker)

    def mat(name: str, dtype) -> np.ndarray:
        a = np.load(p / name, mmap_mode="r")
        if a.dtype != dtype or a.shape != (len(tickers), len(days)):
            raise ValueError(f"{p}/{name}: {a.dtype} {a.shape}, expected {np.dtype(dtype)} {(len(tickers), len(days))}")
        return a

    market = np.load(p / "market.npy")
    if market.shape != days.shape:
        raise ValueError(f"{p}/market.npy: shape {market.shape}, expected {days.shape}")
    if (meta["n"], meta["t"]) != (len(tickers), len(days)):
        raise ValueError(f"{p}/meta.json: n, t = {meta['n']}, {meta['t']} but files have {len(tickers)}, {len(days)}")

    def per_stock(name: str, dtype) -> np.ndarray | None:
        if not (p / name).exists():
            return None
        a = np.load(p / name)
        if a.dtype != dtype or a.shape != (len(tickers),):
            raise ValueError(f"{p}/{name}: {a.dtype} {a.shape}, expected {np.dtype(dtype)} {(len(tickers),)}")
        return a

    delist_day, delist_kind = per_stock("delist_day.npy", np.int32), per_stock("delist_kind.npy", np.int8)
    if (delist_day is None) != (delist_kind is None):
        raise ValueError(f"{p}: delist_day.npy and delist_kind.npy come as a pair")
    if delist_day is not None:
        bad = ((delist_day < -1) | (delist_day >= len(days)) | ((delist_day >= 0) != (delist_kind > 0))
               | ~np.isin(delist_kind, list(DELIST_KINDS)))
        if bad.any():
            raise ValueError(f"{p}: inconsistent delisting for column(s) {np.flatnonzero(bad)[:5].tolist()}")
    return Export(
        path=p,
        meta=meta,
        days=days,
        store_ids=np.asarray(store_ids, dtype=np.int64),
        tickers=tickers,
        v=mat("v.npy", np.float32),
        close=mat("close.npy", np.float32),
        dollar_volume=mat("dollar_volume.npy", np.float32),
        market=market,
        close_unadj=mat("close_unadj.npy", np.float32) if (p / "close_unadj.npy").exists() else None,
        delist_day=delist_day,
        delist_kind=delist_kind,
    )


def next_day_returns(close: np.ndarray) -> np.ndarray:
    """r[:, d] = close[:, d+1] / close[:, d] - 1: the outcome of deciding at day d and holding over d+1.

    NaN where either close is missing, and in the last column.
    """
    c = np.asarray(close, dtype=np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[:, :-1] = c[:, 1:] / c[:, :-1] - 1.0
    r[~np.isfinite(r)] = np.nan
    return r


def empty_scores(export: Export, dtype=np.float32) -> np.ndarray:
    return np.full((export.n, export.t), np.nan, dtype=dtype)


def write_scores(path: str | Path, scores: np.ndarray, export: Export) -> Path:
    """Writes a scores file for `c2 census holdout --family scores` (docs/c2-export.md)."""
    s = np.ascontiguousarray(scores)
    if s.shape != (export.n, export.t):
        raise ValueError(f"scores shape {s.shape}, expected {(export.n, export.t)}")
    if s.dtype not in (np.float32, np.float64):
        raise ValueError(f"scores dtype {s.dtype}, expected float32 or float64")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, s)
    return p
