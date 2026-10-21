"""A small fake `c2 export` (docs/c2-export.md) with a planted signal, for tests and the planted-signal control.

The census variable follows v[d] = rho * v[d-1] + noise for every stock, so a model that works should find
`rho` (rho < 0 plants short-term reversal). Volatility clusters, some values are missing, and prices follow from
the residuals plus the market, so returns and the variable stay consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import FORMAT, str_to_day


def business_days(start: str, end: str) -> np.ndarray:
    d0, d1 = str_to_day(start), str_to_day(end)
    days = np.arange(d0, d1 + 1, dtype=np.int64)
    weekday = (days + 3) % 7  # 1970-01-01 was a Thursday: 0 = Monday
    return days[weekday < 5].astype(np.int32)


def make_synthetic_export(path: str | Path, n: int = 120, start: str = "2004-01-02", end: str = "2025-06-27",
                          rho: float = -0.1, missing: float = 0.02, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    days = business_days(start, end)
    t = len(days)

    market = rng.normal(0.0004, 0.01, t)
    sd = np.exp(rng.normal(np.log(0.02), 0.4, n))[:, None]  # each stock's typical residual move
    vol = np.empty((n, t))
    vol[:, 0] = 1.0
    shocks = rng.normal(0, 1, (n, t))
    for d in range(1, t):  # slow volatility clustering in units of the stock's own sd
        vol[:, d] = np.exp(0.98 * np.log(vol[:, d - 1]) + 0.05 * rng.normal(0, 1, n))
    z = np.empty((n, t))
    z[:, 0] = shocks[:, 0]
    for d in range(1, t):
        z[:, d] = rho * z[:, d - 1] + np.sqrt(1 - rho * rho) * shocks[:, d]
    v = np.clip(z, -5, 5)
    beta = rng.uniform(0.5, 1.5, (n, 1))
    log_ret = beta * market[None, :] + v * sd * vol
    close = 20.0 * np.exp(np.cumsum(log_ret, axis=1))
    gone = rng.random((n, t)) < missing
    v = v.astype(np.float32)
    close = close.astype(np.float32)
    v[gone] = np.nan
    close[gone] = np.nan
    v[:, :252] = np.nan  # the real variable needs a year of history first
    dollar_volume = (close * np.exp(rng.normal(np.log(5e5), 0.5, (n, 1)))).astype(np.float32)

    for name, arr in [("v.npy", v), ("close.npy", close), ("dollar_volume.npy", dollar_volume)]:
        np.save(p / name, np.ascontiguousarray(arr))
    np.save(p / "days.npy", days)
    np.save(p / "market.npy", market.astype(np.float64))
    with open(p / "symbols.tsv", "w") as f:
        f.write("col\tid\tticker\n")
        for j in range(n):
            f.write(f"{j}\t{j}\tSYN{j:04d}\n")
    meta = {
        "format": FORMAT,
        "n": n,
        "t": t,
        "first_day": start,
        "last_day": str(np.datetime64("1970-01-01", "D") + np.timedelta64(int(days[-1]), "D")),
        "fingerprint": "synthetic",
        "git_commit": "synthetic",
        "variable": f"synthetic AR(1) census variable, rho={rho}, seed={seed}",
    }
    (p / "meta.json").write_text(json.dumps(meta, indent=2))
    return p
