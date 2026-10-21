"""reversal.py reproduces correlation2's reversal-family score bit for bit: the G0 cross-check (PLAN.md M0)."""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stockai import data
from stockai.baselines.common import dev_cutoff, load_dev
from stockai.baselines.reversal import libm_log, reversal_scores
from stockai.synthetic import make_synthetic_export

ROOT = Path(__file__).resolve().parents[1]
POS_ZERO = np.int64(0)
NEG_ZERO = np.float64(-0.0).view(np.int64)


def bits(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).view(np.int64)


def c_log(x: float) -> float:
    """The C library's log, which Rust's f64::ln calls: math.log calls it for x > 0 (and raises otherwise)."""
    return math.log(x) if x > 0 else (-math.inf if x == 0 else math.nan)


def rust_reversal(close: np.ndarray, lookback: int) -> np.ndarray:
    """A line-by-line transcription of `reversal_scores` in correlation2's holdout.rs (commit 9335126):

        match d.checked_sub(lookback) {
            Some(a) => { let (p0, p1) = (close[a], close[d]);
                         if p0.is_finite() && p1.is_finite() && p0 > 0.0 { -((p1 as f64 / p0 as f64).ln()) } else { 0.0 } }
            None => 0.0 }
    """
    n, t = close.shape
    out = np.empty((n, t))
    for j in range(n):
        row = close[j].tolist()  # float32 -> Python float is exact, like `as f64`
        for d in range(t):
            if d < lookback:
                out[j, d] = 0.0
                continue
            p0, p1 = row[d - lookback], row[d]
            if math.isfinite(p0) and math.isfinite(p1) and p0 > 0.0:
                out[j, d] = -c_log(p1 / p0)
            else:
                out[j, d] = 0.0
    return out


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory) -> Path:
    return make_synthetic_export(tmp_path_factory.mktemp("syn") / "export", n=60, missing=0.05, seed=4)


@pytest.mark.parametrize("lookback", [5, 21])
def test_matches_correlation2_formula_on_synthetic_data(export_dir, lookback):
    dev = load_dev(export_dir)
    close = np.array(dev.close)
    assert np.isnan(close).mean() > 0.04  # missing closes exercise the 0.0 fallback
    got = reversal_scores(close, lookback)
    want = rust_reversal(close, lookback)
    assert got.dtype == np.float64 and got.shape == close.shape
    assert np.array_equal(bits(got), bits(want))  # bitwise: -0.0 and +0.0 differ
    missing = ~np.isfinite(close)
    fallback = np.zeros(close.shape, dtype=bool)
    fallback[:, :lookback] = True
    fallback[:, lookback:] |= missing[:, lookback:] | missing[:, :-lookback]
    assert fallback.sum() > 0
    assert np.all(bits(got)[fallback] == POS_ZERO)


def test_zero_cases_and_signs():
    nan, inf = np.nan, np.inf
    close = np.array(
        [
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],  # normal
            [10.0, nan, 12.0, 13.0, 14.0, 15.0],  # close[d - lookback] missing at d = 3
            [10.0, 11.0, 12.0, 13.0, nan, 15.0],  # close[d] missing at d = 4
            [10.0, 0.0, 12.0, 13.0, 14.0, 15.0],  # close[d - lookback] = 0 at d = 3
            [10.0, -2.0, 12.0, 13.0, 14.0, 15.0],  # close[d - lookback] < 0 at d = 3
            [10.0, inf, 12.0, 13.0, -inf, 15.0],  # infinite closes
            [10.0, 11.0, 12.0, 11.0, 14.0, 15.0],  # unchanged price at d = 3: -0.0
            [10.0, 11.0, 12.0, 13.0, 0.0, 15.0],  # close[d] = 0 at d = 4: -(ln 0) = +inf, as in Rust
        ],
        dtype=np.float32,
    )
    s = reversal_scores(close, 2)
    b = bits(s)
    assert np.array_equal(b, bits(rust_reversal(close, 2)))
    assert np.all(b[:, :2] == POS_ZERO)  # d < lookback
    assert s[0, 2] == -math.log(12.0 / 10.0) and s[0, 5] == -math.log(15.0 / 13.0)
    assert b[1, 3] == POS_ZERO and s[1, 2] == -math.log(12.0 / 10.0)
    assert b[2, 4] == POS_ZERO
    assert b[3, 3] == POS_ZERO
    assert b[4, 3] == POS_ZERO
    assert b[5, 3] == POS_ZERO and b[5, 4] == POS_ZERO
    assert b[6, 3] == NEG_ZERO  # -(ln 1) is negative zero, which total_cmp orders below +0.0
    assert s[7, 4] == np.inf
    with pytest.raises(TypeError):
        reversal_scores(close.astype(np.float64), 2)  # exactness needs the export's float32 closes


def test_libm_log_matches_math_log():
    x = np.array([1.0, 2.0, 1e-300, 1e300, 0.999999999, 1.000000001, np.inf, 0.0, -1.0, np.nan])
    y = libm_log(x)
    for xi, yi in zip(x[:7], y[:7]):
        assert np.float64(yi).view(np.int64) == np.float64(math.log(xi)).view(np.int64)
    assert y[7] == -np.inf and np.isnan(y[8]) and np.isnan(y[9])


def test_cli_writes_nan_where_correlation2_would_score_infinity(export_dir, tmp_path):
    """The judge refuses infinite scores, so a zero close (+inf in correlation2) is written as NaN, and named."""
    bad = tmp_path / "export"
    shutil.copytree(export_dir, bad)
    close = np.load(bad / "close.npy")
    j, d = 3, 2000
    close[j, d - 5 : d + 1] = [20.0, 21.0, 22.0, 23.0, 24.0, 0.0]
    close[j + 1, d - 5] = close[j + 1, d]  # an unchanged price: -0.0 must survive the file
    np.save(bad / "close.npy", close)
    out = tmp_path / "rev5.npy"
    r = subprocess.run([sys.executable, "-m", "stockai.baselines.reversal", "--export", str(bad), "--lookback", "5",
                        "--out", str(out)], check=True, cwd=ROOT, capture_output=True, text=True)
    assert "SYN0003" in r.stderr
    s = np.load(out)
    assert not np.isinf(s).any() and np.isnan(s[j, d])
    c = dev_cutoff(np.load(bad / "days.npy"))
    want = rust_reversal(close[:, :c], 5)
    want[j, d] = np.nan
    assert np.array_equal(bits(s[:, :c]), bits(want))
    assert bits(s)[j + 1, d] == NEG_ZERO


def test_cli_writes_float64_scores_with_locked_year_nan(export_dir, tmp_path):
    out = tmp_path / "rev5.npy"
    subprocess.run([sys.executable, "-m", "stockai.baselines.reversal", "--export", str(export_dir), "--lookback", "5",
                    "--out", str(out)], check=True, cwd=ROOT, capture_output=True)
    ex = data.load(export_dir)
    s = np.load(out)
    assert s.dtype == np.float64 and s.shape == (ex.n, ex.t) and s.flags.c_contiguous
    c = dev_cutoff(ex.days)
    assert ex.date(c) < "2024-06-28" <= ex.date(c + 1)  # the cutoff is the locked year's first decision day
    assert np.array_equal(bits(s[:, :c]), bits(rust_reversal(np.array(ex.close[:, :c]), 5)))
    assert np.isnan(s[:, c:]).all()
