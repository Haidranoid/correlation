"""12-1 momentum: ln(close[d-21] / close[d-252]), NaN when either close is missing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from stockai.baselines.common import load_dev
from stockai.baselines.momentum import momentum_scores
from stockai.synthetic import make_synthetic_export


def test_matches_formula_on_synthetic_data(tmp_path):
    dev = load_dev(make_synthetic_export(tmp_path / "export", n=30, missing=0.05, seed=2))
    c = np.asarray(dev.close, dtype=np.float64)
    s = momentum_scores(dev.close)
    assert s.dtype == np.float64 and s.shape == c.shape
    assert np.isnan(s[:, :252]).all()
    rng = np.random.default_rng(0)
    for j, d in zip(rng.integers(0, c.shape[0], 2000), rng.integers(252, c.shape[1], 2000)):
        a, b = c[j, d - 252], c[j, d - 21]
        if math.isfinite(a) and math.isfinite(b):
            assert s[j, d] == pytest.approx(math.log(b / a), rel=1e-12, abs=1e-15)
        else:
            assert math.isnan(s[j, d])
    missing = np.isnan(c)
    expect_nan = np.ones(c.shape, dtype=bool)
    expect_nan[:, 252:] = missing[:, 231:-21] | missing[:, :-252]
    assert np.array_equal(np.isnan(s), expect_nan)


def test_non_positive_closes_and_options():
    close = np.array([[1.0, 2.0, 0.0, 4.0, -1.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    s = momentum_scores(close, skip=1, lookback=3)[0]  # s[d] = ln(close[d-1] / close[d-3])
    assert np.isnan(s[:3]).all()  # d < lookback
    assert np.isnan(s[3])  # close[2] = 0
    assert s[4] == pytest.approx(math.log(4.0 / 2.0))
    assert np.isnan(s[5])  # close[4] < 0 and close[2] = 0
    assert s[6] == pytest.approx(math.log(6.0 / 4.0))
    assert np.isnan(s[7])  # close[4] < 0
    with pytest.raises(ValueError):
        momentum_scores(close, skip=3, lookback=3)
