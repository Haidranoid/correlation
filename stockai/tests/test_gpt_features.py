"""The per-day inputs against brute-force definitions, and their point-in-time property."""

import numpy as np
import pytest
import torch

from stockai.gpt import features

DEV = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(0)
    n, t = 40, 700
    close = (20 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, t)), axis=1))).astype(np.float32)
    close[rng.random((n, t)) < 0.05] = np.nan
    close[:5, :400] = np.nan  # late listings
    dv = (close * np.exp(rng.normal(np.log(4e5), 1.0, (n, 1)))).astype(np.float32)
    return close, dv


def test_trailing_log_vol(prices):
    close, _ = prices
    lv = features.trailing_log_vol(close, chunk=7)
    c = close.astype(np.float64)
    r = np.full(c.shape, np.nan)
    r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    for j in range(0, close.shape[0], 3):
        for d in range(0, close.shape[1], 11):
            x = r[j, max(0, d - 251):d + 1]  # the 252 days through d
            x = x[np.isfinite(x)]
            if x.size >= features.VOL_MIN_OBS:
                assert lv[j, d] == pytest.approx(np.log(x.std(ddof=1)), abs=1e-5)
            else:
                assert np.isnan(lv[j, d])


def test_liquid_mask(prices):
    close, dv = prices
    got = features.liquid_mask(close, dv, device=DEV, chunk=50)
    for j in range(close.shape[0]):
        for d in range(0, close.shape[1], 7):
            w = dv[j, max(0, d - 62):d + 1]
            w = np.sort(w[np.isfinite(w)])
            want = (w.size >= features.LIQ_MIN_OBS and w[(w.size - 1) // 2] >= 5e6 and close[j, d] >= 5.0)
            assert got[j, d] == want, (j, d)
    assert 0.05 < got.mean() < 0.95


def test_point_in_time(prices):
    close, dv = prices
    d = 450
    c2, v2 = close.copy(), dv.copy()
    rng = np.random.default_rng(1)
    c2[:, d + 1:] = np.exp(rng.normal(0, 3, c2[:, d + 1:].shape))
    v2[:, d + 1:] = np.exp(rng.normal(20, 3, v2[:, d + 1:].shape))
    np.testing.assert_array_equal(features.trailing_log_vol(c2)[:, :d + 1], features.trailing_log_vol(close)[:, :d + 1])
    np.testing.assert_array_equal(features.liquid_mask(c2, v2, device=DEV)[:, :d + 1],
                                  features.liquid_mask(close, dv, device=DEV)[:, :d + 1])
