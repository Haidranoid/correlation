import numpy as np
import pytest

from stockai.gpt import tokens
from stockai.gpt.tokens import CENTERS, MISSING, N_BINS, VOCAB, decode, encode


def test_vocabulary():
    assert (N_BINS, MISSING, VOCAB) == (101, 101, 102)
    assert CENTERS.shape == (101,)
    np.testing.assert_allclose(CENTERS[[0, 50, 100]], [-5.0, 0.0, 5.0])
    np.testing.assert_allclose(np.diff(CENTERS), 0.1, atol=1e-6)


def test_known_values():
    v = np.array([0.0, 0.04, -0.04, 0.06, -0.06, 1.23, -2.71, 4.97, 5.0, -5.0, 7.5, -100.0], dtype=np.float32)
    assert encode(v).tolist() == [50, 50, 50, 51, 49, 62, 23, 100, 100, 0, 100, 0]
    assert encode(v).dtype == np.uint8


def test_round_trip():
    rng = np.random.default_rng(0)
    v = rng.uniform(-5.0, 5.0, 100_000).astype(np.float32)
    t = encode(v)
    assert t.min() >= 0 and t.max() <= 100
    assert np.abs(decode(t) - v).max() <= tokens.BIN_WIDTH / 2 + 1e-5
    every = np.arange(N_BINS)
    assert np.array_equal(encode(decode(every)), every)  # each center maps back to its own bin
    # the bin formula from PLAN.md §3.2
    assert np.array_equal(t, (np.clip(np.round(v.astype(np.float64) / 0.1), -50, 50) + 50).astype(np.uint8))


def test_missing():
    v = np.array([[np.nan, 0.3], [np.inf, -np.inf]], dtype=np.float32)
    t = encode(v)
    assert t.tolist() == [[MISSING, 53], [MISSING, MISSING]]
    d = decode(t)
    assert np.isnan(d[0, 0]) and np.isnan(d[1]).all() and d[0, 1] == pytest.approx(0.3)
    with pytest.raises(ValueError):
        decode(np.array([MISSING + 1]))
