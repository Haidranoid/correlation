"""One-day census reversal: -v[d], NaN where v is missing, nothing from the locked year's first decision on."""

from __future__ import annotations

import numpy as np

from stockai.baselines import vreversal
from stockai.baselines.common import dev_cutoff, load_dev
from stockai.synthetic import make_synthetic_export


def test_is_minus_v_and_point_in_time(tmp_path):
    dev = load_dev(make_synthetic_export(tmp_path / "export", n=20, missing=0.05, seed=4))
    v = np.asarray(dev.v)
    s = vreversal.vreversal_scores(dev.v)
    assert s.dtype == np.float32 and s.shape == v.shape
    assert np.array_equal(np.isnan(s), np.isnan(v))
    ok = np.isfinite(v)
    assert np.array_equal(s[ok], -v[ok])
    w = v.copy()
    w[:, 500:] = np.nan  # changing data after day 499 leaves scores up to 499 unchanged
    assert np.array_equal(vreversal.vreversal_scores(w)[:, :500], s[:, :500], equal_nan=True)


def test_cli_writes_nan_from_the_cutoff(tmp_path):
    src = make_synthetic_export(tmp_path / "export", n=10, seed=5)
    out = tmp_path / "vrev.npy"
    assert vreversal.main(["--export", str(src), "--out", str(out)]) == 0
    s = np.load(out)
    c = dev_cutoff(np.load(src / "days.npy"))
    assert s.shape[1] == np.load(src / "v.npy").shape[1]
    assert np.isnan(s[:, c:]).all() and np.isfinite(s[:, :c]).any()
