"""No baseline reads or scores the locked year (2024-06-28 .. 2025-06-27) or anything later."""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stockai import data, splits
from stockai.baselines import features as F
from stockai.baselines import lgbm
from stockai.baselines.common import dev_cutoff, load_dev
from stockai.baselines.momentum import momentum_scores
from stockai.baselines.reversal import reversal_scores
from stockai.synthetic import make_synthetic_export

ROOT = Path(__file__).resolve().parents[1]
CFG = dataclasses.replace(lgbm.Config(**lgbm.QUICK), threads=4)


@pytest.fixture(scope="module")
def exports(tmp_path_factory):
    """A synthetic export, and a copy whose every input from the cutoff on is changed adversarially."""
    root = tmp_path_factory.mktemp("locked")
    src = make_synthetic_export(root / "export", n=60, seed=21)
    days = np.load(src / "days.npy")
    c = dev_cutoff(days)
    dst = root / "mutated"
    dst.mkdir()
    for f in src.iterdir():
        shutil.copy(f, dst / f.name)
    rng = np.random.default_rng(3)
    for name in ("v.npy", "close.npy", "dollar_volume.npy"):
        a = np.load(dst / name)
        tail = a[:, c:]
        kind = rng.integers(0, 4, tail.shape)
        tail[kind == 0] = np.nan
        tail[kind == 1] = 0.0
        tail[kind == 2] = -3.0
        tail[kind == 3] = 1e7
        np.save(dst / name, a)
    m = np.load(dst / "market.npy")
    m[c:] = np.nan
    np.save(dst / "market.npy", m)
    return src, dst, c


def test_cutoff_is_the_locked_years_first_decision(exports):
    src, _, c = exports
    days = np.load(src / "days.npy")
    assert splits.split(days, 2023).test_decisions.stop == c
    assert splits.locked_split(days).test_decisions.start == c
    assert data.day_to_str(days[c + 1]) == "2024-06-28"  # synthetic days are business days
    dev = load_dev(src)
    assert dev.v.shape[1] == dev.close.shape[1] == dev.dollar_volume.shape[1] == dev.market.shape[0] == c


def test_changing_the_locked_period_changes_nothing(exports):
    src, dst, c = exports
    a, b = load_dev(src), load_dev(dst)
    assert np.array_equal(reversal_scores(np.asarray(a.close), 5).view(np.int64),
                          reversal_scores(np.asarray(b.close), 5).view(np.int64))
    assert np.array_equal(momentum_scores(a.close), momentum_scores(b.close), equal_nan=True)
    fa = F.compute_features(a.v, a.close, a.dollar_volume, a.market)
    fb = F.compute_features(b.v, b.close, b.dollar_volume, b.market)
    assert fa.stock.tobytes() == fb.stock.tobytes() and fa.market.tobytes() == fb.market.tobytes()
    sa, _ = lgbm.walk_forward(a, [2023], CFG)
    sb, _ = lgbm.walk_forward(b, [2023], CFG)
    assert np.isfinite(sa).any() and sa.tobytes() == sb.tobytes()
    assert np.isnan(sa[:, c:]).all()


@pytest.mark.parametrize("module,args", [("reversal", ["--lookback", "5"]), ("momentum", [])])
def test_rule_scores_are_nan_from_the_cutoff_on(exports, tmp_path, module, args):
    src, _, c = exports
    out = tmp_path / f"{module}.npy"
    subprocess.run([sys.executable, "-m", f"stockai.baselines.{module}", "--export", str(src), *args, "--out",
                    str(out)], check=True, cwd=ROOT, capture_output=True)
    s = np.load(out)
    assert s.dtype == np.float64 and s.shape == (60, np.load(src / "days.npy").size)
    assert np.isnan(s[:, c:]).all() and np.isfinite(s[:, c - 1]).any()
