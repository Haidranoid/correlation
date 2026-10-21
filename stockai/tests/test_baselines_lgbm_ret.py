"""LightGBM with the actual-return target (--target ret): its target is each day's rank-gauss score of the next-day
return, it finds a planted signal in returns, and it reads no outcome after its last validation day."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import ndtri
from scipy.stats import rankdata, spearmanr

from stockai import data, splits
from stockai.baselines import lgbm
from stockai.baselines.common import dev_cutoff, load_dev
from stockai.baselines.features import liquid_mask
from stockai.synthetic import make_synthetic_export

ROOT = Path(__file__).resolve().parents[1]
CFG = dataclasses.replace(lgbm.Config(**lgbm.QUICK), threads=4, target="ret")


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> Path:
    return make_synthetic_export(tmp_path_factory.mktemp("syn_ret") / "planted", n=120, rho=-0.1, seed=11)


@pytest.fixture(scope="module")
def planted_panel(planted):
    dev = load_dev(planted)
    return dev, lgbm.build_panel(dev, CFG)


def split_of(dev, year: int) -> splits.Split:
    return {s.year: s for s in splits.dev_splits(dev.export.days, horizon=1)}[year]


def test_rank_gauss_is_the_normal_quantile_of_each_days_rank():
    x = np.array([[0.3, 1.0], [-2.0, np.nan], [0.3, 5.0], [9.0, -1.0], [0.1, 2.0]])
    rows = np.array([[1, 1], [1, 1], [1, 0], [1, 1], [0, 1]], dtype=bool)
    g = lgbm.rank_gauss(x, rows)
    assert g.dtype == np.float32 and g.shape == x.shape
    # Day 0: rows 0..3 hold 0.3, -2, 0.3, 9: average ranks 2.5, 1, 2.5, 4 of 4 (tied values share theirs).
    assert np.array_equal(g[:4, 0], ndtri((np.array([2.5, 1, 2.5, 4]) - 0.5) / 4).astype(np.float32))
    assert np.isnan(g[4, 0])  # not one of the day's rows
    # Day 1: the missing value and the row left out don't count: 1, -1, 2 rank 2, 1, 3 of 3.
    assert np.array_equal(g[[0, 3, 4], 1], ndtri((np.array([2, 1, 3]) - 0.5) / 3).astype(np.float32))
    assert np.isnan(g[[1, 2], 1]).all()
    y = x.copy()
    y[:, 1] = [-50.0, 3.0, np.nan, 7.0, 0.0]
    assert np.array_equal(lgbm.rank_gauss(y, rows)[:, 0], g[:, 0], equal_nan=True)  # a day uses its own column only


def test_ret_target_is_the_next_day_return_ranked_within_each_day(planted_panel):
    dev, p = planted_panel
    assert p.target_name == "ret" and p.target.dtype == np.float32
    ret = data.next_day_returns(dev.close)
    assert np.array_equal(p.ret_next, ret, equal_nan=True)
    liquid = liquid_mask(dev.close, dev.dollar_volume, CFG.min_price, CFG.min_dollar_volume, CFG.liquidity_window)
    assert np.array_equal(p.trainable, liquid & p.feats.complete & np.isfinite(ret))
    assert np.array_equal(np.isfinite(p.target), p.trainable)  # a target exactly on the rows that can train
    assert np.isnan(p.target[:, -1]).all() and not p.trainable[:, -1].any()
    v = np.asarray(dev.v, dtype=np.float32)
    assert np.array_equal(p.v_next[:, :-1], v[:, 1:], equal_nan=True) and np.isnan(p.v_next[:, -1]).all()
    days = np.flatnonzero(p.trainable.sum(axis=0) >= 50)
    assert days.size > 4000
    for d in days[::97]:
        m = p.trainable[:, d]
        y = p.target[m, d]
        assert np.array_equal(y, ndtri((rankdata(ret[m, d]) - 0.5) / m.sum()).astype(np.float32))
        assert spearmanr(y, ret[m, d]).statistic == pytest.approx(1.0)
        assert abs(float(np.mean(y, dtype=np.float64))) < 1e-6 and 0.98 < float(np.std(y, dtype=np.float64)) < 1.0


def test_ret_target_finds_the_planted_reversal_in_returns(planted_panel):
    dev, panel = planted_panel
    rec = lgbm.fit_year(panel, split_of(dev, 2012), CFG).record
    assert rec["target"] == "ret"
    assert rec["train_rows"] > 50_000 and rec["val_rows"] > 15_000
    assert rec["val_ic_ret"]["days"] >= 240 and rec["val_ic_v"]["days"] >= 240
    assert rec["val_ic_ret"]["mean"] > 0.04 and rec["val_ic_ret"]["t"] > 5  # measured +0.082, t 12.3
    assert rec["val_ic_v"]["mean"] > 0.04 and rec["val_ic_v"]["t"] > 5  # the planted v drives the returns
    assert 0.9 < rec["val_l2_zero"] < 1.05  # about standard normal each day
    assert rec["val_l2"] < rec["val_l2_zero"]
    top = max(rec["gain_share"], key=rec["gain_share"].get)
    assert top in ("vsum_1", "ret_1")  # it found yesterday's move


def test_ret_target_finds_nothing_without_a_planted_signal(tmp_path):
    dev = load_dev(make_synthetic_export(tmp_path / "null", n=120, rho=0.0, seed=12))
    rec = lgbm.fit_year(lgbm.build_panel(dev, CFG), split_of(dev, 2012), CFG).record
    assert abs(rec["val_ic_ret"]["mean"]) < 0.03 and abs(rec["val_ic_v"]["mean"]) < 0.03


@pytest.mark.parametrize("year", [2010, 2017, 2023])
def test_test_year_scores_never_read_a_target(planted_panel, year):
    """Garbage targets, outcomes and training masks on every day outside the year's training and validation
    decisions leave the model and its test scores bit-identical. Those decisions' outcome days end before the test
    year, with the splits' gap."""
    dev, panel = planted_panel
    sp = split_of(dev, year)
    assert sp.train_decisions.stop < sp.val_decisions.start  # last training outcome: day train_decisions.stop
    assert sp.val_decisions.stop < sp.test_decisions.start - sp.gap  # last validation outcome: val_decisions.stop
    used = np.zeros(panel.days.size, dtype=bool)
    used[sp.train_decisions.start : sp.train_decisions.stop] = True
    used[sp.val_decisions.start : sp.val_decisions.stop] = True
    rng = np.random.default_rng(year)
    k = int((~used).sum())
    target, ret_next, v_next, trainable = (panel.target.copy(), panel.ret_next.copy(), panel.v_next.copy(),
                                           panel.trainable.copy())
    for a in (target, ret_next, v_next):
        a[:, ~used] = rng.normal(0.0, 50.0, (a.shape[0], k)).astype(a.dtype)
    trainable[:, ~used] = rng.random((trainable.shape[0], k)) < 0.5
    poisoned = dataclasses.replace(panel, target=target, ret_next=ret_next, v_next=v_next, trainable=trainable)
    a, b = lgbm.fit_year(panel, sp, CFG), lgbm.fit_year(poisoned, sp, CFG)
    assert a.booster.model_to_string() == b.booster.model_to_string()
    assert np.array_equal(a.j, b.j) and np.array_equal(a.d, b.d) and a.pred.tobytes() == b.pred.tobytes()
    assert a.d.min() >= sp.test_decisions.start and a.d.max() < sp.test_decisions.stop
    for key in ("train_rows", "val_rows", "best_iteration", "val_l2", "val_l2_zero", "val_ic_v", "val_ic_ret"):
        assert a.record[key] == b.record[key]


def test_ret_model_ignores_data_after_its_last_validation_outcome(planted, planted_panel, tmp_path):
    """Changing every input after the last validation outcome day leaves the fitted model bit-identical."""
    dev, panel = planted_panel
    sp = split_of(dev, 2014)
    last_outcome = sp.val_decisions.stop  # the last validation decision is stop - 1; its outcome day is stop
    assert last_outcome < sp.test_decisions.start - sp.gap
    mutated = tmp_path / "mutated"
    mutated.mkdir()
    for f in planted.iterdir():
        shutil.copy(f, mutated / f.name)
    rng = np.random.default_rng(6)
    for name in ("v.npy", "close.npy", "dollar_volume.npy"):
        a = np.load(mutated / name)
        a[:, last_outcome + 1 :] *= rng.uniform(0.2, 5.0, a[:, last_outcome + 1 :].shape).astype(np.float32)
        a[:, last_outcome + 1 :: 3] = np.nan
        np.save(mutated / name, a)
    m = np.load(mutated / "market.npy")
    m[last_outcome + 1 :] = rng.normal(0, 0.03, m.size - last_outcome - 1)
    np.save(mutated / "market.npy", m)
    a = lgbm.fit_year(panel, sp, CFG)
    b = lgbm.fit_year(lgbm.build_panel(load_dev(mutated), CFG), sp, CFG)
    assert a.booster.model_to_string() == b.booster.model_to_string()
    for key in ("train_rows", "val_rows", "best_iteration", "val_l2", "val_ic_v", "val_ic_ret"):
        assert a.record[key] == b.record[key]
    assert not np.array_equal(a.pred, b.pred)  # the test-period features did change


def test_ret_scores_cover_only_test_decisions_and_ignore_the_locked_year(tmp_path):
    src = make_synthetic_export(tmp_path / "export", n=60, seed=21)
    days = np.load(src / "days.npy")
    c = dev_cutoff(days)
    dst = tmp_path / "mutated"
    dst.mkdir()
    for f in src.iterdir():
        shutil.copy(f, dst / f.name)
    rng = np.random.default_rng(4)
    for name in ("v.npy", "close.npy", "dollar_volume.npy"):
        a = np.load(dst / name)
        tail = a[:, c:]
        kind = rng.integers(0, 4, tail.shape)
        for k, value in enumerate((np.nan, 0.0, -3.0, 1e7)):
            tail[kind == k] = value
        np.save(dst / name, a)
    m = np.load(dst / "market.npy")
    m[c:] = np.nan
    np.save(dst / "market.npy", m)
    sa, recs = lgbm.walk_forward(load_dev(src), [2023], CFG)
    sb, _ = lgbm.walk_forward(load_dev(dst), [2023], CFG)
    assert recs[0]["target"] == "ret"
    assert np.isfinite(sa).any() and sa.tobytes() == sb.tobytes()
    assert sa.shape == (60, days.size) and sa.dtype == np.float32 and np.isnan(sa[:, c:]).all()
    t = splits.split(days, 2023).test_decisions
    allowed = np.zeros(days.size, dtype=bool)
    allowed[t.start : t.stop] = True
    assert np.array_equal(np.isfinite(sa).any(axis=0), allowed)


def test_target_must_be_known_and_match_the_panel(planted_panel):
    dev, panel = planted_panel
    with pytest.raises(ValueError):
        lgbm.fit_year(panel, split_of(dev, 2012), dataclasses.replace(CFG, target="v"))
    with pytest.raises(ValueError):
        lgbm.build_panel(dev, dataclasses.replace(CFG, target="returns"))
    assert lgbm.Config().target == "v"  # the default stays the POC's target


def test_cli_target_ret_matches_the_library(planted, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "stockai.baselines.lgbm", "--export", str(planted), "--out", str(out),
                        "--years", "2011", "--quick", "--threads", "4", "--target", "ret"], cwd=ROOT, capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in lines] == ["config", "year", "summary"]
    assert lines[0]["config"]["target"] == "ret" and lines[1]["target"] == "ret"
    assert lines[1]["val_ic_ret"]["mean"] > 0.04
    scores, _ = lgbm.walk_forward(load_dev(planted), [2011], CFG)
    assert np.load(out / "scores.npy").tobytes() == scores.tobytes()
