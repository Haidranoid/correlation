"""The LightGBM walk-forward finds a planted signal, scores only test decisions, and never trains on later data."""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from stockai import data, splits
from stockai.baselines import lgbm
from stockai.baselines.common import dev_cutoff, load_dev, parse_years
from stockai.synthetic import make_synthetic_export

ROOT = Path(__file__).resolve().parents[1]
CFG = dataclasses.replace(lgbm.Config(**lgbm.QUICK), threads=4)


@pytest.fixture(scope="module")
def planted(tmp_path_factory) -> Path:
    return make_synthetic_export(tmp_path_factory.mktemp("syn") / "planted", n=120, rho=-0.1, seed=11)


@pytest.fixture(scope="module")
def planted_panel(planted):
    dev = load_dev(planted)
    return dev, lgbm.build_panel(dev, CFG)


def split_of(dev, year: int) -> splits.Split:
    return {s.year: s for s in splits.dev_splits(dev.export.days, horizon=1)}[year]


def test_planted_reversal_gives_clearly_positive_validation_ic(planted_panel):
    dev, panel = planted_panel
    rec = lgbm.fit_year(panel, split_of(dev, 2012), CFG).record
    assert rec["train_rows"] > 50_000 and rec["val_rows"] > 15_000  # the liquidity filter keeps most stocks
    assert rec["val_ic_v"]["days"] >= 240
    assert rec["val_ic_v"]["mean"] > 0.05 and rec["val_ic_v"]["t"] > 5
    assert rec["val_ic_ret"]["mean"] > 0.03 and rec["val_ic_ret"]["t"] > 3
    assert rec["val_l2"] < rec["val_l2_zero"]
    top = max(rec["gain_share"], key=rec["gain_share"].get)
    assert top in ("vsum_1", "ret_1")  # it found yesterday's move


def test_no_planted_signal_gives_validation_ic_near_zero(tmp_path):
    dev = load_dev(make_synthetic_export(tmp_path / "null", n=120, rho=0.0, seed=12))
    rec = lgbm.fit_year(lgbm.build_panel(dev, CFG), split_of(dev, 2012), CFG).record
    assert abs(rec["val_ic_v"]["mean"]) < 0.03


def test_walk_forward_writes_only_test_decisions(planted_panel):
    dev, _ = planted_panel
    by_year = {s.year: s for s in splits.dev_splits(dev.export.days, horizon=1)}
    alone, _ = lgbm.walk_forward(dev, [2011], CFG)
    both, recs = lgbm.walk_forward(dev, [2010, 2011], CFG)
    assert [r["year"] for r in recs] == [2010, 2011]
    for scores, years in ((alone, [2011]), (both, [2010, 2011])):
        allowed = np.zeros(dev.export.t, dtype=bool)
        for y in years:
            allowed[by_year[y].test_decisions.start : by_year[y].test_decisions.stop] = True
        written = np.isfinite(scores).any(axis=0)
        assert np.array_equal(written, allowed)  # every test decision scored, nothing else
        assert scores.shape == (dev.export.n, dev.export.t) and scores.dtype == np.float32
        assert np.isnan(scores[:, dev_cutoff(dev.export.days) :]).all()
    t11 = by_year[2011].test_decisions
    assert np.array_equal(alone[:, t11.start : t11.stop], both[:, t11.start : t11.stop], equal_nan=True)


def test_test_rows_are_every_stock_with_complete_features(planted_panel):
    dev, panel = planted_panel
    sp = split_of(dev, 2013)
    res = lgbm.fit_year(panel, sp, CFG)
    cols = np.arange(sp.test_decisions.start, sp.test_decisions.stop)
    scored = np.zeros((dev.export.n, cols.size), dtype=bool)
    scored[res.j, res.d - cols[0]] = True
    assert np.array_equal(scored, panel.feats.complete[:, cols])
    assert np.isfinite(res.pred).all()


def test_model_ignores_data_after_its_last_validation_outcome(planted, planted_panel, tmp_path):
    """Changing every input after the last validation outcome day leaves the fitted model bit-identical."""
    dev, panel = planted_panel
    sp = split_of(dev, 2014)
    last_outcome = sp.val_decisions.stop  # the last validation decision is stop - 1; its outcome day is stop
    assert last_outcome < sp.test_decisions.start - sp.gap
    mutated = tmp_path / "mutated"
    mutated.mkdir()
    for f in planted.iterdir():
        shutil.copy(f, mutated / f.name)
    rng = np.random.default_rng(5)
    for name in ("v.npy", "close.npy", "dollar_volume.npy"):
        a = np.load(mutated / name)
        a[:, last_outcome + 1 :] *= rng.uniform(0.2, 5.0, a[:, last_outcome + 1 :].shape).astype(np.float32)
        a[:, last_outcome + 1 :: 3] = np.nan
        np.save(mutated / name, a)
    m = np.load(mutated / "market.npy")
    m[last_outcome + 1 :] = rng.normal(0, 0.03, m.size - last_outcome - 1)
    np.save(mutated / "market.npy", m)
    dev2 = load_dev(mutated)
    a = lgbm.fit_year(panel, sp, CFG)
    b = lgbm.fit_year(lgbm.build_panel(dev2, CFG), sp, CFG)
    assert a.booster.model_to_string() == b.booster.model_to_string()
    for key in ("train_rows", "val_rows", "best_iteration", "val_l2", "val_ic_v", "val_ic_ret"):
        assert a.record[key] == b.record[key]
    assert not np.array_equal(a.pred, b.pred)  # the test-period features did change


def test_daily_ic_is_spearman_per_day():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(5), 50)
    pred, out = rng.normal(size=250), rng.normal(size=250)
    out[3] = np.nan
    ics = lgbm.daily_ic(pred, out, day, min_n=20)
    for k in range(5):
        sel = (day == k) & np.isfinite(out)
        assert ics[k] == pytest.approx(spearmanr(pred[sel], out[sel]).statistic)
    assert lgbm.daily_ic(np.ones(50), out[:50], day[:50], 20)[0] == 0.0  # constant predictions count as 0
    assert lgbm.daily_ic(pred[:10], out[:10], day[:10], 20).size == 0


def test_parse_years_refuses_the_locked_year():
    assert parse_years("2010-2012,2015") == [2010, 2011, 2012, 2015]
    assert parse_years(None) == list(splits.DEV_YEARS)
    for bad in ("2024", "2023-2024", "2009", "2030"):
        with pytest.raises(ValueError):
            parse_years(bad)


def test_cli_writes_scores_and_train_log(planted, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "stockai.baselines.lgbm", "--export", str(planted), "--out", str(out),
                        "--years", "2010-2011", "--quick", "--threads", "4"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ex = data.load(planted)
    s = np.load(out / "scores.npy")
    assert s.shape == (ex.n, ex.t) and s.dtype == np.float32
    lines = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in lines] == ["config", "year", "year", "summary"]
    assert lines[0]["features"] == list(lgbm.FEATURES) and lines[0]["quick"] is True
    assert {x["year"] for x in lines[1:3]} == {2010, 2011}
    assert all(x["val_ic_v"]["mean"] > 0.05 for x in lines[1:3])
    assert not (out / "scores.partial.npy").exists()
