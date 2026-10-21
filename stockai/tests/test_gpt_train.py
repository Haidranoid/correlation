"""Training uses only its split's training range; a planted signal is found and the sign-flip null finds nothing."""

import dataclasses

import numpy as np
import pytest
import torch

from stockai import data, splits
from stockai.gpt import features
from stockai.gpt.metrics import daily_rank_ic, ic_summary
from stockai.gpt.model import GPTConfig
from stockai.gpt.tokens import MISSING
from stockai.gpt.train import TrainConfig, lr_at, score_days, train_year
from stockai.gpt.windows import IGNORE, GpuPanel, TrainSet, check_batch
from stockai.synthetic import make_synthetic_export

DEV = "cuda" if torch.cuda.is_available() else "cpu"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    return data.load(make_synthetic_export(tmp_path_factory.mktemp("planted"), n=120, rho=-0.1, seed=0))


@pytest.fixture(scope="module")
def panel(export):
    return features.build_panel(export, device=DEV)


def test_training_positions_stay_in_range(panel):
    gp = GpuPanel(panel, DEV)
    for year in (2010, 2019):
        sp = splits.split(panel.days, year)
        tr = sp.train_decisions
        ts = TrainSet(panel, gp, sp, 256)
        assert ts.c0 + ts.tokens.shape[1] == tr.stop + 1  # nothing after the last training target day
        gen = torch.Generator(device=DEV).manual_seed(year)
        lo, hi, checked = 10**9, -1, 0
        for _ in range(200):
            tok, cont, y, rows, day = ts.sample(64, gen)
            checked += check_batch(panel, sp, y, rows, day)
            use = (y != IGNORE).cpu().numpy()
            d = day.cpu().numpy()
            lo, hi = min(lo, d[use].min()), max(hi, d[use].max())
            # inputs are the panel's tokens through tr.stop, and padding after it
            t, r = tok.cpu().numpy(), rows.cpu().numpy()[:, None].repeat(d.shape[1], 1)
            inside = (d >= 0) & (d <= tr.stop)
            assert np.array_equal(t[inside], panel.tokens[r[inside], d[inside]])
            assert (t[~inside] == MISSING).all()
        assert tr.start <= lo and hi < tr.stop and checked > 100_000
        assert lo < tr.start + 30 and hi > tr.stop - 30  # sampling reaches both ends of the range


def test_training_ignores_data_after_its_range(panel):
    sp = splits.split(panel.days, 2016)
    stop = sp.train_decisions.stop
    rng = np.random.default_rng(0)
    p2 = dataclasses.replace(panel, tokens=panel.tokens.copy(), logvol=panel.logvol.copy(), market=panel.market.copy())
    p2.tokens[:, stop + 1:] = rng.integers(0, 102, p2.tokens[:, stop + 1:].shape)
    p2.logvol[:, stop + 1:] = rng.normal(-3, 2, p2.logvol[:, stop + 1:].shape)
    p2.market[stop + 1:] = 4.0
    a, b = TrainSet(panel, GpuPanel(panel, DEV), sp, 256), TrainSet(p2, GpuPanel(p2, DEV), sp, 256)
    assert (a.mu, a.sd, a.n_valid) == (b.mu, b.sd, b.n_valid)
    np.testing.assert_array_equal(a.hist_logp, b.hist_logp)
    ga, gb = torch.Generator(device=DEV).manual_seed(5), torch.Generator(device=DEV).manual_seed(5)
    for _ in range(20):
        for x, y in zip(a.sample(64, ga), b.sample(64, gb)):
            assert torch.equal(x, y)


def test_lr_schedule():
    assert lr_at(0, 1000, 100, 1e-3, 1e-4) == pytest.approx(1e-5)
    assert lr_at(99, 1000, 100, 1e-3, 1e-4) == pytest.approx(1e-3)
    assert lr_at(1000, 1000, 100, 1e-3, 1e-4) == pytest.approx(1e-4)
    assert 1e-4 < lr_at(550, 1000, 100, 1e-3, 1e-4) < 1e-3


def test_null_flips_every_stock_the_same_way(export):
    p = features.build_panel(export, null="sign-flip", null_seed=3, device=DEV)
    cut = p.t
    flips = features.sign_flips(cut, 3)
    assert set(np.unique(flips)) == {-1.0, 1.0}
    np.testing.assert_array_equal(p.v, np.asarray(export.v[:, :cut]) * flips[None, :])
    assert p.null == "sign-flip" and p.null_seed == 3


@gpu
def test_same_seed_same_scores(panel):
    """Seeded per (seed, year) with deterministic kernels: a rerun reproduces the scores exactly."""
    gp = GpuPanel(panel, DEV)
    sp = splits.split(panel.days, 2013)
    cfg = TrainConfig(batch=32, max_steps=40, eval_every=20)
    mcfg = GPTConfig(n_layer=2, n_head=2, n_embd=64)
    a, b, c = (score_days(train_year(panel, gp, sp, mcfg, cfg, seed=s, log=lambda m: None), gp, panel,
                          sp.test_decisions, cfg) for s in (7, 7, 8))
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, c, equal_nan=True)
    assert not torch.are_deterministic_algorithms_enabled()  # restored after training


def _test_period_ic(res, gp, panel, sp, cfg):
    s = score_days(res, gp, panel, sp.test_decisions, cfg)
    j, k = np.nonzero(np.isfinite(s))
    d = np.asarray(sp.test_decisions)[k]
    return ic_summary(daily_rank_ic(d, s[j, k].astype(np.float64), panel.v[j, d + 1].astype(np.float64)))["mean"]


@gpu
@pytest.mark.parametrize("null", [None, "sign-flip"])
def test_planted_signal_and_null(export, null):
    """rho = -0.1 reversal is found (IC against next-day v ~0.1); with signs flipped per date it is not."""
    panel = features.build_panel(export, null=null, device=DEV)
    gp = GpuPanel(panel, DEV)
    sp = splits.split(panel.days, 2016)
    cfg = TrainConfig(batch=64, max_steps=300, eval_every=100)
    res = train_year(panel, gp, sp, GPTConfig(), cfg, seed=0, log=lambda m: None)
    info = res.info
    test_ic = _test_period_ic(res, gp, panel, sp, cfg)
    if null is None:
        assert info["val_ic_v"] > 0.05, info
        assert info["val_ic_actual"] > 0.02, info
        assert info["bits_gain"] > 0, info
        assert test_ic > 0.05
    else:
        assert abs(info["val_ic_v"]) < 0.03, info
        assert abs(info["val_ic_actual"]) < 0.03, info
        assert abs(test_ic) < 0.03
