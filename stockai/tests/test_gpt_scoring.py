"""No look-ahead: scores and validation metrics at day d depend on nothing after d (after the outcomes, for metrics)."""

import dataclasses

import numpy as np
import pytest
import torch

from stockai import data, splits
from stockai.gpt import features
from stockai.gpt.model import GPTConfig
from stockai.gpt.train import Pairs, TrainConfig, evaluate, score_days, train_year
from stockai.gpt.windows import GpuPanel, TrainSet
from stockai.synthetic import make_synthetic_export

DEV = "cuda" if torch.cuda.is_available() else "cpu"
YEAR = 2016


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    ex = data.load(make_synthetic_export(tmp_path_factory.mktemp("syn"), n=60, seed=3))
    panel = features.build_panel(ex, device=DEV)
    gp = GpuPanel(panel, DEV)
    sp = splits.split(panel.days, YEAR)
    cfg = TrainConfig(batch=32, max_steps=60, eval_every=60)
    res = train_year(panel, gp, sp, GPTConfig(n_layer=2, n_head=2, n_embd=64), cfg, seed=0, log=lambda m: None)
    return ex, panel, gp, sp, cfg, res


def altered_after(ex: data.Export, d: int, seed: int) -> data.Export:
    """A copy of the export with every value after day d replaced by noise (v, close, dollar volume, market)."""
    rng = np.random.default_rng(seed)
    v, close, dv, mkt = (np.array(a) for a in (ex.v, ex.close, ex.dollar_volume, ex.market))
    n, k = v.shape[0], v.shape[1] - d - 1
    v[:, d + 1:] = np.clip(rng.normal(0, 3, (n, k)), -5, 5)
    v[:, d + 1:][rng.random((n, k)) < 0.1] = np.nan
    close[:, d + 1:] = np.exp(rng.normal(3, 2, (n, k)))
    dv[:, d + 1:] = np.exp(rng.normal(16, 3, (n, k)))
    mkt[d + 1:] = rng.normal(0, 0.05, k)
    return dataclasses.replace(ex, v=v.astype(np.float32), close=close.astype(np.float32),
                               dollar_volume=dv.astype(np.float32), market=mkt)


def test_scores_ignore_the_future(trained):
    ex, panel, gp, sp, cfg, res = trained
    test = sp.test_decisions
    d = test.start + 120
    days = range(test.start, d + 1)
    base = score_days(res, gp, panel, days, cfg)
    assert np.array_equal(np.isfinite(base), np.isfinite(panel.v[:, test.start:d + 1]))  # NaN only without v[d]
    assert np.nanstd(base) > 0

    panel2 = features.build_panel(altered_after(ex, d, seed=1), device=DEV)
    for name in ("tokens", "logvol", "liquid", "v"):  # every input through d is unchanged
        np.testing.assert_array_equal(getattr(panel2, name)[:, :d + 1], getattr(panel, name)[:, :d + 1], err_msg=name)
    np.testing.assert_array_equal(panel2.market[:d + 1], panel.market[:d + 1])
    gp2 = GpuPanel(panel2, DEV)
    np.testing.assert_array_equal(score_days(res, gp2, panel2, days, cfg), base)

    later = range(d + 1, test.stop)  # the check can see a change: later scores do move
    a, b = score_days(res, gp, panel, later, cfg), score_days(res, gp2, panel2, later, cfg)
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 100 and np.abs(a[both] - b[both]).max() > 1e-3


def test_validation_ignores_the_future(trained):
    """Validation metrics use outcomes through the day after the last validation decision, and nothing later."""
    ex, panel, gp, sp, cfg, res = trained
    last_outcome = sp.val_decisions.stop  # day d+1 of the last validation decision
    assert last_outcome < sp.test_decisions.start - sp.gap  # the gap before the test period
    panel2 = features.build_panel(altered_after(ex, last_outcome, seed=2), device=DEV)
    gp2 = GpuPanel(panel2, DEV)
    hist = TrainSet(panel, gp, sp, res.model.cfg.ctx).hist_logp
    m1 = evaluate(res.fwd, gp, Pairs(panel, sp.val_decisions, gp.device, True, True), hist, 256, res.mu, res.sd, cfg)
    m2 = evaluate(res.fwd, gp2, Pairs(panel2, sp.val_decisions, gp2.device, True, True), hist, 256, res.mu, res.sd, cfg)
    for k in ("ce", "ce_hist", "ic_actual", "ic_v", "pairs"):
        assert m1[k] == m2[k], k


def test_panel_stops_before_the_locked_year(trained):
    ex, panel, *_ = trained
    cut = features.locked_cut(ex.days)
    assert panel.t == cut < ex.t and ex.days[cut] >= data.str_to_day("2024-06-28") > ex.days[cut - 1]
    assert np.isnan(panel.ret_next[:, -1]).all()  # its outcome would be in the locked year
    with pytest.raises(ValueError):
        splits.split(ex.days, 2024)
