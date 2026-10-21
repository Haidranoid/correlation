"""stockGPT at a horizon of H trading days (stockai/gpt/features.py), on synthetic data: the target is the binned daily
rank of the H-day return; no training input or target reaches past the split's last training target day (the last
training decision + H), and validation reads nothing after its last outcome; a planted signal is found at five days and
the sign-flip null finds nothing; horizon 1 is unchanged."""

import dataclasses
import hashlib
import json

import numpy as np
import pytest
import torch
from scipy.special import ndtri
from scipy.stats import rankdata

from stockai import data, splits
from stockai.baselines.common import dev_cutoff
from stockai.gpt import features
from stockai.gpt.metrics import daily_rank_ic, ic_summary
from stockai.gpt.model import GPTConfig
from stockai.gpt.tokens import MISSING, decode, encode
from stockai.gpt.train import Pairs, TrainConfig, census_sum, evaluate, score_days, train_year
from stockai.gpt.walkforward import main
from stockai.gpt.windows import IGNORE, GpuPanel, TrainSet, check_batch
from stockai.h30.evaluate import load_scores
from stockai.h30.targets import forward_returns
from stockai.synthetic import make_synthetic_export

DEV = "cuda" if torch.cuda.is_available() else "cpu"
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
TINY = ["--layers", "1", "--width", "32", "--heads", "2", "--batch", "16", "--eval-batch", "512"]


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    return data.load(make_synthetic_export(tmp_path_factory.mktemp("h"), n=120, rho=-0.1, seed=0))


@pytest.fixture(scope="module")
def panels(export):
    return {h: features.build_panel(export, device=DEV, horizon=h) for h in (1, 5, 21)}


@pytest.fixture(scope="module")
def export_dir(tmp_path_factory):
    return make_synthetic_export(tmp_path_factory.mktemp("hwf"), n=40, seed=5)  # as tests/test_gpt_walkforward.py


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


@pytest.mark.parametrize("h", [5, 21])
def test_target_is_the_binned_daily_rank_of_the_h_day_return(export, panels, h):
    p = panels[h]
    close = np.asarray(export.close[:, :p.t], dtype=np.float64)
    fwd = forward_returns(close, h)  # the H-day return the evaluation grades
    np.testing.assert_array_equal(p.ret_fwd, fwd)
    both = np.isfinite(close[:, h:]) & np.isfinite(close[:, :-h])
    np.testing.assert_allclose(fwd[:, :-h][both], (close[:, h:] / close[:, :-h] - 1.0)[both], rtol=1e-12)
    for d in range(p.t):  # every day, by brute force: ranks among the stocks with a census value and a known return
        j = np.flatnonzero(np.isfinite(p.v[:, d]) & np.isfinite(fwd[:, d]))
        want = np.full(p.n, MISSING, dtype=np.uint8)
        if j.size:
            want[j] = encode(ndtri((rankdata(fwd[j, d]) - 0.5) / j.size).astype(np.float32))
        assert np.array_equal(p.target[:, d], want), d
    assert (p.target[:, p.t - h:] == MISSING).all()  # no outcome inside the panel, so no target
    z = decode(p.target)
    days = np.flatnonzero(np.isfinite(z).any(axis=0))
    assert days.size > 4000 and np.abs(np.nanmean(z[:, days], axis=0)).max() < 1e-6  # every day centered: no direction
    assert (np.nanargmax(z[:, days], axis=0) == np.nanargmax(np.where(np.isfinite(z), fwd, -np.inf)[:, days], axis=0)).all()


def test_target_ignores_the_markets_move(tmp_path):
    """A move shared by every stock (all closes from day 3000 on scaled by 1.37) changes returns, not the target. (No
    close is missing here: a missing end close would be filled from before the move, which does change a rank.)"""
    ex = data.load(make_synthetic_export(tmp_path / "complete", n=60, missing=0.0, seed=1))
    close = np.array(ex.close)
    close[:, 3000:] *= np.float32(1.37)
    a = features.build_panel(ex, device=DEV, horizon=21)
    b = features.build_panel(dataclasses.replace(ex, close=close), device=DEV, horizon=21)
    assert not np.array_equal(a.ret_fwd, b.ret_fwd, equal_nan=True)
    np.testing.assert_array_equal(a.target, b.target)


@pytest.mark.parametrize("h", [5, 21])
def test_null_flips_the_target_by_its_last_outcome_day(export, panels, h):
    p, pn = panels[h], features.build_panel(export, null="sign-flip", null_seed=3, device=DEV, horizon=h)
    flips = features.sign_flips(p.t, 3)
    np.testing.assert_array_equal(pn.v, np.asarray(export.v[:, :p.t]) * flips[None, :])  # inputs as at horizon 1
    np.testing.assert_array_equal(pn.ret_fwd, p.ret_fwd)  # outcomes are never flipped
    np.testing.assert_array_equal(decode(pn.target)[:, :-h], decode(p.target)[:, :-h] * flips[None, h:])
    assert (pn.target[:, p.t - h:] == MISSING).all()


@pytest.mark.parametrize("h", [5, 21])
def test_splits_keep_every_outcome_window_inside_its_range(panels, h):
    for year in splits.DEV_YEARS:
        sp = splits.split(panels[h].days, year, horizon=h)
        assert sp.horizon == h and sp.gap >= h
        assert (sp.train_decisions.stop - 1 + h) + sp.gap < sp.val_decisions.start
        assert (sp.val_decisions.stop - 1 + h) + sp.gap < sp.test_decisions.start


@pytest.mark.parametrize("h", [5, 21])
def test_training_stays_within_its_last_target_day(panels, h):
    p = panels[h]
    gp = GpuPanel(p, DEV)
    for year in (2010, 2019):
        sp = splits.split(p.days, year, horizon=h)
        tr = sp.train_decisions
        last = tr.stop - 1 + h  # the last training target day
        ts = TrainSet(p, gp, sp, 256)
        assert ts.c0 + ts.tokens.shape[1] == ts.c0 + ts.target.shape[1] == last + 1
        gen = torch.Generator(device=DEV).manual_seed(year)
        lo, hi, checked = 10**9, -1, 0
        for _ in range(200):
            tok, cont, y, rows, day = ts.sample(64, gen)
            checked += check_batch(p, sp, y, rows, day)
            use, d = (y != IGNORE).cpu().numpy(), day.cpu().numpy()
            r = rows.cpu().numpy()[:, None].repeat(d.shape[1], 1)
            assert (d[use] + h <= last).all()
            np.testing.assert_array_equal(y.cpu().numpy()[use], p.target[r[use], d[use]])
            t = tok.cpu().numpy()  # inputs: the panel's tokens through the last target day, padding after it
            inside = (d >= 0) & (d <= last)
            assert np.array_equal(t[inside], p.tokens[r[inside], d[inside]]) and (t[~inside] == MISSING).all()
            lo, hi = min(lo, d[use].min()), max(hi, d[use].max())
        assert tr.start <= lo and hi < tr.stop and checked > 100_000
        assert lo < tr.start + 30 and hi > tr.stop - 30  # sampling reaches both ends of the range


@pytest.mark.parametrize("h", [5, 21])
def test_training_ignores_data_after_its_last_target_day(export, panels, h):
    p = panels[h]
    sp = splits.split(p.days, 2016, horizon=h)
    last = sp.train_decisions.stop - 1 + h
    p2 = features.build_panel(altered_after(export, last, seed=1), device=DEV, horizon=h)
    a, b = TrainSet(p, GpuPanel(p, DEV), sp, 256), TrainSet(p2, GpuPanel(p2, DEV), sp, 256)
    assert (a.mu, a.sd, a.n_valid) == (b.mu, b.sd, b.n_valid)
    np.testing.assert_array_equal(a.hist_logp, b.hist_logp)
    ga, gb = torch.Generator(device=DEV).manual_seed(5), torch.Generator(device=DEV).manual_seed(5)
    for _ in range(20):
        for x, y in zip(a.sample(64, ga), b.sample(64, gb)):
            assert torch.equal(x, y)
    # The boundary is tight: the last training decision's outcome ends on day `last`, so altering it shows.
    p3 = features.build_panel(altered_after(export, last - 1, seed=1), device=DEV, horizon=h)
    c = TrainSet(p3, GpuPanel(p3, DEV), sp, 256)
    assert not torch.equal(a.target, c.target)


def test_census_sum_is_the_judges():
    v = np.array([[1.0, np.nan, 2.0, 3.0, np.nan, np.nan, np.nan, 4.0], [np.nan] * 8], dtype=np.float32)
    rows, days = np.array([0, 0, 0, 1]), np.array([0, 3, 4, 0])
    # days d+1 .. d+3, the finite ones; NaN when none is
    np.testing.assert_array_equal(census_sum(v, rows, days, 3), [5.0, np.nan, 4.0, np.nan])
    np.testing.assert_array_equal(census_sum(v, rows, days, 1), v[rows, days + 1].astype(np.float64))


@gpu
def test_validation_ignores_data_after_its_last_outcome(export, panels):
    h = 5
    p = panels[h]
    sp = splits.split(p.days, 2016, horizon=h)
    gp = GpuPanel(p, DEV)
    cfg = TrainConfig(batch=32, max_steps=60, eval_every=60)
    res = train_year(p, gp, sp, GPTConfig(n_layer=2, n_head=2, n_embd=64), cfg, seed=0, log=lambda m: None)
    hist = TrainSet(p, gp, sp, 256).hist_logp

    def metrics(panel):
        g = GpuPanel(panel, DEV)
        return evaluate(res.fwd, g, Pairs(panel, sp.val_decisions, g.device, True, True), hist, 256, res.mu, res.sd, cfg)

    last = sp.val_decisions.stop - 1 + h  # the last validation decision's outcome ends here
    m1 = metrics(p)
    m2 = metrics(features.build_panel(altered_after(export, last, seed=2), device=DEV, horizon=h))
    for k in ("ce", "ce_hist", "ic_actual", "ic_v", "pairs"):
        assert m1[k] == m2[k], k
    m3 = metrics(features.build_panel(altered_after(export, last - 1, seed=2), device=DEV, horizon=h))
    assert m3["ic_actual"] != m1["ic_actual"] and m3["ce"] != m1["ce"]  # the check can see a change


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    return data.load(make_synthetic_export(tmp_path_factory.mktemp("h5"), n=120, rho=-0.3, seed=0))


@gpu
@pytest.mark.parametrize("null", [None, "sign-flip"])
def test_planted_signal_and_null_at_five_days(planted, null):
    """rho = -0.3 plants a one-day reversal; -v[d] reaches a rank IC of about 0.11 against the 5-day return here. The
    model trained on the 5-day target finds it; with signs flipped per date it finds nothing."""
    h = 5
    p = features.build_panel(planted, null=null, device=DEV, horizon=h)
    gp = GpuPanel(p, DEV)
    sp = splits.split(p.days, 2016, horizon=h)
    cfg = TrainConfig(batch=64, max_steps=300, eval_every=100)
    res = train_year(p, gp, sp, GPTConfig(), cfg, seed=0, log=lambda m: None)
    s = score_days(res, gp, p, sp.test_decisions, cfg)
    j, k = np.nonzero(np.isfinite(s))
    d = np.asarray(sp.test_decisions)[k]
    test_ic = ic_summary(daily_rank_ic(d, s[j, k].astype(np.float64), p.ret_fwd[j, d]))["mean"]
    if null is None:
        assert res.info["val_ic_actual"] > 0.06, res.info
        assert test_ic > 0.06
    else:
        assert abs(res.info["val_ic_actual"]) < 0.05, res.info
        assert abs(test_ic) < 0.03


def test_horizon_mismatch_and_locked_year_refused(export, panels, tmp_path):
    sp1, sp5 = splits.split(panels[5].days, 2016), splits.split(panels[5].days, 2016, horizon=5)
    with pytest.raises(ValueError, match="horizon"):
        TrainSet(panels[5], GpuPanel(panels[5], DEV), sp1, 256)
    ts = TrainSet(panels[5], GpuPanel(panels[5], DEV), sp5, 256)
    with pytest.raises(AssertionError):
        check_batch(panels[1], sp5, *ts.sample(8, torch.Generator(device=DEV).manual_seed(0))[2:])
    with pytest.raises(ValueError, match="locked"):
        splits.split(panels[21].days, 2024, horizon=21)
    with pytest.raises(ValueError, match="locked"):
        main(["--export", str(export.path), "--out", str(tmp_path / "out"), "--horizon", "21", "--years", "2023-2024",
              *TINY])


@gpu
@pytest.mark.parametrize("null", [None, "sign-flip"])
def test_walkforward_at_five_days(export_dir, tmp_path, null):
    out = tmp_path / "out"
    assert main(["--export", str(export_dir), "--out", str(out), "--horizon", "5", "--quick", "--steps", "20",
                 "--eval-every", "10", "--seeds", "0", *TINY] + (["--null", null] if null else [])) == 0
    ex = data.load(export_dir)
    s = np.load(out / "scores_seed0.npy")
    assert s.shape == (ex.n, ex.t) and s.dtype == np.float32
    expected = np.zeros(s.shape, dtype=bool)
    for y in (2022, 2023):
        test = splits.split(ex.days, y, horizon=5).test_decisions
        expected[:, test.start:test.stop] = np.isfinite(ex.v[:, test.start:test.stop])
    assert np.array_equal(np.isfinite(s), expected)  # every test decision day, wherever v[d] is finite
    cut = dev_cutoff(ex.days)
    assert not np.isfinite(s[:, cut:]).any()
    load_scores(str(out / "scores_ensemble.npy"), cut)  # what `python -m stockai.h30.evaluate --scores` reads
    lines = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
    assert [(x["year"], x["horizon"]) for x in lines] == [(2022, 5), (2023, 5)]
    assert all("horizon 5" in x["split"] for x in lines)
    run = json.loads((out / "run.json").read_text())
    assert run["horizon"] == 5 and run["target"]["horizon"] == 5 and run["null"] == null


def test_horizon_one_targets_are_the_next_days(export, panels):
    p = panels[1]
    assert p.horizon == 1 and p.ret_fwd is p.ret_next
    np.testing.assert_array_equal(p.target[:, :-1], p.tokens[:, 1:])
    assert (p.target[:, -1] == MISSING).all()
    np.testing.assert_array_equal(p.ret_next, data.next_day_returns(export.close[:, :p.t]).astype(np.float32))
    sp = splits.split(p.days, 2016)
    ts = TrainSet(p, GpuPanel(p, DEV), sp, 256)
    assert ts.c0 + ts.tokens.shape[1] == sp.train_decisions.stop + 1  # ends on the last training decision's next day
    tok, cont, y, rows, day = ts.sample(64, torch.Generator(device=DEV).manual_seed(0))
    use, d, r = (y != IGNORE).cpu().numpy(), day.cpu().numpy(), rows.cpu().numpy()[:, None].repeat(256, 1)
    np.testing.assert_array_equal(y.cpu().numpy()[use], p.tokens[r[use], d[use] + 1])
    pairs = Pairs(p, sp.val_decisions, DEV, True, True)
    np.testing.assert_array_equal(pairs.target, p.tokens[pairs.rows_np, pairs.days_np + 1])
    np.testing.assert_array_equal(pairs.ret, p.ret_next[pairs.rows_np, pairs.days_np].astype(np.float64))
    np.testing.assert_array_equal(pairs.v_next, p.v[pairs.rows_np, pairs.days_np + 1].astype(np.float64))


# The H = 1 walk-forward's scores before the horizon option existed (stockai/gpt as of the POC, which produced
# runs/poc-quick-new), recorded on this machine: deterministic kernels reproduce them bit for bit on the same GPU and
# torch, and nowhere else is promised, so the comparison is skipped elsewhere.
GOLDEN_DEVICE, GOLDEN_TORCH = "NVIDIA GeForce RTX 5080", "2.14.1+cu130"
GOLDEN = {None: "b08fce1822c2920312ac1e46ed6e30a1e64cdfe6b3de4ab0ecc099592a8aa08c",
          "sign-flip": "98706f1522ece7f3a6162d73ee18958dc0dcb6fea1348f532387f5b8b7ec49a9"}
# (year, best step, validation IC against actual returns, validation cross-entropy) from the same runs' train logs
GOLDEN_VAL = {None: [(2015, 30, -0.005675688377430484, 4.1531222681956725),
                     (2022, 10, -0.002263889488541867, 4.391730572390498)],
              "sign-flip": [(2015, 10, -0.0009500211857682837, 4.385234772760778),
                            (2022, 20, 0.00024475392634851786, 4.214743222748573)]}


@gpu
@pytest.mark.parametrize("null", [None, "sign-flip"])
def test_horizon_one_is_unchanged(export_dir, tmp_path, null):
    if torch.cuda.get_device_name() != GOLDEN_DEVICE or torch.__version__ != GOLDEN_TORCH:
        pytest.skip(f"the reference scores were recorded on {GOLDEN_DEVICE} with torch {GOLDEN_TORCH}")
    args = ["--export", str(export_dir), "--years", "2015", "2022", "--seeds", "0", "--steps", "30", "--eval-every",
            "10", "--device", "cuda", *TINY] + (["--null", null] if null else [])
    for flag in ([], ["--horizon", "1"]):
        out = tmp_path / f"out{len(flag)}"
        assert main([*args, "--out", str(out), *flag]) == 0
        assert hashlib.sha256(np.load(out / "scores_seed0.npy").tobytes()).hexdigest() == GOLDEN[null]
        log = [json.loads(x) for x in (out / "train_log.jsonl").read_text().splitlines()]
        assert [(x["year"], x["best_step"], x["val_ic_actual"], x["val_ce"]) for x in log] == GOLDEN_VAL[null]
