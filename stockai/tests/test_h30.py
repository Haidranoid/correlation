"""stockai-30d: forward returns, the grader's universe and schedule, top-10 grading with its chance levels, the rules'
point-in-time inputs, and a LightGBM walk-forward on a synthetic export that never reads an outcome across a gap."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from stockai import splits
from stockai.baselines import lgbm
from stockai.baselines.common import load_dev
from stockai.h30 import metrics, model, rules, targets
from stockai.synthetic import make_synthetic_export


def test_forward_returns_use_the_end_close_or_the_last_one_before_it():
    c = np.array([[10.0, 11.0, 12.0, np.nan, 14.0, 15.0, np.nan, np.nan],
                  [5.0, -1.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]])
    f = targets.forward_returns(c, h=2, fill=1)
    assert f.shape == c.shape
    assert f[0, 0] == pytest.approx(12 / 10 - 1)
    assert f[0, 1] == pytest.approx(12 / 11 - 1)  # close[3] missing: close[2], one day earlier, stands in
    assert f[0, 2] == pytest.approx(14 / 12 - 1)
    assert np.isnan(f[0, 3])  # no close on the decision day
    assert f[0, 4] == pytest.approx(15 / 14 - 1)  # close[6] missing: close[5]
    assert np.isnan(f[0, 5])  # close[7] and close[6] missing: nothing within one day
    assert np.isnan(f[1, 1])  # a non-positive close
    assert np.isnan(f[:, -2:]).all()  # the window would leave the columns


def test_forward_returns_never_read_past_the_window():
    rng = np.random.default_rng(0)
    c = rng.uniform(5, 50, (6, 40))
    f = targets.forward_returns(c, h=21)
    g = c.copy()
    g[:, 30:] = rng.uniform(5, 50, (6, 10))  # garble days 30 on: decisions d <= 8 only read through d + 21 <= 29
    assert np.array_equal(f[:, :9], targets.forward_returns(g, h=21)[:, :9])


def reference_universe(v, close, dv, h0, window=756):
    """holdout.rs `liquid_universe`, one stock at a time."""
    out = []
    w0 = max(h0 - window, 0)
    tail = min(h0 - w0, 252)
    for j in range(v.shape[0]):
        vv = v[j, w0:h0]
        if h0 == w0 or np.isfinite(vv).sum() * 10 < (h0 - w0) * 9:
            continue
        pc = np.sort(close[j, h0 - tail:h0][np.isfinite(close[j, h0 - tail:h0])])
        pv = np.sort(dv[j, h0 - tail:h0][np.isfinite(dv[j, h0 - tail:h0])])
        med = lambda x: x[len(x) // 2] if len(x) else 0.0  # noqa: E731
        if med(pc) >= 5.0 and med(pv) >= 5e6:
            out.append(j)
    return np.array(out, dtype=np.int64)


def test_judge_universe_matches_the_graders_rule():
    rng = np.random.default_rng(3)
    n, t = 60, 900
    v = rng.normal(0, 1, (n, t)).astype(np.float32)
    v[rng.random((n, t)) < rng.uniform(0, 0.2, (n, 1))] = np.nan  # some stocks miss more than 10% of days
    close = np.exp(rng.normal(np.log(rng.uniform(2, 30, (n, 1))), 0.2, (n, t))).astype(np.float32)
    close[rng.random((n, t)) < 0.03] = np.nan
    dv = (close * rng.uniform(1e5, 2e6, (n, 1))).astype(np.float32)
    for h0 in (300, 800, 900):
        got = targets.judge_universe(v, close, dv, h0)
        want = reference_universe(v, close, dv, h0)
        assert np.array_equal(got, want), h0
    assert 0 < targets.judge_universe(v, close, dv, 800).size < n


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return load_dev(make_synthetic_export(tmp_path_factory.mktemp("syn30") / "x", n=150, rho=-0.1, seed=5))


def test_year_spec_is_the_graders_monthly_schedule_inside_the_development_columns(syn):
    close = np.asarray(syn.close)
    for year in (2015, 2023):
        ys = targets.year_spec(syn.days, syn.cutoff, np.asarray(syn.v), close, np.asarray(syn.dollar_volume), year)
        allreb = np.r_[ys.decisions, ys.dropped]
        assert allreb[0] == ys.h0 - 1 and np.all(np.diff(np.sort(allreb)) == targets.H)
        assert np.all(ys.decisions + targets.H <= syn.cutoff - 1)
        assert np.all(ys.dropped + targets.H > syn.cutoff - 1)
        if year == 2015:
            assert ys.dropped.size == 0 and ys.decisions.size >= 12
    assert targets.year_spec(syn.days, syn.cutoff, np.asarray(syn.v), close, np.asarray(syn.dollar_volume),
                             2023).dropped.size >= 1  # the last month reaches the locked year
    with pytest.raises(ValueError):
        targets.year_spec(syn.days, syn.cutoff, np.asarray(syn.v), close, np.asarray(syn.dollar_volume), 2024)


def test_top_by_breaks_ties_toward_the_lower_column():
    vals = np.array([1.0, 3.0, 3.0, 2.0])
    cols = np.array([7, 9, 4, 1])
    assert metrics.top_by(vals, cols, 2).tolist() == [4, 9]
    assert metrics.top_by(vals, cols, 2, largest=False).tolist() == [7, 1]


def day_arrays(seed=1, n=400):
    rng = np.random.default_rng(seed)
    vol = np.exp(rng.normal(np.log(0.02), 0.5, n))
    fwd = rng.normal(0, 1, n) * vol * 4  # volatile stocks fill both tails
    return vol, fwd


def test_grade_day_counts_hits_and_chance_levels():
    vol, fwd = day_arrays()
    n = vol.size
    as2 = lambda x: x[:, None].astype(np.float64)  # noqa: E731  (one decision column)
    uni = np.arange(n)
    perfect = metrics.grade_day(as2(fwd), as2(fwd), as2(vol), uni, 0)
    assert perfect.hits_gain == 10 and perfect.hits_loss == 0 and perfect.top_decile == 10 and perfect.ic == pytest.approx(1)
    worst = metrics.grade_day(as2(-fwd), as2(fwd), as2(vol), uni, 0)
    assert worst.hits_loss == 10 and worst.hits_gain == 0 and worst.bottom_decile == 10
    assert perfect.exp_uniform == pytest.approx(100 / n)
    # The volatility-matched chance level is the mean over random swaps within each pick's volatility group.
    hv = metrics.grade_day(as2(vol), as2(fwd), as2(vol), uni, 0)
    order = np.argsort(np.argsort(vol, kind="stable"), kind="stable")
    dec = order * metrics.VOL_GROUPS // n
    gain = set(metrics.top_by(fwd, uni, 10).tolist())
    rng = np.random.default_rng(9)
    sims = []
    for _ in range(4000):
        hits = 0
        for c in hv.picks:
            pool = np.flatnonzero(dec == dec[c])
            hits += int(rng.choice(pool) in gain)
        sims.append(hits)
    assert hv.exp_vol == pytest.approx(np.mean(sims), abs=0.05)
    assert hv.exp_vol > hv.exp_uniform  # the most volatile stocks hold more than their share of the gainers


def test_grade_uses_only_the_years_universe_and_known_outcomes(syn):
    close = np.asarray(syn.close, dtype=np.float32)
    fwd = targets.forward_returns(close)
    vol = metrics.trailing_vol(close)
    ys = targets.year_spec(syn.days, syn.cutoff, np.asarray(syn.v), close, np.asarray(syn.dollar_volume), 2016)
    score = np.where(np.isfinite(fwd), fwd, np.nan).astype(np.float32)  # the outcome itself
    r = metrics.grade(score, fwd, vol, [ys])
    o = r["overall"]
    assert o["days"] >= 10
    if o["days"]:
        assert o["hits_gain_per_day"] == 10 and o["hits_loss"] == 0 and o["ic_mean"] == pytest.approx(1)
        for day in r["days"]:
            assert set(day["picks"]) <= set(ys.universe.tolist())


def test_rules_read_closes_through_d_only(syn):
    close = np.array(syn.close, dtype=np.float32)
    d = 3000
    a = rules.rules(close)
    g = close.copy()
    g[:, d + 1:] = np.float32(1.0) + np.random.default_rng(2).random(g[:, d + 1:].shape).astype(np.float32)
    b = rules.rules(g)
    for name in ("momentum", "reversal", "high_vol", "low_vol"):
        assert np.array_equal(a[name][:, :d + 1], b[name][:, :d + 1], equal_nan=True), name


def test_model_walk_forward_scores_only_test_days_and_reads_no_later_outcome(syn):
    cfg = dataclasses.replace(lgbm.Config(**lgbm.QUICK), threads=4)
    scores, records = model.walk_forward(syn, [2016], cfg)
    split = {s.year: s for s in splits.dev_splits(syn.export.days, horizon=targets.H)}[2016]
    fin = np.flatnonzero(np.isfinite(scores).any(axis=0))
    assert fin.min() >= split.test_decisions.start and fin.max() < split.test_decisions.stop
    rec = records[0]
    assert rec["horizon"] == targets.H and rec["test_scored"] > 0
    # Training and validation outcomes end before the next range starts (gaps of at least H days).
    assert split.train_decisions.stop - 1 + targets.H < split.val_decisions.start
    assert split.val_decisions.stop - 1 + targets.H < split.test_decisions.start
    # Garbling every close after the last validation outcome leaves the model and its test scores unchanged
    # (test scores read features through d only, which features.py checks bit for bit).
    last = split.val_decisions.stop - 1 + targets.H
    g = dataclasses.replace(syn, close=np.array(syn.close), v=np.array(syn.v))
    g.close[:, last + 1:] = np.where(np.isfinite(g.close[:, last + 1:]), g.close[:, last + 1:] * 1.3, np.nan)
    p1 = model.build_panel(syn, cfg)
    p2 = model.build_panel(g, cfg)
    j1, d1, pred1, _ = model.fit_year(p1, split, cfg)
    _, _, _, rec2 = model.fit_year(p2, split, cfg)
    assert rec2["best_iteration"] == rec["best_iteration"] and rec2["val_l2"] == rec["val_l2"]
    assert pred1.size == j1.size and math.isfinite(rec["val_l2"])


def test_sign_flip_null_flips_each_days_target(syn):
    cfg = dataclasses.replace(lgbm.Config(**lgbm.QUICK), threads=4)
    p = model.build_panel(syn, cfg)
    q = model.build_panel(syn, cfg, null="sign-flip", null_seed=1)
    both = np.isfinite(p.target) & np.isfinite(q.target)
    ratio = np.where(both, q.target / np.where(p.target == 0, 1, p.target), np.nan)
    for d in np.flatnonzero(both.any(axis=0))[:50]:
        vals = ratio[both[:, d], d]
        vals = vals[np.isfinite(vals) & (p.target[both[:, d], d] != 0)]
        assert np.allclose(vals, q.flips[d])


def test_tied_scores_share_the_last_slots():
    n = 200
    fwd = np.linspace(-0.5, 0.5, n)[:, None]
    score = np.zeros((n, 1))
    score[-5:, 0] = 2.0  # five clear picks (the five biggest gainers)
    score[100:110, 0] = 1.0  # ten tied for the five slots left
    vol = np.full((n, 1), 0.02)
    r = metrics.grade_day(score, fwd, vol, np.arange(n), 0)
    w = dict(zip(r.picks, r.weights))
    assert sum(r.weights) == pytest.approx(10) and r.tied == 10
    assert all(w[c] == 1.0 for c in range(195, 200)) and all(w[c] == pytest.approx(0.5) for c in range(100, 110))
    assert r.hits_gain == pytest.approx(5.0)  # the tied stocks are mid-ranked: none is a top-10 gainer
    assert r.pick_ret == pytest.approx((fwd[195:, 0].sum() + 0.5 * fwd[100:110, 0].sum()) / 10)
    acc = metrics.account([r, r], cost_bps=20)
    assert acc["end"] == pytest.approx(10000 * (1 + r.pick_ret - 20e-4) * (1 + r.pick_ret))  # buys once, then holds


def test_non_overlapping_drops_a_month_that_runs_into_the_next_years_first():
    u = np.arange(3)
    a = targets.YearSpec(year=2010, h0=100, t=353, universe=u, decisions=np.arange(99, 352, 21), dropped=np.array([], int))
    b = targets.YearSpec(year=2011, h0=353, t=605, universe=u, decisions=np.arange(352, 604, 21), dropped=np.array([], int))
    out = targets.non_overlapping([a, b])
    # 2010's last rebalance (99 + 12 * 21 = 351) would end 20 days into 2011's first month (from 352): dropped
    assert out[0].decisions[-1] == 330 and 351 in out[0].dropped.tolist()
    assert np.array_equal(out[1].decisions, b.decisions)
    for ys, nxt in zip(out, out[1:]):
        assert ys.decisions[-1] + targets.H <= nxt.decisions[0]


def test_forward_returns_count_a_stock_delisted_inside_its_window_by_the_rule():
    gone = [np.nan] * 4
    c = np.array([[10.0, 11.0, 12.0, 8.0] + gone,  # a buyout: last close on day 3
                  [10.0, 11.0, 12.0, 8.0] + gone,  # a failure
                  [10.0, 11.0, 12.0, 8.0] + gone,  # unknown
                  [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]])  # still listed
    dday, kind = np.array([3, 3, 3, -1], dtype=np.int32), np.array([1, 2, 3, 0], dtype=np.int8)
    main = targets.forward_returns(c, 2, delist_day=dday, delist_kind=kind)
    # Decisions with the last close inside the window (d <= 3 < d + 2): days 2 and 3.
    assert main[0, 2] == pytest.approx(8 / 12 - 1) and main[0, 3] == pytest.approx(0.0)  # a buyout at its last price
    assert main[1, 2] == -1 and main[1, 3] == -1 and main[2, 3] == -1  # failure and unknown: a total loss
    assert main[1, 1] == pytest.approx(8 / 11 - 1)  # the window ends on the last close itself: an ordinary outcome
    assert np.isnan(main[:3, 4:]).all()  # no close on the decision day
    last = targets.forward_returns(c, 2, delist_day=dday, delist_kind=kind, rule="last-price")
    loss = targets.forward_returns(c, 2, delist_day=dday, delist_kind=kind, rule="total-loss")
    assert last[1, 3] == pytest.approx(0.0) and last[2, 2] == pytest.approx(8 / 12 - 1)
    assert loss[0, 2] == -1 and loss[0, 3] == -1
    plain = targets.forward_returns(c, 2)  # no delisting data: the stock has no outcome once nothing fills the end
    assert np.isnan(plain[0, 3]) and np.isnan(plain[1, 3])
    for f in (main, last, loss):
        assert np.array_equal(f[3], plain[3], equal_nan=True)  # the listed stock is untouched
        assert np.isnan(f[:, -2:]).all()  # windows that leave the columns stay ungraded
    with pytest.raises(ValueError):
        targets.forward_returns(c, 2, delist_day=dday, delist_kind=kind, rule="optimistic")


def test_forward_returns_fill_never_reaches_back_to_the_decision_day():
    # At a 5-day horizon the end close may come from at most 4 days before it, never from the decision day itself.
    assert np.isnan(targets.forward_returns(np.array([[10.0] + [np.nan] * 5 + [12.0]]), h=5)[0, 0])
    c = np.array([[10.0, 11.0] + [np.nan] * 4 + [12.0]])
    assert targets.forward_returns(c, h=5)[0, 0] == pytest.approx(11 / 10 - 1)
    # At 21 days the fill is still FILL_DAYS = 5 (unchanged since the monthly results).
    c = np.r_[10.0, np.full(15, 11.0), 13.0, np.full(5, np.nan)][None, :]
    assert targets.forward_returns(c, h=21)[0, 0] == pytest.approx(13 / 10 - 1)


def test_judge_universe_price_test_uses_closes_as_traded():
    rng = np.random.default_rng(11)
    n, t = 40, 900
    v = rng.normal(0, 1, (n, t)).astype(np.float32)
    traded = np.exp(rng.normal(np.log(rng.uniform(2, 30, (n, 1))), 0.1, (n, t))).astype(np.float32)
    traded[:10] = np.exp(rng.normal(np.log(3.0), 0.1, (10, t))).astype(np.float32)  # really under $5
    adjusted = traded.copy()
    adjusted[:10] *= 20  # a later 1-for-20 reverse split multiplies their earlier prices
    dv = (traded * 1e7).astype(np.float32)  # every stock trades enough
    lifted = targets.judge_universe(v, adjusted, dv, 800)
    honest = targets.judge_universe(v, adjusted, dv, 800, price_close=traded)
    assert set(range(10)) <= set(lifted.tolist()) and not set(range(10)) & set(honest.tolist())
    assert np.array_equal(honest, targets.judge_universe(v, traded, dv, 800))


def test_comparable_excess_compares_each_pick_with_its_nearest_non_picks():
    rng = np.random.default_rng(4)
    n = 300
    vv, rr = rng.uniform(0.01, 0.05, n), rng.uniform(0.3, 2.0, n)
    w = np.zeros(n)
    w[:10] = 1.0
    smooth = 2 * vv + 0.1 * rr  # returns explained by volatility and past-year fall alone
    base = metrics.comparable_excess(smooth, w, vv, rr)
    assert abs(base) < 0.02
    lifted = smooth.copy()
    lifted[:10] += 0.05  # each pick beats its look-alikes by 5 points
    assert metrics.comparable_excess(lifted, w, vv, rr) == pytest.approx(base + 0.05)
    lifted[10:] += 1.0  # moving every non-pick moves the yardstick, not the picks
    assert metrics.comparable_excess(lifted, w, vv, rr) == pytest.approx(base + 0.05 - 1.0)
    half = w.copy()
    half[:2] = 0.5  # tied picks count at their weight
    assert np.isfinite(metrics.comparable_excess(smooth, half, vv, rr))


def add_sharadar_extras(p, delisted: dict) -> None:
    """Turns a synthetic export into a Sharadar-like one: as-traded closes (equal to the adjusted ones here) and the
    given delistings {column: (last day, kind)}, with nothing after the last day."""
    close, v = np.load(p / "close.npy"), np.load(p / "v.npy")
    n = close.shape[0]
    dday, kind = np.full(n, -1, dtype=np.int32), np.zeros(n, dtype=np.int8)
    for j, (last, k) in delisted.items():
        close[j, last + 1:] = np.nan
        v[j, last + 1:] = np.nan
        dday[j], kind[j] = last, k
    for name, arr in [("close.npy", close), ("v.npy", v), ("close_unadj.npy", close.copy()),
                      ("delist_day.npy", dday), ("delist_kind.npy", kind)]:
        np.save(p / name, arr)


def test_load_reads_the_optional_sharadar_arrays(tmp_path):
    from stockai import data
    p = make_synthetic_export(tmp_path / "x", n=20, seed=2)
    ex = data.load(p)
    assert ex.close_unadj is None and ex.delist_day is None and ex.delist_kind is None
    add_sharadar_extras(p, {3: (4000, 2)})
    ex = data.load(p)
    assert ex.close_unadj.shape == ex.close.shape and ex.delist_day[3] == 4000 and ex.delist_kind[3] == 2
    assert (ex.delist_day[np.arange(20) != 3] == -1).all()
    dev = load_dev(p)
    assert dev.default_price_test() == "traded" and dev.price_close("traded").shape == dev.close.shape
    np.save(p / "delist_kind.npy", np.zeros(20, dtype=np.int8))  # a delisted stock without a kind
    with pytest.raises(ValueError):
        data.load(p)


def test_evaluate_counts_delisted_stocks_and_reports_the_bounds(tmp_path):
    import json

    from stockai.h30 import evaluate
    p = make_synthetic_export(tmp_path / "x", n=150, rho=-0.1, seed=5)
    dev = load_dev(p)
    close = np.asarray(dev.close)
    ys = targets.year_spec(dev.days, dev.cutoff, np.asarray(dev.v), close, np.asarray(dev.dollar_volume), 2016)
    d = int(ys.decisions[0])
    victims = ys.universe[:3]  # a buyout, a failure and an unknown, each delisted 3 days into the first graded month
    add_sharadar_extras(p, {int(j): (d + 3, k) for j, k in zip(victims, (1, 2, 3))})
    score = np.full((dev.export.n, dev.export.t), np.nan, dtype=np.float32)
    score[:, :dev.cutoff] = np.random.default_rng(0).normal(0, 1, (dev.export.n, dev.cutoff))
    score[victims, d] = 10.0  # picked
    np.save(tmp_path / "s.npy", score)
    out = tmp_path / "out"
    evaluate.main(["--export", str(p), "--years", "2016", "--scores", f"s={tmp_path / 's.npy'}", "--out", str(out)])
    g = json.loads((out / "grades.json").read_text())
    assert g["meta"]["delist"] == "main" and g["meta"]["price_test"] == "traded"
    assert g["meta"]["delisted"]["universe_kind"] == {"1": 1, "2": 1, "3": 1}
    s = g["scores"]["s"]
    lo, mid, hi = (s["delist_bounds"]["total-loss"]["mean_pick_ret"], s["overall"]["mean_pick_ret"],
                   s["delist_bounds"]["last-price"]["mean_pick_ret"])
    assert lo < mid < hi  # the failure and the unknown lose everything under main; the buyout keeps its last price
    assert s["delisted_pick_share"] == pytest.approx(3 / (10 * s["overall"]["days"]))
    text = (out / "summary.md").read_text()
    assert "Delisted stocks: the rule and its bounds" in text and "Against comparable stocks" in text
