"""The POC's controls and G0 comparison: canaries move scores only as intended, the tripwire is point in time and
never reads the locked year, and the G0 score comparison counts ulps correctly."""

from __future__ import annotations

import json

import numpy as np
import pytest

from stockai import data, splits
from stockai.baselines.common import dev_cutoff
from stockai.gpt.features import locked_cut
from stockai.poc import controls, g0
from stockai.synthetic import make_synthetic_export

YEARS = [2015, 2016]


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    return data.load(make_synthetic_export(tmp_path_factory.mktemp("poc") / "export", n=80, seed=11))


def scores_like(export, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(export.n, export.t)).astype(np.float32)
    s[rng.random(s.shape) < 0.1] = np.nan
    s[:, dev_cutoff(export.days):] = np.nan
    return s


def test_shift_uses_scores_from_20_days_earlier_only_inside_test_years(export):
    s = scores_like(export)
    out = controls.shift_scores(s, export.days, YEARS, 20)
    cols = np.zeros(export.t, dtype=bool)
    for y in YEARS:
        r = splits.split(export.days, y).test_decisions
        cols[r.start:r.stop] = True
        np.testing.assert_array_equal(out[:, r.start:r.stop], s[:, r.start - 20:r.stop - 20])
    assert np.isnan(out[:, ~cols]).all()
    assert np.isnan(controls.shift_scores(s, export.days, [2023], 20)[:, dev_cutoff(export.days):]).all()
    with pytest.raises(ValueError):
        controls.shift_scores(s, export.days, YEARS, 0)


def test_permute_shuffles_columns_within_each_year(export):
    s = scores_like(export, 1)
    out = controls.permute_scores(s, export.days, YEARS, seed=0)
    inside = np.zeros(export.t, dtype=bool)
    for y in YEARS:
        r = splits.split(export.days, y).test_decisions
        inside[r.start:r.stop] = True
        a, b = s[:, r.start:r.stop], out[:, r.start:r.stop]
        # the same set of day columns, in a different order
        assert sorted(np.nan_to_num(a, nan=99.0).T.tolist()) == sorted(np.nan_to_num(b, nan=99.0).T.tolist())
        assert not np.array_equal(a, b, equal_nan=True)
    assert np.isnan(out[:, ~inside]).all()
    np.testing.assert_array_equal(out, controls.permute_scores(s, export.days, YEARS, seed=0))


def test_lowvol_reference_is_minus_trailing_log_vol_on_test_days(export):
    from stockai.gpt.features import trailing_log_vol

    out = controls.lowvol_scores(export, YEARS)
    lv = trailing_log_vol(np.asarray(export.close[:, :dev_cutoff(export.days)]))
    inside = np.zeros(export.t, dtype=bool)
    for y in YEARS:
        r = splits.split(export.days, y).test_decisions
        inside[r.start:r.stop] = True
        np.testing.assert_array_equal(out[:, r.start:r.stop], -lv[:, r.start:r.stop])
    assert np.isnan(out[:, ~inside]).all()


def test_trailing_adv_is_point_in_time_and_matches_brute_force():
    rng = np.random.default_rng(2)
    dv = rng.lognormal(15, 1, (5, 200)).astype(np.float32)
    dv[rng.random(dv.shape) < 0.2] = np.nan
    adv = controls.trailing_adv(dv, window=63, min_obs=32)
    for j in range(5):
        for d in range(200):
            w = dv[j, max(0, d - 62):d + 1].astype(np.float64)
            w = w[np.isfinite(w)]
            expect = w.mean() if w.size >= 32 else np.nan
            np.testing.assert_allclose(adv[j, d], expect, rtol=1e-9, equal_nan=True)
    late = dv.copy()
    late[:, 120:] = 1e12
    np.testing.assert_array_equal(controls.trailing_adv(late)[:, :120], adv[:, :120])


def test_tripwire_ignores_everything_from_the_locked_year(export, tmp_path):
    s = scores_like(export, 2)
    base = controls.tripwire(s, export, [2022, 2023], top=40, min_stocks=20)
    lc = locked_cut(export.days)
    mutated = tmp_path / "mutated"
    mutated.mkdir()
    for f in export.path.iterdir():
        (mutated / f.name).write_bytes(f.read_bytes())
    for name in ("close.npy", "dollar_volume.npy"):
        a = np.load(mutated / name)
        a[:, lc:] = np.float32(1e6) * np.random.default_rng(3).random(a[:, lc:].shape, dtype=np.float32)
        np.save(mutated / name, a)
    again = controls.tripwire(s, data.load(mutated), [2022, 2023], top=40, min_stocks=20)
    assert json.dumps(base, sort_keys=True) == json.dumps(again, sort_keys=True)
    assert base["overall"]["days"] > 400 and abs(base["overall"]["mean"]) < 0.05


def test_tripwire_finds_a_planted_large_cap_signal(export):
    ret = data.next_day_returns(np.asarray(export.close[:, :locked_cut(export.days)]))
    s = np.full((export.n, export.t), np.nan, dtype=np.float32)
    s[:, :ret.shape[1]] = ret  # the next day's return itself: maximal leakage
    s[:, dev_cutoff(export.days):] = np.nan
    res = controls.tripwire(s, export, [2020], top=40, min_stocks=20)
    assert res["tripped"] and res["overall"]["mean"] > 0.9


def test_g0_score_comparison_counts_ulps(tmp_path):
    a = np.array([[1.0, 2.0, np.nan], [0.5, -0.0, 3.0]])
    dump = a.copy()
    dump[0, 2] = np.nan
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "d.npy", dump)
    assert g0.compare_scores(tmp_path / "a.npy", tmp_path / "d.npy")["passed"]
    b = a.copy()
    b[1, 2] = np.nextafter(3.0, 4.0)
    np.save(tmp_path / "b.npy", b)
    r = g0.compare_scores(tmp_path / "b.npy", tmp_path / "d.npy")
    assert r["passed"] and r["max_ulp"] == 1 and r["bitwise_equal"] == 4
    b[1, 2] = np.nextafter(np.nextafter(3.0, 4.0), 4.0)
    np.save(tmp_path / "b.npy", b)
    assert not g0.compare_scores(tmp_path / "b.npy", tmp_path / "d.npy")["passed"]
    b = a.copy()
    b[1, 1] = 0.0  # +0.0 against the dump's -0.0: different bits, one step apart in the judge's total order
    np.save(tmp_path / "b.npy", b)
    r = g0.compare_scores(tmp_path / "b.npy", tmp_path / "d.npy")
    assert r["bitwise_equal"] == 4 and r["max_ulp"] == 1 and r["signed_zero_mismatches"] == 1
    b[1, 1] = -1.0
    b[0, 0] = np.nan  # a score the judge computed but Python did not
    np.save(tmp_path / "b.npy", b)
    r = g0.compare_scores(tmp_path / "b.npy", tmp_path / "d.npy")
    assert not r["passed"] and r["py_nan_where_judge_scored"] == 1
    assert g0.ulp_distance(np.array([-1e300]), np.array([1e300]))[0] > 2**62  # no overflow across signs


def test_percentile_keeps_each_days_order_and_makes_every_score_positive(export):
    from scipy.stats import rankdata

    s = scores_like(export, 4)
    s[:5, 300] = 0.25  # ties stay ties
    out = controls.percentile_scores(s, export.days)
    np.testing.assert_array_equal(np.isfinite(out), np.isfinite(s))
    fin = np.isfinite(out)
    assert (out[fin] > 0).all() and (out[fin] <= 1).all()
    for d in (0, 300, 1500, dev_cutoff(export.days) - 1):
        m = np.isfinite(s[:, d])
        np.testing.assert_array_equal(rankdata(out[m, d], method="average"), rankdata(s[m, d], method="average"))
    assert np.unique(out[:5, 300]).size == 1
    assert np.isnan(out[:, dev_cutoff(export.days):]).all()
