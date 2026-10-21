"""The baseline features are point in time and compute what features.py says they do."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from stockai.baselines import features as F
from stockai.baselines.common import load_dev
from stockai.synthetic import make_synthetic_export


def mutate_after(src: Path, dst: Path, d: int, seed: int = 0) -> Path:
    """A copy of export `src` with every input after day column d changed adversarially (NaN, 0, < 0, huge, rescaled)."""
    dst.mkdir(parents=True)
    for f in src.iterdir():
        shutil.copy(f, dst / f.name)
    rng = np.random.default_rng(seed)
    for name in ("v.npy", "close.npy", "dollar_volume.npy"):
        a = np.load(dst / name)
        tail = a[:, d + 1 :]
        kind = rng.integers(0, 5, tail.shape)
        tail *= rng.uniform(0.2, 5.0, tail.shape).astype(np.float32)
        tail[kind == 0] = np.nan
        tail[kind == 1] = 0.0
        tail[kind == 2] = -1.0
        tail[kind == 3] = 1e6
        np.save(dst / name, a)
    m = np.load(dst / "market.npy")
    m[d + 1 :] = rng.normal(0, 0.05, m.size - d - 1)
    m[d + 1 :: 7] = np.nan
    np.save(dst / "market.npy", m)
    return dst


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def panel_of(path: Path):
    dev = load_dev(path)
    return dev, F.compute_features(dev.v, dev.close, dev.dollar_volume, dev.market), F.liquid_mask(dev.close,
                                                                                                    dev.dollar_volume)


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    path = make_synthetic_export(tmp_path_factory.mktemp("syn") / "export", n=40, missing=0.03, seed=7)
    return path, *panel_of(path)


@pytest.mark.parametrize("d", [260, 1500, 4000])
def test_features_at_d_ignore_data_after_d(base, tmp_path, d):
    path, dev, feats, liquid = base
    dev2, feats2, liquid2 = panel_of(mutate_after(path, tmp_path / "mutated", d, seed=d))
    assert not same_bits(np.asarray(dev.close[:, d + 1 :]), np.asarray(dev2.close[:, d + 1 :]))
    k = d + 1
    assert same_bits(feats.stock[:, :, :k], feats2.stock[:, :, :k])
    assert same_bits(feats.market[:, :k], feats2.market[:, :k])
    assert same_bits(feats.complete[:, :k], feats2.complete[:, :k])
    assert same_bits(liquid[:, :k], liquid2[:, :k])
    # ... and the mutation did reach the later features, so the comparison is not vacuous.
    assert not same_bits(feats.stock[:, :, k:], feats2.stock[:, :, k:])


def upper_median(x: np.ndarray) -> float:
    x = np.sort(x[np.isfinite(x)])
    return float(x[x.size // 2]) if x.size else np.nan


def test_features_match_direct_formulas(base):
    _, dev, feats, liquid = base
    v = np.asarray(dev.v, dtype=np.float64)
    c = np.asarray(dev.close, dtype=np.float64)
    dv = np.asarray(dev.dollar_volume, dtype=np.float64)
    mk = np.asarray(dev.market)
    rng = np.random.default_rng(1)
    checked = 0
    for j, d in zip(rng.integers(0, dev.v.shape[0], 400), rng.integers(F.HISTORY, dev.v.shape[1], 400)):
        if not feats.complete[j, d]:
            continue
        checked += 1
        x = dict(zip(F.FEATURES, feats.rows(np.array([j]), np.array([d]))[0].astype(np.float64)))
        for h in F.VSUM_H:
            assert x[f"vsum_{h}"] == pytest.approx(np.nansum(v[j, d - h + 1 : d + 1]) / np.sqrt(h), rel=1e-5, abs=1e-6)
        for h in F.RET_H:
            back = c[j, max(d - h - F.FILL_DAYS, 0) : d - h + 1]
            base_close = back[np.isfinite(back)][-1]
            assert x[f"ret_{h}"] == pytest.approx(np.log(c[j, d] / base_close), rel=1e-5, abs=1e-6)
        r = np.log(c[j, 1:] / c[j, :-1])  # r[k - 1] is day k's return
        for h in F.VOL_H:
            w = r[d - h : d]
            assert x[f"vol_{h}"] == pytest.approx(np.std(w[np.isfinite(w)]), rel=1e-4)
        assert x["log_dv_21"] == pytest.approx(np.log1p(upper_median(dv[j, d - 20 : d + 1])), rel=1e-5)
        for k in F.V_LAGS:
            assert x[f"v_lag{k}"] == pytest.approx(np.nan_to_num(v[j, d - k]), abs=1e-6)
        for h in F.MKT_H:
            assert x[f"mkt_{h}"] == pytest.approx(mk[d - h + 1 : d + 1].sum(), rel=1e-5, abs=1e-7)
        want_liquid = c[j, d] >= 5.0 and upper_median(dv[j, d - 62 : d + 1]) >= 5e6
        assert bool(liquid[j, d]) == want_liquid
    assert checked > 200


def test_complete_needs_todays_data_and_a_year_of_history(base):
    _, dev, feats, _ = base
    v, c = np.asarray(dev.v), np.asarray(dev.close)
    assert not feats.complete[:, : F.HISTORY].any()
    assert not feats.complete[~np.isfinite(v)].any()
    assert not feats.complete[~np.isfinite(c)].any()
    finite = np.ones(feats.complete.shape, dtype=bool)
    for name in F.FEATURES:
        finite &= np.isfinite(feats.panel(name))
    assert np.array_equal(finite, feats.complete)
    assert feats.complete[:, F.HISTORY + 252 :].mean() > 0.85


def test_window_helpers_edge_cases():
    x = np.arange(10, dtype=np.float64)[None, :]
    s = F.window_sum(x, 3)
    assert np.isnan(s[0, :2]).all() and s[0, 2] == 3 and s[0, 9] == 24
    assert np.isnan(F.window_sum(x, 11)).all()
    m = np.array([[1, np.nan, 3, 4, np.nan, np.nan, np.nan, 8]], dtype=np.float32)
    med = F.window_median(m, 4, min_count=2)
    assert np.isnan(med[0, :3]).all()  # d < h - 1
    assert med[0, 3] == 3  # {1, 3, 4}: upper median of 3 values
    assert med[0, 4] == 4 and med[0, 5] == 4  # {3, 4}: upper median of 2 values
    assert np.isnan(med[0, 6]) and np.isnan(med[0, 7])  # {4} and {8}: fewer than min_count
    f = F.fill_forward(np.array([[1.0, np.nan, np.nan, 4.0, np.nan]]), limit=1)
    assert f[0, 1] == 1.0 and np.isnan(f[0, 2]) and f[0, 4] == 4.0
