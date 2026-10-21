"""correlation2's outside series as time-only LightGBM inputs (stockai.h30.external): point in time, the category and
window rules, the shift control, and a walk-forward that uses them."""

from __future__ import annotations

import csv
import dataclasses
import json

import numpy as np
import pytest

from stockai import data
from stockai.baselines import lgbm
from stockai.baselines.common import load_dev
from stockai.h30 import external, model
from stockai.synthetic import make_synthetic_export


def write_external(p, days, cats, seed=0, live=0.9):
    """A fake export-external folder: per category, a few series of standardized moves with some missing days."""
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for c, n in cats.items():
        for k in range(n):
            x = rng.normal(0, 1, days.size).astype(np.float32)
            x[rng.random(days.size) > live] = np.nan
            rows.append(x)
            meta.append({"row": len(meta), "id": f"{c}_{k}", "name": f"{c} {k}", "category": c, "subcategory": "", "region": "",
                         "catalog": "test", "transform": "logdiff", "frequency": "daily", "known_after_days": "0",
                         "status": "headline", "first": "", "last": "", "observations": "0"})
    rows.append(np.zeros(days.size, dtype=np.float32))  # a noise series: never used
    meta.append({**meta[-1], "row": len(meta), "id": "noise_0", "category": "noise_test", "status": "noise"})
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "ext_v.npy", np.stack(rows))
    with open(p / "ext_meta.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0]), delimiter="\t")
        w.writeheader()
        w.writerows(meta)
    (p / "meta.json").write_text(json.dumps({"first_day": data.day_to_str(int(days[0])), "last_day": data.day_to_str(int(days[-1]))}))
    return p


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = tmp_path_factory.mktemp("ext")
    dev = load_dev(make_synthetic_export(root / "x", n=120, rho=-0.1, seed=7))
    ext = external.load(write_external(root / "ext", np.asarray(dev.export.days), {"rates": 4, "weather": 5, "thin": 2}), dev.export.days)
    return dev, ext


def test_features_follow_the_category_and_window_rules(syn):
    dev, ext = syn
    names, x = external.features(ext, dev.cutoff)
    assert names == tuple(f"ext_{c}_{h}" for c in ("rates", "thin", "weather") for h in (1, 5, 21))
    assert x.shape == (9, dev.cutoff) and x.dtype == np.float32
    v = np.asarray(ext.v[:, :dev.cutoff])
    rates = v[0:4]
    n = np.isfinite(rates).sum(axis=0)
    mean = np.where(n >= 3, np.nansum(rates, axis=0) / np.maximum(n, 1), np.nan)
    d = 3000
    assert x[0, d] == pytest.approx(mean[d], nan_ok=True)  # one-day window: the day's mean
    win = mean[d - 4:d + 1]
    if np.isfinite(win).any():
        assert x[1, d] == pytest.approx(np.nansum(win) / np.sqrt(5), rel=1e-5)  # missing days count as 0
    assert np.isnan(x[3:6]).all()  # "thin" never has 3 live series


def test_features_never_read_past_the_day(syn):
    dev, ext = syn
    _, a = external.features(ext, dev.cutoff)
    d = 2500
    garbled = dataclasses.replace(ext, v=np.array(ext.v))
    garbled.v[:, d + 1:] = np.random.default_rng(3).normal(0, 1, garbled.v[:, d + 1:].shape)
    _, b = external.features(garbled, dev.cutoff)
    assert np.array_equal(a[:, :d + 1], b[:, :d + 1], equal_nan=True)


def test_shift_rolls_every_series_together(syn):
    dev, ext = syn
    _, plain = external.features(ext, dev.cutoff)
    _, shifted = external.features(ext, dev.cutoff, shift=1000)
    rolled = dataclasses.replace(ext, v=np.roll(np.asarray(ext.v[:, :dev.cutoff]), 1000, axis=1))
    _, want = external.features(rolled, dev.cutoff)
    assert np.array_equal(shifted, want, equal_nan=True) and not np.array_equal(shifted, plain, equal_nan=True)
    _, excl = external.features(ext, dev.cutoff, exclude=("weather",))
    assert excl.shape[0] == 6


def test_walk_forward_with_outside_features_reports_their_gain(syn):
    dev, ext = syn
    cfg = dataclasses.replace(lgbm.Config(**lgbm.QUICK), threads=4)
    names, x = external.features(ext, dev.cutoff)
    scores, records = model.walk_forward(dev, [2016], cfg, ext=(names, x))
    rec = records[0]
    assert set(names) <= set(rec["gain_share"]) and len(rec["gain_share"]) == 21 + len(names)
    assert np.isfinite(scores).any()
    panel = model.build_panel(dev, cfg, ext=(names, x))
    j, d = np.array([0, 1]), np.array([3000, 3001])
    assert panel.feats.rows(j, d).shape == (2, 21 + len(names))
    assert np.array_equal(panel.feats.rows(j, d)[:, 21:], x[:, d].T, equal_nan=True)
