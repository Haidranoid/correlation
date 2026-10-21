"""LightGBM for stockai-30d: from the close of day d, rank the stocks by their return over the next H = 21 trading days.

For each development test year (stockai.splits.dev_splits with horizon H, so every gap between training, validation
and test outcomes is at least H days) one model is trained on that split's training decision days, early-stopped on the
L2 of its validation days, and then scores every test decision day for every stock with complete features; NaN
elsewhere. The 21 point-in-time features are the POC's (stockai/baselines/features.py) and the training rows its
liquidity filter (close >= $5, 63-day median dollar volume >= $5M at d).

The target is the H-day forward return (targets.forward_returns) as a daily rank-gauss score among the day's training
rows, so the model learns the order of next month's returns, not their market-wide level. On a Sharadar export a stock
delisted inside its window counts by the delisting rule (`--delist`, default main: a failure is a total loss), so the
model also learns from the stocks that went under; and the $5 liquidity test uses as-traded closes (`--price-test`,
default traded when the export has them). `--null sign-flip`
multiplies that target by a seeded random sign per decision day (the same for every stock): a model of it can learn
nothing real about the order of returns, so whatever it shows is what chance and the evaluation produce.

    uv run python -m stockai.h30.model --out DIR [--years 2010-2023] [--null sign-flip] [--quick]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .. import data, splits
from ..baselines.common import DevData, describe_scores, load_dev, parse_years
from ..baselines.features import FEATURES, Features, compute_features, liquid_mask
from ..baselines.lgbm import QUICK, Config, _git_state, daily_ic, ic_summary, mask_rows, rank_gauss
from . import external
from .external import REVISED, WithExternal
from .targets import DELIST_RULES, H, forward_returns

NULLS = ("sign-flip",)


@dataclass
class Panel:
    days: np.ndarray  # int32 [C]
    feats: Features
    trainable: np.ndarray  # bool [N, C]: liquid at d, complete features, known target
    target: np.ndarray  # float32 [N, C]: rank-gauss of the H-day forward return (sign-flipped per day under the null)
    fwd: np.ndarray  # float64 [N, C]: the H-day forward return itself (NaN where its window leaves the columns)
    flips: np.ndarray | None = None


def sign_flips(c: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x30D]))
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=c)


def build_panel(dev: DevData, cfg: Config, h: int = H, null: str | None = None, null_seed: int = 0,
                price_test: str | None = None, delist: str = "main",
                ext: tuple[tuple[str, ...], np.ndarray] | None = None) -> Panel:
    """`ext`: (names, float32 [F, C]) time-only outside features (stockai.h30.external), appended to the stock ones."""
    feats = compute_features(dev.v, dev.close, dev.dollar_volume, dev.market)
    if ext is not None:
        feats = WithExternal(base=feats, ext=ext[1], ext_names=ext[0])
    price_close = dev.price_close(price_test or dev.default_price_test())
    liquid = liquid_mask(price_close, dev.dollar_volume, cfg.min_price, cfg.min_dollar_volume, cfg.liquidity_window)
    fwd = forward_returns(dev.close, h, delist_day=dev.delist_day, delist_kind=dev.delist_kind, rule=delist)
    rows = liquid & feats.complete & np.isfinite(fwd)
    target = rank_gauss(fwd, rows)
    flips = None
    if null == "sign-flip":
        flips = sign_flips(target.shape[1], null_seed)
        target = (target * flips[None, :]).astype(np.float32)
    elif null is not None:
        raise ValueError(f"unknown null {null!r}")
    trainable = rows & np.isfinite(target)
    return Panel(days=np.asarray(dev.days), feats=feats, trainable=trainable, target=target, fwd=fwd, flips=flips)


def fit_year(panel: Panel, split: splits.Split, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Trains on split.train_decisions, early-stops on split.val_decisions, scores split.test_decisions.

    Returns (stock rows, decision days, predictions, record)."""
    c = panel.days.size
    if split.test_decisions.stop > c:
        raise ValueError(f"year {split.year}: test decisions run past the development cutoff ({c} columns)")
    if split.val_decisions.stop - 1 + split.horizon >= split.test_decisions.start:
        raise AssertionError("a validation outcome would reach the test period")  # splits.py rules it out
    t0 = time.time()
    tr_all = np.arange(split.train_decisions.start, split.train_decisions.stop)
    per_day = panel.trainable[:, tr_all].sum(axis=0)
    available = int(per_day.sum())
    tr_cols = tr_all
    if available > cfg.max_train_rows:  # a seeded random subset of whole days
        order = np.random.default_rng([cfg.seed, split.year]).permutation(tr_all.size)
        tr_cols = np.sort(tr_all[order[np.cumsum(per_day[order]) <= cfg.max_train_rows]])
    j_tr, d_tr = mask_rows(panel.trainable, tr_cols)
    va_cols = np.arange(split.val_decisions.start, split.val_decisions.stop)
    j_va, d_va = mask_rows(panel.trainable, va_cols)
    if j_tr.size == 0 or j_va.size == 0:
        raise ValueError(f"year {split.year}: {j_tr.size} training rows and {j_va.size} validation rows")
    # Every outcome read for training or validation ends before the test period's first decision.
    if d_va.max() + split.horizon >= split.test_decisions.start or d_tr.max() + split.horizon >= va_cols[0]:
        raise AssertionError(f"year {split.year}: an outcome window crosses into a later range")
    x_tr, y_tr = panel.feats.rows(j_tr, d_tr), panel.target[j_tr, d_tr]
    x_va, y_va = panel.feats.rows(j_va, d_va), panel.target[j_va, d_va]

    names = list(panel.feats.names)
    dtr = lgb.Dataset(x_tr, label=y_tr, feature_name=names, params={"max_bin": cfg.max_bin, "verbosity": -1},
                      free_raw_data=True)
    dva = lgb.Dataset(x_va, label=y_va, reference=dtr)
    booster = lgb.train(cfg.params(), dtr, num_boost_round=cfg.rounds, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(cfg.early_stopping, first_metric_only=True, verbose=False)])
    best = int(booster.best_iteration or booster.current_iteration())
    t_fit = time.time()
    p_va = booster.predict(x_va, num_iteration=best)
    ic_va = daily_ic(p_va, panel.fwd[j_va, d_va], d_va, cfg.min_ic_stocks)

    te_cols = np.arange(split.test_decisions.start, split.test_decisions.stop)
    j_te, d_te = mask_rows(panel.feats.complete, te_cols)
    pred = booster.predict(panel.feats.rows(j_te, d_te), num_iteration=best) if j_te.size else np.empty(0)
    gain = booster.feature_importance("gain", iteration=best)
    total_gain = float(gain.sum()) or 1.0
    y64 = y_va.astype(np.float64)
    span = lambda cols: f"{data.day_to_str(panel.days[cols[0]])}..{data.day_to_str(panel.days[cols[-1]])}"  # noqa: E731
    record = {
        "kind": "year", "year": split.year, "horizon": split.horizon,
        "train": span(tr_all), "train_days": int(tr_all.size), "train_days_used": int(tr_cols.size),
        "train_rows_available": available, "train_rows": int(j_tr.size),
        "val": span(va_cols), "val_rows": int(j_va.size), "test": span(te_cols), "test_scored": int(j_te.size),
        "best_iteration": best,
        "val_l2": float(np.mean((p_va - y64) ** 2)), "val_l2_zero": float(np.mean(y64 ** 2)),
        "val_ic_fwd": ic_summary(ic_va),
        "gain_share": {name: float(g) / total_gain for name, g in zip(names, gain)},
        "seconds": {"fit": t_fit - t0, "total": time.time() - t0},
    }
    return j_te, d_te, pred, record


def walk_forward(dev: DevData, years: list[int], cfg: Config, h: int = H, null: str | None = None, null_seed: int = 0,
                 on_year=None, price_test: str | None = None, delist: str = "main",
                 ext: tuple[tuple[str, ...], np.ndarray] | None = None) -> tuple[np.ndarray, list[dict]]:
    """Scores float32 [N, T] (NaN outside the chosen years' test decisions) and one record per year."""
    t0 = time.time()
    panel = build_panel(dev, cfg, h, null, null_seed, price_test, delist, ext)
    t_panel = time.time() - t0
    by_year = {s.year: s for s in splits.dev_splits(dev.export.days, horizon=h)}
    scores = np.full((dev.export.n, dev.export.t), np.nan, dtype=np.float32)
    records = []
    for y in years:
        split = by_year[y]
        j, d, pred, rec = fit_year(panel, split, cfg)
        if d.size and not (split.test_decisions.start <= d.min() and d.max() < split.test_decisions.stop):
            raise RuntimeError(f"year {y}: scores outside its test decisions")
        scores[j, d] = pred
        rec["seconds"]["panel"] = t_panel if not records else 0.0
        records.append(rec)
        if on_year:
            on_year(rec, scores)
    if np.isfinite(scores[:, dev.cutoff:]).any():
        raise AssertionError("a score at or after the development cutoff")
    return scores, records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.h30.model", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None, help="c2 export directory (default: $C2_DATA_DIR/export/full)")
    ap.add_argument("--out", required=True, help="output directory: scores.npy and train_log.jsonl")
    ap.add_argument("--years", default=None, help="test years, e.g. 2010-2023 (default)")
    ap.add_argument("--horizon", type=int, default=H, help=f"trading days ahead (default {H}: a calendar month)")
    ap.add_argument("--null", choices=NULLS, default=None, help="negative control: sign-flip the target per day")
    ap.add_argument("--null-seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="smoke run: " + ", ".join(f"{k}={v}" for k, v in QUICK.items()))
    ap.add_argument("--threads", type=int, default=None, help="LightGBM threads (default: min(16, cores))")
    ap.add_argument("--price-test", choices=("traded", "adjusted"), default=None,
                    help="closes for the $5 liquidity test (default: traded when the export has them, else adjusted)")
    ap.add_argument("--delist", choices=DELIST_RULES, default="main", help="outcome of a stock delisted in its window")
    ap.add_argument("--external", default=None, metavar="DIR",
                    help="add correlation2's outside series as time-only features (stockai.h30.external; DIR from "
                         "`stockai-judge export-external`, or 'default' for ~/.local/share/stockai/external)")
    ap.add_argument("--external-shift", type=int, default=0, metavar="DAYS",
                    help="control: roll every outside series by DAYS trading days before building the features")
    ap.add_argument("--external-exclude", default="", metavar="CATS",
                    help="comma list of outside categories to leave out ('revised' = " + ",".join(REVISED) + ")")
    a = ap.parse_args(argv)
    fields = dict(QUICK) if a.quick else {}
    if a.threads is not None:
        fields["threads"] = a.threads
    cfg = Config(**fields)
    years = parse_years(a.years)
    dev = load_dev(a.export)
    price_test = a.price_test or dev.default_price_test()
    ext, ext_record = None, None
    if a.external:
        exclude = tuple(REVISED) if a.external_exclude == "revised" else tuple(c for c in a.external_exclude.split(",") if c)
        xt = external.load(None if a.external == "default" else a.external, dev.export.days, dev.export.meta.get("fingerprint"))
        ext = external.features(xt, dev.cutoff, a.external_shift, exclude)
        ext_record = {"dir": str(xt.path), "shift": a.external_shift, "exclude": list(exclude), "features": len(ext[0]),
                      "categories": external.categories(xt, exclude), "correlation2_commit": xt.info.get("correlation2_commit")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path, scores_path = out / "train_log.jsonl", out / "scores.npy"

    def write(rec: dict, mode: str = "a") -> None:
        with open(log_path, mode) as f:
            f.write(json.dumps(rec) + "\n")

    ex = dev.export
    write({"kind": "config", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv[1:] if argv is None else argv,
           "export": str(ex.path), "fingerprint": ex.meta.get("fingerprint"), "horizon": a.horizon, "null": a.null,
           "null_seed": a.null_seed if a.null else None, "years": years, "quick": a.quick,
           "price_test": price_test, "delist": a.delist if dev.delist_day is not None else None, "external": ext_record,
           "dev_cutoff": f"columns < {dev.cutoff} ({ex.date(dev.cutoff - 1)} is the last day read)",
           "features": list(FEATURES) + list(ext[0] if ext else ()), "config": dataclasses.asdict(cfg), "lgb_params": cfg.params(),
           "lightgbm": lgb.__version__, "stockai_git": _git_state()}, mode="w")
    print(f"stockai-30d LightGBM, horizon {a.horizon} days{', null ' + a.null if a.null else ''}: years {years[0]}..{years[-1]}"
          f"{' (quick)' if a.quick else ''}; reading days through {ex.date(dev.cutoff - 1)} only", flush=True)
    t0 = time.time()

    def on_year(rec: dict, scores: np.ndarray) -> None:
        write(rec)
        os.replace(data.write_scores(out / "scores.partial.npy", scores, ex), scores_path)
        print(f"  {rec['year']}: {rec['train_rows']:,} train rows, {rec['val_rows']:,} val rows, best iteration "
              f"{rec['best_iteration']}; val IC vs {a.horizon}-day returns {rec['val_ic_fwd']['mean']:+.4f} "
              f"(t {rec['val_ic_fwd']['t']:.1f}); {rec['seconds']['total'] + rec['seconds']['panel']:.0f}s", flush=True)

    scores, records = walk_forward(dev, years, cfg, a.horizon, a.null, a.null_seed, on_year, price_test, a.delist, ext)
    write({"kind": "summary", "years": years, "seconds": time.time() - t0, "scores": describe_scores(scores, ex.days)})
    print(f"wrote {scores_path} ({describe_scores(scores, ex.days)}) [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
