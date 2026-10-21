"""LightGBM walk-forward baseline: the "something else" every AI result is compared against (PLAN.md §2, §3.1, M1).

For each development test year (stockai.splits.dev_splits, horizon 1), one model is trained on that split's training
decision days, early-stopped on the L2 of its validation days, and then scores every test decision day for every
stock with complete features (features.py); NaN elsewhere. The target (`--target`) is one of:

- `v` (the default, and the POC's run): v[d+1], the next day's census-variable move. v subtracts each stock's
  trailing 252-day mean residual, so part of v[d+1] is known at d; a model of v learns that part, which is not in
  returns (test-results.md, docs/c2-export.md).
- `ret`: the next day's actual return close[d+1] / close[d] - 1, which is what the account earns, as a daily
  cross-sectional rank-gauss score: on each day, the return's rank among that day's training rows (tied returns share
  their average rank) mapped to the standard normal quantile ndtri((rank - 0.5) / n). That is robust to fat tails
  and to market-wide days. Validation L2, and so early stopping, is then in those units; the rank IC against
  returns is the rank IC against this target, since the transform keeps each day's order.

Training and validation rows are stocks that pass a point-in-time liquidity filter at day d (close >= $5 and a 63-day
median dollar volume >= $5M), with complete features and a known target. When a year has more training rows than
`max_train_rows`, a seeded random subset of whole training days is kept (every stock on a kept day). Either target
reads day d+1 only for training and validation decision days d, which splits.py ends early enough that every such
outcome is known before the first test decision; test rows read features only.

Model code never grades itself on test years: train_log.jsonl holds validation metrics only (L2 against the zero
forecast, and the mean daily Spearman rank IC against next-day actual returns and next-day v, for either target).
The test years are graded by `c2 census holdout --family scores`.

    uv run python -m stockai.baselines.lgbm --export DIR --out DIR [--years 2010-2023] [--target v|ret] [--quick]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

from .. import data, splits
from .common import DevData, describe_scores, load_dev, parse_years
from .features import FEATURES, Features, compute_features, liquid_mask

TARGETS = ("v", "ret")  # v[d+1]; or the next-day actual return as a daily rank-gauss score (module docstring)


@dataclass
class Config:
    target: str = "v"
    min_price: float = 5.0
    min_dollar_volume: float = 5e6
    liquidity_window: int = 63
    # 4M rows and 1,000 rounds keep a year's fit near 10 s and at most ~1 minute on the Core Ultra 9 285K (measured
    # 0.04-0.06 s per round at 4M rows with no other load).
    max_train_rows: int = 4_000_000
    rounds: int = 1000
    early_stopping: int = 100
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 2000
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.7
    bagging_freq: int = 1
    lambda_l2: float = 10.0
    max_bin: int = 255
    # 0: min(16, cores). On the 8 P + 16 E core 285K, 16 threads fit faster than 24 (7.6 s against 10.4 s) or 8.
    threads: int = 0
    seed: int = 0
    min_ic_stocks: int = 20  # validation days with fewer scored stocks are left out of the IC

    def params(self) -> dict:
        return {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "max_bin": self.max_bin,
            "num_threads": self.threads or min(16, os.cpu_count() or 1),
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }


# --quick: a smoke run, not a result.
QUICK = {"max_train_rows": 300_000, "rounds": 300, "early_stopping": 30, "learning_rate": 0.1, "num_leaves": 31,
         "min_data_in_leaf": 200}


@dataclass
class Panel:
    """Everything the walk-forward reads, for the day columns [0, cutoff) of a DevData only."""

    days: np.ndarray  # int32 [C]
    feats: Features
    trainable: np.ndarray  # bool [N, C]: liquid at d, complete features, known target
    # float32 [N, C]: the model's target (target_name): v_next itself for "v"; for "ret", rank_gauss of ret_next over
    # the day's liquid rows with complete features and a next-day return. NaN where unknown and in the last column.
    target: np.ndarray
    ret_next: np.ndarray  # float64 [N, C]: close[d+1] / close[d] - 1 (data.next_day_returns)
    v_next: np.ndarray  # float32 [N, C]: v[j, d+1]; NaN where unknown and in the last column
    target_name: str = "v"


def rank_gauss(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """float32 [N, C]: on each day (column) d, the finite values x[j, d] of the rows j with rows[j, d] become the
    standard normal quantile of their rank among that day's such rows, ndtri((rank - 0.5) / n), with tied values
    sharing their average rank; NaN elsewhere. Each column uses that column only."""
    ok = rows & np.isfinite(x)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for d in np.flatnonzero(ok.any(axis=0)):
        j = np.flatnonzero(ok[:, d])
        out[j, d] = ndtri((rankdata(x[j, d], method="average") - 0.5) / j.size)
    return out


def build_panel(dev: DevData, cfg: Config) -> Panel:
    if cfg.target not in TARGETS:
        raise ValueError(f"target {cfg.target!r}: expected one of {TARGETS}")
    feats = compute_features(dev.v, dev.close, dev.dollar_volume, dev.market)
    liquid = liquid_mask(dev.close, dev.dollar_volume, cfg.min_price, cfg.min_dollar_volume, cfg.liquidity_window)
    v = np.asarray(dev.v, dtype=np.float32)
    v_next = np.full(v.shape, np.nan, dtype=np.float32)
    v_next[:, :-1] = v[:, 1:]
    ret_next = data.next_day_returns(dev.close)
    if cfg.target == "v":
        target = v_next
    else:  # ranked among the rows that can train that day, so the rows a model sees are standard normal each day
        target = rank_gauss(ret_next, liquid & feats.complete & np.isfinite(ret_next))
    trainable = liquid & feats.complete & np.isfinite(target)
    return Panel(days=np.asarray(dev.days), feats=feats, trainable=trainable, target=target, ret_next=ret_next,
                 v_next=v_next, target_name=cfg.target)


def mask_rows(mask: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(stock j, day d) of the True entries of mask[:, cols], ordered by day, then stock."""
    k, j = np.nonzero(mask[:, cols].T)
    return j, cols[k]


def daily_ic(pred: np.ndarray, outcome: np.ndarray, day: np.ndarray, min_n: int) -> np.ndarray:
    """Spearman rank IC of pred against outcome on each day (rows grouped by day), over rows with a finite outcome.

    Days with fewer than `min_n` such rows are left out; a day with constant ranks counts as 0, as in correlation2.
    """
    ok = np.isfinite(outcome)
    p, o, dd = pred[ok], outcome[ok], day[ok]
    if dd.size == 0:
        return np.empty(0)
    cut = np.flatnonzero(np.diff(dd)) + 1
    out = []
    for a, b in zip(np.r_[0, cut], np.r_[cut, dd.size]):
        if b - a < min_n:
            continue
        rp, ro = rankdata(p[a:b]), rankdata(o[a:b])
        rp, ro = rp - rp.mean(), ro - ro.mean()
        den = np.sqrt((rp * rp).sum() * (ro * ro).sum())
        out.append(float((rp * ro).sum() / den) if den > 0 else 0.0)
    return np.asarray(out)


def ic_summary(ics: np.ndarray) -> dict:
    if ics.size == 0:
        return {"mean": None, "t": None, "days": 0}
    m, sd = float(ics.mean()), float(ics.std())
    return {"mean": m, "t": m / max(sd, 1e-12) * np.sqrt(ics.size), "days": int(ics.size)}


def _span(days: np.ndarray, cols: np.ndarray) -> str:
    return f"{data.day_to_str(days[cols[0]])}..{data.day_to_str(days[cols[-1]])}" if cols.size else "none"


@dataclass
class YearResult:
    booster: lgb.Booster
    record: dict
    j: np.ndarray  # test rows: stock
    d: np.ndarray  # test rows: decision day, always inside split.test_decisions
    pred: np.ndarray  # float64


def fit_year(panel: Panel, split: splits.Split, cfg: Config) -> YearResult:
    """Trains on split.train_decisions, early-stops on split.val_decisions, scores split.test_decisions."""
    if split.horizon != 1:
        raise ValueError("the target is the next day's move: horizon must be 1")
    if cfg.target != panel.target_name:
        raise ValueError(f"the panel holds target {panel.target_name!r} but the config asks for {cfg.target!r}")
    c = panel.days.size
    if split.test_decisions.stop > c:
        raise ValueError(f"year {split.year}: test decisions run past the development cutoff ({c} columns)")
    t0 = time.time()
    tr_all = np.arange(split.train_decisions.start, split.train_decisions.stop)
    per_day = panel.trainable[:, tr_all].sum(axis=0)
    available = int(per_day.sum())
    tr_cols = tr_all
    if available > cfg.max_train_rows:  # keep a seeded random subset of whole days
        order = np.random.default_rng([cfg.seed, split.year]).permutation(tr_all.size)
        tr_cols = np.sort(tr_all[order[np.cumsum(per_day[order]) <= cfg.max_train_rows]])
    j_tr, d_tr = mask_rows(panel.trainable, tr_cols)
    va_cols = np.arange(split.val_decisions.start, split.val_decisions.stop)
    j_va, d_va = mask_rows(panel.trainable, va_cols)
    if j_tr.size == 0 or j_va.size == 0:
        raise ValueError(f"year {split.year}: {j_tr.size} training rows and {j_va.size} validation rows")
    x_tr, y_tr = panel.feats.rows(j_tr, d_tr), panel.target[j_tr, d_tr]
    x_va, y_va = panel.feats.rows(j_va, d_va), panel.target[j_va, d_va]
    t_rows = time.time()

    params = cfg.params()
    dtr = lgb.Dataset(x_tr, label=y_tr, feature_name=list(FEATURES), params={"max_bin": cfg.max_bin, "verbosity": -1},
                      free_raw_data=True)
    dva = lgb.Dataset(x_va, label=y_va, reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=cfg.rounds, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(cfg.early_stopping, first_metric_only=True, verbose=False)])
    best = int(booster.best_iteration or booster.current_iteration())
    t_fit = time.time()

    p_va = booster.predict(x_va, num_iteration=best)
    ic_v = daily_ic(p_va, panel.v_next[j_va, d_va].astype(np.float64), d_va, cfg.min_ic_stocks)  # y_va for "v"
    ic_r = daily_ic(p_va, panel.ret_next[j_va, d_va], d_va, cfg.min_ic_stocks)

    te_cols = np.arange(split.test_decisions.start, split.test_decisions.stop)
    j_te, d_te = mask_rows(panel.feats.complete, te_cols)
    pred = booster.predict(panel.feats.rows(j_te, d_te), num_iteration=best) if j_te.size else np.empty(0)
    t_pred = time.time()

    gain = booster.feature_importance("gain", iteration=best)
    total_gain = float(gain.sum()) or 1.0
    y64 = y_va.astype(np.float64)
    record = {
        "kind": "year",
        "year": split.year,
        "target": panel.target_name,  # val_l2* are in this target's units
        "train": _span(panel.days, tr_all),
        "train_days": int(tr_all.size),
        "train_days_used": int(tr_cols.size),
        "train_rows_available": available,
        "train_rows": int(j_tr.size),
        "val": _span(panel.days, va_cols),
        "val_rows": int(j_va.size),
        "test": _span(panel.days, te_cols),
        "test_days": int(te_cols.size),
        "test_scored": int(j_te.size),
        "best_iteration": best,
        "val_l2": float(np.mean((p_va - y64) ** 2)),
        "val_l2_zero": float(np.mean(y64**2)),
        "val_l2_train_mean": float(np.mean((y64 - float(np.mean(y_tr, dtype=np.float64))) ** 2)),
        "val_ic_v": ic_summary(ic_v),
        "val_ic_ret": ic_summary(ic_r),
        "gain_share": {name: float(g) / total_gain for name, g in zip(FEATURES, gain)},
        "seconds": {"rows": t_rows - t0, "fit": t_fit - t_rows, "predict": t_pred - t_fit, "total": t_pred - t0},
    }
    return YearResult(booster=booster, record=record, j=j_te, d=d_te, pred=pred)


def walk_forward(dev: DevData, years: list[int], cfg: Config, on_year=None) -> tuple[np.ndarray, list[dict]]:
    """Scores float32 [N, T] (NaN outside the chosen years' test decisions) and one record per year.

    `on_year(record, scores)` is called after each year, e.g. to save progress.
    """
    t0 = time.time()
    panel = build_panel(dev, cfg)
    t_panel = time.time() - t0
    by_year = {s.year: s for s in splits.dev_splits(dev.export.days, horizon=1)}
    scores = np.full((dev.export.n, dev.export.t), np.nan, dtype=np.float32)
    records = []
    for y in years:
        split = by_year[y]
        res = fit_year(panel, split, cfg)
        if res.d.size and not (split.test_decisions.start <= res.d.min() and res.d.max() < split.test_decisions.stop):
            raise RuntimeError(f"year {y}: scores outside its test decisions")  # never reached; a hard rule
        scores[res.j, res.d] = res.pred
        res.record["seconds"]["panel"] = t_panel if not records else 0.0
        records.append(res.record)
        if on_year:
            on_year(res.record, scores)
    return scores, records


def _git_state() -> dict:
    root = Path(__file__).resolve().parents[2]
    git = ["git", "--no-optional-locks", "-C", str(root)]
    try:
        head = subprocess.run([*git, "rev-parse", "--verify", "-q", "HEAD"], capture_output=True, text=True, timeout=10)
        dirty = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": head.stdout.strip() if head.returncode == 0 else None,
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None}


def _fmt(ic: dict) -> str:
    return "n/a" if ic["mean"] is None else f"{ic['mean']:+.4f} (t {ic['t']:.1f})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m stockai.baselines.lgbm", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None, help="c2 export directory (default: $C2_DATA_DIR/export/full)")
    ap.add_argument("--out", required=True, help="output directory: scores.npy and train_log.jsonl")
    ap.add_argument("--years", default=None, help="test years, e.g. 2010-2023 (default), 2015 or 2010,2012")
    ap.add_argument("--quick", action="store_true", help="smoke run: " + ", ".join(f"{k}={v}" for k, v in QUICK.items()))
    ap.add_argument("--target", choices=TARGETS, default=None,
                    help="v: next-day v (default); ret: next-day actual return as a daily rank-gauss score")
    ap.add_argument("--threads", type=int, default=None, help="LightGBM threads (default: min(16, cores))")
    ap.add_argument("--max-train-rows", type=int, default=None, help="cap on training rows per year")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--min-price", type=float, default=None, help="liquidity filter: close >= this (default 5)")
    ap.add_argument("--min-dollar-volume", type=float, default=None, help="63-day median dollar volume (default 5e6)")
    a = ap.parse_args(argv)

    fields = dict(QUICK) if a.quick else {}
    for name in ("target", "threads", "max_train_rows", "seed", "min_price", "min_dollar_volume"):
        if getattr(a, name) is not None:
            fields[name] = getattr(a, name)
    cfg = Config(**fields)
    years = parse_years(a.years)
    dev = load_dev(a.export)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path, scores_path = out / "train_log.jsonl", out / "scores.npy"

    def write(rec: dict, mode: str = "a") -> None:
        with open(log_path, mode) as f:
            f.write(json.dumps(rec) + "\n")

    ex = dev.export
    write({"kind": "config", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv[1:] if argv is None
           else argv, "export": str(ex.path), "fingerprint": ex.meta.get("fingerprint"),
           "export_git_commit": ex.meta.get("git_commit"), "n": ex.n, "t": ex.t,
           "dev_cutoff": f"columns < {dev.cutoff} ({ex.date(dev.cutoff - 1)} is the last day read)",
           "years": years, "quick": a.quick, "features": list(FEATURES), "config": dataclasses.asdict(cfg),
           "lgb_params": cfg.params(), "lightgbm": lgb.__version__, "numpy": np.__version__, "stockai_git": _git_state()},
          mode="w")
    print(f"LightGBM walk-forward, target {cfg.target}: {ex.n} stocks x {ex.t} days from {ex.path}; years "
          f"{years[0]}..{years[-1]}{' (quick)' if a.quick else ''}; reading days through {ex.date(dev.cutoff - 1)} only",
          flush=True)
    t0 = time.time()

    def on_year(rec: dict, scores: np.ndarray) -> None:
        write(rec)
        tmp = data.write_scores(out / "scores.partial.npy", scores, ex)
        os.replace(tmp, scores_path)  # scores.npy always holds the years finished so far
        s = rec["seconds"]
        print(f"  {rec['year']}: {rec['train_rows']:,} train rows ({rec['train_days_used']}/{rec['train_days']} days), "
              f"{rec['val_rows']:,} val rows, best iteration {rec['best_iteration']}; val L2 {rec['val_l2']:.5f} vs "
              f"zero {rec['val_l2_zero']:.5f}; val IC v {_fmt(rec['val_ic_v'])}, returns {_fmt(rec['val_ic_ret'])}; "
              f"{rec['test_scored']:,} test scores; {s['total'] + s['panel']:.0f}s", flush=True)

    scores, records = walk_forward(dev, years, cfg, on_year=on_year)
    ic_v = [r["val_ic_v"]["mean"] for r in records if r["val_ic_v"]["mean"] is not None]
    ic_r = [r["val_ic_ret"]["mean"] for r in records if r["val_ic_ret"]["mean"] is not None]
    write({"kind": "summary", "years": years, "seconds": time.time() - t0,
           "mean_val_ic_v": float(np.mean(ic_v)) if ic_v else None,
           "mean_val_ic_ret": float(np.mean(ic_r)) if ic_r else None, "scores": describe_scores(scores, ex.days)})
    print(f"wrote {scores_path} ({describe_scores(scores, ex.days)}) and {log_path} [{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
