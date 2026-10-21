"""Walk-forward stockGPT: one model per test year and seed, written as scores files for correlation2's judge.

    uv run python -m stockai.gpt.walkforward --export DIR --out DIR [--horizon H] [--years 2010-2023] [--seeds 0 1 2]
        [--null sign-flip] [--quick] [--layers N --width N --heads N] [--compile]

`--horizon H` (trading days, default 1) sets what the model predicts from the close of day d (features.py): at 1 the
next day's census token (stockGPT v1, unchanged), above 1 how each stock's return over the next H days ranks among the
day's stocks (5: a week, 21: a month). The splits then keep gaps of max(5, H) days, so no outcome window of a
training or validation decision crosses into the next range, and early stopping uses the rank IC against the actual
H-day return. Grade H-day scores with `python -m stockai.h30.evaluate --horizon H --scores NAME=OUT/scores_ensemble.npy`
or the judge's `holdout --family scores --rebalance H`; this module grades nothing.

For each seed and each test year Y (stockai/splits.py), a model is trained on that split's training decisions,
early-stopped on its validation decisions, and scores the year's test decisions. Outputs in OUT:
- scores_seed{k}.npy: float32 [N, T] aligned to the export (stockai.data.write_scores), NaN where no score;
  rewritten after every year, so it always holds the years finished so far (train_log.jsonl lists them);
- scores_ensemble.npy: the mean of the seeds' scores, written at the end;
- train_log.jsonl: one line per (year, seed), appended as each finishes;
- checkpoints/seed{k}_year{Y}.pt: the best weights with their config and volatility constants;
- run.json: the command, configs, export identity and timings.
OUT must not hold a previous run's train_log.jsonl (pass --overwrite to replace it).

The locked year (2024-06-28 on) is never loaded: features.build_panel cuts it off, splits.split refuses it, and
every scores file is checked to have no finite value from the development cutoff (the locked year's first decision
day, stockai.baselines.common.dev_cutoff) on before writing.

Step budget: `--passes` (4) passes over the year's training positions, but at least `--min-steps` (400). On the real
export 4 passes is ~120 steps for 2010 (1.8M positions) and ~780 for 2023 (12.2M); 2010's validation showed the short
budget undertrained (+35 mbit and early-stopped at step 25, against +178 mbit at 400 steps), so the floor applies
to 2010-2016. Warmup is min(--warmup, 10% of the steps). Training uses deterministic CUDA kernels, so a (seed, year)
rerun reproduces its scores exactly (--nondeterministic trades that for ~8% faster steps).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .. import data, splits
from ..baselines.common import dev_cutoff
from . import features
from .model import GPTConfig
from .train import TrainConfig, score_days, train_year
from .windows import GpuPanel

QUICK_STEPS = 300
QUICK_EVAL_EVERY = 100


def parse_years(specs: list[str]) -> list[int]:
    """'2010-2023', '2010,2012', '2015' or several of these."""
    years: list[int] = []
    for spec in specs:
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                if hi < lo:
                    raise ValueError(f"bad year range {part!r}")
                years.extend(range(lo, hi + 1))
            else:
                years.append(int(part))
    out = sorted(set(years))
    if not out:
        raise ValueError("no years given")
    return out


def _clean(x):
    """JSON-safe: NaN/inf become null, numpy scalars become Python numbers."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, np.integer):
        return int(x)
    return x


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def check_locked_untouched(scores: np.ndarray, export: data.Export) -> None:
    cut = dev_cutoff(export.days)  # the locked year's first decision day, one before its first day
    if np.isfinite(scores[:, cut:]).any():
        raise AssertionError("a score was written on or after the development cutoff (the locked year's first "
                             "decision day)")


def ensemble(per_seed: list[np.ndarray]) -> np.ndarray:
    masks = [np.isfinite(s) for s in per_seed]
    for m in masks[1:]:
        if not np.array_equal(m, masks[0]):
            raise AssertionError("seeds scored different (stock, day) cells")
    return np.mean(np.stack(per_seed), axis=0).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m stockai.gpt.walkforward", description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", default=None, help="c2 export directory (default: $C2_DATA_DIR/export/full)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--years", nargs="+", default=["2010-2023"], help="test years: 2010-2023, 2010,2012 or 2015")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--horizon", type=int, default=1,
                    help="trading days ahead: 1 = the next day's census token (stockGPT v1); H > 1 = the day's rank of "
                         "each stock's H-day return (features.py)")
    ap.add_argument("--null", choices=features.NULLS, default=None, help="negative control: sign-flip v per date")
    ap.add_argument("--null-seed", type=int, default=0, help="seed of the null's per-date signs (one per run)")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke run: the last 2 years, the first seed, {QUICK_STEPS} steps")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--passes", type=float, default=4.0, help="step budget in passes over the training positions")
    ap.add_argument("--steps", type=int, default=None, help="fixed number of steps (overrides --passes)")
    ap.add_argument("--min-steps", type=int, default=400,
                    help="floor on the step budget (4 passes is ~120 steps for 2010 on the real export)")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-stride", type=int, default=1, help="evaluate every k-th validation day while training")
    ap.add_argument("--eval-batch", type=int, default=2048)
    ap.add_argument("--compile", action="store_true", help="torch.compile, after checking it against eager")
    ap.add_argument("--nondeterministic", action="store_true",
                    help="allow nondeterministic CUDA kernels in training (~8%% faster steps, not reproducible)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-checkpoints", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="replace a previous run's train_log.jsonl in OUT")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t_run = time.time()
    years = parse_years(args.years)
    seeds = list(dict.fromkeys(args.seeds))
    if args.quick:
        years, seeds = years[-2:], seeds[:1]
    mcfg = GPTConfig(n_layer=args.layers, n_head=args.heads, n_embd=args.width, ctx=args.ctx, dropout=args.dropout)
    tcfg = TrainConfig(batch=args.batch, lr=args.lr, min_lr=args.min_lr, warmup=args.warmup, passes=args.passes,
                       max_steps=args.steps, min_steps=args.min_steps, eval_every=args.eval_every,
                       patience=args.patience, val_stride=args.val_stride, eval_batch=args.eval_batch,
                       compile=args.compile, deterministic=not args.nondeterministic)
    if args.quick:
        tcfg = replace(tcfg, max_steps=min(args.steps or QUICK_STEPS, QUICK_STEPS),
                       eval_every=args.eval_every or QUICK_EVAL_EVERY)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train_log.jsonl").exists():
        if not args.overwrite:
            raise SystemExit(f"{out}/train_log.jsonl exists: use a new --out, or --overwrite to replace that run")
        (out / "train_log.jsonl").unlink()

    export = data.load(args.export)
    # Gaps of max(5, H) days; refuses the locked year and later.
    split_list = [splits.split(export.days, y, horizon=args.horizon) for y in years]
    _log(f"export {export.path}: {export.n} stocks x {export.t} days ({export.meta.get('first_day')} .. "
         f"{export.meta.get('last_day')}), fingerprint {export.meta.get('fingerprint')}")
    _log(f"years {years}, seeds {seeds}, horizon {args.horizon}, null {args.null}, device {device}"
         + (f" ({torch.cuda.get_device_name(device)}, {torch.cuda.mem_get_info(device)[0] / 1e9:.1f} GB free)"
            if device.type == "cuda" else ""))

    t0 = time.time()
    panel = features.build_panel(export, null=args.null, null_seed=args.null_seed,
                                 device=str(device) if device.type == "cuda" else "cpu", horizon=args.horizon)
    gp = GpuPanel(panel, device)
    panel_seconds = time.time() - t0
    _log(f"panel: {panel.n} x {panel.t} days (cut before {splits.LOCKED_YEAR}-{splits.PERIOD_START}) "
         f"in {panel_seconds:.1f}s")

    run = {
        "argv": sys.argv if argv is None else ["python", "-m", "stockai.gpt.walkforward", *argv],
        "years": years,
        "seeds": seeds,
        "horizon": args.horizon,
        "target": features.target_info(export, args.horizon),
        "null": args.null,
        "null_seed": args.null_seed if args.null else None,
        "quick": args.quick,
        "model": mcfg.to_dict(),
        "train": tcfg.to_dict(),
        "export": {"path": str(export.path), **{k: export.meta.get(k) for k in
                                                ("format", "n", "t", "first_day", "last_day", "fingerprint",
                                                 "git_commit")}},
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "panel_seconds": panel_seconds,
    }
    (out / "run.json").write_text(json.dumps(_clean(run), indent=2))
    log_path = out / "train_log.jsonl"
    per_seed: list[np.ndarray] = []
    for seed in seeds:
        scores = data.empty_scores(export)
        for sp in split_list:
            _log(f"seed {seed} year {sp.year}")
            res = train_year(panel, gp, sp, mcfg, tcfg, seed, log=_log)
            t1 = time.time()
            scores[:, sp.test_decisions.start:sp.test_decisions.stop] = score_days(res, gp, panel, sp.test_decisions,
                                                                                    tcfg)
            score_seconds = time.time() - t1
            info = {**res.info, "score_seconds": score_seconds,
                    "test_scored": int(np.isfinite(scores[:, sp.test_decisions.start:sp.test_decisions.stop]).sum()),
                    "total_seconds": res.info["train_seconds"] + score_seconds,
                    "model": mcfg.to_dict(), "train_config": tcfg.to_dict()}
            with open(log_path, "a") as f:
                f.write(json.dumps(_clean(info)) + "\n")
            if not args.no_checkpoints:
                (out / "checkpoints").mkdir(exist_ok=True)
                torch.save({"state_dict": res.model.state_dict(), "model": mcfg.to_dict(), "vol_mu": res.mu,
                            "vol_sd": res.sd, "info": {k: v for k, v in info.items() if k != "history"}},
                           out / "checkpoints" / f"seed{seed}_year{sp.year}.pt")
            check_locked_untouched(scores, export)
            data.write_scores(out / f"scores_seed{seed}.npy", scores, export)
            _log(f"  done: best step {info['best_step']}/{info['steps']}, val ic actual {info['val_ic_actual']:+.4f}, "
                 f"ic v {info['val_ic_v']:+.4f}, {info['bits_gain'] * 1000:+.2f} mbit; train {info['train_seconds']:.0f}s"
                 f" (eval {info['eval_seconds']:.0f}s) + score {score_seconds:.0f}s")
        per_seed.append(scores)
    ens = ensemble(per_seed)
    check_locked_untouched(ens, export)
    data.write_scores(out / "scores_ensemble.npy", ens, export)
    run["total_seconds"] = time.time() - t_run
    (out / "run.json").write_text(json.dumps(_clean(run), indent=2))
    _log(f"wrote {out} in {run['total_seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
