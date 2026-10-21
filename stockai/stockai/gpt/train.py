"""Training, validation and scoring of stockGPT for one walk-forward split (PLAN.md §3.2).

The target of decision day d looks H = split.horizon trading days ahead (features.py): at H = 1 the token of day d+1
(next-token prediction, stockGPT v1), at H > 1 the bin of the day's rank-gauss score of the H-day forward return.
- Training: cross-entropy of each position's target bin on windows sampled on the GPU from the split's training
  tensor (windows.py). AdamW, linear warmup then cosine decay, grad clipping, bf16 autocast. The step budget is a
  number of passes over the training positions.
- Validation (split.val_decisions only): for every validation day d and every stock with a finite v[d] passing the
  point-in-time liquidity filter, the score from the full-context window ending at d. Reports cross-entropy of the
  target token against the training-target bin histogram, and the mean daily Spearman rank IC of the score against
  the actual return over the next H days and against the sum of v over them (the finite ones; the judge's census-unit
  outcome; at H = 1 the next day's v). Early stopping on the actual-return IC keeps the best weights. At H > 1 the
  validation days' outcomes overlap (one H-day window a day), so the IC's t overstates its precision, by roughly
  sqrt(H).
- Scoring: score[j, d] = the expected target under the predicted bin probabilities (at H = 1 the expected v at d+1; at
  H > 1 the expected rank-gauss score of the H-day return), from the window ending at d, for every stock with a finite
  v[d]; NaN elsewhere.
"""

from __future__ import annotations

import contextlib
import math
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .. import splits
from .features import Panel
from .metrics import daily_rank_ic, ic_summary
from .model import GPTConfig, StockGPT
from .tokens import MISSING, N_BINS
from .windows import IGNORE, GpuPanel, TrainSet, check_batch, windows_ending


@dataclass
class TrainConfig:
    batch: int = 256
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 200
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    passes: float = 4.0  # step budget: passes over the training positions
    max_steps: int | None = None  # overrides `passes` and `min_steps`
    min_steps: int = 400  # floor for the small early years (4 passes is ~120 steps for 2010; see walkforward docs)
    eval_every: int | None = None  # default: about max_steps / 6, between 25 and 250
    patience: int = 3  # evaluations without a better validation IC before stopping
    val_stride: int = 1  # evaluate every k-th validation day during training (the final metrics use all days)
    eval_batch: int = 2048
    bf16: bool = True
    compile: bool = False
    compile_tol: float = 1e-2
    deterministic: bool = True  # torch.use_deterministic_algorithms while training (~8% slower steps)

    def to_dict(self) -> dict:
        return asdict(self)


def run_seed(seed: int, year: int) -> int:
    """The torch/numpy seed for one (seed, year) model."""
    return int(np.random.SeedSequence([int(seed), int(year)]).generate_state(1)[0])


def seed_everything(s: int) -> None:
    np.random.seed(s % (2**32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def lr_at(step: int, max_steps: int, warmup: int, lr: float, min_lr: float) -> float:
    """Linear warmup over `warmup` steps, then cosine decay from lr to min_lr at max_steps (step counts from 0)."""
    if step < warmup:
        return lr * (step + 1) / warmup
    p = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * p))


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


@contextlib.contextmanager
def _deterministic(enabled: bool):
    """Deterministic CUDA kernels (embedding and attention backward) for the duration, then the previous setting.

    Needs CUBLAS_WORKSPACE_CONFIG before CUDA starts; stockai/gpt/__init__.py sets it."""
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def census_sum(v: np.ndarray, rows: np.ndarray, days: np.ndarray, h: int) -> np.ndarray:
    """float64: the sum of the finite v[j, d+1 .. d+h] of each pair (j, d), NaN where none is finite (the judge's
    forward census-variable sum, holdout.rs `fwd_resid`); at h = 1, v[j, d+1]."""
    x = v[rows[:, None], days[:, None] + np.arange(1, h + 1)[None, :]].astype(np.float64)
    ok = np.isfinite(x)
    return np.where(ok.any(axis=1), np.where(ok, x, 0.0).sum(axis=1), np.nan)


class Pairs:
    """(stock, decision day) pairs to score, with their outcomes over days d+1 .. d+H (validation only)."""

    def __init__(self, panel: Panel, days, device: torch.device, liquid_only: bool, outcomes: bool):
        rows, ds = [], []
        for d in days:
            ok = np.isfinite(panel.v[:, d])
            if liquid_only:
                ok &= panel.liquid[:, d]
            j = np.flatnonzero(ok)
            rows.append(j)
            ds.append(np.full(j.size, d, dtype=np.int64))
        self.rows_np = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        self.days_np = np.concatenate(ds) if ds else np.zeros(0, np.int64)
        self.rows = torch.as_tensor(self.rows_np, device=device)
        self.days = torch.as_tensor(self.days_np, device=device)
        if outcomes:
            h = panel.horizon
            if self.days_np.size and self.days_np.max() + h >= panel.t:
                raise ValueError("an outcome day is outside the panel")
            self.target = panel.target[self.rows_np, self.days_np].astype(np.int64)
            self.ret = panel.ret_fwd[self.rows_np, self.days_np].astype(np.float64)
            self.v_next = census_sum(panel.v, self.rows_np, self.days_np, h)

    def __len__(self) -> int:
        return int(self.rows_np.size)


@torch.no_grad()
def predict(fwd, gp: GpuPanel, rows: torch.Tensor, ends: torch.Tensor, ctx: int, mu: float, sd: float,
            targets: np.ndarray | None = None, batch: int = 2048, bf16: bool = True, pad: bool = False):
    """Scores (the expected target: next-day v at horizon 1) for windows ending at `ends`, and log-probabilities of
    `targets` if given."""
    model = getattr(fwd, "_orig_mod", fwd)
    model.eval()
    p = rows.numel()
    scores = torch.empty(p, dtype=torch.float32, device=gp.device)
    tgt = torch.as_tensor(targets, device=gp.device) if targets is not None else None
    logp = torch.full((p,), float("nan"), dtype=torch.float32, device=gp.device) if tgt is not None else None
    centers = model.centers
    for a in range(0, p, batch):
        r, e = rows[a:a + batch], ends[a:a + batch]
        n = r.numel()
        if pad and n < batch:  # fixed shapes for torch.compile
            r = torch.cat([r, r[:1].expand(batch - n)])
            e = torch.cat([e, e[:1].expand(batch - n)])
        tok, cont = windows_ending(gp, r, e, ctx, mu, sd)
        with _autocast(gp.device, bf16):
            logits = fwd(tok, cont, last_only=True)
        lp = torch.log_softmax(logits.float(), dim=-1)[:n]
        scores[a:a + n] = lp.exp() @ centers
        if tgt is not None:
            t = tgt[a:a + n]
            ok = t != MISSING
            got = lp.gather(1, t.clamp(max=N_BINS - 1)[:, None])[:, 0]
            logp[a:a + n] = torch.where(ok, got, torch.nan)
    return scores, logp


def evaluate(fwd, gp: GpuPanel, pairs: Pairs, hist_logp: np.ndarray, ctx: int, mu: float, sd: float,
             cfg: TrainConfig) -> dict:
    t0 = time.time()
    scores, logp = predict(fwd, gp, pairs.rows, pairs.days, ctx, mu, sd, targets=pairs.target, batch=cfg.eval_batch,
                           bf16=cfg.bf16, pad=cfg.compile)
    s = scores.cpu().numpy().astype(np.float64)
    lp = logp.cpu().numpy().astype(np.float64)
    ok = np.isfinite(lp)
    ce = float(-lp[ok].mean()) if ok.any() else float("nan")
    ce_hist = float(-hist_logp[pairs.target[ok]].mean()) if ok.any() else float("nan")
    ic_a = ic_summary(daily_rank_ic(pairs.days_np, s, pairs.ret))
    ic_v = ic_summary(daily_rank_ic(pairs.days_np, s, pairs.v_next))
    return {
        "ce": ce,
        "ce_hist": ce_hist,
        "bits_gain": (ce_hist - ce) / math.log(2),
        "ic_actual": ic_a["mean"],
        "ic_actual_t": ic_a["t"],
        "ic_v": ic_v["mean"],
        "ic_v_t": ic_v["t"],
        "ic_days": ic_a["days"],
        "pairs": len(pairs),
        "ce_tokens": int(ok.sum()),
        "score_sd": float(np.nanstd(s)) if s.size else float("nan"),
        "eval_seconds": time.time() - t0,
    }


def compile_checked(model: StockGPT, tok: torch.Tensor, cont: torch.Tensor, cfg: TrainConfig):
    """torch.compile the model after checking it against eager on a fixed batch (pytorch#191433).

    The bug silently corrupts bf16 embedding + Linear + norm blocks on the RTX 5080 (relative errors ~2.2 in the
    report), so outputs are compared under the same bf16 autocast used for training. Compared, each against
    `cfg.compile_tol`: the predicted bin probabilities and the expected-value scores (max abs difference; both O(1)
    whatever the weights), and the logits as a relative RMS difference ||compiled - eager|| / ||eager||, full and
    last-position. bf16 rounding alone gives ~1e-3 on the latter; a max-abs logit criterion would trip on single
    bf16 ulps. For context the stats also give each one's relative RMS error against an fp32 eager reference.
    The comparison runs the eval-mode graphs (no dropout); training then runs the compiled train-mode graph.
    """
    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).norm() / b.norm().clamp_min(1e-12))

    compiled = torch.compile(model)
    model.eval()
    with torch.no_grad():
        ref = model(tok, cont).float()
        with _autocast(tok.device, cfg.bf16):
            le = model(tok, cont).float()
            lc = compiled(tok, cont).float()
            se = model(tok, cont, last_only=True).float()
            sc = compiled(tok, cont, last_only=True).float()
    checked = {
        "prob_max_abs": float((torch.softmax(le, -1) - torch.softmax(lc, -1)).abs().max()),
        "score_max_abs": float((model.expected_value(se) - model.expected_value(sc)).abs().max()),
        "logit_rel_rms": rel(lc, le),
        "last_logit_rel_rms": rel(sc, se),
    }
    stats = {**checked, "eager_vs_fp32_rel_rms": rel(le, ref), "compiled_vs_fp32_rel_rms": rel(lc, ref)}
    worst = max(checked.values())
    if not math.isfinite(worst) or worst > cfg.compile_tol:
        raise RuntimeError(f"torch.compile output differs from eager by {worst:.3g} > {cfg.compile_tol} ({stats}); "
                           "see pytorch#191433. Run without --compile.")
    model.train()
    return compiled, stats


@dataclass
class YearResult:
    model: StockGPT
    fwd: object  # the module used for forward passes (compiled or the model itself)
    mu: float
    sd: float
    info: dict = field(default_factory=dict)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def train_year(panel: Panel, gp: GpuPanel, split: splits.Split, mcfg: GPTConfig, cfg: TrainConfig, seed: int,
               log=_log) -> YearResult:
    """Trains one model on split.train_decisions, early-stopped on split.val_decisions; returns the best weights."""
    with _deterministic(cfg.deterministic):
        return _train_year(panel, gp, split, mcfg, cfg, seed, log)


def _train_year(panel: Panel, gp: GpuPanel, split: splits.Split, mcfg: GPTConfig, cfg: TrainConfig, seed: int,
                log) -> YearResult:
    t_start = time.time()
    dev = gp.device
    s = run_seed(seed, split.year)
    seed_everything(s)
    ts = TrainSet(panel, gp, split, mcfg.ctx)
    model = StockGPT(mcfg).to(dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(s)
    aux = torch.Generator(device=dev)  # for the fill estimate and the compile check, so the training stream is fixed
    aux.manual_seed(s ^ 0x5A5A5A5A)

    fwd, compile_stats = model, None
    if cfg.compile:
        tok, cont, *_ = ts.sample(min(cfg.batch, 64), aux)
        fwd, compile_stats = compile_checked(model, tok, cont, cfg)
        log(f"  compile check passed: {compile_stats}")

    fill = float(np.mean([(ts.sample(cfg.batch, aux)[2] != IGNORE).float().mean().item() for _ in range(8)]))
    tokens_per_step = cfg.batch * mcfg.ctx * fill
    budget = int(math.ceil(cfg.passes * ts.n_valid / tokens_per_step))
    max_steps = cfg.max_steps if cfg.max_steps else max(cfg.min_steps, budget)
    warmup = min(cfg.warmup, max(1, max_steps // 10))
    eval_every = cfg.eval_every or min(250, max(25, math.ceil(max_steps / 6)))

    pairs = Pairs(panel, split.val_decisions[::cfg.val_stride], dev, liquid_only=True, outcomes=True)
    opt = torch.optim.AdamW(model.param_groups(cfg.weight_decay), lr=cfg.lr, betas=cfg.betas,
                            fused=dev.type == "cuda")
    log(f"  {split.describe(panel.days)}")
    log(f"  params {model.n_params():,}  train positions {ts.n_valid:,}  fill {fill:.3f}  steps {max_steps} "
        f"(warmup {warmup}, eval every {eval_every})  val pairs {len(pairs):,}")

    best_ic, best_step, best_state, best_metrics = -math.inf, 0, None, None
    history, bad, step, checked = [], 0, 0, 0
    loss_sum = torch.zeros((), device=dev)
    loss_n = 0
    eval_seconds = 0.0
    t_train = time.time()
    for step in range(1, max_steps + 1):
        lr = lr_at(step - 1, max_steps, warmup, cfg.lr, cfg.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        tok, cont, y, rows, day = ts.sample(cfg.batch, gen)
        if step == 1:
            checked += check_batch(panel, split, y, rows, day)
        model.train()
        with _autocast(dev, cfg.bf16):
            logits = fwd(tok, cont)
        loss = F.cross_entropy(logits.float().reshape(-1, N_BINS), y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        loss_sum += loss.detach()
        loss_n += 1
        if step % eval_every == 0 or step == max_steps:
            checked += check_batch(panel, split, *ts.sample(cfg.batch, aux)[2:])
            m = evaluate(fwd, gp, pairs, ts.hist_logp, mcfg.ctx, ts.mu, ts.sd, cfg)
            eval_seconds += m["eval_seconds"]
            m.update(step=step, lr=lr, train_loss=float(loss_sum.item() / max(1, loss_n)), elapsed=time.time() - t_start)
            loss_sum.zero_()
            loss_n = 0
            history.append(m)
            log(f"  step {step:5d}  train {m['train_loss']:.4f}  val ce {m['ce']:.4f} ({m['bits_gain'] * 1000:+.2f} mbit)"
                f"  ic actual {m['ic_actual']:+.4f} (t {m['ic_actual_t']:+.1f})  ic v {m['ic_v']:+.4f}"
                f"  [{m['elapsed']:.0f}s]")
            ic = m["ic_actual"] if math.isfinite(m["ic_actual"]) else -math.inf
            if best_state is None or ic > best_ic:
                best_ic, best_step, best_metrics = ic, step, m
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    log(f"  early stop at step {step}: no better validation IC since step {best_step}")
                    break
    train_loop_seconds = time.time() - t_train
    model.load_state_dict(best_state)
    final = best_metrics
    if cfg.val_stride > 1:  # the logged validation numbers use every validation day
        full = Pairs(panel, split.val_decisions, dev, liquid_only=True, outcomes=True)
        final = evaluate(fwd, gp, full, ts.hist_logp, mcfg.ctx, ts.mu, ts.sd, cfg)
        eval_seconds += final["eval_seconds"]
    info = {
        "year": split.year,
        "horizon": split.horizon,
        "seed": seed,
        "torch_seed": s,
        "null": panel.null,
        "null_seed": panel.null_seed,
        "split": split.describe(panel.days),
        "params": model.n_params(),
        "train_positions": ts.n_valid,
        "fill": fill,
        "max_steps": max_steps,
        "steps": step,
        "best_step": best_step,
        "warmup": warmup,
        "eval_every": eval_every,
        "val_ce": final["ce"],
        "val_ce_hist": final["ce_hist"],
        "bits_gain": final["bits_gain"],
        "val_ic_actual": final["ic_actual"],
        "val_ic_actual_t": final["ic_actual_t"],
        "val_ic_v": final["ic_v"],
        "val_ic_v_t": final["ic_v_t"],
        "val_pairs": final["pairs"],
        "vol_mu": ts.mu,
        "vol_sd": ts.sd,
        "train_seconds": time.time() - t_start,
        "train_loop_seconds": train_loop_seconds,
        "eval_seconds": eval_seconds,
        "compile_check": compile_stats,
        "checked_positions": checked,
        "history": [{k: v for k, v in h.items()} for h in history],
    }
    return YearResult(model=model, fwd=fwd, mu=ts.mu, sd=ts.sd, info=info)


def score_days(res: YearResult, gp: GpuPanel, panel: Panel, days: range, cfg: TrainConfig) -> np.ndarray:
    """float32 [N, len(days)]: the expected target (at horizon 1, v at d+1) from the window ending at d, for every stock
    with a finite v[d]."""
    idx = np.asarray(days, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= panel.t or np.any(np.diff(idx) <= 0)):
        raise ValueError("decision days must be increasing and inside the panel")
    pairs = Pairs(panel, idx, gp.device, liquid_only=False, outcomes=False)
    out = np.full((panel.n, idx.size), np.nan, dtype=np.float32)
    if len(pairs):
        scores, _ = predict(res.fwd, gp, pairs.rows, pairs.days, res.model.cfg.ctx, res.mu, res.sd,
                            batch=cfg.eval_batch, bf16=cfg.bf16, pad=cfg.compile)
        out[pairs.rows_np, np.searchsorted(idx, pairs.days_np)] = scores.cpu().numpy()
    return out
