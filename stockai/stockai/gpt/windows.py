"""Model inputs on the GPU: training batches for one split, and full-context windows ending at decision days.

A window ending at column e holds days e-ctx+1 .. e; the model's output at its last position is the forecast for
decision day e: the target over days e+1 .. e+H (features.py; at H = 1 the token of day e+1). Columns before the data
start are padding (MISSING token, zero features, missing flag set).

Training (`TrainSet`) copies only columns up to the split's last training target day (the last training decision
plus H) to the device, so nothing later can reach a batch, with the targets of the training decisions beside them
(MISSING for every other column). Windows are sampled on the device: draw a valid training position (stock j, decision
day d with d in split.train_decisions and a non-MISSING target) uniformly, then place the window so that d sits at a
uniformly random offset in it. The loss covers every position of the window whose decision day is a training
decision and whose target is not MISSING; `check_batch` verifies that against the panel.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import splits
from .features import CLIP, Panel
from .tokens import MISSING, N_BINS

IGNORE = -100  # cross-entropy ignore_index


class GpuPanel:
    """Inputs for every stock and day of a Panel on the device: tokens, log volatility, market feature."""

    def __init__(self, panel: Panel, device: str | torch.device):
        self.device = torch.device(device)
        self.n, self.t = panel.n, panel.t
        self.tokens = torch.as_tensor(panel.tokens, device=self.device)  # uint8 [N, T]
        self.logvol = torch.as_tensor(panel.logvol, device=self.device)  # float32 [N, T]
        self.market = torch.as_tensor(panel.market, device=self.device)  # float32 [T]


def vol_constants(panel: Panel, split: splits.Split) -> tuple[float, float]:
    """Mean and sd of log volatility over the training decisions only (the standardization of the vol input)."""
    tr = split.train_decisions
    lv = panel.logvol[:, tr.start:tr.stop]
    x = lv[np.isfinite(lv)].astype(np.float64)
    if x.size < 2:
        raise ValueError("no finite volatility in the training range")
    return float(x.mean()), float(max(x.std(), 1e-6))


def gather(tokens: torch.Tensor, logvol: torch.Tensor, market: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor,
           mu: float, sd: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs at (rows[b], cols[b, i]); columns outside 0 .. width-1 are padding.

    Returns tokens long [B, L] and continuous features float32 [B, L, 3]: market move, standardized log
    volatility (clipped, 0 when missing) and the volatility-missing flag.
    """
    width = tokens.shape[1]
    ok = (cols >= 0) & (cols < width)
    c = cols.clamp(0, width - 1)
    r = rows[:, None]
    tok = torch.where(ok, tokens[r, c].long(), MISSING)
    lv = torch.where(ok, logvol[r, c], torch.nan)
    miss = torch.isnan(lv)
    z = torch.where(miss, 0.0, ((lv - mu) / sd).clamp(-CLIP, CLIP))
    m = torch.where(ok, market[c], 0.0)
    return tok, torch.stack([m, z, miss.float()], dim=-1)


def windows_ending(gp: GpuPanel, rows: torch.Tensor, ends: torch.Tensor, ctx: int, mu: float, sd: float):
    """Full-context inputs for windows ending at columns `ends` (inclusive): nothing after `ends` is read."""
    cols = ends[:, None] + torch.arange(-(ctx - 1), 1, device=ends.device)[None, :]
    return gather(gp.tokens, gp.logvol, gp.market, rows, cols, mu, sd)


class TrainSet:
    """The training tensor for one split on the device, and its window sampler."""

    def __init__(self, panel: Panel, gp: GpuPanel, split: splits.Split, ctx: int):
        if split.horizon != panel.horizon:
            raise ValueError(f"the split's horizon ({split.horizon}) is not the panel's targets' ({panel.horizon})")
        tr, h = split.train_decisions, split.horizon
        self.split, self.ctx, self.device = split, ctx, gp.device
        self.c0 = max(0, tr.start - (ctx - 1))  # earliest column a window needs as context
        self.c1 = tr.stop + h  # exclusive: the last training decision tr.stop-1 has its outcome through day tr.stop-1+h
        if self.c1 > panel.t:
            raise ValueError("the training range runs past the panel")
        self.width = self.c1 - self.c0
        # Only these columns are on the device for training: nothing after the last training target day.
        self.tokens = gp.tokens[:, self.c0:self.c1].clone()
        self.logvol = gp.logvol[:, self.c0:self.c1].clone()
        self.market = gp.market[self.c0:self.c1].clone()
        targets = panel.target[:, tr.start:tr.stop]  # the target of each decision d in tr (outcome days d+1 .. d+h)
        local = np.full((panel.n, self.width), MISSING, dtype=np.uint8)  # no target outside the training decisions
        local[:, tr.start - self.c0:tr.stop - self.c0] = targets
        self.target = torch.as_tensor(local, device=self.device)
        widths = {self.tokens.shape[1], self.logvol.shape[1], self.market.shape[0], self.target.shape[1]}
        if widths != {(tr.stop - 1 + h) + 1 - self.c0}:
            raise AssertionError("the training tensor must end on the last training target day")
        self.mu, self.sd = vol_constants(panel, split)

        valid = targets != MISSING
        j, k = np.nonzero(valid)
        self.n_valid = int(j.size)
        if self.n_valid == 0:
            raise ValueError("no training positions")
        local_d = (tr.start - self.c0) + k
        self.anchors = torch.as_tensor(j.astype(np.int64) * self.width + local_d, device=self.device)
        # Unconditional bin histogram of the training targets (add-one smoothed): the no-skill baseline.
        counts = np.bincount(targets[valid].astype(np.int64), minlength=N_BINS).astype(np.float64)
        self.hist_logp = np.log((counts + 1.0) / (counts.sum() + N_BINS))
        self.tr_lo = tr.start - self.c0  # local column range of training decisions: [tr_lo, tr_hi)
        self.tr_hi = tr.stop - self.c0
        self._offsets = torch.arange(ctx, device=self.device)

    def sample(self, batch: int, gen: torch.Generator):
        """One batch: tokens [B, ctx], cont [B, ctx, 3], targets [B, ctx] (IGNORE where no loss), stock rows [B] and
        each position's decision day [B, ctx] as an index into the panel's days."""
        k = torch.randint(self.anchors.numel(), (batch,), device=self.device, generator=gen)
        a = self.anchors[k]
        rows, d = a // self.width, a % self.width
        off = torch.randint(self.ctx, (batch,), device=self.device, generator=gen)
        cols = (d - off)[:, None] + self._offsets[None, :]  # input columns; the anchor d is at position `off`
        tok, cont = gather(self.tokens, self.logvol, self.market, rows, cols, self.mu, self.sd)
        t_ok = (cols >= 0) & (cols < self.width)
        y = torch.where(t_ok, self.target[rows[:, None], cols.clamp(0, self.width - 1)].long(), MISSING)
        use = (cols >= self.tr_lo) & (cols < self.tr_hi) & (y != MISSING)
        y = torch.where(use, y, IGNORE)
        return tok, cont, y, rows, cols + self.c0


def check_batch(panel: Panel, split: splits.Split, y: torch.Tensor, rows: torch.Tensor, day: torch.Tensor) -> int:
    """Asserts, against the panel on the CPU, that every position carrying a loss is a training sample: its decision
    day d is in split.train_decisions, its outcome (days d+1 .. d+H) ends before the first validation decision, and its
    target is the panel's for that stock and day: the token of day d+1 at H = 1; at H > 1 the binned rank of the H-day
    return, for a stock with a census value at d and a known H-day return. Returns the positions checked."""
    h = split.horizon
    if panel.horizon != h:
        raise AssertionError(f"the panel's targets look {panel.horizon} days ahead, the split {h}")
    y, rows, day = y.cpu().numpy(), rows.cpu().numpy(), day.cpu().numpy()
    b, i = np.nonzero(y != IGNORE)
    d = day[b, i]
    tr = split.train_decisions
    if d.size == 0:
        raise AssertionError("a training batch without any loss positions")
    end = d.max() + h  # the last outcome day of any loss position
    if d.min() < tr.start or d.max() >= tr.stop or end > tr.stop - 1 + h or end >= split.val_decisions.start:
        raise AssertionError(f"training position on day {d.min()}..{d.max()} (outcomes through day {end}) outside "
                             f"split.train_decisions {tr.start}..{tr.stop - 1} or reaching the validation decisions "
                             f"from {split.val_decisions.start}")
    j = rows[b]
    if h == 1:
        if not np.array_equal(y[b, i], panel.tokens[j, d + 1].astype(np.int64)):
            raise AssertionError("a training target is not the panel's token for the next day")
    else:
        if not np.array_equal(y[b, i], panel.target[j, d].astype(np.int64)):
            raise AssertionError(f"a training target is not the panel's {h}-day target")
        if not (np.isfinite(panel.v[j, d]).all() and np.isfinite(panel.ret_fwd[j, d]).all()):
            raise AssertionError(f"a training target without a census value at d or a known {h}-day return")
    return int(d.size)
