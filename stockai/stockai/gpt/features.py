"""Per-stock, per-day inputs, training targets and outcomes for stockGPT, built once from an export.

Inputs at day d use data through the close of day d only:
- tokens: the census variable v[d] as a bin token (tokens.py), optionally sign-flipped per date (the null);
- logvol: log of the sd of daily log returns of close over the 252 days through d (NaN with < 126 returns);
- market: market[d] / 0.01, clipped at +-5 (0 where missing).

The training target of decision day d looks `horizon` (H) trading days ahead, as a bin token like the inputs:
- H = 1 (stockGPT v1): the next day's token, tokens[d+1].
- H > 1: the day's cross-sectional rank of the H-day forward return among the stocks with a census value at d and a
  known return, as the standard normal quantile ndtri((rank - 0.5) / n), tied returns sharing their average rank
  (rank-gauss, stockai.baselines.lgbm.rank_gauss: the target of the H-day LightGBM in stockai/h30/model.py), binned like
  v. For a few thousand stocks it spans about +-3.6, inside the bins' +-5. The forward return is
  stockai.h30.targets.forward_returns (close[d+H] / close[d] - 1), given the export's delistings when it has them
  (Sharadar exports, docs/sharadar.md). Each day's targets have the same distribution whatever the market did that
  week or month, so the model learns which stocks will beat the others, not the market's direction, and from actual
  returns: what the account earns and what the H-day evaluation grades. A sum of the next H census values would carry
  H times the part of v[d+1] known at d (v subtracts each stock's trailing mean residual; test-results.md, item 1)
  against sqrt(H) times its noise, and would reward learning that part. The stocks ranked are the ones that can train
  that day (as LightGBM ranks among its training rows), and every one of them is scored at d. Under the sign-flip null
  the target of decision d is multiplied by the sign of day d+H, the last day of its outcome, which no input at d has
  seen; at H = 1 that is the flipped next-day token, as before.
Outcomes, used only for validation metrics: `ret_fwd`, the actual return of deciding at d and holding H days
(stockai.data.next_day_returns at H = 1, the forward return above otherwise; never flipped), and v over days
d+1 .. d+H (read by train.Pairs; under the null, the flipped v). `liquid[j, d]` is a point-in-time filter (close >= $5
and 63-day median dollar volume >= $5M as of d).

Everything is cut before the locked year: no array here has a column on or after LOCKED_YEAR-06-28, and a target or
outcome whose window would reach it is unknown (MISSING, NaN).
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass

import numpy as np
import torch

from .. import data, splits
from ..baselines.lgbm import rank_gauss
from ..h30.targets import forward_returns
from .tokens import MISSING, encode

VOL_WINDOW = 252
VOL_MIN_OBS = 126
MARKET_SCALE = 0.01
CLIP = 5.0
LIQ_WINDOW = 63
LIQ_MIN_OBS = 32
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 5e6
NULLS = ("sign-flip",)


@dataclass
class Panel:
    days: np.ndarray  # int32 [T] (T = columns kept, all before the locked year)
    v: np.ndarray  # float32 [N, T]: the variable the tokens encode (sign-flipped under the null)
    tokens: np.ndarray  # uint8 [N, T]
    logvol: np.ndarray  # float32 [N, T]; NaN = unavailable
    market: np.ndarray  # float32 [T]: scaled, clipped market move; 0 where missing
    liquid: np.ndarray  # bool [N, T]
    ret_next: np.ndarray  # float32 [N, T]: actual return over day d+1 (an outcome, never an input)
    t_full: int  # columns in the export (scores files keep this width)
    null: str | None = None
    null_seed: int | None = None
    flips: np.ndarray | None = None  # [T] +-1 under the sign-flip null
    horizon: int = 1  # trading days the target looks ahead
    target: np.ndarray | None = None  # uint8 [N, T]: the training target token of decision day d; MISSING if unknown
    ret_fwd: np.ndarray | None = None  # [N, T]: actual return of deciding at d over the next `horizon` days (an
    # outcome): ret_next itself at horizon 1, the float64 forward returns that were ranked into `target` otherwise

    @property
    def n(self) -> int:
        return self.v.shape[0]

    @property
    def t(self) -> int:
        return self.v.shape[1]


def locked_cut(days: np.ndarray) -> int:
    """Index of the first day of the locked year: nothing at or after it may be used."""
    return int(np.searchsorted(days, data.str_to_day(f"{splits.LOCKED_YEAR}-{splits.PERIOD_START}"), side="left"))


def trailing_log_vol(close: np.ndarray, window: int = VOL_WINDOW, min_obs: int = VOL_MIN_OBS, chunk: int = 1024) -> np.ndarray:
    """log sd of daily log returns over days d-window+1 .. d (through d); NaN with fewer than `min_obs` returns."""
    n, t = close.shape
    out = np.full((n, t), np.nan, dtype=np.float32)
    lo = np.maximum(np.arange(1, t + 1) - window, 0)
    for a in range(0, n, chunk):
        c = np.asarray(close[a:a + chunk], dtype=np.float64)
        r = np.full(c.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
        ok = np.isfinite(r)
        r = np.where(ok, r, 0.0)

        def window_sum(x: np.ndarray) -> np.ndarray:
            cs = np.zeros((x.shape[0], t + 1))
            np.cumsum(x, axis=1, out=cs[:, 1:])
            return cs[:, 1:] - cs[:, lo]

        k = window_sum(ok.astype(np.float64))
        s1, s2 = window_sum(r), window_sum(r * r)
        with np.errstate(divide="ignore", invalid="ignore"):
            var = (s2 - s1 * s1 / k) / (k - 1)
            lv = 0.5 * np.log(var)
        good = (np.round(k) >= min_obs) & (var > 1e-14) & np.isfinite(lv)
        out[a:a + chunk] = np.where(good, lv, np.nan)
    return out


def liquid_mask(close: np.ndarray, dollar_volume: np.ndarray, min_price: float = MIN_PRICE,
                min_dollar_volume: float = MIN_DOLLAR_VOLUME, window: int = LIQ_WINDOW, min_obs: int = LIQ_MIN_OBS,
                device: str | None = None, chunk: int = 256) -> np.ndarray:
    """close[d] >= min_price and the median dollar volume over days d-window+1 .. d >= min_dollar_volume."""
    n, t = close.shape
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = np.zeros((n, t), dtype=bool)
    for a in range(0, t, chunk):
        b = min(t, a + chunk)
        lo = max(0, a - window + 1)
        dv = torch.tensor(np.array(dollar_volume[:, lo:b], dtype=np.float32), device=dev)
        pad = window - 1 - (a - lo)
        if pad:
            dv = torch.cat([torch.full((n, pad), float("nan"), device=dev), dv], dim=1)
        w = dv.unfold(1, window, 1)  # [n, b - a, window]: days d-window+1 .. d
        cnt = torch.isfinite(w).sum(-1)
        # Lower median of the finite values (as torch.nanmedian), via sort so it is allowed in deterministic mode.
        srt = torch.sort(torch.nan_to_num(w, nan=float("inf")), dim=-1).values
        med = srt.gather(-1, ((cnt - 1).clamp(min=0) // 2)[..., None])[..., 0]
        ok = (cnt >= min_obs) & (med >= min_dollar_volume)
        c = np.asarray(close[:, a:b], dtype=np.float32)
        with np.errstate(invalid="ignore"):
            out[:, a:b] = ok.cpu().numpy() & (c >= min_price)
    return out


def sign_flips(t: int, seed: int) -> np.ndarray:
    """The null's random +-1 per date (same for every stock), seeded."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x5F11F]))
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=t)


def next_token_targets(tokens: np.ndarray) -> np.ndarray:
    """uint8 [N, T]: the horizon-1 target of decision day d, tokens[d+1]; MISSING in the last column."""
    out = np.full(tokens.shape, MISSING, dtype=np.uint8)
    out[:, :-1] = tokens[:, 1:]
    return out


def h_day_returns(close: np.ndarray, h: int, export: data.Export) -> np.ndarray:
    """float64 [N, C]: stockai.h30.targets.forward_returns of `close` over h days, with the export's delistings when it
    has them (Sharadar exports; None for Stooq's, which then get the plain forward return)."""
    return forward_returns(close, h, delist_day=getattr(export, "delist_day", None),
                           delist_kind=getattr(export, "delist_kind", None))


def rank_targets(v: np.ndarray, fwd: np.ndarray, sign: np.ndarray | None = None) -> np.ndarray:
    """uint8 [N, C]: on each day d, the rank-gauss score of fwd[:, d] among the stocks with a finite v[:, d] and a
    finite fwd[:, d], times sign[d] if given, as a bin token; MISSING for every other stock. Column d reads column d
    only, so a target is known once its forward return is."""
    z = rank_gauss(fwd, np.isfinite(v) & np.isfinite(fwd))
    if sign is not None:
        z = z * np.asarray(sign, dtype=np.float32)[None, :]
    return encode(z)


def null_target_signs(flips: np.ndarray, h: int) -> np.ndarray:
    """float32 [T]: the sign-flip null's sign for the target of decision day d, flips[d+h] (the last day of its
    outcome); NaN where d+h is past the columns (no such outcome)."""
    out = np.full(flips.shape, np.nan, dtype=np.float32)
    out[:-h] = flips[h:]
    return out


def target_info(export: data.Export, horizon: int) -> dict:
    """What the training target is, for run.json; at H > 1 also which forward-return code made it, since
    stockai.h30.targets.forward_returns is shared with the H-day evaluation and may change between runs."""
    if horizon == 1:
        return {"horizon": 1, "target": "the next day's census token"}
    return {"horizon": horizon, "target": f"the day's rank-gauss score of the {horizon}-day forward return, binned",
            "forward_returns": f"stockai.h30.targets.forward_returns{inspect.signature(forward_returns)}",
            "forward_returns_sha256": hashlib.sha256(inspect.getsource(forward_returns).encode()).hexdigest(),
            "delistings": getattr(export, "delist_day", None) is not None}


def build_panel(export: data.Export, null: str | None = None, null_seed: int = 0, min_price: float = MIN_PRICE,
                min_dollar_volume: float = MIN_DOLLAR_VOLUME, device: str | None = None, horizon: int = 1) -> Panel:
    """Inputs, training targets and outcomes for every stock and every day before the locked year."""
    if null is not None and null not in NULLS:
        raise ValueError(f"unknown null {null!r}; expected one of {NULLS}")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    cut = locked_cut(export.days)
    days = np.asarray(export.days[:cut])
    v = np.array(export.v[:, :cut], dtype=np.float32)
    close = np.array(export.close[:, :cut], dtype=np.float32)
    dollar_volume = np.array(export.dollar_volume[:, :cut], dtype=np.float32)
    market = np.array(export.market[:cut], dtype=np.float64)

    flips = None
    if null == "sign-flip":
        flips = sign_flips(cut, null_seed)
        v *= flips[None, :]
    with np.errstate(invalid="ignore"):
        mkt = np.clip(market / MARKET_SCALE, -CLIP, CLIP)
    mkt = np.where(np.isfinite(mkt), mkt, 0.0).astype(np.float32)
    tokens = encode(v)
    ret_next = data.next_day_returns(close).astype(np.float32)
    if horizon == 1:
        target, ret_fwd = next_token_targets(tokens), ret_next
    else:  # computed on the cut columns, so every window that would reach the locked year has no outcome
        ret_fwd = h_day_returns(close, horizon, export)
        target = rank_targets(v, ret_fwd, None if flips is None else null_target_signs(flips, horizon))
    return Panel(
        days=days,
        v=v,
        tokens=tokens,
        logvol=trailing_log_vol(close),
        market=mkt,
        liquid=liquid_mask(close, dollar_volume, min_price, min_dollar_volume, device=device),
        ret_next=ret_next,
        t_full=export.t,
        null=null,
        null_seed=null_seed if null else None,
        flips=flips,
        horizon=horizon,
        target=target,
        ret_fwd=ret_fwd,
    )
