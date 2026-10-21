"""Did a score's top 10 land among the month's 10 biggest gainers? Exact chance levels, with and without volatility.

On each graded decision day d of a test year (targets.YearSpec), the eligible stocks are the year's universe with a
known H-day forward return and a finite score. The score's picks are its 10 highest. When stocks tie at the cut, the
k tied stocks share the m slots left, each with weight m / k: every count and return below is the average over all ways
of breaking the tie (the grader instead takes the lower columns, which is ticker order, so a score with few distinct
values would otherwise be graded partly on the alphabet). The realized gainers are the 10 eligible stocks with the highest forward return, the losers the
10 lowest; the top and bottom deciles are the highest and lowest ceil(n / 10).

Chance levels, for picks drawn at random from the same eligible stocks:
- uniform: hits on the 10 gainers are hypergeometric, mean 10 * 10 / n;
- volatility-matched: each pick replaced by a random eligible stock from its own fiftieth of the day's stocks by
  trailing 63-day volatility. The month's biggest gainers are mostly volatile stocks, so a rule that just picks volatile
  stocks lands among them far more often than uniform chance without saying anything about direction; this level
  removes most of that (tenths were too coarse: the most volatile stocks inside the top tenth still beat it in both
  tails). Its mean is exact (the sum over picks of the share of gainers in the pick's group); its variance treats the
  picks as independent.

A pick is also checked against the 10 biggest losers: picking volatile stocks puts names in both tails, while knowing
the direction puts more among the gainers than among the losers.

Comparable stocks (with `ratio`, the past-year price ratio close[d] / close[d-252]): each pick's return against the
mean of the COMPARABLE eligible stocks it didn't pick that are nearest to it that day in percentile rank of trailing
volatility and of past-year fall, weighted like the picks and summed over the top 10. Stocks that fell (or rose) and
swung alike as a group cancel out, so what is left is the choice among them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from .targets import YearSpec

TOP = 10
VOL_GROUPS = 50
VOL_WINDOW = 63
VOL_MIN = 42
COMPARABLE = 20


def trailing_vol(close: np.ndarray, window: int = VOL_WINDOW, min_obs: int = VOL_MIN) -> np.ndarray:
    """float64 [N, C]: sd of daily log returns over days d-window+1 .. d (both closes present), from at least min_obs."""
    c = np.asarray(close, dtype=np.float64)
    r = np.full(c.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[:, 1:] = np.log(c[:, 1:] / c[:, :-1])
    ok = np.isfinite(r)
    x = np.where(ok, r, 0.0)
    cs = np.zeros((c.shape[0], c.shape[1] + 1))
    cs2 = np.zeros_like(cs)
    cn = np.zeros_like(cs)
    np.cumsum(x, axis=1, out=cs[:, 1:])
    np.cumsum(x * x, axis=1, out=cs2[:, 1:])
    np.cumsum(ok, axis=1, out=cn[:, 1:])
    hi = np.arange(1, c.shape[1] + 1)
    lo = np.maximum(hi - window, 0)
    k = cn[:, hi] - cn[:, lo]
    s1 = cs[:, hi] - cs[:, lo]
    s2 = cs2[:, hi] - cs2[:, lo]
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (s2 - s1 * s1 / k) / (k - 1)
    out = np.sqrt(np.where(var > 0, var, np.nan))
    out[k < min_obs] = np.nan
    return out


def top_by(values: np.ndarray, cols: np.ndarray, n: int, largest: bool = True) -> np.ndarray:
    """The n entries of cols with the largest (or smallest) values; ties to the lower column."""
    key = -values if largest else values
    order = np.lexsort((cols, key))  # by key, then column
    return cols[order[:n]]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def past_year_ratio(close: np.ndarray) -> np.ndarray:
    """float64 [N, C]: close[d] / close[d - 252]; NaN for the first year and where either close is missing."""
    c = np.asarray(close, dtype=np.float64)
    out = np.full(c.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[:, 252:] = c[:, 252:] / c[:, :-252]
    out[~np.isfinite(out)] = np.nan
    return out


def pct_rank(x: np.ndarray) -> np.ndarray:
    """(rank + 0.5) / n, ties broken by position."""
    order = np.argsort(x, kind="stable")
    r = np.empty(x.size)
    r[order] = np.arange(x.size)
    return (r + 0.5) / x.size


def comparable_excess(f: np.ndarray, w: np.ndarray, vv: np.ndarray, rr: np.ndarray, top: int = TOP,
                      k: int = COMPARABLE) -> float:
    """One day's picks against their comparable stocks (module docstring). f, w, vv, rr: the eligible stocks' forward
    returns, pick weights, volatilities and past-year ratios. A pick without both measures is left out."""
    ok = np.isfinite(vv) & np.isfinite(rr)
    cand = np.flatnonzero(ok & (w == 0))
    if cand.size < k:
        return float("nan")
    pv, pr = np.full(f.size, np.nan), np.full(f.size, np.nan)
    idx = np.flatnonzero(ok)
    pv[idx], pr[idx] = pct_rank(vv[idx]), pct_rank(rr[idx])
    e = 0.0
    for i in np.flatnonzero((w > 0) & ok):
        nb = cand[np.argsort((pv[cand] - pv[i]) ** 2 + (pr[cand] - pr[i]) ** 2, kind="stable")[:k]]
        e += w[i] * (f[i] - f[nb].mean())
    return e / top


def hyper_var(n_draw: int, k: int, n: int) -> float:
    if n <= 1:
        return 0.0
    return n_draw * (k / n) * ((n - k) / n) * ((n - n_draw) / (n - 1))


@dataclass
class DayResult:
    day: int
    n: int
    hits_gain: float
    hits_loss: float
    top_decile: float
    bottom_decile: float
    exp_uniform: float
    var_uniform: float
    exp_vol: float
    var_vol: float
    exp_vol_loss: float
    pick_ret: float
    all_ret: float
    ic: float
    tied: int = 0  # stocks sharing the slots at the cut (0: no tie at the cut)
    picks: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    comparable: float = float("nan")  # the picks' excess over their comparable stocks (comparable_excess)


def grade_day(score: np.ndarray, fwd: np.ndarray, vol: np.ndarray, universe: np.ndarray, d: int,
              top: int = TOP, subset: np.ndarray | None = None, ratio: np.ndarray | None = None) -> DayResult | None:
    """One decision day. score, fwd, vol: [N, C] arrays; universe: the year's candidate columns; subset: an optional
    [N, C] mask known at d (for example the largest stocks) that further restricts who is eligible that day; ratio:
    past_year_ratio, for the comparable-stocks yardstick."""
    u = universe[np.isfinite(fwd[universe, d]) & np.isfinite(score[universe, d])]
    if subset is not None:
        u = u[subset[u, d]]
    n = u.size
    if n < 5 * top:
        return None
    s, f = score[u, d].astype(np.float64), fwd[u, d]
    cut = np.sort(s)[::-1][top - 1]
    above = s > cut
    tied = s == cut
    w = np.where(above, 1.0, np.where(tied, (top - above.sum()) / tied.sum(), 0.0))
    pick_idx = np.flatnonzero(w > 0)
    pset = u[pick_idx].tolist()
    wset = w[pick_idx].tolist()
    gain = set(top_by(f, u, top).tolist())
    loss = set(top_by(f, u, top, largest=False).tolist())
    q = math.ceil(n / 10)
    top_dec = set(top_by(f, u, q).tolist())
    bot_dec = set(top_by(f, u, q, largest=False).tolist())
    # volatility groups (VOL_GROUPS equal shares) among the eligible stocks (missing volatility: its own group)
    vv = vol[u, d]
    dec = np.full(n, -1)
    fin = np.isfinite(vv)
    if fin.any():
        r = rankdata(vv[fin], method="ordinal") - 1
        dec[fin] = (r * VOL_GROUPS // fin.sum()).astype(int)
    pos = {c: i for i, c in enumerate(u.tolist())}
    is_gain = np.array([c in gain for c in u.tolist()])
    is_loss = np.array([c in loss for c in u.tolist()])
    exp_vol = var_vol = exp_vol_loss = 0.0
    for c, wc in zip(pset, wset):
        group = dec == dec[pos[c]]
        p = float(is_gain[group].mean())
        exp_vol += wc * p
        var_vol += wc * p * (1 - p)
        exp_vol_loss += wc * float(is_loss[group].mean())
    wsum = lambda members: float(sum(wc for c, wc in zip(pset, wset) if c in members))  # noqa: E731
    return DayResult(
        day=int(d), n=int(n),
        hits_gain=wsum(gain), hits_loss=wsum(loss), top_decile=wsum(top_dec), bottom_decile=wsum(bot_dec),
        exp_uniform=top * top / n, var_uniform=hyper_var(top, top, n),
        exp_vol=exp_vol, var_vol=var_vol, exp_vol_loss=exp_vol_loss,
        pick_ret=float(np.dot(w[pick_idx], f[pick_idx]) / top), all_ret=float(f.mean()), ic=spearman(s, f),
        tied=int(tied.sum()) if tied.sum() > 1 else 0, picks=pset, weights=wset,
        comparable=comparable_excess(f, w, vv, ratio[u, d], top) if ratio is not None else float("nan"),
    )


def grade(score: np.ndarray, fwd: np.ndarray, vol: np.ndarray, years: list[YearSpec], top: int = TOP,
          subset: np.ndarray | None = None, ratio: np.ndarray | None = None) -> dict:
    """Per-year and overall results for one [N, C] score array."""
    by_year, all_days = [], []
    for ys in years:
        days = [r for d in ys.decisions
                if (r := grade_day(score, fwd, vol, ys.universe, int(d), top, subset, ratio)) is not None]
        all_days += days
        by_year.append({"year": ys.year, **summarize(days)})
    return {"years": by_year, "overall": summarize(all_days), "days": [_day_dict(r) for r in all_days]}


def summarize(days: list[DayResult]) -> dict:
    if not days:
        return {"days": 0}
    k = len(days)
    hg = sum(r.hits_gain for r in days)
    eu, vu = sum(r.exp_uniform for r in days), sum(r.var_uniform for r in days)
    ev, vv = sum(r.exp_vol for r in days), sum(r.var_vol for r in days)
    ics = np.array([r.ic for r in days])
    ex = np.array([r.pick_ret - r.all_ret for r in days])
    sd_ic, sd_ex = float(ics.std(ddof=1)) if k > 1 else float("nan"), float(ex.std(ddof=1)) if k > 1 else float("nan")
    # Month-level statistics: each graded month is one observation (the picks within a month are not independent).
    hit_ex = np.array([r.hits_gain - r.exp_vol for r in days])
    sd_h = float(hit_ex.std(ddof=1)) if k > 1 else float("nan")
    order = np.argsort(-ex)
    rest = ex[order[5:]] if k > 6 else ex
    sd_rest = float(rest.std(ddof=1)) if rest.size > 1 else float("nan")
    # The comparable-stocks yardstick, overall and without the same 5 best periods (by raw excess).
    comp = np.array([r.comparable for r in days])
    comp_rest = comp[order[5:]] if k > 6 else comp
    tstat = lambda x: float(x.mean() / x.std(ddof=1) * math.sqrt(x.size)) if x.size > 1 and x.std(ddof=1) > 0 else float("nan")  # noqa: E731
    comp, comp_rest = comp[np.isfinite(comp)], comp_rest[np.isfinite(comp_rest)]
    return {
        "days": k,
        "picks": k * TOP,
        "hits_t_month": float(hit_ex.mean() / sd_h * math.sqrt(k)) if k > 1 and sd_h > 0 else float("nan"),
        "median_excess": float(np.median(ex)),
        "months_ahead": int((ex > 0).sum()),
        "top5_share_of_excess": float(ex[order[:5]].sum() / ex.sum()) if ex.sum() > 0 and k > 6 else float("nan"),
        "excess_without_top5": float(rest.mean()),
        "excess_without_top5_t": float(rest.mean() / sd_rest * math.sqrt(rest.size)) if rest.size > 1 and sd_rest > 0 else float("nan"),
        "hits_gain": hg,
        "hits_loss": sum(r.hits_loss for r in days),
        "gain_share_of_tail_hits": hg / (hg + sum(r.hits_loss for r in days)) if hg + sum(r.hits_loss for r in days) else float("nan"),
        "hits_gain_per_day": hg / k,
        "exp_uniform_per_day": eu / k,
        "exp_vol_per_day": ev / k,
        "exp_vol_loss_per_day": sum(r.exp_vol_loss for r in days) / k,
        "z_vs_uniform": (hg - eu) / math.sqrt(vu) if vu > 0 else float("nan"),
        "z_vs_vol": (hg - ev) / math.sqrt(vv) if vv > 0 else float("nan"),
        "days_tied_at_cut": sum(r.tied > 0 for r in days),
        "top_decile_share": sum(r.top_decile for r in days) / (k * TOP),
        "bottom_decile_share": sum(r.bottom_decile for r in days) / (k * TOP),
        "mean_pick_ret": float(np.mean([r.pick_ret for r in days])),
        "mean_all_ret": float(np.mean([r.all_ret for r in days])),
        "mean_excess": float(ex.mean()),
        "excess_t": float(ex.mean() / sd_ex * math.sqrt(k)) if k > 1 and sd_ex > 0 else float("nan"),
        "ic_mean": float(ics.mean()),
        "ic_t": float(ics.mean() / sd_ic * math.sqrt(k)) if k > 1 and sd_ic > 0 else float("nan"),
        "ic_positive": int((ics > 0).sum()),
        "mean_n": float(np.mean([r.n for r in days])),
        "comparable_days": int(comp.size),
        "comparable_excess": float(comp.mean()) if comp.size else float("nan"),
        "comparable_t": tstat(comp),
        "comparable_ahead": int((comp > 0).sum()),
        "comparable_without_top5": float(comp_rest.mean()) if comp_rest.size else float("nan"),
        "comparable_without_top5_t": tstat(comp_rest),
    }


def account(days: list[DayResult], cost_bps: float, top: int = TOP) -> dict:
    """$10,000 in the picks (each pick's weight / 10 of the account; tied picks share their slots), from one graded
    month to the next (they tile the test years), paying cost_bps per side on what is bought and sold; against the same
    account in every eligible stock (equal weight, no costs)."""
    value, base, prev = 10000.0, 10000.0, {}
    for r in days:
        now = {c: wc / top for c, wc in zip(r.picks, r.weights)}
        traded = sum(abs(now.get(c, 0.0) - prev.get(c, 0.0)) for c in set(now) | set(prev))  # sells plus buys
        value *= 1 + r.pick_ret - traded * cost_bps / 1e4
        base *= 1 + r.all_ret
        prev = now
    return {"end": value, "all_end": base}


def _day_dict(r: DayResult) -> dict:
    return {"day": r.day, "n": r.n, "hits_gain": r.hits_gain, "hits_loss": r.hits_loss, "exp_uniform": r.exp_uniform,
            "exp_vol": r.exp_vol, "pick_ret": r.pick_ret, "all_ret": r.all_ret, "ic": r.ic, "tied": r.tied,
            "picks": r.picks, "weights": r.weights, "comparable": r.comparable}
