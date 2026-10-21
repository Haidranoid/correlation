"""Seasonality diagnostic: share of the variance of monthly (or quarterly) log changes explained by
calendar-month (quarter) means. Used to choose anomaly_log1p (strongly seasonal) vs logdiff."""
import math


def seasonal_r2(rows, period="month"):
    """rows: sorted [(date, value)], value >= 0. Returns (r2, n)."""
    xs = [(d, math.log1p(v)) for d, v in rows]
    dl = []
    for (d0, a), (d1, b) in zip(xs, xs[1:]):
        gap = (d1 - d0).days
        if (period == "month" and 27 <= gap <= 32) or (period == "quarter" and 88 <= gap <= 93):
            dl.append((d1.month, b - a))
    if len(dl) < 24:
        return float("nan"), len(dl)
    mean = sum(v for _, v in dl) / len(dl)
    tot = sum((v - mean) ** 2 for _, v in dl)
    groups = {}
    for m, v in dl:
        groups.setdefault(m, []).append(v)
    between = sum(len(g) * (sum(g) / len(g) - mean) ** 2 for g in groups.values())
    return (between / tot if tot > 0 else float("nan")), len(dl)
