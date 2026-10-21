"""Baselines graded through the same judge as stockGPT (PLAN.md §3.1, M1).

- `reversal`: correlation2's own reversal-family score, reproduced exactly (the G0 cross-check).
- `momentum`: 12-1 momentum.
- `lgbm`: LightGBM walk-forward on simple point-in-time price features (`features`).

Every baseline reads only the days before the locked year's first decision (`common.dev_cutoff`) and writes NaN
from there on, so the locked year (2024-06-28 .. 2025-06-27) and anything later is never used or scored.
"""
