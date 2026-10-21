"""stockGPT v1 (PLAN.md §3.2): a small GPT over census-variable tokens, trained walk-forward, scored for the judge.

- tokens.py: the census variable as 101 bins of 0.1 plus a MISSING token;
- features.py: per-stock inputs, training targets at a horizon of H days (the next day's token at H = 1, the binned
  daily rank of the H-day return above it) and validation outcomes, cut before the locked year;
- model.py: the nanoGPT-style decoder;
- windows.py: the training tensor and window sampling on the GPU, full-context windows for scoring;
- train.py: training, validation (CE, bits gain, rank IC), early stopping and scoring for one split;
- walkforward.py: the CLI (python -m stockai.gpt.walkforward [--horizon H]); scripts/gpt_horizon.sh runs it at H days.
"""

import os as _os

# Deterministic cuBLAS for training (train.TrainConfig.deterministic); must be set before CUDA starts.
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
