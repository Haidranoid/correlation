# Training a small GPT-style stock-price model from scratch, and fine-tuning existing ones, on one RTX 5080 (state as of 2026-10-18)

Scope: the practical engineering only: software stack, data encoding, model and data sizing, training recipe, code bases, and experiment tracking, for one RTX 5080 (16 GB, sm_120) on Linux with uv. Version facts were checked on 2026-10-18 against PyPI's JSON API, the PyTorch wheel index, GitHub's API and Hugging Face model cards, unless a finding says otherwise. "Inference" bullets are my own reasoning or arithmetic, built from the cited findings.

---

## 1. Software stack: which PyTorch/CUDA wheel runs on the RTX 5080 (sm_120), Python 3.14 or 3.12/3.13, and the status of bf16, torch.compile, SDPA and FlashAttention

### Takeaway
Use **PyTorch 2.14.1** (released 2026-09-30). The plain PyPI wheel on Linux is the **CUDA 13.0 build**, and it includes Blackwell sm_120. A `cu132` wheel also exists, and the user's 595.99 driver (CUDA 13.2) runs either. **Do not use `cu126`**: the 2.14 cu126 builds stop at Hopper (sm_90). The `cu128` and `cu129` indexes stopped at torch 2.11 and 2.13. **Python 3.14: yes.** torch 2.14.1, numpy, pandas, polars, pyarrow, transformers and lightgbm all ship wheels that install on 3.14, and torch.compile has supported 3.14 since 2.10. Even so, **pin Python 3.12 with uv** for the project, for three reasons: Qlib, Kronos's pinned requirements and Aim have no wheels above cp312, 3.14 changed the default multiprocessing start method, and 3.12 has the longest track record. bf16, torch.compile and PyTorch's built-in SDPA flash kernels all work on sm_120. Skip the separate `flash-attn` package. One open, silent bf16 + torch.compile correctness bug was reproduced on an RTX 5080, so check compiled output against eager output.

### Cited Findings

#### PyTorch releases and CUDA wheels
- Latest PyTorch is **2.14.1, uploaded 2026-09-30**. Earlier: 2.14.0 on 2026-09-02, 2.13.0 on 2026-07-08, 2.12.1 on 2026-06-17. The PyPI wheel has `cp314` and `cp314t` manylinux_2_28 x86_64 builds, and its classifiers list Python 3.10–3.14 — [PyPI torch](https://pypi.org/project/torch/2.14.1/#files)
- On Linux, the PyPI wheel for torch 2.14.1 depends on `cuda-toolkit[...]==13.0.3`, `nvidia-cudnn-cu13==9.24.0.43`, `nvidia-nccl-cu13==2.30.7` and `triton~=3.8.0` (Python < 3.15). The default PyPI install is therefore the CUDA 13.0 build — [PyPI torch 2.14.1 metadata](https://pypi.org/project/torch/2.14.1/)
- uv's PyTorch guide: PyPI hosts "GPU-accelerated wheels on Linux (targeting CUDA 13.0, as of PyTorch 2.11.0)" — [uv docs: Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- PyTorch's release matrix for **2.14**: Python ">=3.10, <=(3.15, 3.15t experimental)", C++20, stable CUDA 12.6 (cuDNN 9.10.2.21), CUDA 13.0 (cuDNN 9.24.0.43) and CUDA 13.2 (cuDNN 9.24.0.43), no experimental CUDA — [PyTorch RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- Architectures compiled into the 2.14 Linux x86 builds:
  - CUDA 12.6.3: Maxwell 5.0, Pascal 6.0, Volta 7.0, Turing 7.5, Ampere 8.0 and 8.6, Hopper 9.0. **No Blackwell.**
  - CUDA 13.0.3 and 13.2.1: Turing 7.5, Ampere 8.0 and 8.6, Hopper 9.0, "Blackwell(10.0, 12.0+PTX)".
  - Source: [PyTorch RELEASE.md, "PyTorch CUDA Support Matrix"](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- Release schedule from the same file:
  - 2.14: RC on 10 Aug 2026, final on 2 Sept 2026.
  - **2.15: final planned for 28 Oct 2026** (tentative). It moves to Python ">=3.11", stable CUDA 13.2 only, and experimental CUDA 13.4.
  - 2.16: planned for 22 Dec 2026.
  - Source: [PyTorch RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
- What each CUDA index on download.pytorch.org currently offers (listing parsed 2026-10-18):
  - `cu126`: up to 2.14.1.
  - `cu128`: newest is **2.11.0**.
  - `cu129`: newest is **2.13.0**.
  - `cu130`: up to 2.14.1.
  - `cu132`: up to 2.14.1.
  - `cu131` and `cu133` do not exist (HTTP 403).
  - All of these carry `cp314` builds, and 2.13 onward also carry `cp315`.
  - Sources: [cu128 index](https://download.pytorch.org/whl/cu128/torch/), [cu129 index](https://download.pytorch.org/whl/cu129/torch/), [cu130 index](https://download.pytorch.org/whl/cu130/torch/), [cu132 index](https://download.pytorch.org/whl/cu132/torch/), [cu126 index](https://download.pytorch.org/whl/cu126/torch/)
- Using too old a build fails like this: "GeForce RTX 5090 D with CUDA capability sm_120 is not compatible with the current PyTorch installation" (closed issue, 2025-04-05) — [pytorch#150733](https://github.com/pytorch/pytorch/issues/150733). A later reply in another thread says that "from our testing perspective `sm_120` is already added to stable builds" — [pytorch#164342](https://github.com/pytorch/pytorch/issues/164342)
- CUDA 13.0 GA needs a Linux driver ≥ 580.65.06. CUDA 13.x minor-version compatibility needs a driver ≥ 580 — [CUDA 13.0 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
- **uv integration**:
  - A project can pin torch to a PyTorch index with `[[tool.uv.index]] name="pytorch-cu130" url="https://download.pytorch.org/whl/cu130" explicit=true` plus `[tool.uv.sources] torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' ..." }]`.
  - `uv pip install --torch-backend=auto` detects the driver and picks the matching index. It is "only available in the `uv pip` interface".
  - Source: [uv docs: Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- Triton 3.8.0 (2026-08-28) has cp314 wheels and requires Python <3.15 — [PyPI triton](https://pypi.org/project/triton/3.8.0/#files)

#### Python 3.14 support in the main libraries (PyPI, checked 2026-10-18)
| Package | Latest (upload date) | Installs on CPython 3.14 / Linux x86_64? | Source |
|---|---|---|---|
| torch | 2.14.1 (2026-09-30) | **Yes**, cp314 and cp314t wheels | [PyPI](https://pypi.org/project/torch/#files) |
| numpy | 2.5.4 (2026-10-10) | **Yes**, cp314. Requires Python ≥3.12 | [PyPI](https://pypi.org/project/numpy/#files) |
| pandas | 3.0.6 (2026-09-17) | **Yes**, cp314. pandas 2.3.3 (2025-09-29) also has cp314 | [PyPI 3.0.6](https://pypi.org/project/pandas/#files), [PyPI 2.3.3](https://pypi.org/project/pandas/2.3.3/#files) |
| polars | 2.0.0 (2026-10-06) | **Yes**. A pure wrapper over `polars-runtime-32` cp310-abi3 wheels. Classifiers stop at 3.13, but abi3 wheels install on 3.14 | [PyPI polars](https://pypi.org/project/polars/#files), [PyPI polars-runtime-32](https://pypi.org/project/polars-runtime-32/#files) |
| pyarrow | 26.0.0 (2026-10-09) | **Yes**, cp314 | [PyPI](https://pypi.org/project/pyarrow/#files) |
| transformers | 5.20.1 (2026-10-16) | **Yes**. Pure Python, classifiers include 3.14. tokenizers 0.23.3 (abi3 + cp314t) and safetensors 0.8.0 (abi3) have compatible wheels | [PyPI transformers](https://pypi.org/project/transformers/#files), [PyPI tokenizers](https://pypi.org/project/tokenizers/#files), [PyPI safetensors](https://pypi.org/project/safetensors/#files) |
| lightgbm | 4.7.0 (2026-07-18) | **Yes**. py3-none manylinux wheel, classifiers include 3.14 | [PyPI](https://pypi.org/project/lightgbm/#files) |
| xgboost / catboost | 3.4.1 (2026-08-15) / 1.2.10 | Yes (xgboost needs Python ≥3.12). catboost has cp314 wheels | [PyPI xgboost](https://pypi.org/project/xgboost/#files), [PyPI catboost](https://pypi.org/project/catboost/#files) |
| scipy / scikit-learn / numba | 1.18.1 / 1.9.1 / 0.68.0 | Yes, cp314 wheels | [scipy](https://pypi.org/project/scipy/#files), [scikit-learn](https://pypi.org/project/scikit-learn/#files), [numba](https://pypi.org/project/numba/#files) |
| chronos-forecasting | 2.3.2 (2026-09-08) | Yes. Pure Python, needs torch<3, transformers<6, pandas<4 | [PyPI](https://pypi.org/project/chronos-forecasting/#files) |
| timesfm | 3.0.2 (2026-09-09) | Yes, pure Python (`timesfm[torch]` extra) | [PyPI](https://pypi.org/project/timesfm/#files) |
| gluonts | 0.17.0 (2026-07-31) | Yes, but requires `pandas<3`, so pandas 2.3.3 | [PyPI](https://pypi.org/project/gluonts/0.17.0/) |
| neuralforecast | 3.3.0 (2026-10-07) | Probably. Classifiers stop at 3.13, but its dependencies `ray` 2.59.0 and `coreforecast` 0.0.18 have cp314 wheels. Requires torch ≥2.9.1 | [PyPI](https://pypi.org/project/neuralforecast/3.3.0/) |
| pyqlib (Qlib) | 0.9.7 (2025-08-15) | **No**. Wheels cp38–cp312 only. The README support table lists Python 3.8–3.12. GitHub tagged v0.9.8 on 2026-10-08, but PyPI still serves 0.9.7 | [PyPI](https://pypi.org/project/pyqlib/0.9.7/#files), [Qlib README](https://github.com/microsoft/qlib) |
| Kronos repo `requirements.txt` | (git only) | **Not as pinned**. Pins `pandas==2.2.2` (wheels cp39–cp312 only) and `matplotlib==3.9.3` (cp39–cp313) | [Kronos requirements.txt](https://github.com/shiyu-coder/Kronos/blob/master/requirements.txt), [PyPI pandas 2.2.2](https://pypi.org/project/pandas/2.2.2/#files), [PyPI matplotlib 3.9.3](https://pypi.org/project/matplotlib/3.9.3/#files) |
| aim | 3.29.1 (2025-05-08) | **No**. Wheels cp37–cp312, and `aimrocks` 0.5.2 has wheels only to cp312 | [PyPI aim](https://pypi.org/project/aim/3.29.1/#files), [PyPI aimrocks](https://pypi.org/project/aimrocks/#files) |
| flash-attn | 2.8.3.post2 (2026-10-14) | Source only (sdist), so it has to be compiled | [PyPI](https://pypi.org/project/flash-attn/#files) |

- The PyPI package named `kronos` is an unrelated Django cron app ("Kronos is a Django application that makes it easy to define and schedule tasks with cron"). The Kronos forecasting model is installed from its git repository — [PyPI kronos](https://pypi.org/project/kronos/)
- The Kronos paper's own environment was Python 3.13.2, PyTorch 2.7.0, NumPy 1.26.2, Pandas 2.2.2 and Matplotlib 3.9.3 on Ubuntu 24.04.1 — [Kronos paper, Appendix "Experimental Environment"](https://arxiv.org/abs/2508.02739)

#### torch.compile and Python 3.14
- PyTorch dev-discuss, 2025-12-05: torch.compile support for Python 3.14 is complete and "will be included in the next PyTorch release, 2.10". The free-threaded 3.14t build has "basic support". torch.compile runs there, "but multi-threading with torch.compile is still not supported". 3.14 changes that bit PyTorch:
  - `__annotations__` changes "broke TorchScript". jit tests are skipped and TorchScript warns on 3.14+.
  - **The default multiprocessing start method changed from `fork` to `forkserver`**, which affected `DataLoader` tests. PyTorch fixed its tests by setting the method back to `fork`.
  - Source: [dev-discuss: torch.compile support for Python 3.14 completed](https://dev-discuss.pytorch.org/t/torch-compile-support-for-python-3-14-completed/3276)

#### bf16, SDPA and FlashAttention on consumer Blackwell (sm_120)
- RTX 5080 (GB203) tensor throughput from NVIDIA's whitepaper (dense / sparse): **BF16 with FP32 accumulate 112.6 / 225.1 TFLOPS**, FP16 with FP16 accumulate 225.1 / 450.2, FP8 with FP32 accumulate 225.1 / 450.2, TF32 56.3 / 112.6, FP4 900.4 / 1801. The rest of the card: non-tensor FP32 56.3 TFLOPS, 84 SMs, 336 fifth-gen tensor cores, boost 2617 MHz, 16 GB GDDR7, 256-bit bus at 30 Gbps (960 GB/s) — [NVIDIA RTX Blackwell GPU Architecture whitepaper, Table 4 / Appendix B](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
- PyTorch's SDPA hardware gates on `main` (October 2026), in [sdp_utils.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp):
  - The flash backend accepts "[sm80, sm121]". Memory-efficient attention accepts "[sm50, sm121]". sm_120 therefore gets PyTorch's built-in flash kernels.
  - A separate constraint (`check_requires_grad_and_head_dim_gt192_constraints_on_sm86_89_or_120_121`) limits training with head_dim > 192 on sm86/89/120/121.
  - cuDNN attention is preferred only when the compute-capability major version is 9 or 10 (Hopper or datacenter Blackwell) and cuDNN is newer than 9.15.
- nanochat chooses its compute dtype automatically: "CUDA SM 80+ (A100, H100, ...) → `bfloat16`" — [nanochat README](https://github.com/karpathy/nanochat)
- The standalone FlashAttention package on sm_120:
  - **FA2 from source.** Building with `FLASH_ATTN_CUDA_ARCHS=120` crashes nvcc on backward kernels with CUDA 12.8 and 12.9. v2.8.3's prebuilt wheels only cover torch ≤2.8. Open issue filed 2026-03-17 — [flash-attention#2361](https://github.com/Dao-AILab/flash-attention/issues/2361)
  - **FA3** support for sm_120 is still an open request — [flash-attention#1549](https://github.com/Dao-AILab/flash-attention/issues/1549)
  - **FA4 (CuTe DSL).**
    - An SM120 kernel (`flash_fwd_sm120.py`) exists and JIT-compiles, but paged KV and `num_splits` blocked vLLM serving as of 2026-08-03. sm_120 uses `mma.sync` (MMAv2), not WGMMA or tcgen05 — [flash-attention#2307](https://github.com/Dao-AILab/flash-attention/issues/2307)
    - A backward-pass bug on SM12x in v4.0.0b16 was fixed on main by #2671 (2026-07-24) — [flash-attention#2650](https://github.com/Dao-AILab/flash-attention/issues/2650)
    - One user measured FA4 about 5% *slower* than FA2 on an RTX 5090 (2026-04-06) — [flash-attention#2440](https://github.com/Dao-AILab/flash-attention/issues/2440)
- On A100s, flash-attention raised Time-MoE's training speed by 23% and inference by 19%. BF16 instead of FP32 cut the base model's training memory from 2.21 to 1.77 GB — [Time-MoE paper, Table 6](https://arxiv.org/abs/2409.16040)

#### Known sm_120 / RTX 5080 gotchas (GitHub issues)
- **Open, silent-correctness bug on an RTX 5080** — [pytorch#191433](https://github.com/pytorch/pytorch/issues/191433)
  - Filed 2026-07-29, labelled "module: correctness (silent)".
  - Pattern: `torch.compile` with a BF16-autocast `Linear` output added to FP32 embedding outputs, feeding an RMSNorm-style reduction.
  - Result: max absolute error ≈2.2 against eager, on PyTorch 2.13.0+cu130, Python 3.14.6, RTX 5080.
  - Workaround found: converting weights to native BF16 before compiling cut the error to 0.0156.
- A closed bug: "[inductor] Mix-order reduction silently corrupts conv-bias grads (bf16, sm_120) in 2.10.0 — fixed in 2.11" — [pytorch#190796](https://github.com/pytorch/pytorch/issues/190796)
- FP8: on sm_120, `scaled_mm` with mxfp8 accepts only the TN operand layout. NT and NN are rejected by cuBLAS. Open, 2026-09-22 — [pytorch#198126](https://github.com/pytorch/pytorch/issues/198126)

### Inferences
- **Recommended stack:** Python 3.12 (`uv python install 3.12`), torch 2.14.1 from PyPI (the CUDA 13.0 build, sm_120 included), numpy 2.x, pandas 2.3.3 (or 3.0.6 if no GluonTS or Kronos), polars, pyarrow, lightgbm 4.7, transformers 5.x. Python 3.14 also works for the core stack. 3.12 is chosen because it is the only version on which every code base in this report installs as published: Qlib and Kronos's pinned requirements stop at cp312, Aim too.
- Two uv environments avoid dependency fights. Kronos's pins (pandas 2.2.2, huggingface_hub 0.33.1, safetensors 0.6.2) conflict with transformers 5.x and pandas 3.x, so:
  - `stockai/` holds the user's own GPT, LightGBM baselines, Chronos-2 and TimesFM 2.5.
  - `kronos/` is a clone with its pins relaxed or kept, on Python 3.12.
- Minimal commands (inferred from the uv and PyTorch facts above):
  ```
  uv python install 3.12
  uv init stockai --python 3.12 && cd stockai
  uv add torch numpy pandas polars pyarrow lightgbm scikit-learn transformers mlflow tensorboard
  uv run python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0), 'sm_120' in torch.cuda.get_arch_list())"
  ```
  Expected output: `2.14.1`, CUDA `13.0`, `(12, 0)`, `True`. To use cu132 instead, add a `pytorch-cu132` index (`https://download.pytorch.org/whl/cu132`, `explicit = true`) and point `[tool.uv.sources] torch` at it. Never pick `cu126` for this GPU. Following old "use cu128" guides now caps torch at 2.11.
- PyTorch 2.15 (due 2026-10-28) moves stable CUDA to 13.2 only. The default PyPI wheel will then probably switch to CUDA 13.2, which the 595.99 driver supports. A uv lockfile keeps 2.14.1 until the project deliberately upgrades.
- Use `F.scaled_dot_product_attention(..., is_causal=True)` or `nn.MultiheadAttention`. Do not install `flash-attn`: for head_dim ≤ 192, PyTorch already dispatches its own flash kernels on sm_120, and the separate package is source-only and crashes nvcc when built for sm_120.
- Use bf16 autocast (or nanochat-style explicit bf16 compute), not fp16: no loss scaling is needed and the tensor-core rate is the same. Skip FP8 at these model sizes: the gain is small, and mxfp8 on sm_120 has layout limits.
- Because of pytorch#191433, compare `torch.compile(model)` against eager on a fixed batch at the start of each run (max abs difference on logits and loss). Embedding + bf16 Linear + RMSNorm is exactly a GPT input block, which is what that bug hit. If the difference is large, keep weights in native bf16 (the workaround reported in the issue) or drop compile; for a 1–10M-parameter model, compile mostly cuts launch overhead. Swapping RMSNorm for LayerNorm is an untested alternative.
- Python 3.14's `forkserver` default matters with `DataLoader(num_workers>0)`: each worker gets its dataset by pickling, and pickling an `np.memmap` produces a full in-memory copy. With about 26.6M bars × a few float32 features (well under 1 GB), the simplest design keeps the whole panel on the GPU and samples windows with tensor indexing, so no worker processes are needed. If workers are used, pass `multiprocessing_context="fork"`.

### Gaps
- Not checked whether pytorch#191433 is fixed in 2.14.1: the issue was still open on 2026-10-18 and no fix commit was identified.
- Not fetched: PyTorch's 2.14 release blog and pytorch.org's install selector. The CUDA variants and Python support above come from RELEASE.md, the wheel indexes and PyPI metadata.
- No measured SDPA flash-backend throughput on an RTX 5080 was found. A comment in [flash-attention#2307](https://github.com/Dao-AILab/flash-attention/issues/2307) claims "PyTorch SDPA does achieve ~160 TFLOPS on the 5070 Ti", which exceeds NVIDIA's listed dense BF16/FP32-accumulate rates for that class of card. It is unverified and not used here.
- neuralforecast on 3.14 was judged from wheel availability, not by installing it.

---

## 2. Encoding prices for a transformer: bins (StockGPT), mean scaling + uniform bins (Chronos), BSQ tokens over OHLCV (Kronos), patches (TimesFM, PatchTST), continuous heads (Student-t, quantiles); volatility normalization; splits, bad prints and outliers

### Takeaway
For daily stock data the model should see **returns, not price levels**, normalized by recent volatility and clipped. The simplest "next-token" design is StockGPT's: one discrete token per daily return, a cross-entropy loss, and an expected return read from the predicted bin probabilities. Bins defined on a volatility-scaled return work better than fixed 50 bps bins. The user's existing 252-day-vol-scaled market residual, clipped at ±5, is already a good input. Kronos's learned BSQ tokenizer is the strongest published *financial* tokenizer and uses OHLCV, but it adds a second model to train. Patching cuts sequence length (cheap long contexts) without adding signal. On finance data, two studies found discrete/classification objectives beat MSE or Gaussian-NLL regression:
- Kronos's ablation: discrete cross-entropy gave higher IC and RankIC than continuous MSE or NLL heads.
- Re(Visiting): Chronos (binned tokens) pre-trained from scratch beat TimesFM (patches, regression) at equal window length, before costs.

### Cited Findings

#### Quantized returns: StockGPT
- StockGPT data: CRSP common stocks (share codes 10/11, NYSE/AMEX/NASDAQ), about 70M daily returns from 1926–2023. Training on 1926–2000 (about 50M returns): 1926–1990 for parameters, 1991–2000 for hyperparameters. The test period 2001–2023 holds about 20M returns. The model was "trained only once" — [StockGPT, arXiv 2404.05101](https://arxiv.org/abs/2404.05101)
- Tokenization (same source):
  - Returns become integer basis points, cut into **50 bps intervals** closed on the right.
  - The first bin is −10,000 bps (−100%). The last bin, (10,000, Inf), treats every return above +100% as 100%.
  - Each token stands for its bin midpoint, giving **402 tokens**.
  - Example: returns (−2.4%, 0%, 0%, 5%, 4.8%) become tokens (196, 200, 200, 210, 210).
  - The 1st–99th percentile of training daily returns is −9.6% to +11.1%.
- Architecture and training (same source):
  - Block size 256, embedding size 128, 4 attention blocks × 4 heads, dropout 0.2, **0.93M parameters**.
  - 10,000 steps × 64 sequences, with each stock sampled in proportion to its number of observations.
  - Cross-entropy settles around 2.5 after 5,000 steps; with 402 labels the maximum is ln 402.
  - Trained "fully ... locally on a MacBook M2 with 64GB RAM and 30 GPU cores".
- Readout: "the expected return ... will then be the weighted average of return bin midpoints weighted by the corresponding bin probabilities". Sampling many forecasts gives the same answer at higher cost — [StockGPT](https://arxiv.org/abs/2404.05101)

#### Mean scaling + uniform bins: Chronos (and its successors)
- Chronos maps values with an affine transform x̃ = (x − m)/s and uses **mean scaling** (s = mean absolute value of the context). It then quantizes into B = 4094 equal-width bins on [−15, +15]. With PAD and EOS the vocabulary is 4096, and the loss is cross-entropy, i.e. "regression via classification" — [Chronos, arXiv 2403.07815](https://arxiv.org/abs/2403.07815)
- Chronos's stated limitation: "tokens are spaced 30s/(B − 1) from each other ... values closer than that ... may be mapped to the same token". A very small s relative to the series range causes overflow at ±15s. The authors suggest standardization as an inference-time workaround for series with a large level and small variance — [Chronos](https://arxiv.org/abs/2403.07815)
- Chronos-Bolt "chunks the historical time series context into patches" and "directly generate[s] quantile forecasts across multiple future steps". It is up to 250× faster and 20× more memory-efficient than Chronos at the same size. Chronos-2 (120M; a 28M "small" also exists) adds multivariate and covariate support — [chronos-forecasting README](https://github.com/amazon-science/chronos-forecasting)
- Chronos-2 "employs a group attention mechanism that facilitates in-context learning ... across multiple time series within a group". It was trained on synthetic data that imposes multivariate structure — [Chronos-2, arXiv 2510.15821](https://arxiv.org/abs/2510.15821)

#### Learned tokenizer over OHLCV: Kronos (Binary Spherical Quantization)
- What Kronos is: decoder-only models pre-trained on "over 12 billion K-line records from 45 global exchanges" across 7 frequencies. Each OHLCVA bar becomes **one token made of a coarse and a fine subtoken**. The model predicts the coarse subtoken first, then the fine one conditioned on it — [Kronos, arXiv 2508.02739](https://arxiv.org/abs/2508.02739)
- The tokenizer (same source):
  - A small Transformer autoencoder (encoder and decoder each 3 layers, d=256, FFN 512, 4 heads) with a Binary Spherical Quantization (BSQ) layer.
  - Codes are k = 20 bits, split into two 10-bit subtokens, so two 1,024-way predictions replace one 1,048,576-way prediction.
  - Paper Table 12: the split cuts vocabulary-dependent parameters from about 1.7B to 3.4M, and total parameters from 1,842.3M to 102.3M for Kronos-base.
  - Codebook usage is 97.66% (coarse) and 85.25% (fine).
- Input preprocessing: per-window **z-score of each feature** (Open, High, Low, Close, Volume, Amount), then clipping. The fine-tuning config uses `clip = 5.0`. Five learnable time embeddings are added: minute, hour, weekday, day-of-month, month — [Kronos paper, Appendix C](https://arxiv.org/abs/2508.02739); [Kronos finetune/config.py](https://github.com/shiyu-coder/Kronos/blob/master/finetune/config.py)
- The paper argues that discretization makes learning "a more robust, classification" problem, citing "flash-crash" outliers that "lead to unbounded approximation errors" in regression. It says BSQ's quantization error is bounded — [Kronos, Appendix H](https://arxiv.org/abs/2508.02739)
- **Ablation at similar parameter counts (IC / RankIC on price-series forecasting):** Direct-AR with continuous MSE 0.0212 / 0.0149; Prob-AR with continuous NLL 0.0179 / 0.0102; Kronos-Parallel (discrete cross-entropy) 0.0345 / 0.0226; Kronos (discrete cross-entropy, sequential subtokens) 0.0431 / 0.0254. Return forecasting IC: 0.0416, 0.0356, 0.0529, 0.0665 for the same four models — [Kronos, Table 2](https://arxiv.org/abs/2508.02739)
- Averaging more sampled paths raises IC and RankIC steadily (paper Figure 7). Inference settings for return forecasting: T = 0.6, top-p = 0.9, N = 10 samples — [Kronos](https://arxiv.org/abs/2508.02739)

#### Patching: TimesFM and PatchTST
- TimesFM is a patched decoder. Example: input patch length 32 and output patch length 128, so each step uses the history to forecast the next 128 points. 200M parameters, pre-trained on O(100B) time points — [TimesFM, arXiv 2310.10688](https://arxiv.org/abs/2310.10688)
- TimesFM 2.5 (Sept 2025): 200M parameters, context up to 16k, and an optional 30M **quantile head** for continuous quantile forecasts out to 1k steps — [TimesFM README](https://github.com/google-research/timesfm)
- PatchTST: patches as tokens, plus **channel independence** (every univariate channel shares the embedding and Transformer weights). Patch P = 16 with stride S = 8 on a 336-step look-back cut training time "as much as 22 times" on large datasets — [PatchTST, arXiv 2211.14730](https://arxiv.org/abs/2211.14730)

#### Continuous output heads
- Lag-Llama: a decoder-only model with lagged values as covariates and a **Student's t** head that outputs degrees of freedom, mean and scale — [Lag-Llama, arXiv 2310.08278](https://arxiv.org/abs/2310.08278)
- A time-series scaling-law study used a Student-t head and found it "enables significantly more stable training than a Gaussian head or MSE loss" on heavy-tailed data — [Edwards et al., arXiv 2405.13867](https://arxiv.org/abs/2405.13867)
- Time-MoE trains with a **Huber loss** "which provides greater robustness to outliers and improves training stability" — [Time-MoE, arXiv 2409.16040](https://arxiv.org/abs/2409.16040)
- Sundial avoids discrete tokens altogether: a flow-matching "TimeFlow Loss" over the next patch, pre-trained on about 1 trillion time points — [Sundial, arXiv 2502.00816](https://arxiv.org/abs/2502.00816)

#### Volatility normalization
- Deep Momentum Networks put learned trading rules "into the volatility scaling framework of time series momentum" and train directly on the Sharpe ratio. On 88 futures, the Sharpe-optimized LSTM more than doubled traditional methods without costs and still beat them up to 2–3 bps of costs. The paper adds a turnover regularization term for less liquid assets — [Lim, Zohren, Roberts, arXiv 1904.04912](https://arxiv.org/abs/1904.04912)
- The user's prior project defines its variable as the market-residual daily log return divided by the stock's trailing 252-day residual volatility, clipped at ±5. Beta, volatility and liquidity use only data up to t−1 — [correlation2 test-results.md](/home/bhoult/programming/correlation/test-results.md). The same project quantizes the ±5 sd values to INT8 at 25 steps per sd for its GPU correlation engine — [correlation2 SPEC.md](/home/bhoult/programming/correlation/SPEC.md)

#### Splits, bad prints, outliers (the user's Stooq data and published pipelines)
- Stooq bulk prices are **split- and dividend-adjusted**. Known split dates show no jump (NVDA 10:1 on 2024-06-10: 120.86 → 121.76). AAPL on 2020-01-02 reads 72.72, against an actual close of $300.35 (75.09 after the split) — [correlation2 ANALYSIS.md §3](/home/bhoult/programming/correlation/ANALYSIS.md)
- Contamination and survivorship in the same data, per [correlation2 ANALYSIS.md §3](/home/bhoult/programming/correlation/ANALYSIS.md):
  - The test ticker ZVZZT moves 3,000× in a day.
  - There are **69 one-day bad prints** since mid-2021 (a move of more than 2× that undoes itself the next day). Most other split-sized moves are real crashes and should be kept.
  - Before decimalization (2001), ticks plus decades of adjustment produce fake 0% days and stair-stepped returns.
  - **Survivorship:** Stooq ships only symbols listed when the snapshot was taken.
  - The prior project's default analysis start is 2005-01-01 — [correlation2 SPEC.md](/home/bhoult/programming/correlation/SPEC.md)
- Kronos's cleaning, per [Kronos, Appendix B](https://arxiv.org/abs/2508.02739):
  - A missing price is treated as a hard boundary, splitting the series into segments.
  - Missing volume or amount becomes 0, and volume is randomly zeroed in 5% of training samples.
  - Segments with "abnormal price spikes or prolonged periods of" stagnation are filtered with frequency-specific thresholds (Daily row of Table 4: 128, 0.30, 1, 3).
- StockGPT treats any daily return above +100% as +100% (its last bin). −100% is the natural floor and has its own bin — [StockGPT](https://arxiv.org/abs/2404.05101)

#### Head-to-head evidence on financial data
- "Re(Visiting) Time Series Foundation Models in Finance" (2025-11-23) used daily excess returns from 94 countries, 1990–2023: up to 176.96M US and 496.89M global observations. Chronos tiny/mini/small and TimesFM at about 8M and 20M were pre-trained from scratch every year — [arXiv 2511.18578](https://arxiv.org/abs/2511.18578). Results (window 512, before costs):
  - Chronos (small): annual return 36.84%, Sharpe 5.42.
  - TimesFM (20M): annual return 30.36%, Sharpe 3.66.
  - TSFMs improved with longer windows, while tree/linear benchmarks did relatively better on short windows. TimesFM (20M) returned −18.22% a year with 5 past returns, against 30.36% with 512.

### Inferences
- **Fixed 50 bps bins (StockGPT) are coarse and ignore volatility.** For a large cap with daily σ ≈ 1–1.5%, ±1σ covers only about 4–6 bins. A volatile small cap uses many more bins, so the same bins mean different things for different stocks.
- **Prefer bins on a volatility-scaled value.** On the user's variable z ∈ [−5, 5], uniform 0.05σ bins give 201 tokens and 0.1σ bins give 101. Each token then has the same meaning across stocks and regimes, and ±5 clipping already takes care of extreme values and bad prints. The forecast is E[z] = Σ p_i·mid_i (and P(z > 0)), multiplied by the stock's σ_252 to get a return. Equal-frequency (quantile) bins are another option. They give every token equal training mass but should be fitted on the training years only, to avoid look-ahead.
- **Chronos mean scaling on raw prices is a poor fit for daily changes.** With s ≈ the price level, the bin spacing 30s/4093 is about 0.73% of the price, so typical daily moves span only one or two bins. On returns (s ≈ mean |r| ≈ 1–2%), the spacing is about 1–1.5 bps, which is fine. Feed returns or z, not prices, to Chronos-style tokenizers.
- **Kronos uses more information than closes alone.** The O/H/L/C/V fields carry intraday range and volume, which help volatility forecasts. The costs:
  - Two-stage training (the tokenizer, then the predictor).
  - Per-window z-scoring throws away absolute level and scale, which is fine for returns-like targets.
  - A 512-token context limit for small/base.
  - A forecast that is a sampled price path, from which a return signal has to be derived.
- **Patching is mainly a compute and context lever.** Patch-32 cuts tokens 32× (section 3), allowing contexts of several years. But each daily-return patch is mostly noise, and the only financial comparison found (Re(Visiting)) had the binned Chronos ahead of patched TimesFM at equal window.
- **Continuous heads are a sensible second experiment.** A Student-t head on z (fat tails) or a quantile/pinball head (direct intervals) both work. Plain MSE on raw returns lets a few tail days dominate the gradient, so use Huber or Student-t, or clip as the user already does.
- **Pros and cons in a low signal-to-noise setting:**

| Encoding | Pros | Cons |
|---|---|---|
| Uniform bins on raw returns (StockGPT) | Simplest. LLM code works unchanged. Gives a full predictive distribution. Cross-entropy is robust to outliers | Resolution differs by stock volatility. Tail bins are sparse. The mean is a small difference between large probabilities |
| Bins on vol-scaled residual z (recommended start) | Same meaning across stocks and time. Removes market beta. The user already computes and clips it | Adds dependence on the σ and β estimates, which must use t−1 data only |
| Chronos mean-scaling + 4094 bins | Proven pipeline and training scripts | Badly suited to price levels. Per-window scaling hides level. Large vocabulary is wasted on returns |
| Kronos BSQ over OHLCV | Uses range and volume. Pretrained on financial data. Discrete loss beat regression in its ablation | Two models to train. Code comments partly AI-written. Max context 512 (small/base) |
| Patches (PatchTST / TimesFM) | Far fewer tokens, so long contexts are cheap | Little evidence of benefit on daily returns. Regression heads are sensitive to tails |
| Continuous Student-t or quantile head | Calibrated tails and intervals | Weaker than cross-entropy on IC in the one finance ablation found (Kronos) |

### Gaps
- No published head-to-head of these encodings on **US daily** returns with a common protocol was found. Kronos's ablation pools its own financial benchmarks, and Re(Visiting) compares only Chronos and TimesFM variants.
- Kronos's paper does not state (in the parts extracted) the end date of its pre-training data, or whether US daily bars through 2025 are in it. That is needed to judge look-ahead risk when testing on 2024-06-28 to 2025-06-27.
- No source was found on the best bin width or bin count for vol-scaled daily returns. The 0.05–0.1σ suggestion is a reasoned default, not a tested result.

---

## 3. Model size vs data size: scaling laws, tokens from 26.6M / 100M bars, FLOPs, wall-clock and VRAM on an RTX 5080

### Takeaway
Scaling laws hold for time series. But financial return forecasting is **data-limited**: the largest finance study found the gains from more parameters flatten once parameters outrun the effective sample. With one token per daily bar, the US file (26.6M bars) is about 27M tokens and all markets (about 100M bars) about 100M tokens. LLM-style compute-optimal sizing (about 20 tokens per parameter, with up to about 4 epochs of repetition nearly free) then points to **about 1–5M parameters for US-only and about 5–20M for all markets**. A from-scratch 100M model is far past that. Compute is not the constraint:
- An epoch on the US data costs about 1.5e14 FLOPs at 1M parameters, 1.8e15 at 10M and 1.4e16 at 100M.
- On a 112.6-TFLOPS (bf16) RTX 5080 that is roughly seconds to a minute (1M), a few minutes (10M) and 5–40 minutes (100M) per epoch.
- 16 GB of VRAM is ample for anything up to about 100M parameters, and allows fine-tuning Kronos-base (102M).

### Cited Findings

#### General time-series scaling results
- Edwards et al. (2024, revised 2025-01-08), about 8B data points in 30,211,687 series from 38 sources, published at [arXiv 2405.13867](https://arxiv.org/abs/2405.13867):
  - Power-law scaling with parameters, data and compute "spanning five orders of magnitude".
  - Aspect ratio and head count matter little.
  - The data was "large enough so that for our largest models (∼100M parameters) we are still operating in the ∼infinite data limit".
- "Scaling Law for Time Series Forecasting" (NeurIPS 2024), [arXiv 2405.15124](https://arxiv.org/abs/2405.15124):
  - "More capable models do not always outperform less capable models, and longer input horizons may hurt performance".
  - "Optimal horizon does exist, and it will increase if the amount of available training data increases."
- Yao et al. (ICLR 2025) used 16.8B time points across 7 domains, enough for models of 10^3–10^8 parameters. Findings, from [arXiv 2410.12360](https://arxiv.org/abs/2410.12360):
  - Encoder-only Transformers scale better than decoder-only ones.
  - Architectural enhancements in two state-of-the-art TSFMs mainly improve in-distribution results and reduce out-of-distribution scaling.
  - In one emergent-behaviour example, 10K and 1M models failed to capture a periodic pattern that 10M and 100M models reproduced.
- Timer (ICML 2024): GPT-style next-token pre-training on curated datasets of "up to 1 billion time points" (UTSD) — [arXiv 2402.02368](https://arxiv.org/abs/2402.02368)
- Time-MoE (ICLR 2025): Time-300B (309.09B observations), models up to 2.4B parameters. "Our results validate the applicability of scaling laws for training tokens and model size". Trained on 128 × A100-80G in BF16 — [arXiv 2409.16040](https://arxiv.org/abs/2409.16040)
- Kronos: "as the model size scales up, performance on these tasks consistently improves", from small (24.7M) through base (102.3M) to large (499.2M). Configurations: small 8 layers, d 512, FFN 1024, 8 heads; base 12/832/2048/16; large 18/1664/3072/32. Pre-trained on 24 RTX 4090D GPUs (3 pods × 8) — [Kronos paper](https://arxiv.org/abs/2508.02739); [Kronos README model zoo](https://github.com/shiyu-coder/Kronos)
- LLM heuristics:
  - Chinchilla: model size and training tokens "should be scaled equally" (the 70B model was trained on 1.4T tokens, about 20 tokens per parameter) — [Hoffmann et al., arXiv 2203.15556](https://arxiv.org/abs/2203.15556)
  - Data-constrained scaling: "training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero" — [Muennighoff et al., arXiv 2305.16264](https://arxiv.org/abs/2305.16264)

#### Finance-specific evidence on size vs data
- Re(Visiting) TSFMs in Finance, from [arXiv 2511.18578](https://arxiv.org/abs/2511.18578):
  - "adding more observations measurably improves both forecasting accuracy and portfolio performance".
  - "performance remains distinctly data limited: when moving to substantially larger models, such as TSFMs, and parameter counts outpace the effective sample, marginal returns to complexity flatten ... the frontier ... is defined by the scale and alignment of finance-native data. Synthetic data can offer partial relief, but it is no substitute".
  - Hyperparameter tuning let TSFMs beat the benchmarks "even without scaling the dataset".
- Compute behind that study, from the same paper:
  - About 50,000 GPU-hours on GH200 (H100 96 GB), covering 396 Chronos and 264 TimesFM pre-trainings plus inference.
  - Reported Chronos pre-training times were about 25 h (tiny, mini) to about 50 h (small). TimesFM took about 40 h for the larger configurations.
  - Benchmarks took 2–15 CPU-hours.
- "Pretrained Time-Series Foundation Models for Financial Return Forecasting" (2026-06-25): five liquid US stocks; TSFMs won 8 of 10 task rankings. "Gains over the random-walk benchmark are small and sparse": the Diebold-Mariano test rejected only for Chronos on AMZN and Moirai-2.0 on GOOG — [arXiv 2606.27100](https://arxiv.org/abs/2606.27100)
- A preprint pre-trained on 10B synthetic chaotic series and reported "a scaling law-like phenomenon" for zero-shot Bitcoin prediction — [arXiv 2509.04921](https://arxiv.org/abs/2509.04921). Weak evidence: synthetic data and a single asset.

#### Throughput and compute anchors
- Training FLOPs ≈ 6N per token for an N-parameter (non-embedding) transformer, plus an attention term of 2·n_layer·n_ctx·d_attn per token in the forward pass (×3 for training) — [Kaplan et al., arXiv 2001.08361](https://arxiv.org/abs/2001.08361)
- nanoGPT "baby GPT", per [nanoGPT README](https://github.com/karpathy/nanoGPT) and [config](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py):
  - 6 layers, 6 heads, 384 dims, context 256, batch 64, 5,000 iterations, dropout 0.2, lr 1e-3.
  - "On one A100 GPU this training run takes about 3 minutes", best val loss 1.4697.
- Chronos training: 200K steps at effective batch 256, context 512 + 64 predicted. On one 8×A100-40GB p4d.24xlarge: Mini 7.68 h, Small 7.73 h, Base 17.96 h, Large 63.05 h (TF32 + model compilation) — [Chronos paper, §5.2 and Table 6](https://arxiv.org/abs/2403.07815)
- StockGPT (0.93M parameters) trained 10,000 steps × 64 sequences × 256 tokens on a MacBook M2 — [StockGPT](https://arxiv.org/abs/2404.05101)
- nanochat: a GPT-2-grade model (around d24–d26) on 8×H100. Early-2026 leaderboard rows show 3.04 h (2026-01-29) and 2.91 h with fp8 (2026-02-02), and the README now says the run "takes ~1.5 hours (down from 3h)". "The speedrun is a 4e19 FLOPs capability model." On a single GPU "you'll have to wait 8 times longer". Below 80 GB, reduce `--device-batch-size` — [nanochat README](https://github.com/karpathy/nanochat)
- RTX 5080 peak: 112.6 dense BF16 tensor TFLOPS (FP32 accumulate), 16 GB, 960 GB/s — [NVIDIA whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)

#### Memory accounting rules
- Mixed-precision Adam keeps 16 bytes per parameter of model state: 2 for fp16/bf16 parameters, 2 for gradients, 12 for fp32 master weights, momentum and variance — [ZeRO, arXiv 1910.02054](https://arxiv.org/abs/1910.02054)
- Activation memory per transformer layer is about s·b·h·(34 + 5·a·s/h) bytes in 16-bit without recomputation. The 5as/h term comes from storing attention scores, softmax and dropout, and selective recomputation removes it — [Korthikanti et al., arXiv 2205.05198](https://arxiv.org/abs/2205.05198)
- Kronos's attention modules call `F.scaled_dot_product_attention` — [Kronos model/module.py](https://github.com/shiyu-coder/Kronos/blob/master/model/module.py)

### Inferences
- **Tokens produced by each encoding** (arithmetic; "bars" means stock-days):

| Encoding | Tokens per bar | US file, 26.6M bars | All markets, ~100M bars | Note |
|---|---|---|---|---|
| One return-bin token per day (StockGPT style, or z-bins) | 1 | **26.6M** | **100M** | Every position is a next-token target |
| Chronos, one value per token, close only | 1 | 26.6M | 100M | Five tokens per bar if each OHLCV field becomes its own token |
| Kronos, one OHLCVA token per bar (2 subtokens) | 1 (2 predictions) | 26.6M tokens / 53.2M subtoken predictions | 100M / 200M | Each token encodes 6 fields |
| Patch 16, no overlap | 1/16 | 1.66M | 6.25M | 16 values per token |
| Patch 16, stride 8 (PatchTST) | 1/8 | 3.33M | 12.5M | Overlapping |
| Patch 32 (TimesFM input patch) | 1/32 | 0.83M | 3.1M | |
| Cross-sectional day token (all stocks per day) | about 6,400 days for 2000–2025 | | | Very few "time" samples |

  - Usable counts are smaller than the raw rows. The 26.6M rows include ETFs and pre-2005 history. The prior project's liquid universe (2,155 stocks) over about 20 years gives at most about 2,155 × 252 × 20 ≈ 11M bars.
  - Stocks on the same day are strongly correlated, so the effective number of independent observations is far below the token count.
  - With patches, the count of supervised scalar targets is still about the number of bars, even though the token count shrinks. Token-based sizing rules therefore fit patch models poorly.
- **Compute-optimal sizing as a rough guide.** At about 20 tokens per parameter with ≤4 epochs:
  - US-only, 26.6M tokens: about 1.3M parameters for one pass, about 5M with 4 epochs.
  - All markets, 100M tokens: about 5M for one pass, about 20M with 4 epochs.
  - So StockGPT's 0.93M is near "optimal" for US data, Kronos-mini (4.1M) or Kronos-small (24.7M) are at or above the ceiling for all markets, and a fresh 100M model would need about 64 US epochs to reach Chinchilla's ratio. That is well past the 4 epochs where repetition stops helping, so expect overfitting unless dropout, weight decay and early stopping are strong.
  - The low signal-to-noise ratio of daily returns means the useful information per token is far lower than in text, which pushes the optimum smaller still.
- **FLOPs per epoch and wall-clock on the RTX 5080.** Formula: 6N + 6·L·T·d per token. MFU means the share of the 112.6 TFLOPS bf16 peak actually achieved. Tiny models are kernel-launch- and Python-bound, so they sit at the low end of the MFU range:

| Model (config) | Non-embedding params | FLOPs per token | 1 epoch, US 26.6M | Time per US epoch | 1 epoch, all 100M | Time per all-market epoch |
|---|---|---|---|---|---|---|
| ~1M (4 layers, d=128, ctx 256) | 0.79M | 5.5M | 1.5e14 | about 0.1–1.1 min (MFU 40% down to 2%) | 5.5e14 | about 0.2–4 min |
| ~10M (6 layers, d=384, ctx 256) | 10.6M | 67M | 1.8e15 | about 1–5 min (MFU 20%–5%) | 6.7e15 | about 5–20 min |
| ~100M (12 layers, d=768, ctx 512) | 84.9M | 538M | 1.4e16 | about 5–21 min (MFU 40%–10%) | 5.4e16 | about 20–80 min |

  - Throughput check: the nanoGPT baby-GPT run is about 10.6M parameters × 82M tokens, roughly 5.5e15 FLOPs including the attention term. Finishing in about 3 min on an A100 means about 30 TFLOP/s sustained. That would be about 27% MFU on a 5080, so 3–6 min on the 5080 is a reasonable guess for the same job.
  - Even a 100M model × 10 epochs × 100M tokens (5.4e17 FLOPs) is roughly 3–13 hours.
  - nanochat's 4e19-FLOP GPT-2 speedrun would take about 10 days on one 5080 at about 40% MFU. That marks the scale that is out of reach.
- **VRAM on 16 GB.** Rules: 16 bytes per parameter of model state, plus activations of about 34·s·b·h·L bytes when fused SDPA avoids storing the s×s scores:
  - 1M model: about 0.01 GB of state, about 0.3 GB of activations at batch 64 × 256.
  - 10M model: about 0.2 GB of state, about 1.3 GB at batch 64 × 256.
  - 100M model: about 1.4 GB of state. Activations are about 5.1 GB at batch 32 × 512, about 10.3 GB at 64 × 512 (tight but feasible), and about 20.5 GB at 128 × 512, which needs gradient accumulation or activation checkpointing.
  - Kronos-base (102M) fine-tuning at the default batch of 50: with the CSV example's 512-step window, about 1.6 GB of state plus about 8.7 GB of activations, so it fits. With the Qlib demo's 90 + 10-step windows, activations are about 1.7 GB. Kronos-large (499M; weights not released) would need about 8 GB of model state alone.
  - Memory limits sit around 300–500M parameters. Data size limits models far earlier.

### Gaps
- No measured tokens/second or MFU for small GPT-style models on an RTX 5080 (or any RTX 50-series card) was found. The time ranges above are estimates from peak FLOPs and assumed MFU.
- No published scaling-law fit specific to daily equity returns was found. Re(Visiting) gives a qualitative "data limited" conclusion, and Kronos shows monotone gains on its own financial benchmarks without a fitted exponent.
- The figure of about 100M bars for all markets comes from the user and was not verified.
- Chinchilla and data-repetition results come from language modelling. How well they transfer to low signal-to-noise return series is unknown.

---

## 4. Training recipe: context length, single-stock vs cross-sectional context, market/sector context, walk-forward retraining, early stopping, regularization, target and loss

### Takeaway
A defensible beginner recipe:
- **Data:** univariate sequences of the vol-scaled residual return z for each stock, binned into about 100–200 tokens. Context 256 trading days.
- **Model:** a 1–10M-parameter causal transformer trained with cross-entropy on all positions.
- **Splits:** train / validation / test strictly by date with a gap, and early stopping on validation loss *and* validation rank IC.
- **Retraining:** expanding windows, refit each year.
- **Evaluation:** cross-sectional rank IC and net-of-cost portfolio returns, against SPY/QQQ and random-guess accounts (the correlation2 protocol).
- **Later experiments:** cross-series attention (Chronos-2-style groups or iTransformer) and market or sector tokens.

Every published finance result reviewed shows edge that is small or vanishes after realistic costs. Measure against costs from day one.

### Cited Findings

#### Context length
- Contexts in published models:
  - StockGPT: 256 trading days, "which approximates the number of trading days in a year" — [StockGPT](https://arxiv.org/abs/2404.05101)
  - Kronos small/base: max context 512; Kronos-mini: 2048 — [Kronos README](https://github.com/shiyu-coder/Kronos)
  - Chronos: 512 context, 64 predicted — [Chronos](https://arxiv.org/abs/2403.07815)
  - TimesFM 2.5: up to 16k — [TimesFM README](https://github.com/google-research/timesfm)
- Kronos's fine-tuning defaults: `lookback_window = 90`, `predict_window = 10`, `max_context = 512`. The CSV pipeline example uses lookback 512 and predict 48 on 5-minute bars — [Kronos finetune/config.py](https://github.com/shiyu-coder/Kronos/blob/master/finetune/config.py); [Kronos finetune_csv README](https://github.com/shiyu-coder/Kronos/blob/master/finetune_csv/README.md)
- In Re(Visiting), with windows of 5, 21, 252 and 512 days, TSFM performance rose with window length, while benchmarks did relatively better on short windows — [arXiv 2511.18578](https://arxiv.org/abs/2511.18578)
- The best look-back grows with the amount of training data — [Scaling Law for TSF, arXiv 2405.15124](https://arxiv.org/abs/2405.15124)

#### Single-stock (channel-independent) vs cross-sectional context
- PatchTST's channel independence (one shared model over univariate channels) "can improve the long-term forecasting accuracy significantly" — [PatchTST](https://arxiv.org/abs/2211.14730)
- iTransformer inverts the axes. "The time points of individual series are embedded into variate tokens which are utilized by the attention mechanism to capture multivariate correlations" — [iTransformer, arXiv 2310.06625](https://arxiv.org/abs/2310.06625)
- Chronos-2's group attention shares information across "sets of related series, variates of a multivariate series, or targets and covariates" — [Chronos-2](https://arxiv.org/abs/2510.15821)
- Timer-XL's TimeAttention captures "intra- and inter-series dependencies of flattened time series tokens (patches)" — [Timer-XL, arXiv 2410.04803](https://arxiv.org/abs/2410.04803)
- TimesFM 3.0 (Aug 2026) adds native multivariate forecasting with past-only and past-and-future covariates. Its weights are non-commercial — [TimesFM README](https://github.com/google-research/timesfm)
- In finance, Chronos-2 with multivariate inputs beat univariate on the Magnificent-7 and US Treasury panels (2000–2025 rolling). But "mixing time series across equity and interest rate markets reduces forecast accuracy, indicating that adding noisy context degrades model performance" — [arXiv 2605.21504](https://arxiv.org/abs/2605.21504)
- MASTER alternates intra-stock and inter-stock aggregation and "leverages market information for automatic feature selection" — [MASTER, arXiv 2312.15235](https://arxiv.org/abs/2312.15235)
- Kronos is univariate per asset. Its README warns that production use needs portfolio optimization and "risk factor neutralization" (market beta, size, value) to isolate "pure alpha" — [Kronos README](https://github.com/shiyu-coder/Kronos)

#### Walk-forward retraining, validation splits, early stopping
- Re(Visiting) pre-trained "from scratch for each year using only the financial data available up to that point ... preventing look-ahead bias, an issue that may arise when employing off-the-shelf TSFMs". That produced 23 yearly models (out-of-sample 2001–2023) on expanding windows from 1990. Performance decayed over time for all models, more slowly for TSFMs — [arXiv 2511.18578](https://arxiv.org/abs/2511.18578)
- StockGPT used a single fit: 1926–1990 training, 1991–2000 validation and tuning, 2001–2023 test — [StockGPT](https://arxiv.org/abs/2404.05101)
- Kronos's demo splits: train 2011-01-01 to 2022-12-31, validation 2022-09-01 to 2024-06-30, test 2024-04-01 to 2025-06-05, backtest 2024-07-01 to 2025-06-05. The overlaps are deliberate, "to account for the `lookback_window`". The best checkpoint is chosen by validation loss — [Kronos finetune/config.py](https://github.com/shiyu-coder/Kronos/blob/master/finetune/config.py); [finetune_csv README](https://github.com/shiyu-coder/Kronos/blob/master/finetune_csv/README.md)
- Qlib provides "Online serving and automatic model rolling" (rolling retraining) and the DDG-DA concept-drift method — [Qlib README](https://github.com/microsoft/qlib)

#### Optimizer and regularization settings used by reference models
- StockGPT: dropout 0.2, batch 64, 10,000 steps — [StockGPT](https://arxiv.org/abs/2404.05101)
- nanoGPT baby GPT: lr 1e-3 decayed to 1e-4 over 5,000 iterations, warmup 100, β2 = 0.99 "because number of tokens per iter is small", dropout 0.2 — [nanoGPT config](https://github.com/karpathy/nanoGPT/blob/master/config/train_shakespeare_char.py)
- Kronos pre-training: AdamW, cosine schedule with warmup from 10% of peak over 15,000 steps. "As model scale increases, we decrease the peak learning rate and dropout probability while increasing the weight decay" (paper Table 5 lists dropouts of 0.25, 0.20 and 0.00 and weight decays of 0.01, 0.05 and 0.10). Volume is randomly zeroed for 5% of samples — [Kronos paper](https://arxiv.org/abs/2508.02739)
- Kronos fine-tuning defaults, from [Kronos finetune/config.py](https://github.com/shiyu-coder/Kronos/blob/master/finetune/config.py):
  - Tokenizer lr 2e-4, predictor lr 4e-5.
  - AdamW β = (0.9, 0.95), weight decay 0.1.
  - 30 epochs of 2,000 × batch samples each, batch 50 per GPU.
  - `clip = 5.0`.
  - `use_comet = True` (Comet ML logging) by default.
- Chronos: AdamW with weight decay 0.01, lr 1e-3 decayed linearly to 0 over 200K steps — [Chronos](https://arxiv.org/abs/2403.07815)
- Chronos-2 `Chronos2Pipeline.fit()`: `finetune_mode` "full" or "lora", `learning_rate` default 1e-6 (about 1e-5 recommended for LoRA), `num_steps` default 1000 — [chronos-forecasting src/chronos/chronos2/pipeline.py](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py)
- Time-MoE: weight decay 0.1, β1 0.9, β2 0.95, 10,000-step linear warmup then cosine, BF16, sequence packing — [Time-MoE](https://arxiv.org/abs/2409.16040)

#### Targets, losses and evaluation metrics
- Targets and losses used:
  - Cross-entropy over return bins (StockGPT, Chronos), or over hierarchical subtokens (Kronos) — [StockGPT](https://arxiv.org/abs/2404.05101); [Chronos](https://arxiv.org/abs/2403.07815); [Kronos](https://arxiv.org/abs/2508.02739)
  - Huber (Time-MoE) — [Time-MoE](https://arxiv.org/abs/2409.16040)
  - Student-t NLL (Lag-Llama; Edwards et al.) — [Lag-Llama](https://arxiv.org/abs/2310.08278); [Edwards](https://arxiv.org/abs/2405.13867)
  - Quantile (pinball) heads (Chronos-Bolt; TimesFM 2.5) — [chronos-forecasting README](https://github.com/amazon-science/chronos-forecasting); [TimesFM README](https://github.com/google-research/timesfm)
  - Training on the Sharpe ratio directly, with a turnover penalty — [Deep Momentum Networks](https://arxiv.org/abs/1904.04912)
- Kronos measures forecasts with IC and RankIC between predicted and realized values. Its investment simulation ranks stocks daily on predicted return, holds the top names equal-weight with a minimum 5-day holding period and a daily cap on names traded, and charges 0.15% per trade — [Kronos paper, Appendix D](https://arxiv.org/abs/2508.02739)
- Re(Visiting) reports out-of-sample R², directional accuracy and macro-F1, then daily-rebalanced equal-weight long-short portfolios — [arXiv 2511.18578](https://arxiv.org/abs/2511.18578)

#### What results look like, especially after costs
- StockGPT (all before costs unless noted), from [StockGPT](https://arxiv.org/abs/2404.05101):
  - Equal-weight daily long-short: 119% a year, Sharpe 6.5 (2001–2023).
  - 69% a year net in a worst case of full daily replacement (400% turnover) at 5 bps per trade.
  - Value-weighted: 27% a year, Sharpe about 1.0, "consistent with ... small stocks are more predictable".
- Re(Visiting), from [arXiv 2511.18578](https://arxiv.org/abs/2511.18578):
  - Benchmarks: US out-of-sample R² of −0.47% for linear regression and −0.10% for CatBoost. Directional accuracy "just above 51%".
  - Chronos (small) pre-trained from scratch, window 512: R² −0.59%.
  - After tuning, Chronos (small) reached 51.74% directional accuracy, 41.89% a year and Sharpe 6.78 before costs, against CatBoost's 51.16%, 47.25% and 6.46.
  - With costs: "Moving from 0 bps to 20 bps (and beyond) drives all long–short strategies to negative Sharpe ratios across window sizes, regardless of model class."
- Continual pre-training of TimesFM on 100M financial time points improved price-prediction accuracy. Mock-trading Sharpe ratios ranged from −0.483 to 1.679 depending on horizon, "assuming no trading costs" — [arXiv 2412.09880](https://arxiv.org/abs/2412.09880)
- Off-the-shelf TSFMs "perform poorly in zero-shot and fine-tuning settings, whereas models pre-trained from scratch on financial data achieve substantial forecasting and economic improvements" — [arXiv 2511.18578](https://arxiv.org/abs/2511.18578)
- On five large US stocks, TSFM gains over a random walk were "small and sparse" — [arXiv 2606.27100](https://arxiv.org/abs/2606.27100)
- The user's prior project: liquid US stocks, a 2024-06-28 to 2025-06-27 holdout, daily top-20 rebalancing, baselines of SPY, QQQ, equal-weight and 1,000 random guessers — [correlation2 ANALYSIS.md §4](/home/bhoult/programming/correlation/ANALYSIS.md). It found no lead-lag edge beyond chance, and nothing survived 5–20 bps of costs (from the assignment brief).

### Inferences
- **Step-by-step beginner recipe** (built from the cited settings; numbers are starting points, not tuned results):
  1. **Data.**
     - Export from the correlation2 store: one array per stock of z (vol-scaled market-residual return, clipped ±5), raw log return, σ_252, dollar volume, and a date index.
     - Universe and dates: start 2005-01-01, drop ZVZZT and ETFs, repair the 69 bad prints, and reuse the liquid-universe filter (price ≥ $5, median dollar volume ≥ $5M, ≥95% of days traded) for evaluation. Training can also include less liquid names.
     - Record survivorship as a known bias.
  2. **Tokenize.** Uniform 0.1σ bins on [−5, 5] give 101 tokens (or 0.05σ for 201), plus a PAD token. Optionally add a second embedding that is summed in (Kronos-style): day-of-week, month, and a binned market-return or VIX-proxy token for the same day t−1.
  3. **Splits.**
     - Train 2005 to 2021-06-27.
     - Validation 2021-06-28 to 2023-06-27 (two years, including the 2022 drawdown).
     - Embargo of at least one day so no target date sits in two splits.
     - Optional "dev test" 2023-06-28 to 2024-06-27.
     - **Final holdout 2024-06-28 to 2025-06-27, touched once** (correlation2's protocol).
     - Then a walk-forward check: refit each year for 2015–2024 on expanding windows and report every out-of-sample year.
  4. **Baselines first:**
     - A zero forecast and the unconditional bin histogram. The validation cross-entropy of the histogram is the floor; report the model's gain in nats or bits per token over it.
     - Last-return reversal and momentum.
     - LightGBM on lagged z, |z|, volume and calendar features.
     - Random-guess accounts.
  5. **Model v0 (StockGPT-size).**
     - Architecture: nanoGPT-style decoder, 4–6 layers, d = 128–256, 4 heads, context 256, dropout 0.1–0.2.
     - Optimizer: AdamW, lr 1e-3 with cosine decay to 1e-4, warmup 100–500 steps, β = (0.9, 0.95–0.99), weight decay 0.1.
     - Batches: 128–512 windows sampled in proportion to series length.
     - bf16 autocast and SDPA `is_causal=True`.
     - Cross-entropy on every position.
     - Evaluate every N steps on validation cross-entropy and on daily cross-sectional rank IC of E[z]. Stop early when rank IC stops improving: cross-entropy can keep falling by modelling volatility while saying nothing about direction.
  6. **Signal to portfolio.** Each day, rank by E[z] (or E[z]·σ). Trade top-N long, or top/bottom deciles, with daily or weekly rebalancing. Report gross and net returns at 5, 10 and 20 bps, turnover, and comparisons with SPY, QQQ and random guessers.
  7. **v1 experiments, one change at a time:**
     - A longer context (512).
     - Training on all markets with a market-ID embedding.
     - A Student-t head on z instead of bins.
     - Cross-sectional context: attention over the stocks of a sector on the same day (iTransformer- or Chronos-2-group-style), or a market/sector summary token.
  8. **Fine-tuning comparison:**
     - Chronos-2 zero-shot, then `fit(finetune_mode="lora", learning_rate=1e-5, num_steps≈1000–5000)` on z series.
     - TimesFM 2.5 zero-shot (Apache-2.0 weights).
     - Kronos-small or Kronos-mini fine-tuned on US daily OHLCV with `finetune_csv` (single GPU: `torchrun --standalone --nproc_per_node=1`, `use_comet=False`).
     - All on the same holdout. Off-the-shelf models may have seen 2024–2025 data in pre-training, so the user's own from-scratch model is the only fully clean comparison.
- **Target choice.**
  - Next-day raw return lets small, volatile stocks dominate the loss.
  - The vol-scaled residual z puts all stocks on one scale and removes beta, so it is the recommended training target.
  - Cross-sectional rank of next-day return is the target that matches a long-short or top-N portfolio. It is best used for evaluation (rank IC) or as a fine-tuning loss on a second stage, because a per-stock autoregressive language model cannot see the cross-section.
- **Regularization.**
  - At 1–10M parameters on 10–27M tokens: dropout 0.1–0.2 and weight decay about 0.1.
  - At most about 4 passes over the data.
  - Random window crops, and optionally random masking of volume features (Kronos's 5% trick).
  - Early stopping on a later time period, never a random split.
  - Expect validation cross-entropy to beat the histogram by only a tiny margin. A large improvement usually means leakage, for example a σ or β estimated with day-t data.
- **Leakage canaries** (as in correlation2's v2 plan): shift targets by +1 day, or permute dates, and confirm that rank IC and P&L collapse to chance.

### Gaps
- No paper was found that trains a StockGPT-style model on vol-scaled residual returns specifically, so the recipe's bin widths, context and learning rates are reasoned defaults.
- Not verified: how much of the 2024–2025 holdout period is inside the pre-training corpora of Chronos-2, TimesFM 2.5 and Kronos. The pre-training cutoffs were not established, which matters for zero-shot evaluation.
- No source quantifies the benefit of adding market or sector tokens to a per-stock GPT on US daily data. The available evidence is MASTER (other markets) and Chronos-2 multivariate studies (Mag-7 and Treasuries).

---

## 5. Starting code bases: Kronos, chronos-forecasting, TimesFM, nanoGPT/nanochat, HF Transformers time-series models, GluonTS, NeuralForecast, Qlib; licenses, maintenance in 2026, ease for a beginner

### Takeaway
For a *from-scratch next-token model*, start from **nanoGPT's ~300-line `model.py` and `train.py`** (MIT; deprecated but frozen and readable) or a stripped-down nanochat (MIT, uv-based, maintained). Feed it return-bin tokens instead of characters. That is the StockGPT recipe, and the easiest for a beginner to understand end to end. For *fine-tuning a financial foundation model*, **Kronos** (MIT code and weights) is the only finance-native option with published fine-tuning scripts. Its CSV pipeline is the easy entry. It needs Python 3.12 or relaxed pins, has had no commits since 2026-04-13, and its comments are partly AI-generated. **chronos-forecasting** (Apache-2.0, active, v2.3.2) is the best-maintained general option: Chronos-2 has a one-call `fit()` with LoRA. TimesFM 2.5 (Apache-2.0 weights) has a PEFT LoRA example. TimesFM 3.0 and Moirai 2.0 weights are **non-commercial**. Qlib (MIT, active, Python ≤3.12) is best as an evaluation and backtest scaffold, not as the model code.

### Cited Findings

#### Repository status (GitHub API, 2026-10-18)
| Repository | Code license | Stars | Last commit | Latest release | Source |
|---|---|---|---|---|---|
| shiyu-coder/Kronos | MIT | 41,130 | 2026-04-13 (merge "fix/batch-dimension-training") | none tagged | [GitHub](https://github.com/shiyu-coder/Kronos) |
| amazon-science/chronos-forecasting | Apache-2.0 | 6,136 | 2026-10-05 | v2.3.2 (2026-09-08) | [GitHub](https://github.com/amazon-science/chronos-forecasting) |
| google-research/timesfm | Apache-2.0 | 34,602 | 2026-09-29 | v3.0.0 (2026-08-28); PyPI 3.0.2 (2026-09-09) | [GitHub](https://github.com/google-research/timesfm) |
| karpathy/nanoGPT | MIT | 63,907 | 2025-11-12 ("mention nanochat and deprecation") | none | [GitHub](https://github.com/karpathy/nanoGPT) |
| karpathy/nanochat | MIT | 58,751 | 2026-07-03 (pushed 2026-09-07) | none | [GitHub](https://github.com/karpathy/nanochat) |
| KellerJordan/modded-nanogpt | MIT | 5,955 | 2026-09-28 | none | [GitHub](https://github.com/KellerJordan/modded-nanogpt) |
| huggingface/transformers | Apache-2.0 | 167,823 | 2026-10-17 | v5.20.1 (2026-10-16) | [GitHub](https://github.com/huggingface/transformers) |
| awslabs/gluonts | Apache-2.0 | 5,249 | 2026-10-13 | v0.17.0 (2026-07-31) | [GitHub](https://github.com/awslabs/gluonts) |
| Nixtla/neuralforecast | Apache-2.0 | 4,287 | 2026-10-14 | v3.3.0 (2026-10-07) | [GitHub](https://github.com/Nixtla/neuralforecast) |
| microsoft/qlib | MIT | 49,484 | 2026-10-08 | v0.9.8 (2026-10-08; PyPI still 0.9.7) | [GitHub](https://github.com/microsoft/qlib) |
| thuml/Time-Series-Library | MIT | 12,968 | 2026-04-18 | none | [GitHub](https://github.com/thuml/Time-Series-Library) |
| thuml/Large-Time-Series-Model (Timer) | MIT | 1,023 | 2026-03-22 | none | [GitHub](https://github.com/thuml/Large-Time-Series-Model) |
| Time-MoE/Time-MoE | Apache-2.0 | 1,006 | 2026-03-21 ("Support transformers == 4.57.0") | none | [GitHub](https://github.com/Time-MoE/Time-MoE) |
| SalesforceAIResearch/uni2ts (Moirai) | Apache-2.0 | 1,610 | 2026-06-02 | 2.0.0 (2025-11-04) | [GitHub](https://github.com/SalesforceAIResearch/uni2ts) |
| time-series-foundation-models/lag-llama | Apache-2.0 | 1,600 | 2025-06-06 | none | [GitHub](https://github.com/time-series-foundation-models/lag-llama) |
| yuqinie98/PatchTST | Apache-2.0 | 2,728 | 2023-08-11 | none | [GitHub](https://github.com/yuqinie98/PatchTST) |
| thuml/iTransformer | MIT | 2,232 | 2025-07-17 | none | [GitHub](https://github.com/thuml/iTransformer) |

#### Pretrained-weight licenses (Hugging Face model cards, checked 2026-10-18)
- Kronos weights (NeoQuasar/Kronos-mini, Kronos-small, Kronos-base, Kronos-Tokenizer-base): **MIT** (last modified 2025-09-09) — [HF Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small)
- Chronos weights: amazon/chronos-2, autogluon/chronos-2-small and amazon/chronos-bolt-small are **Apache-2.0** — [HF chronos-2](https://huggingface.co/amazon/chronos-2)
- TimesFM weights:
  - google/timesfm-2.5-200m-pytorch is **Apache-2.0** — [HF TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
  - google/timesfm-3.0-pytorch is **"other"**. The README says TimesFM 3.0 weights are under `timesfm-non-commercial-license-v1.0`, "restricted to non-commercial, non-production use", while code and weights up to 2.5 stay Apache-2.0 — [TimesFM README](https://github.com/google-research/timesfm)
- Other model weights:
  - Salesforce/moirai-2.0-R-small is **CC-BY-NC-4.0** — [HF Moirai 2.0 small](https://huggingface.co/Salesforce/moirai-2.0-R-small)
  - thuml/sundial-base-128m and Maple728/TimeMoE-50M are Apache-2.0 — [HF Sundial](https://huggingface.co/thuml/sundial-base-128m), [HF TimeMoE-50M](https://huggingface.co/Maple728/TimeMoE-50M)

#### Per code base
- **Kronos.**
  - Model zoo: Kronos-mini 4.1M (context 2048, Kronos-Tokenizer-2k), Kronos-small 24.7M (512), Kronos-base 102.3M (512). Kronos-large (499.2M) is not open-sourced.
  - Inference goes through `KronosPredictor` with T, top_p and `sample_count`, plus `predict_batch`.
  - Two fine-tuning pipelines:
    - (a) Qlib-based `finetune/` for the A-share demo: preprocess, `train_tokenizer.py`, `train_predictor.py` (written for `torchrun`), `qlib_test.py` backtest.
    - (b) `finetune_csv/` for any CSV with timestamps and OHLCV/amount columns, YAML config, `train_sequential.py` (tokenizer then predictor), with flags to skip either stage.
  - The README calls the pipeline "a demonstration ... not a production-ready quantitative trading system". It warns that "many of the code comments within the `finetune/` directory were generated by an AI assistant (Gemini 2.5 Pro) ... may contain inaccuracies".
  - Accepted at AAAI 2026.
  - Sources: [Kronos README](https://github.com/shiyu-coder/Kronos); [finetune_csv README](https://github.com/shiyu-coder/Kronos/blob/master/finetune_csv/README.md)
- **chronos-forecasting.**
  - Training scripts for original Chronos: `scripts/training/train.py` with configs for chronos-t5 tiny through large and `chronos-gpt2.yaml`.
  - `scripts/kernel-synth.py` generates synthetic data.
  - Chronos-2 fine-tunes through `Chronos2Pipeline.fit(... finetune_mode="full"|"lora" ...)`.
  - Install with `pip install chronos-forecasting`.
  - Sources: [repo tree](https://github.com/amazon-science/chronos-forecasting/tree/main/scripts/training); [pipeline.py](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py); [README](https://github.com/amazon-science/chronos-forecasting)
- **TimesFM.**
  - Fine-tuning: "Added fine-tuning example using HuggingFace Transformers + PEFT (LoRA)" (2026-04-09), in `timesfm-forecasting/examples/finetuning/`.
  - Install with `pip install timesfm[torch]`, or `uv venv && uv pip install -e .[torch]`.
  - TimesFM 3.0 uses a new `timesfm3` module.
  - Sources: [TimesFM README](https://github.com/google-research/timesfm)
- **nanoGPT and nanochat.**
  - nanoGPT: "`train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition". Since Nov 2025, "nanoGPT (this repo) is now very old and deprecated" in favour of nanochat — [nanoGPT README](https://github.com/karpathy/nanoGPT)
  - nanochat:
    - Uses uv (`uv sync --extra gpu`).
    - One `--depth` dial sets width, heads, learning rates and training horizon "so that the trained model comes out compute optimal".
    - Runs on a single GPU without `torchrun` (gradient accumulation, 8× longer).
    - Under 80 GB, reduce `--device-batch-size`.
    - Precision is controlled through a global `COMPUTE_DTYPE` instead of autocast.
    - Source: [nanochat README](https://github.com/karpathy/nanochat)
- **Hugging Face Transformers v5** ships model folders `autoformer`, `informer`, `patchtsmixer`, `patchtst`, `time_series_transformer`, `timesfm` and `timesfm2_5` — [transformers/src/transformers/models](https://github.com/huggingface/transformers/tree/main/src/transformers/models)
- **NeuralForecast** (Apache-2.0) covers NBEATS, NHITS, TFT, PatchTST, iTransformer, TimesNet, TSMixer, DeepAR, LSTM, MLP, Informer and TimeLLM among others. It depends on torch ≥2.9.1, pytorch-lightning ≥2.6.6, ray[train,tune] and optuna — [NeuralForecast README](https://github.com/Nixtla/neuralforecast); [PyPI neuralforecast 3.3.0](https://pypi.org/project/neuralforecast/3.3.0/)
- **GluonTS** 0.17.0 (Apache-2.0) requires `pandas<3` — [PyPI gluonts 0.17.0](https://pypi.org/project/gluonts/0.17.0/)
- **Qlib** (MIT):
  - Full pipeline from data through training to backtest (`qrun` with YAML workflows such as LightGBM on Alpha158).
  - Model zoo includes LightGBM, DoubleEnsemble, Transformer and Localformer, TRA and DDG-DA. Rolling retraining. Supported Python 3.8–3.12.
  - 2026-10-08 commit "fix(security): constrain config-driven code execution". The README asks users to opt in with `trusted=True` for certain artifact-loading paths.
  - Sources: [Qlib README](https://github.com/microsoft/qlib); [commit list](https://github.com/microsoft/qlib/commits/main)
- **Time-Series-Library** (thuml; MIT) collects many forecasting architectures in one framework. Last commit 2026-04-18 — [GitHub](https://github.com/thuml/Time-Series-Library)

### Inferences
- **Easiest path for a beginner, in order:**
  1. **nanoGPT's model.py and train.py, adapted to return tokens.** No framework to learn, and it is exactly the "next token" mental model. Replace the char dataset with a GPU-resident sampler over z-token arrays, change `vocab_size`, and add a validation-IC hook. nanoGPT is frozen, which suits a tutorial base. nanochat is the maintained successor, but it carries tokenizer, chat, SFT and eval machinery a return model does not need.
  2. **Kronos finetune_csv**, for a pretrained financial model on OHLCV. Use a separate Python 3.12 env, either keeping its pins or relaxing pandas and matplotlib. Write one CSV per symbol, or adapt `dataset.py` to read the panel directly. Set `use_comet=False`, and check its data-split code by reading it, given the AI-generated comments.
  3. **chronos-forecasting Chronos-2**, the strongest general baseline with the smallest amount of code (`predict_df` and `fit`), Apache-2.0 and actively maintained.
  4. **TimesFM 2.5**, a zero-shot baseline with LoRA available (avoid 3.0 weights if commercial use is possible).
  5. **Qlib or NeuralForecast** only if their workflow is wanted: Qlib for backtest reports, NeuralForecast for quick PatchTST and iTransformer baselines. Neither is required for the GPT itself.
- Maintenance risk:
  - Kronos: 6 months without commits and 282 open issues. Pin the commit.
  - Lag-Llama, the original PatchTST repo and iTransformer are dormant. Use the HF Transformers or NeuralForecast implementations of PatchTST instead.
  - Time-MoE pins `transformers == 4.57.0`, which likely conflicts with transformers 5.x.

### Gaps
- None of these code bases were installed or run on the RTX 5080 for these notes. Compatibility is judged from wheels, metadata and documentation only.
- Kronos's `finetune_csv` throughput and time-to-train on a single 16 GB GPU are not documented.
- The licences of the pre-training *data* behind each pretrained model, and their pre-training cutoff dates, were not established.

---

## 6. Experiment tracking and reproducibility tools that run locally (MLflow, W&B offline, TensorBoard, Aim, plus Trackio)

### Takeaway
Use **MLflow 3.17** (local, no account, browser UI) or **Trackio 0.42** (Hugging Face, MIT, local-first SQLite, a drop-in for the `wandb` API) for runs, metrics, configs and checkpoints. Add **TensorBoard** only for quick loss curves. **W&B** works offline but needs its cloud, or a W&B server, to view runs. **Avoid Aim**: its last release was 2025-05-08 and it has no wheels for Python 3.13 or 3.14. For reproducibility:
- Commit `uv.lock`.
- Log the git SHA, config, data snapshot hash and seed.
- Set PyTorch's determinism flags for checks.
- Accept that runs are not bit-identical across PyTorch versions or GPUs.

### Cited Findings
- **MLflow:**
  - Latest 3.17.1 (2026-10-13), pure-Python wheel — [PyPI mlflow](https://pypi.org/project/mlflow/#files)
  - The docs say "By default, MLflow Tracking logs data to the local `mlruns` directory". Start the UI with `mlflow server --port 5000`.
  - Logging: `mlflow.autolog()` (PyTorch supported), `mlflow.log_param`, `mlflow.log_metric(..., step=epoch)`, `mlflow.pytorch.log_model`.
  - Source: [MLflow Tracking docs](https://mlflow.org/docs/latest/ml/tracking/)
- **Weights & Biases:**
  - Latest SDK 0.31.0 (2026-10-16), classifiers include 3.14. The SDK is MIT — [PyPI wandb](https://pypi.org/project/wandb/#files); [GitHub wandb/wandb](https://github.com/wandb/wandb)
  - Offline use: set `WANDB_MODE="offline"`; runs are written under `wandb/`. Later run `wandb sync wandb/[DRYRUN-FOLDER-NAME]` "to sync your locally stored run data to the cloud" — [W&B support article (docs now served from docs.coreweave.com)](https://docs.coreweave.com/support/models/articles/is-it-possible-to-save-metrics-offline-a)
  - The docs.wandb.ai FAQ URL now 301-redirects to docs.coreweave.com — [docs.coreweave.com redirect target](https://docs.coreweave.com/guides/technical-faq/setup/)
- **TensorBoard:** 2.21.0 (2026-06-29), pure Python, Apache-2.0. Last commit 2026-10-08 ("Migrate TensorBoard builds to Bazel 8") — [PyPI tensorboard](https://pypi.org/project/tensorboard/#files); [GitHub tensorboard](https://github.com/tensorflow/tensorboard)
- **Aim:** last release v3.29.1 (2025-05-08), last commit 2025-12-31. PyPI wheels only for cp37–cp312, and its `aimrocks` dependency also stops at cp312 — [GitHub aimhubio/aim](https://github.com/aimhubio/aim); [PyPI aim 3.29.1](https://pypi.org/project/aim/3.29.1/#files); [PyPI aimrocks](https://pypi.org/project/aimrocks/#files)
- **Trackio:**
  - 0.42.0 (2026-10-09), MIT, "a lightweight, free experiment tracking library built by Hugging Face".
  - "Local-first, because you shouldn't need to make an account". "Logs are stored in SQLite database (with support for 'freezing' logs to Parquet)".
  - API-compatible with `wandb.init`, `wandb.log` and `wandb.finish` ("`import trackio as wandb`"). The dashboard runs locally, and `trackio query project --project <name> --sql "SELECT ..."` queries runs from the command line.
  - Sources: [Trackio README](https://github.com/gradio-app/trackio); [PyPI trackio](https://pypi.org/project/trackio/#files)
- **Kronos** logs to Comet ML by default (`self.use_comet = True`). Set it to False to stay local — [Kronos finetune/config.py](https://github.com/shiyu-coder/Kronos/blob/master/finetune/config.py)
- **PyTorch reproducibility notes (2.14 docs):**
  - "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms."
  - Use `torch.manual_seed()`; `torch.use_deterministic_algorithms(True)` (it errors on nondeterministic ops); `torch.backends.cudnn.benchmark = False`.
  - For DataLoader workers, pass a `worker_init_fn` that seeds NumPy and `random` from `torch.initial_seed()`, and a seeded `torch.Generator`.
  - Source: [PyTorch docs: Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
- **Environment locking:** nanochat ships `uv.lock` and installs with `uv sync` — [nanochat README](https://github.com/karpathy/nanochat)

### Inferences
- A minimal local setup:
  - `uv add mlflow` (or `trackio`).
  - One `mlflow.start_run()` per training run, logging the full config dict, git SHA, `uv.lock` hash, data snapshot ID (for example a hash of the exported Parquet), and the evaluation metrics: validation cross-entropy minus the histogram's, rank IC, and net P&L at 5/10/20 bps.
  - Save the best checkpoint as an artifact.
  - Trackio fits if the user wants SQL over runs. Since Claude-style agents may analyze the experiments, its CLI and SQL interface is convenient.
- Keep a ledger of every configuration tried, as correlation2 did with FDR-adjusted p-values. With many runs on one holdout, the best run's holdout score is biased upward, so pick the run on validation and test it once.
- Determinism mode slows training and some SDPA paths may error. Use it to confirm reproducibility on short runs, not on every run.

### Gaps
- Not verified: whether current MLflow 3.x releases default to `./mlruns` files or to a SQLite `mlflow.db` (the docs page says `mlruns`). Also not verified: W&B's current terms for a self-hosted local server.
- The `CUBLAS_WORKSPACE_CONFIG` setting (needed for deterministic cuBLAS) was not on the fetched docs page, so its current recommended value is not cited here.
