"""stockGPT: a small nanoGPT-style decoder over census-variable tokens (PLAN.md §3.2).

Each day's input is the token embedding plus a learned position embedding plus a linear embedding of
continuous context known at that day's close (the market move, the stock's standardized log trailing
volatility and a missing flag). Pre-LayerNorm blocks, causal self-attention through
`scaled_dot_product_attention(is_causal=True)`, GELU MLP. The head predicts the bin of each position's target
(101 classes; the target is never MISSING): the next day's token at horizon 1, the binned rank of the H-day return
above it (features.py).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokens import CENTERS, N_BINS, VOCAB

N_CONT = 3  # market move, standardized log volatility, volatility-missing flag


@dataclass
class GPTConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    ctx: int = 256
    dropout: float = 0.1
    vocab: int = VOCAB
    n_out: int = N_BINS
    n_cont: int = N_CONT
    bias: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError(f"width {cfg.n_embd} is not a multiple of heads {cfg.n_head}")
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        h = self.n_head
        q = q.view(b, t, h, c // h).transpose(1, 2)
        k = k.view(b, t, h, c // h).transpose(1, 2)
        v = v.view(b, t, h, c // h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.c_proj(y))

    def forward_last(self, x: torch.Tensor) -> torch.Tensor:
        """The output at the last position only, [B, 1, C]: the last position attends to every position, which is
        what the causal mask allows it, so no mask is needed (eval only)."""
        b, t, c = x.shape
        h = self.n_head
        w, bias = self.c_attn.weight, self.c_attn.bias
        q = F.linear(x[:, -1:], w[:c], None if bias is None else bias[:c])
        k, v = F.linear(x, w[c:], None if bias is None else bias[c:]).split(c, dim=2)
        q = q.view(b, 1, h, c // h).transpose(1, 2)
        k = k.view(b, t, h, c // h).transpose(1, 2)
        v = v.view(b, t, h, c // h).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.c_proj(y.transpose(1, 2).reshape(b, 1, c))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))

    def forward_last(self, x: torch.Tensor) -> torch.Tensor:
        """forward(x)[:, -1:], computing the attention output and MLP for the last position only."""
        x = x[:, -1:] + self.attn.forward_last(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class StockGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.ctx, cfg.n_embd)
        self.wce = nn.Linear(cfg.n_cont, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.n_out, bias=False)
        self.register_buffer("centers", torch.tensor(CENTERS), persistent=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():  # GPT-2 scaled init of the residual projections
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor, cont: torch.Tensor, last_only: bool = False) -> torch.Tensor:
        """Logits [B, L, 101] (or [B, 101] with `last_only`) for the bin of each position's target (at horizon 1, the
        next day's token).

        tokens: int [B, L] in 0..101; cont: float [B, L, n_cont]. Position i only sees positions <= i.
        """
        b, t = tokens.shape
        if t > self.cfg.ctx:
            raise ValueError(f"sequence length {t} exceeds context {self.cfg.ctx}")
        pos = torch.arange(t, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)[None] + self.wce(cont)
        x = self.drop(x)
        if last_only and not self.training:  # scoring: the last block only needs the last position's output
            for block in self.blocks[:-1]:
                x = block(x)
            x = self.blocks[-1].forward_last(x)[:, 0]
        else:
            for block in self.blocks:
                x = block(x)
            if last_only:
                x = x[:, -1]
        return self.head(self.ln_f(x))

    def expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected target under the predicted bin probabilities (the score; at horizon 1 the next day's v)."""
        return torch.softmax(logits.float(), dim=-1) @ self.centers

    def param_groups(self, weight_decay: float) -> list[dict]:
        """nanoGPT's split: weight decay on matrices and embeddings, none on biases and norms."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
