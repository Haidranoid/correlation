import pytest
import torch

from stockai.gpt.model import GPTConfig, StockGPT
from stockai.gpt.tokens import CENTERS, MISSING, VOCAB

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def inputs(b, t, gen):
    tok = torch.randint(0, VOCAB, (b, t), generator=gen)
    tok[:, ::7] = MISSING
    cont = torch.randn(b, t, 3, generator=gen)
    cont[..., 2] = (cont[..., 2] > 1).float()
    return tok, cont


def test_default_size():
    m = StockGPT(GPTConfig())
    assert 0.7e6 < m.n_params() < 1.0e6  # ~0.8M (PLAN.md §3.2)
    assert m.cfg.ctx == 256 and m.cfg.n_layer == 4 and m.cfg.n_head == 4 and m.cfg.n_embd == 128


@pytest.mark.parametrize("device", DEVICES)
def test_causal(device):
    """Changing inputs after position p never changes the outputs at positions <= p."""
    gen = torch.Generator().manual_seed(0)
    m = StockGPT(GPTConfig(n_layer=3, n_head=4, n_embd=64, ctx=64)).to(device).eval()
    tok, cont = inputs(3, 64, gen)
    with torch.no_grad():
        base = m(tok.to(device), cont.to(device)).cpu()
        for p in (0, 5, 31, 62):
            t2, c2 = tok.clone(), cont.clone()
            t2[:, p + 1:] = torch.randint(0, VOCAB, t2[:, p + 1:].shape, generator=gen)
            c2[:, p + 1:] = torch.randn(c2[:, p + 1:].shape, generator=gen) * 3
            out = m(t2.to(device), c2.to(device)).cpu()
            torch.testing.assert_close(out[:, :p + 1], base[:, :p + 1], rtol=0, atol=1e-5)
            assert (out[:, p + 1:] - base[:, p + 1:]).abs().max() > 1e-4  # later positions do see the change


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_causal_bf16():
    """The same under the bf16 autocast used for training and scoring."""
    gen = torch.Generator().manual_seed(1)
    m = StockGPT(GPTConfig()).cuda().eval()
    tok, cont = inputs(4, 256, gen)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        base = m(tok.cuda(), cont.cuda()).float()
        p = 200
        t2, c2 = tok.clone(), cont.clone()
        t2[:, p + 1:] = MISSING
        c2[:, p + 1:] = 9.0
        out = m(t2.cuda(), c2.cuda()).float()
    torch.testing.assert_close(out[:, :p + 1], base[:, :p + 1], rtol=0, atol=1e-2)


@pytest.mark.parametrize("device", DEVICES)
def test_last_only_and_expected_value(device):
    """Scoring's last-position shortcut gives the full forward's last output."""
    gen = torch.Generator().manual_seed(2)
    m = StockGPT(GPTConfig(n_layer=2, n_head=2, n_embd=32, ctx=40)).to(device).eval()
    tok, cont = inputs(2, 40, gen)
    with torch.no_grad():
        full = m(tok.to(device), cont.to(device))
        last = m(tok.to(device), cont.to(device), last_only=True)
    torch.testing.assert_close(last, full[:, -1], rtol=1e-5, atol=1e-5)
    if device == "cuda":  # the scoring shapes, in fp32 and under bf16 autocast
        m = StockGPT(GPTConfig()).cuda().eval()
        for p in m.parameters():
            p.data.normal_(0, 0.1, generator=None)
        tok, cont = (x.cuda() for x in inputs(64, 256, gen))
        with torch.no_grad():
            full, last = m(tok, cont), m(tok, cont, last_only=True)
            torch.testing.assert_close(last, full[:, -1], rtol=1e-4, atol=1e-4)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                full, last = m(tok, cont).float(), m(tok, cont, last_only=True).float()
        assert (last - full[:, -1]).norm() / full[:, -1].norm() < 1e-2
    onehot = torch.full((3, 101), -1e4, device=device)
    onehot[0, 0] = onehot[1, 50] = onehot[2, 100] = 0.0
    torch.testing.assert_close(m.expected_value(onehot).cpu(), torch.tensor(CENTERS[[0, 50, 100]]))


def test_context_limit():
    m = StockGPT(GPTConfig(n_layer=1, n_head=1, n_embd=16, ctx=8))
    with pytest.raises(ValueError):
        m(torch.zeros(1, 9, dtype=torch.long), torch.zeros(1, 9, 3))
