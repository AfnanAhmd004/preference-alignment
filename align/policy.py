"""Policy utilities: sampling and per-sequence log-probabilities under a TinyGPT."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import GPTConfig, TinyGPT
from .task import BOS, RESP_LEN, VOCAB


def new_policy(seed: int = 0) -> TinyGPT:
    torch.manual_seed(seed)
    return TinyGPT(GPTConfig(vocab_size=len(VOCAB), block_size=16, n_layer=2, n_head=4, n_embd=64))


def sequence_logprob(model: TinyGPT, seqs: torch.Tensor) -> torch.Tensor:
    """Sum of log p(response tokens | prompt) for sequences [bos, prompt, r1..r8]."""
    logits, _ = model(seqs[:, :-1])
    resp_logits = logits[:, 1:, :10]  # response positions; the action space is the 10 digits
    logp = F.log_softmax(resp_logits, -1).gather(-1, seqs[:, 2:, None]).squeeze(-1)
    return logp.sum(-1)


@torch.no_grad()
def sample(model: TinyGPT, prompts: list[int], temperature: float = 1.0) -> torch.Tensor:
    seqs = torch.tensor([[BOS, p] for p in prompts])
    for _ in range(RESP_LEN):
        logits, _ = model(seqs)
        logits = logits[:, -1, :10] / temperature  # restrict to digit tokens
        seqs = torch.cat([seqs, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return seqs


def sft_uniform(model: TinyGPT, steps: int = 300, batch: int = 128, seed: int = 0) -> TinyGPT:
    """Supervised 'pretraining' on uniformly random digit strings: the unaligned reference policy."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(steps):
        prompts = torch.randint(10, 12, (batch, 1), generator=g)
        resp = torch.randint(0, 10, (batch, RESP_LEN), generator=g)
        seqs = torch.cat([torch.full((batch, 1), BOS), prompts, resp], 1)
        loss = -sequence_logprob(model, seqs).mean() / RESP_LEN
        opt.zero_grad(); loss.backward(); opt.step()
    return model
