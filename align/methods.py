"""Reward modelling, DPO and KL-regularised policy-gradient RLHF."""
from __future__ import annotations

import copy
import random

import torch
from torch import nn
from torch.nn import functional as F

from .model import GPTConfig, TinyGPT
from .policy import sample, sequence_logprob
from .task import VOCAB, random_prompt, to_tensor, true_reward


# ------------------------------------------------------------------------------ reward model
class RewardModel(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.body = TinyGPT(GPTConfig(vocab_size=len(VOCAB), block_size=16, n_layer=2, n_head=4, n_embd=64))
        self.body.head = nn.Identity()
        self.value = nn.Linear(64, 1)

    def forward(self, seqs):
        h, _ = self.body(seqs)
        return self.value(h[:, -1]).squeeze(-1)


def train_reward_model(pairs, epochs: int = 4, batch: int = 128, lr: float = 1e-3, seed: int = 0) -> RewardModel:
    """Bradley–Terry loss: -log sigmoid(r(chosen) - r(rejected))."""
    rng = random.Random(seed)
    rm = RewardModel(seed)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr)
    for _ in range(epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), batch):
            p, c, r = zip(*pairs[i : i + batch])
            loss = -F.logsigmoid(rm(to_tensor(p, c)) - rm(to_tensor(p, r))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return rm.eval()


@torch.no_grad()
def pairwise_accuracy(rm: RewardModel, pairs) -> float:
    p, c, r = zip(*pairs)
    return float((rm(to_tensor(p, c)) > rm(to_tensor(p, r))).float().mean())


# ------------------------------------------------------------------------------ DPO
def dpo(policy: TinyGPT, pairs, beta: float = 0.1, epochs: int = 3, batch: int = 64, lr: float = 5e-4, seed: int = 0) -> TinyGPT:
    """Direct Preference Optimisation: no reward model, no sampling, just a classification-style loss
    on the log-ratio of policy to frozen reference for chosen vs rejected responses."""
    ref = copy.deepcopy(policy).eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), batch):
            p, c, r = zip(*pairs[i : i + batch])
            yc, yr = to_tensor(p, c), to_tensor(p, r)
            margin = (sequence_logprob(policy, yc) - sequence_logprob(ref, yc)) - (sequence_logprob(policy, yr) - sequence_logprob(ref, yr))
            loss = -F.logsigmoid(beta * margin).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return policy


# ------------------------------------------------------------------------------ RLHF (policy gradient)
def rlhf(policy: TinyGPT, rm: RewardModel, steps: int = 150, batch: int = 64, k: int = 4, kl_coef: float = 0.05,
         lr: float = 5e-4, seed: int = 0, log_every: int = 25):
    """KL-regularised REINFORCE with a leave-one-out baseline (RLOO), a simpler cousin of PPO.

    Each prompt gets k samples; advantage = reward minus the mean reward of the other k-1 samples.
    The reward is the reward model's score minus kl_coef * log(pi/pi_ref).
    """
    ref = copy.deepcopy(policy).eval()
    rng = random.Random(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    log = []
    for step in range(steps):
        prompts = [random_prompt(rng) for _ in range(batch // k) for _ in range(k)]
        seqs = sample(policy, prompts)
        logp = sequence_logprob(policy, seqs)
        with torch.no_grad():
            kl = logp.detach() - sequence_logprob(ref, seqs)
            r_rm = rm(seqs)
            reward = (r_rm - kl_coef * kl).view(-1, k)
            adv = (reward - (reward.sum(1, keepdim=True) - reward) / (k - 1)).view(-1)
        loss = -(adv * logp).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0 or step == steps - 1:
            tr = sum(true_reward(p, s[2:].tolist()) for p, s in zip(prompts, seqs)) / len(prompts)
            log.append({"step": step, "rm_reward": float(r_rm.mean()), "true_reward": tr, "kl": float(kl.mean())})
    return policy, log


@torch.no_grad()
def evaluate(policy: TinyGPT, n: int = 1024, seed: int = 123) -> dict[str, float]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    prompts = [random_prompt(rng) for _ in range(n)]
    seqs = sample(policy, prompts)
    resp = [s[2:].tolist() for s in seqs]
    from .task import STOI

    by = {E_: [true_reward(p, r) for p, r in zip(prompts, resp) if p == E_] for E_ in (STOI["E"], STOI["O"])}
    return {"true_reward": sum(true_reward(p, r) for p, r in zip(prompts, resp)) / n,
            "reward_E": sum(by[STOI["E"]]) / max(1, len(by[STOI["E"]])),
            "reward_O": sum(by[STOI["O"]]) / max(1, len(by[STOI["O"]])),
            "violations": sum(STOI["7"] in r for r in resp) / n}
