"""DPO loss and sequence log-probs (torch). Kept dependency-light so it can be unit-tested on CPU."""
from __future__ import annotations


def seq_logp(logits, labels, mask):
    """Sum of token log-probs over completion tokens. logits: (B,T,V) aligned so logits[:, i] predicts labels[:, i]."""
    import torch
    lp = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1)


def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta: float = 0.1):
    import torch.nn.functional as F
    z = beta * ((pol_c - ref_c) - (pol_r - ref_r))
    return -F.logsigmoid(z).mean(), (z > 0).float().mean()
