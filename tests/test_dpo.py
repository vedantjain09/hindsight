import math

import pytest

torch = pytest.importorskip("torch")
from hindsight.train_utils import dpo_loss, seq_logp


def test_dpo_loss_is_ln2_at_reference_and_decreases_when_policy_prefers_chosen():
    z = torch.zeros(4)
    loss, acc = dpo_loss(z, z, z, z, 0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6
    better, acc2 = dpo_loss(z + 5, z - 5, z, z, 0.1)
    assert better.item() < loss.item() and acc2.item() == 1.0
    worse, _ = dpo_loss(z - 5, z + 5, z, z, 0.1)
    assert worse.item() > loss.item()


def test_dpo_matches_numpy_reference():
    import numpy as np
    pc, pr, rc, rr = torch.tensor([1.2]), torch.tensor([-0.4]), torch.tensor([0.3]), torch.tensor([0.1])
    z = 0.2 * ((1.2 - 0.3) - (-0.4 - 0.1))
    want = -np.log(1 / (1 + np.exp(-z)))
    assert abs(dpo_loss(pc, pr, rc, rr, 0.2)[0].item() - want) < 1e-6


def test_seq_logp_masks_prompt_tokens():
    logits = torch.zeros(1, 3, 4)                       # uniform over 4 tokens
    labels = torch.tensor([[1, 2, 3]])
    full = seq_logp(logits, labels, torch.ones(1, 3)).item()
    part = seq_logp(logits, labels, torch.tensor([[0.0, 0.0, 1.0]])).item()
    assert abs(full - 3 * math.log(0.25)) < 1e-5 and abs(part - math.log(0.25)) < 1e-5
