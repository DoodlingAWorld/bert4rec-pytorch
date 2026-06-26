"""Contract for the Cloze loss."""

import torch
import torch.nn.functional as F

from bert4rec.losses import cloze_loss


def _setup(B=2, L=4, V=7, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, L, V, requires_grad=True)
    labels = torch.zeros(B, L, dtype=torch.int64)
    return logits, labels


def test_returns_scalar_and_grad_flows():
    logits, labels = _setup()
    labels[0, 1] = 3
    labels[1, 2] = 5
    loss = cloze_loss(logits, labels)
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_only_masked_positions_contribute():
    """ignore_index=0 + mean over masked positions only."""
    torch.manual_seed(1)
    logits = torch.randn(1, 3, 7)
    labels = torch.tensor([[0, 4, 0]])  # only position 1 is masked (target id 4)
    loss = cloze_loss(logits, labels)
    manual = F.cross_entropy(logits[0, 1].unsqueeze(0), torch.tensor([4]))
    assert torch.allclose(loss, manual, atol=1e-5)
