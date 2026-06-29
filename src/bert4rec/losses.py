"""The Cloze objective: negative log-likelihood over masked positions only (Eq. 8)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cloze_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked positions.

    Parameters
    ----------
    logits : [B, L, V] scores over the full id space (V = num_items + 2) at every position.
    labels : [B, L] int64 targets. Masked positions hold the original item id; every other
        position (non-masked real items AND padding) is 0 and must be IGNORED.

    Returns
    -------
    A scalar: the mean cross-entropy over the masked positions (use ``ignore_index=0`` so
    that id 0 is never a target and non-masked positions do not contribute).
    """
    V = logits.size(-1)
    return F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index = 0)
