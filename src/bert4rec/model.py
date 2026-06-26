"""BERT4Rec: a bidirectional Transformer encoder for sequential recommendation.

This is the architecture from Sun et al. (CIKM 2019), Section 3. It differs from SASRec in
four ways, all of which matter:

  * Bidirectional attention: every position attends to every non-pad position (left AND
    right). There is NO causal mask; the only mask is for padding.
  * Multi-head attention: ``num_heads`` heads, each of size ``hidden_dim / num_heads``,
    scaled by 1/sqrt(head_dim) (Eq. 1-2).
  * Post-LN Transformer block (Eq. 5-6):
        A = LayerNorm(x + Dropout(MultiHeadSelfAttn(x)))
        out = LayerNorm(A + Dropout(PFFN(A)))
  * Position-wise FFN with GELU and a 4x inner expansion: d -> 4d -> d (Eq. 3).

Vocabulary / ids: 0 is padding, 1..num_items are real items, and ``num_items + 1`` is the
special ``[mask]`` token. So the embedding table and output layer span ``num_items + 2`` ids.

Output layer (Eq. 7): a two-layer head that scores every item id at every position:
        logits = GELU(H @ W_P + b_P) @ E^T + b_O
where E is the (shared) item embedding table and b_O is a per-item output bias. ``forward``
returns logits at every position; the Cloze loss selects the masked positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import BERT4RecConfig


class BERT4Rec(nn.Module):
    """Bidirectional Transformer encoder with a tied-embedding output head.

    Required attributes (the tests and the rest of the repo rely on these):
      * ``self.item_emb``: nn.Embedding(num_items + 2, hidden_dim, padding_idx=0) — the
        shared item embedding table (also used to score outputs).
      * ``self.mask_id``: int == num_items + 1.

    Required method:
      * ``forward(seq) -> logits``: ``seq`` is an int64 tensor [B, L] of item ids (with
        ``mask_id`` at positions to predict and 0 for padding). Returns logits
        [B, L, num_items + 2] over the full id space. Attention is bidirectional with a
        padding mask (positions where ``seq == 0`` may not be attended to, and pad rows may
        be left unconstrained since the loss ignores them).
    """

    def __init__(self, num_items: int, cfg: BERT4RecConfig):
        super().__init__()
        self.num_items = num_items
        self.mask_id = num_items + 1
        self.vocab_size = num_items + 2
        self.cfg = cfg
        # IMPLEMENT: embeddings (item + positional), dropout, L post-LN Transformer blocks
        # (multi-head bidirectional attention + GELU 4x FFN), and the two-layer output head.
        raise NotImplementedError(
            "Implement the BERT4Rec encoder (see module docstring)"
        )

    def forward(
        self, seq: torch.Tensor
    ) -> torch.Tensor:  # [B, L] -> [B, L, num_items + 2]
        raise NotImplementedError("Implement BERT4Rec.forward (see module docstring)")
