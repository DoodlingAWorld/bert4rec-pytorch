"""The Cloze (masked-item) training pipeline and the eval-time input builder.

Training (Cloze task, Section 3.6): for each user sequence, randomly replace a fraction
``mask_prob`` of the real items with the special ``[mask]`` token and ask the model to
predict the originals from both-side context. Sequences are left-padded to ``max_len``
(pad id 0). The label tensor holds the original id at masked positions and 0 everywhere
else (0 is the ignore index used by the Cloze loss).

Inference (Section 3.6, "Test"): the Cloze objective predicts masked positions, but the
recommendation task predicts the *next* item. To bridge this, append a single ``[mask]``
token to the end of the user's history and read the prediction off that final position.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from .data import left_pad_sequence


class ClozeMaskingDataset(Dataset):
    """Yields (tokens, labels) for the Cloze task, one user per item.

    For a user's training items, after left-padding to ``max_len``:
      * each REAL position is masked independently with probability ``mask_prob`` (replaced
        with ``mask_id``); its label is set to the original item id.
      * non-masked real positions and all pad positions have label 0.
      * at least one real position is masked (so every sample contributes to the loss).

    ``__getitem__`` returns two int64 tensors of shape [max_len]: ``tokens`` and ``labels``.
    Use a fresh mask each epoch (sample in ``__getitem__``, not once up front).

    Left-padding convention: for a user with ``n`` real items, the first ``max_len - n``
    positions are 0 (padding) and the last ``n`` positions hold the items (truncate to the
    most recent ``max_len`` if ``n > max_len``). Only those last ``n`` positions are eligible
    for masking. ``mask_id`` is ``num_items + 1``.
    """

    def __init__(
        self,
        user_train: dict[int, list[int]],
        num_items: int,
        max_len: int,
        mask_prob: float,
        seed: int = 0,
    ):
        self.users = [u for u, s in user_train.items() if len(s) >= 1]
        self.user_train = user_train
        self.num_items = num_items
        self.mask_id = num_items + 1
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        ts = self.user_train[self.users[idx]][-self.max_len:]
        n = len(ts)
        start = self.max_len - n # real items occupy the LAST n positions
        tokens = np.zeros(self.max_len, dtype=np.int64)
        labels = np.zeros(self.max_len, dtype=np.int64)

        # Walk the real positions. For each, draw self.rng.random() < self.mask_prob:
        # - masked -> tokens[pos] = self.mask_id, labels[pos] = original_item
        # - not masked -> tokens[pos] = original_item, labels[pos] = 0 (already 0)

        for j, item in enumerate(ts):
            pos = start + j
            if self.rng.random() < self.mask_prob:
                tokens[pos] = self.mask_id
                labels[pos] = item
            else:
                tokens[pos] = item

        # Guarantee at least one mask: if nothing got masked (labels.any() is False), force-mask one random real position
        # (This keeps every sample contributing to the loss, and is what test_masked_positions_consistent checks.)
        if not labels.any():
            j = int(self.rng.integers(0, n))
            pos = start + j
            tokens[pos] = self.mask_id
            labels[pos] = ts[j]

        return torch.from_numpy(tokens), torch.from_numpy(labels)



def build_eval_input(history: list[int], max_len: int, mask_id: int) -> np.ndarray:
    """Build the inference input: the user's recent history followed by a trailing ``[mask]``.

    Returns an int64 array of shape [max_len], left-padded with 0, whose LAST position is
    ``mask_id`` (the position whose prediction is the recommendation). Keep the most recent
    ``max_len - 1`` history items (drop the oldest if truncation is needed), then left-pad
    with 0 if fewer than ``max_len - 1`` remain. Example: history [1,2,3,4,5,6], max_len 4
    -> [4, 5, 6, mask_id].
    """
    return left_pad_sequence(history[-(max_len - 1):] + [mask_id], max_len)
