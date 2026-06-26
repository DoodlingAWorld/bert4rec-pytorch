"""Contract for the Cloze masking pipeline and the eval input builder."""

import numpy as np
import torch

from bert4rec.masking import ClozeMaskingDataset, build_eval_input


def _ds(mask_prob=0.5, max_len=8, num_items=20, seed=0):
    user_train = {
        u: list(range(1, 6)) for u in range(1, 4)
    }  # each user: items [1,2,3,4,5]
    return (
        ClozeMaskingDataset(user_train, num_items, max_len, mask_prob, seed=seed),
        num_items,
        max_len,
    )


def test_shapes_and_dtypes():
    ds, _num_items, max_len = _ds()
    tokens, labels = ds[0]
    assert tuple(tokens.shape) == (max_len,)
    assert tuple(labels.shape) == (max_len,)
    assert tokens.dtype == torch.int64 and labels.dtype == torch.int64


def test_left_padding_untouched():
    ds, _num_items, max_len = _ds()
    tokens, labels = ds[0]
    n_real = 5
    pad = max_len - n_real
    assert torch.equal(tokens[:pad], torch.zeros(pad, dtype=torch.int64))
    assert torch.equal(labels[:pad], torch.zeros(pad, dtype=torch.int64))


def test_masked_positions_consistent():
    ds, num_items, _max_len = _ds(seed=1)
    tokens, labels = ds[0]
    mask_id = num_items + 1
    masked = labels != 0
    # at masked positions the token is the [mask] id and the label is a real item
    assert torch.all(tokens[masked] == mask_id)
    assert torch.all((labels[masked] >= 1) & (labels[masked] <= num_items))
    # at least one masked position
    assert masked.sum() >= 1


def test_full_masking():
    ds, num_items, max_len = _ds(mask_prob=1.0, seed=2)
    tokens, labels = ds[0]
    real = torch.arange(max_len) >= (max_len - 5)
    assert torch.all(tokens[real] == num_items + 1)  # every real position masked
    assert torch.equal(labels[real], torch.tensor([1, 2, 3, 4, 5]))


def test_build_eval_input_trailing_mask():
    out = build_eval_input([1, 2, 3], max_len=6, mask_id=99)
    assert out.shape == (6,)
    assert out[-1] == 99  # trailing [mask]
    assert list(out[:-1]) == [0, 0, 1, 2, 3]  # left-padded history before it


def test_build_eval_input_truncates_to_recent():
    out = build_eval_input([1, 2, 3, 4, 5, 6], max_len=4, mask_id=99)
    # keep most recent (max_len - 1) = 3 history items, then the mask
    assert list(out) == [4, 5, 6, 99]
