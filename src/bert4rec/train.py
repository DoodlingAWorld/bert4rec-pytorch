"""Training orchestration for BERT4Rec (Cloze objective + popularity-sampled evaluation).

``load_dataset`` and ``set_seed`` are provided. ``train`` is the loop you implement: it wires
the Cloze dataset, the bidirectional model, and the masked-LM loss together, evaluates with
the popularity-sampled protocol, and returns the best metrics plus the trained model.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from .config import BERT4RecConfig
from .data import (
    build_user_sequences,
    download_ml1m,
    item_popularity,
    leave_one_out_split,
    parse_ml1m,
)
from .eval import evaluate
from .losses import cloze_loss
from .masking import ClozeMaskingDataset, build_eval_input
from .model import BERT4Rec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(cfg: BERT4RecConfig, raw_dir: str = "data/raw"):
    """Download + preprocess + split. Returns (user_train, user_valid, user_test, num_items)."""
    ratings_path = download_ml1m(raw_dir)
    rows = parse_ml1m(ratings_path)
    user_seqs, _num_users, num_items = build_user_sequences(rows, cfg.min_count)
    user_train, user_valid, user_test = leave_one_out_split(user_seqs)
    return user_train, user_valid, user_test, num_items


def train(
    cfg: BERT4RecConfig, data=None, raw_dir: str = "data/raw", verbose: bool = True
) -> dict:
    """Train BERT4Rec and return results.

    Parameters
    ----------
    cfg : hyperparameters.
    data : optional ``(user_train, user_valid, user_test, num_items)`` to skip the download
        (used by tests). When None, calls ``load_dataset(cfg, raw_dir)``.

    Returns a dict with at least:
        {"model": BERT4Rec, "best_val_ndcg": float, "test_metrics": dict, "num_items": int}

    What to implement
    -----------------
    1. ``set_seed(cfg.seed)``; get the splits (from ``data`` or ``load_dataset``).
    2. Build ``ClozeMaskingDataset`` + a DataLoader (shuffle, cfg.batch_size).
    3. Build ``BERT4Rec(num_items, cfg)`` and an Adam optimizer with
       ``betas=(cfg.beta1, cfg.beta2)`` and ``weight_decay=cfg.weight_decay``; optionally a
       linear LR decay schedule.
    4. Each epoch: for (tokens, labels) batches, ``logits = model(tokens)``,
       ``loss = cloze_loss(logits, labels)``, backward, clip grads to ``cfg.grad_clip``
       (``torch.nn.utils.clip_grad_norm_``), optimizer step.
    5. Every ``cfg.eval_every`` epochs, evaluate on validation with the popularity protocol
       using a ``rank_fn`` that, for each batch of histories, builds inputs with
       ``build_eval_input(h, cfg.max_len, model.mask_id)``, runs the model, takes the logits
       at the trailing ``[mask]`` (last position), and gathers the candidate columns. Keep the
       best checkpoint by val NDCG@10; early-stop after ``cfg.patience`` non-improving evals.
    6. Restore the best checkpoint and evaluate on test (input = train + validation item via
       ``extra_context=user_valid``). Return the dict above.

    Use ``item_popularity(user_train, num_items)`` for the evaluation's negative sampling.
    """
    raise NotImplementedError("Implement the BERT4Rec training loop (see docstring)")
