"""Leave-one-out evaluation with popularity-sampled negatives (BERT4Rec paper, Section 4.2).

For each user the held-out item is ranked against 100 negatives sampled by item popularity
(not uniform), excluding items in the user's history. We report HR@{1,5,10}, NDCG@{5,10},
and MRR. The scoring is model-agnostic: callers pass a ``rank_fn`` that maps a batch of
histories + candidate ids to scores, so the same evaluation serves both BERT4Rec and SASRec.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def hit_at_k(rank: int, k: int) -> float:
    return 1.0 if rank < k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    return (1.0 / math.log2(rank + 2)) if rank < k else 0.0


def mrr(rank: int) -> float:
    return 1.0 / (rank + 1)


def _sample_pop_negatives(
    exclude: set[int],
    num_items: int,
    n: int,
    prob: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    """Sample n distinct item ids in [1, num_items] with P(item) proportional to popularity,
    excluding ids in ``exclude``. ``prob`` is a normalized distribution over ids 0..num_items.
    """
    available = num_items - len([i for i in exclude if 1 <= i <= num_items])
    if n > available:
        raise ValueError(
            f"cannot sample {n} negatives: only {available} items available after exclusions"
        )
    negs: list[int] = []
    seen = set(exclude)
    # Oversample in blocks and filter; popularity is skewed so a few rounds suffice. Cap the
    # rounds so a pathologically skewed distribution fails loudly instead of hanging.
    max_rounds = 1000
    for _ in range(max_rounds):
        draws = rng.choice(len(prob), size=max(n * 2, 16), p=prob)
        for c in draws:
            c = int(c)
            if c != 0 and c not in seen:
                seen.add(c)
                negs.append(c)
                if len(negs) == n:
                    return negs
    raise RuntimeError(
        f"could not sample {n} negatives in {max_rounds} rounds (popularity too skewed?)"
    )


def evaluate(
    rank_fn: Callable[[list[list[int]], np.ndarray], np.ndarray],
    user_train: dict[int, list[int]],
    user_target: dict[int, list[int]],
    num_items: int,
    popularity: np.ndarray,
    *,
    extra_context: dict[int, list[int]] | None = None,
    num_neg: int = 100,
    topks: tuple[int, ...] = (1, 5, 10),
    seed: int = 0,
    batch_size: int = 256,
) -> dict[str, float]:
    """Compute ranking metrics over all users that have a held-out target.

    Parameters
    ----------
    rank_fn : callable
        ``rank_fn(histories, candidates) -> scores``. ``histories`` is a list of B item-id
        lists (the user history to condition on); ``candidates`` is an int array [B, C]
        whose column 0 is the ground-truth item. Returns a float array [B, C] of scores
        (higher = more relevant). The model wrapper handles padding / [mask] / inference.
    user_train : history used as model input.
    user_target : the held-out item per user (valid or test).
    popularity : interaction counts per id (length num_items + 1); used to sample negatives.
    extra_context : items appended to the input after train (e.g. the validation item is part
        of the input when evaluating on the test item).
    """
    rng = np.random.default_rng(seed)
    prob = popularity.astype(np.float64).copy()
    prob[0] = 0.0
    prob = prob / prob.sum()

    users = [
        u
        for u, t in user_target.items()
        if len(t) > 0 and len(user_train.get(u, [])) > 0
    ]

    histories: list[list[int]] = []
    candidates: list[list[int]] = []
    for u in users:
        hist = list(user_train[u])
        if extra_context is not None:
            hist = hist + list(extra_context.get(u, []))
        target = user_target[u][0]
        negs = _sample_pop_negatives(
            set(hist) | {target}, num_items, num_neg, prob, rng
        )
        histories.append(hist)
        candidates.append([target] + negs)

    cand_arr = np.asarray(candidates, dtype=np.int64)

    # NDCG@1 == HR@1, so we only track NDCG for k > 1 (matching the paper's tables).
    ndcg_ks = [k for k in topks if k > 1]
    totals = {f"HR@{k}": 0.0 for k in topks}
    totals.update({f"NDCG@{k}": 0.0 for k in ndcg_ks})
    totals["MRR"] = 0.0
    n = len(users)

    for start in range(0, n, batch_size):
        hb = histories[start : start + batch_size]
        cb = cand_arr[start : start + batch_size]
        scores = np.asarray(rank_fn(hb, cb))
        # rank of the ground-truth item (column 0): how many candidates score strictly higher.
        ranks = (scores > scores[:, :1]).sum(axis=1)
        for r in ranks:
            r = int(r)
            for k in topks:
                totals[f"HR@{k}"] += hit_at_k(r, k)
            for k in ndcg_ks:
                totals[f"NDCG@{k}"] += ndcg_at_k(r, k)
            totals["MRR"] += mrr(r)

    return {key: val / n for key, val in totals.items()}
