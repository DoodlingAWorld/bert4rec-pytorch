"""Dataset download, preprocessing, and the leave-one-out split.

The download + preprocessing + split logic is shared with the SASRec reproduction
(sibling repo `sasrec-pytorch`): MovieLens-1M, 5-core filtering, contiguous re-indexing,
chronological ordering, and the standard leave-one-out split. The Cloze training pipeline
and BERT4Rec-specific evaluation live in their own modules.

Item id 0 is reserved for padding. Real items are 1..num_items. The Cloze pipeline adds
one extra id (num_items + 1) for the special ``[mask]`` token.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_HF_RATINGS = "https://huggingface.co/datasets/nasserCha/movielens_rating_1m/resolve/main/ratings.dat"


def _make_opener(proxy: str | None):
    if proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        )
    return urllib.request.build_opener()


def download_ml1m(raw_dir: str, proxy: str | None = None) -> str:
    """Download MovieLens-1M and return the path to ratings.dat.

    Tries the canonical GroupLens zip first; on failure falls back to a Hugging Face
    mirror of ratings.dat (same `user::item::rating::ts` format). Pass an HTTP proxy URL
    if your network requires one to reach the internet.
    """
    os.makedirs(raw_dir, exist_ok=True)
    ratings_path = os.path.join(raw_dir, "ml-1m", "ratings.dat")
    if os.path.exists(ratings_path):
        return ratings_path

    urllib.request.install_opener(_make_opener(proxy))
    zip_path = os.path.join(raw_dir, "ml-1m.zip")
    try:
        print(f"Downloading {ML1M_URL} ...")
        urllib.request.urlretrieve(ML1M_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        return ratings_path
    except Exception as e:  # noqa: BLE001 - any network/zip failure -> try the mirror
        print(f"Primary source failed ({e}); falling back to Hugging Face mirror ...")
        os.makedirs(os.path.dirname(ratings_path), exist_ok=True)
        urllib.request.urlretrieve(ML1M_HF_RATINGS, ratings_path)
        return ratings_path


def parse_ml1m(ratings_path: str) -> list[tuple[int, int, int]]:
    """Parse ratings.dat -> list of (user, item, timestamp). Rating is ignored."""
    rows = []
    with open(ratings_path, "r", encoding="latin-1") as f:
        for line in f:
            u, i, _rating, ts = line.strip().split("::")
            rows.append((int(u), int(i), int(ts)))
    return rows


def kcore_filter(
    rows: list[tuple[int, int, int]], min_count: int
) -> list[tuple[int, int, int]]:
    """Iteratively drop users/items with < min_count interactions until stable."""
    rows = list(rows)
    while True:
        user_cnt: dict[int, int] = defaultdict(int)
        item_cnt: dict[int, int] = defaultdict(int)
        for u, i, _ in rows:
            user_cnt[u] += 1
            item_cnt[i] += 1
        keep = [
            (u, i, ts)
            for (u, i, ts) in rows
            if user_cnt[u] >= min_count and item_cnt[i] >= min_count
        ]
        if len(keep) == len(rows):
            return keep
        rows = keep


def build_user_sequences(
    rows: list[tuple[int, int, int]], min_count: int = 5
) -> tuple[dict[int, list[int]], int, int]:
    """k-core filter, re-index ids to 1..N (0 = padding), order each user by timestamp.

    Returns (user_seqs, num_users, num_items).
    """
    rows = kcore_filter(rows, min_count)
    user_map: dict[int, int] = {}
    item_map: dict[int, int] = {}
    by_user: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for u, i, ts in rows:
        if u not in user_map:
            user_map[u] = len(user_map) + 1
        if i not in item_map:
            item_map[i] = len(item_map) + 1
        by_user[user_map[u]].append((ts, item_map[i]))

    user_seqs: dict[int, list[int]] = {}
    for u, pairs in by_user.items():
        pairs.sort(key=lambda p: p[0])
        user_seqs[u] = [item for _ts, item in pairs]
    return user_seqs, len(user_map), len(item_map)


def leave_one_out_split(
    user_seqs: dict[int, list[int]],
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[int]]]:
    """Split each user's chronological sequence: test = last, valid = 2nd-last, train = rest.

    Users with < 3 interactions go entirely to train (empty valid/test).
    """
    user_train: dict[int, list[int]] = {}
    user_valid: dict[int, list[int]] = {}
    user_test: dict[int, list[int]] = {}
    for u, seq in user_seqs.items():
        if len(seq) < 3:
            user_train[u], user_valid[u], user_test[u] = seq, [], []
        else:
            user_train[u], user_valid[u], user_test[u] = seq[:-2], [seq[-2]], [seq[-1]]
    return user_train, user_valid, user_test


def left_pad_sequence(items: list[int], max_len: int) -> np.ndarray:
    """Left-pad/truncate a list of item ids to a fixed-length int64 array (keep most recent)."""
    seq = np.zeros(max_len, dtype=np.int64)
    if len(items) == 0:
        return seq
    items = items[-max_len:]
    seq[max_len - len(items) :] = items
    return seq


def item_popularity(user_train: dict[int, list[int]], num_items: int) -> np.ndarray:
    """Interaction counts per item id over training sequences.

    Returns a float array of length ``num_items + 1`` (index 0 = padding, count 0). Used by
    the evaluation to sample negatives by popularity, as in the BERT4Rec paper.
    """
    counts = np.zeros(num_items + 1, dtype=np.float64)
    for seq in user_train.values():
        for i in seq:
            counts[i] += 1.0
    return counts
