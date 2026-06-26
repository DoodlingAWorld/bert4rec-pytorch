#!/usr/bin/env python3
"""Download + preprocess MovieLens-1M and print dataset statistics (paper Table 1)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bert4rec.data import build_user_sequences, download_ml1m, parse_ml1m  # noqa: E402


def main() -> None:
    path = download_ml1m("data/raw")
    seqs, num_users, num_items = build_user_sequences(parse_ml1m(path), min_count=5)
    n_actions = sum(len(s) for s in seqs.values())
    print("MovieLens-1M (after preprocessing)")
    print(f"  users        : {num_users}")
    print(f"  items        : {num_items}")
    print(f"  actions      : {n_actions:,}")
    print(f"  avg length   : {n_actions / num_users:.1f}")


if __name__ == "__main__":
    main()
