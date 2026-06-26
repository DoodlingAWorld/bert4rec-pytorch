#!/usr/bin/env python3
"""Train BERT4Rec on MovieLens-1M and save metrics + checkpoint.

python scripts/train.py                 # paper-style ML-1M config
python scripts/train.py --fast-dev      # quick smoke test (minutes)
python scripts/train.py --epochs 100 --hidden-dim 128
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bert4rec.config import BERT4RecConfig, fast_dev_config  # noqa: E402
from bert4rec.train import train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fast-dev", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="experiments/runs")
    args = p.parse_args()

    cfg = (
        fast_dev_config(seed=args.seed)
        if args.fast_dev
        else BERT4RecConfig(seed=args.seed)
    )
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim

    out = train(cfg, verbose=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = "fastdev" if args.fast_dev else "ml1m"
    name = f"{tag}_seed{cfg.seed}_d{cfg.hidden_dim}"
    record = {
        "config": cfg.to_dict(),
        "best_val_ndcg": out["best_val_ndcg"],
        "test_metrics": out["test_metrics"],
        "num_items": out["num_items"],
    }
    with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)
    torch.save(out["model"].state_dict(), os.path.join(args.out_dir, f"{name}.pt"))
    print("\nTEST:", {k: round(v, 4) for k, v in out["test_metrics"].items()})
    print(f"Saved -> {args.out_dir}/{name}.*")


if __name__ == "__main__":
    main()
