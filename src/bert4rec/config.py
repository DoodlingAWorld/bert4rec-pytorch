"""Configuration for BERT4Rec. Defaults follow the paper's MovieLens-1M setup (Section 4.3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BERT4RecConfig:
    # ---- data ----
    dataset: str = "ml-1m"
    max_len: int = 200  # N: max sequence length (paper uses 200 for ML-1m)
    min_count: int = 5  # k-core

    # ---- model ----
    hidden_dim: int = 64  # d (paper: d>=64 is satisfactory; main table tunes up to 256)
    num_layers: int = 2  # L
    num_heads: int = 2  # h (head dim = hidden_dim / num_heads)
    dropout: float = 0.2

    # ---- Cloze objective ----
    mask_prob: float = 0.2  # rho: fraction of items masked per sequence (ML-1m)

    # ---- training ----
    lr: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 5.0  # clip gradients when L2 norm exceeds this
    batch_size: int = 256
    num_epochs: int = 200
    patience: int = 10
    eval_every: int = 10

    # ---- evaluation ----
    num_neg_eval: int = 100  # negatives sampled per user (by popularity)
    topks: tuple[int, ...] = (1, 5, 10)

    # ---- misc ----
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0

    # Note: the special [mask] token id is `num_items + 1` (0 is padding, 1..num_items are
    # real items). It depends on the dataset, so it is computed where num_items is known
    # (the masking pipeline and the model), not stored on the config.

    def to_dict(self) -> dict:
        return asdict(self)


def fast_dev_config(**overrides) -> BERT4RecConfig:
    """Tiny config to validate the pipeline end-to-end in minutes (not for real metrics)."""
    cfg = BERT4RecConfig(
        max_len=50,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        num_epochs=5,
        eval_every=1,
        patience=5,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
