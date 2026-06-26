"""Integration contract for the training loop (tiny synthetic data, no download).

This passes only once the model, masking, loss, AND train loop are implemented.
"""

from bert4rec.config import fast_dev_config
from bert4rec.train import train


def _tiny_data(num_users=20, num_items=30, seq_len=6):
    seqs = {
        u: [((u + t) % num_items) + 1 for t in range(seq_len)]
        for u in range(1, num_users + 1)
    }
    user_train = {u: s[:-2] for u, s in seqs.items()}
    user_valid = {u: [s[-2]] for u, s in seqs.items()}
    user_test = {u: [s[-1]] for u, s in seqs.items()}
    return user_train, user_valid, user_test, num_items


def test_train_runs_and_returns_metrics():
    cfg = fast_dev_config(
        max_len=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        num_epochs=1,
        eval_every=1,
        batch_size=8,
        seed=0,
    )
    out = train(cfg, data=_tiny_data(), verbose=False)

    assert "model" in out and "best_val_ndcg" in out and "test_metrics" in out
    assert 0.0 <= out["best_val_ndcg"] <= 1.0
    assert "HR@10" in out["test_metrics"] and "NDCG@10" in out["test_metrics"]
