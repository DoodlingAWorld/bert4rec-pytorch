"""Contract for the BERT4Rec model: shapes, vocab, [mask] id, and bidirectionality."""

import torch

from bert4rec.config import BERT4RecConfig
from bert4rec.model import BERT4Rec


def _model(seed=0, num_items=50):
    torch.manual_seed(seed)
    cfg = BERT4RecConfig(
        max_len=10, hidden_dim=16, num_layers=2, num_heads=2, dropout=0.0
    )
    return BERT4Rec(num_items=num_items, cfg=cfg).eval(), cfg, num_items


def test_vocab_and_mask_id():
    model, _cfg, num_items = _model()
    assert model.mask_id == num_items + 1
    # embedding spans 0..num_items+1  => num_items + 2 rows
    assert model.item_emb.num_embeddings == num_items + 2
    assert model.item_emb.padding_idx == 0


def test_forward_shape_is_over_full_vocab():
    model, cfg, num_items = _model()
    seq = torch.randint(1, num_items + 1, (4, cfg.max_len))
    logits = model(seq)
    assert logits.shape == (4, cfg.max_len, num_items + 2)


def test_padding_embedding_is_zero():
    model, _cfg, _ = _model()
    assert torch.allclose(
        model.item_emb.weight[0], torch.zeros_like(model.item_emb.weight[0])
    )


def test_attention_is_bidirectional_not_causal():
    """Changing a LATER position must affect an EARLIER position's output (info flows both
    ways). This is the opposite of SASRec's causal property.

    This works on a randomly-initialised model (no training needed): embedding rows are
    independent, so swapping the last token gives position 0 a genuinely different value to
    attend to. A correct bidirectional model always changes position 0; a causal model never
    does (position 0 would attend only to itself). So the test discriminates the two.
    """
    model, cfg, num_items = _model()
    L = cfg.max_len
    seq = torch.randint(1, num_items + 1, (1, L))
    out1 = model(seq)

    seq2 = seq.clone()
    seq2[0, -1] = (seq2[0, -1] % num_items) + 1
    if seq2[0, -1] == seq[0, -1]:
        seq2[0, -1] = (seq2[0, -1] % num_items) + 1
    out2 = model(seq2)

    # an earlier position (index 0) should change because it can attend to the future
    assert not torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-6)
