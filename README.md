# bert4rec-pytorch: bidirectional sequential recommendation

A from-scratch **PyTorch** implementation of
[**BERT4Rec** (Sun et al., CIKM 2019)](https://arxiv.org/abs/1904.06690): a bidirectional
Transformer encoder trained with a Cloze (masked-item) objective for sequential
recommendation. It is the masked-LM counterpart to the autoregressive
[SASRec](https://github.com/DoodlingAWorld/sasrec-pytorch), and this repo compares the two
head-to-head on MovieLens-1M under one evaluation protocol.

## SASRec vs BERT4Rec

The paper puts it plainly: *"SASRec is a left-to-right unidirectional version of our
BERT4Rec with single-head attention and causal attention mask."* The differences are exactly
the GPT-vs-BERT distinction, applied to recommendation:

| | SASRec (autoregressive) | BERT4Rec (masked-LM) |
|---|---|---|
| Attention | causal (left-to-right) | **bidirectional** |
| Heads | single | **multi-head** |
| Block | pre-LN | **post-LN** |
| FFN | ReLU, d->d | **GELU, d->4d->d** |
| Objective | predict next at each step | **Cloze**: mask items, predict from both sides |
| Inference | last position | append a **`[mask]`** token, predict it |

Because bidirectional attention would let a position "see" its own target, BERT4Rec cannot
train autoregressively. The Cloze task fixes this: randomly mask a fraction of items and
predict them from the surrounding context, then at test time append a trailing `[mask]` and
read the recommendation off that position.

## Results

Measured on CPU with MovieLens-1M, popularity-sampled negatives (paper protocol).

| Model | HR@10 | NDCG@10 | MRR |
|---|---|---|---|
| Paper SASRec (Table 2) | 0.6629 | 0.4368 | 0.3790 |
| Paper BERT4Rec (Table 2) | 0.6970 | 0.4818 | 0.4254 |
| **This repo (BERT4Rec)** | _pending_ | _pending_ | _pending_ |
| **This repo (SASRec, same eval)** | _pending_ | _pending_ | _pending_ |

> Reproducibility note: BERT4Rec's published numbers are known to require very long training
> (see Petrov & Macdonald, RecSys 2022). On CPU we train as long as is practical and report
> honestly; the controlled SASRec-vs-BERT4Rec comparison under one protocol is the headline.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

pytest                              # component contracts
python scripts/prepare_data.py     # download + preprocess, print stats
python scripts/train.py --fast-dev # smoke test in minutes
python scripts/train.py            # full ML-1M run
```

If your network needs an HTTP proxy to reach the internet, pass it to `download_ml1m(proxy=...)`.

## Layout

```
src/bert4rec/
  data.py       ML-1M download, preprocessing, leave-one-out split, item popularity
  config.py     hyperparameters (paper ML-1m defaults)
  eval.py       leave-one-out eval with popularity-sampled negatives (HR/NDCG/MRR)
  model.py      bidirectional multi-head Transformer encoder + tied-embedding output head
  masking.py    Cloze masking dataset + trailing-[mask] eval input
  losses.py     Cloze (masked-LM) cross-entropy over masked positions
  train.py      training loop (Cloze objective, popularity eval, early stopping)
scripts/        prepare_data.py, train.py
tests/          one contract file per module (model, masking, losses, train)
```

## Reference

```bibtex
@inproceedings{sun2019bert4rec,
  title={BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer},
  author={Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao and Ou, Wenwu and Jiang, Peng},
  booktitle={CIKM},
  year={2019}
}
```

Builds on the SASRec reproduction in the sibling repo `sasrec-pytorch`. Reference
implementation consulted for correctness (not copied): the authors' `FeiSun/BERT4Rec`.
