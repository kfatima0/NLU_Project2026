# NLU — Part A

Joint intent classification and slot filling on ATIS with a decoder-only Transformer
trained from scratch. Two heads: one per-token for slots, one on an appended `cls` token
for the intent (it goes at the end because attention is causal, so only the last position
has seen the whole utterance).

Run with `python main.py`.

Every configuration runs with three seeds and is reported as mean ± std. Models are picked
on the dev joint score, J = F1/2 + Acc/2.

**Result:** `step5_dropout01` — test slot F1 90.39 ± 0.48, intent accuracy 92.83 ± 0.85,
joint 91.61 ± 0.36.

Dropout was the single largest gain. The lowest learning rate tried (1e-4 in early runs)
failed outright, which is why the rate is searched before anything else.
