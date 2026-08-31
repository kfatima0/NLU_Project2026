# NLU — Part B

The same two ATIS tasks, but fine-tuning pretrained GPT2 and BERT instead of training from
scratch.

Sub-tokenization is handled by giving each word's label to its **first** sub-token and
masking the rest with -100, so the loss skips them; predictions are read back at those
same positions to recover one tag per word for `conll.evaluate`.

Intent comes from `[CLS]` for BERT and from the last non-padding token for GPT2, because
of how each model attends.

Run with `python main.py`.

**Result:** `bert_lr5e5` — test slot F1 95.61 ± 0.07, intent accuracy 97.69 ± 0.13,
joint 96.65 ± 0.08.

BERT beats GPT2 by 2.57 points on slots but only 0.94 on intent. Tagging a token benefits
from seeing the words after it, which GPT2 structurally cannot do; a single sentence-level
decision does not, since GPT2's final token has already read everything.
