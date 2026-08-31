# LM — Part A

GPT2-style language model trained from scratch on Penn Treebank.

Run with `python main.py`. The corpus downloads itself into `dataset/` on first run.

Steps: learning-rate search, then `d_model` / `n_heads` / `num_layers` / `ff_dim` one at a
time, then dropout, then weight tying.

**Result:** best dev PPL 40.49, test PPL 36.44 (`A5_add_dropout_0.1`). Every configuration
is under the required PPL < 250.

**Note on weight tying (A6).** It scores worse than A5, and the notebook includes a cell
showing why: `init_weights` only touches `nn.Linear`, and when tying is on
`lm_head.weight` *is* `token_embed.weight`, so the uniform init also overwrites the token
embeddings and leaves them ~172x smaller than the positional ones. A6 therefore changes
two things at once and is not a clean test of tying on its own.
