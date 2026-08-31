# LM — Part B

GPT2 fine-tuned with LoRA adapters, implemented by hand (no PEFT). Adapters go on the
query, key and value projections; the pretrained backbone stays frozen.

Run with `python main.py`.

**Result:** best dev PPL 22.94, test PPL 20.87 (`B2_LoRA_r32_a64_lr5e-4`), training 1.77M
parameters out of 126M. That is 42.7% lower test perplexity than Part A while training
11.7x fewer weights, so both mandatory requirements (PPL < 250, better than Part A) are met.

A check in the code confirms the adapter is a no-op before training: with `B = 0` the
wrapped model scores 137.0948 dev PPL, identical to plain GPT2.

Note the two parts batch differently — Part A pads individual sentences, Part B packs the
corpus into 64-token blocks — so a little of the gap comes from the extra context packing
gives rather than from LoRA alone.
