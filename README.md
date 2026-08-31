# NLU Project 2026

Course project for Natural Language Understanding, University of Trento.

## Layout

```
LM/A    language modelling from scratch (GPT2-style transformer, Penn Treebank)
LM/B    the same task with LoRA adapters on pretrained GPT2
NLU/A   joint intent classification and slot filling from scratch (ATIS)
NLU/B   the same task fine-tuning pretrained GPT2 and BERT
```

Each part holds `model.py` (architecture), `utils.py` (data), `functions.py`
(training and evaluation), `main.py` (runs the experiments), `plots.py` (figures),
plus `results/` with the measured numbers and `plots/` with the generated charts.

`notebooks/` keeps the executed notebooks the results came from.

## Results

| Part | Selected model | Metric |
|------|----------------|--------|
| LM/A | `step5_dropout01` | test PPL 36.44 |
| LM/B | `lora_r32_a64_lr5e4` | test PPL 20.87 |
| NLU/A | `step5_dropout01` | test slot F1 90.39, intent acc 92.83 |
| NLU/B | `bert_lr5e5` | test slot F1 95.61, intent acc 97.69 |

## Running

```bash
pip install -r requirements.txt
cd LM/A && python main.py
```

Datasets download themselves on first run (Penn Treebank) or are read from
`dataset/` (ATIS).
