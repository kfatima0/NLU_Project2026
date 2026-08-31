# Runs the experiments and prints the results.
from functions import *


# clear this before re-running the whole sweep
results = []


BASELINE = dict(d_model=128, n_heads=2, num_layers=2, ff_dim=512,
                dropout=0.0, weight_tying=False)

for lr, tag in [(1e-3, '1e-3'), (5e-4, '5e-4'), (3e-4, '3e-4'), (1e-4, '1e-4')]:
    run_experiment(name=f'base_lr{tag}', lr=lr, **BASELINE)


best_lr = 1e-3  # selected by dev PPL (Table 2)

INCREMENTAL = [
    ('step0_baseline',  dict(d_model=128, n_heads=2, num_layers=2, ff_dim=512, dropout=0.0, weight_tying=False)),
    ('step1_width192',  dict(d_model=192, n_heads=2, num_layers=2, ff_dim=512, dropout=0.0, weight_tying=False)),
    ('step2_heads4',    dict(d_model=192, n_heads=4, num_layers=2, ff_dim=512, dropout=0.0, weight_tying=False)),
    ('step3_depth3', dict(d_model=192, n_heads=4, num_layers=3, ff_dim=512, dropout=0.0, weight_tying=False)),
    ('step4_ffn768',   dict(d_model=192, n_heads=4, num_layers=3, ff_dim=768, dropout=0.0, weight_tying=False)),
    ('step5_dropout01',     dict(d_model=192, n_heads=4, num_layers=3, ff_dim=768, dropout=0.1, weight_tying=False)),
    ('step6_tying',    dict(d_model=192, n_heads=4, num_layers=3, ff_dim=768, dropout=0.1, weight_tying=True)),
]

trained = {}
for exp_name, cfg in INCREMENTAL:
    trained[exp_name] = run_experiment(name=exp_name, lr=best_lr, **cfg)
