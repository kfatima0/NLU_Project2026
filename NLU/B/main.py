# Runs the experiments and prints the results.
from functions import *


for name, kind, model_name, lr in [
    ('gpt2_lr5e5', 'gpt2', 'openai-community/gpt2',          5e-5),
    ('gpt2_lr3e5', 'gpt2', 'openai-community/gpt2',          3e-5),
    ('bert_lr5e5', 'bert', 'google-bert/bert-base-uncased',  5e-5),
    ('bert_lr3e5', 'bert', 'google-bert/bert-base-uncased',  3e-5),
]:
    run_experiment(name, kind, model_name, lr)
