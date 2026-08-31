# Data loading, preprocessing and experiment settings.
import math
import random
import numpy as np
import pandas as pd

# this code came out of a notebook; outside one, display() is just print
try:
    from IPython.display import display
except ImportError:
    display = print
import os
import urllib.request
from functools import partial
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# settings shared by every experiment below
BLOCK_SIZE   = 128     # maximum sequence length in tokens
N_EPOCHS     = 5
PATIENCE     = 3       # early stopping on dev PPL
WEIGHT_DECAY = 0.01    # AdamW decoupled weight decay
CLIP         = 1.0     # gradient-norm clipping

print(f'Device      : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU         : {torch.cuda.get_device_name(0)}')
print(f'Block size  : {BLOCK_SIZE} tokens')
print(f'Epochs      : {N_EPOCHS} (patience {PATIENCE})')


os.makedirs('dataset/PennTreeBank', exist_ok=True)
base = 'https://raw.githubusercontent.com/massimo-rizzoli/NLU-2026-Labs/main/labs/dataset/PennTreeBank'
for split in ['ptb.train.txt', 'ptb.valid.txt', 'ptb.test.txt']:
    dst = f'dataset/PennTreeBank/{split}'
    if not os.path.exists(dst):
        urllib.request.urlretrieve(f'{base}/{split}', dst)
        print(f'Downloaded {split}')
    else:
        print(f'Already present: {split}')


def read_file(path, eos_token='<eos>'):
    output = []
    with open(path, 'r') as f:
        for line in f.readlines():
            output.append(line.strip() + ' ' + eos_token)
    return output


train_raw = read_file('dataset/PennTreeBank/ptb.train.txt')
dev_raw   = read_file('dataset/PennTreeBank/ptb.valid.txt')
test_raw  = read_file('dataset/PennTreeBank/ptb.test.txt')

print(f'Train: {len(train_raw)} | Dev: {len(dev_raw)} | Test: {len(test_raw)}')


# GPT2 has no pad token of its own, so I reuse its EOS as padding
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token
vocab_len = len(tokenizer)

print(f'Vocabulary size : {vocab_len}')
print(f'EOS / PAD id    : {tokenizer.pad_token_id}')


class PennTreeBank(data.Dataset):
    def __init__(self, corpus):
        self.sents = list(corpus)

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        return self.sents[idx]


def collate_fn(batch, tokenizer, device, block_size=BLOCK_SIZE):
    tokenized = tokenizer(batch, padding=True, truncation=True,
                          max_length=block_size, return_tensors='pt')
    ids = tokenized.input_ids

    input_ids = ids[:, :-1].contiguous().to(device)   # [w1 ... w_{n-1}]
    labels    = ids[:, 1:].contiguous().to(device)    # [w2 ... w_n]

    # only real tokens count towards the loss, padding is ignored
    n_tokens = int((ids[:, 1:] != tokenizer.pad_token_id).sum())

    return input_ids, labels, n_tokens


train_dataset = PennTreeBank(train_raw)
dev_dataset   = PennTreeBank(dev_raw)
test_dataset  = PennTreeBank(test_raw)

_collate = partial(collate_fn, tokenizer=tokenizer, device=DEVICE)

train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=_collate, shuffle=True)
dev_loader   = DataLoader(dev_dataset,   batch_size=64, collate_fn=_collate)
test_loader  = DataLoader(test_dataset,  batch_size=64, collate_fn=_collate)

print(f'Batches -- train: {len(train_loader)} | dev: {len(dev_loader)} | test: {len(test_loader)}')
