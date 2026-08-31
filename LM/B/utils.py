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
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SEED         = 42
BLOCK_SIZE   = 64      # how many tokens per block
BATCH_SIZE   = 64
N_EPOCHS     = 5
PATIENCE     = 3
WEIGHT_DECAY = 0.01
CLIP         = 1.0
LORA_DROPOUT = 0.05
AMP          = torch.cuda.is_available()   # fp16 on the T4 tensor cores

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f'Device     : {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU        : {torch.cuda.get_device_name(0)}')
print(f'Block size : {BLOCK_SIZE} tokens | batch {BATCH_SIZE} | epochs {N_EPOCHS}')
print(f'Mixed precision: {AMP}')


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
    with open(path, 'r') as f:
        return [line.strip() + ' ' + eos_token for line in f.readlines()]


train_raw = read_file('dataset/PennTreeBank/ptb.train.txt')
dev_raw   = read_file('dataset/PennTreeBank/ptb.valid.txt')
test_raw  = read_file('dataset/PennTreeBank/ptb.test.txt')

tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
tokenizer.pad_token = tokenizer.eos_token

print(f'\nSentences -- train: {len(train_raw)} | dev: {len(dev_raw)} | test: {len(test_raw)}')
print(f'Vocabulary: {len(tokenizer)} | EOS/PAD id: {tokenizer.pad_token_id}')


def pack_blocks(raw, tokenizer, block_size=BLOCK_SIZE):
    # glue everything together, then cut it into equal chunks
    ids = []
    for line in raw:
        ids.extend(tokenizer(line).input_ids)
    n_blocks = len(ids) // block_size
    return torch.tensor(ids[:n_blocks * block_size], dtype=torch.long).view(n_blocks, block_size)


train_blocks = pack_blocks(train_raw, tokenizer)
dev_blocks   = pack_blocks(dev_raw, tokenizer)
test_blocks  = pack_blocks(test_raw, tokenizer)

train_loader = DataLoader(train_blocks, batch_size=BATCH_SIZE, shuffle=True)
dev_loader   = DataLoader(dev_blocks,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_blocks,  batch_size=BATCH_SIZE)

table1 = pd.DataFrame([
    {'Split': 'Train',       'Sentences': len(train_raw), 'Packed blocks': len(train_blocks),
     'Main purpose': 'Adapter learning'},
    {'Split': 'Development', 'Sentences': len(dev_raw),   'Packed blocks': len(dev_blocks),
     'Main purpose': 'Model selection'},
    {'Split': 'Test',        'Sentences': len(test_raw),  'Packed blocks': len(test_blocks),
     'Main purpose': 'Final held-out evaluation'},
])
display(table1)
print(f'Batches -- train: {len(train_loader)} | dev: {len(dev_loader)} | test: {len(test_loader)}')
