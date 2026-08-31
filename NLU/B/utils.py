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
import json
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

IGNORE = -100
MAX_LEN = 64


import os
import json
import math
import copy
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel

plt.rcParams['figure.dpi'] = 110
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)

from conll import evaluate

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

SEEDS        = [42, 43, 44]
SEED         = SEEDS[0]
IGNORE       = -100     # CrossEntropyLoss skips these positions
MAX_LEN      = 64
BATCH_SIZE   = 32
N_EPOCHS     = 15
PATIENCE     = 4
WEIGHT_DECAY = 0.01
CLIP         = 1.0
AMP          = torch.cuda.is_available()    # fp16 on the tensor cores

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU   : {torch.cuda.get_device_name(0)}')
print(f'Mixed precision: {AMP}')


def load_json(path):
    with open(path) as f:
        return json.load(f)


tmp_train = load_json('dataset/ATIS/train.json')
test_raw  = load_json('dataset/ATIS/test.json')

counts = Counter(x['intent'] for x in tmp_train)
single  = [x for x in tmp_train if counts[x['intent']] == 1]
multi   = [x for x in tmp_train if counts[x['intent']] > 1]

train_raw, dev_raw = train_test_split(
    multi, test_size=0.10, random_state=SEED, shuffle=True,
    stratify=[x['intent'] for x in multi])
train_raw += single

# label spaces come from train only; anything new later falls back to 'unk'
slot2id = {'pad': 0, 'unk': 1}
for s in sorted({s for x in train_raw for s in x['slots'].split()}):
    slot2id.setdefault(s, len(slot2id))
intent2id = {'unk': 0}
for v in sorted({x['intent'] for x in train_raw}):
    intent2id.setdefault(v, len(intent2id))
id2slot = {v: k for k, v in slot2id.items()}

table1 = pd.DataFrame([
    {'Split': 'Train', 'Utterances': len(train_raw), 'Purpose': 'Fine-tuning'},
    {'Split': 'Dev',   'Utterances': len(dev_raw),   'Purpose': 'Model selection'},
    {'Split': 'Test',  'Utterances': len(test_raw),  'Purpose': 'Final evaluation'},
])
display(table1)
print(f'Slot labels: {len(slot2id)} | Intent labels: {len(intent2id)}')


class AtisSubword(data.Dataset):
    def __init__(self, rows, tokenizer):
        self.items = []
        for x in rows:
            words = x['utterance'].split()
            slots = x['slots'].split()
            enc = tokenizer(words, is_split_into_words=True,
                            truncation=True, max_length=MAX_LEN)
            word_ids = enc.word_ids()

            labels, first_pos, prev = [], [], None
            for pos, wid in enumerate(word_ids):
                if wid is None or wid == prev:      # special token, or a continuation
                    labels.append(IGNORE)
                else:
                    labels.append(slot2id.get(slots[wid], slot2id['unk']))
                    first_pos.append(pos)
                prev = wid

            n = len(first_pos)                      # words that survived truncation
            self.items.append({
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'labels': labels,
                'intent': intent2id.get(x['intent'], intent2id['unk']),
                'first_pos': first_pos,
                'words': words[:n],
                'gold': slots[:n],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(pad_id):
    def collate(batch):
        n = max(len(b['input_ids']) for b in batch)
        ids  = torch.full((len(batch), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        lab  = torch.full((len(batch), n), IGNORE, dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b['input_ids'])
            ids[i, :k]  = torch.tensor(b['input_ids'])
            mask[i, :k] = torch.tensor(b['attention_mask'])
            lab[i, :k]  = torch.tensor(b['labels'])
        return {'input_ids': ids.to(DEVICE),
                'attention_mask': mask.to(DEVICE),
                'labels': lab.to(DEVICE),
                'intents': torch.tensor([b['intent'] for b in batch]).to(DEVICE),
                'first_pos': [b['first_pos'] for b in batch],
                'words': [b['words'] for b in batch],
                'gold':  [b['gold'] for b in batch]}
    return collate


def build_loaders(name):
    kw = {'add_prefix_space': True} if 'gpt2' in name else {}
    tok = AutoTokenizer.from_pretrained(name, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token       # GPT2 ships without one
    pad_id = tok.pad_token_id
    loaders = [DataLoader(AtisSubword(rows, tok), batch_size=BATCH_SIZE,
                          shuffle=shuf, collate_fn=make_collate(pad_id))
               for rows, shuf in [(train_raw, True), (dev_raw, False), (test_raw, False)]]
    return tok, loaders


# quick look at what the alignment actually produces
_tok, _ = build_loaders('google-bert/bert-base-uncased')
_ex = AtisSubword(train_raw[:1], _tok)[0]
print('words     :', _ex['words'][:8])
print('tokens    :', _tok.convert_ids_to_tokens(_ex['input_ids'])[:12])
print('labels    :', _ex['labels'][:12], '  (-100 = skipped)')
print('first_pos :', _ex['first_pos'][:8])
