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


import os
import json
import math
import copy
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

plt.rcParams['figure.dpi'] = 110
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 220)

from conll import evaluate

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SEEDS        = [42, 43, 44]     # three seeds per config
SEED         = SEEDS[0]
PAD_TOKEN    = 0
N_EPOCHS     = 40
PATIENCE     = 6
WEIGHT_DECAY = 0.01
CLIP         = 5.0
BATCH_SIZE   = 128

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

print(f'Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'GPU   : {torch.cuda.get_device_name(0)}')


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

print(f'Train: {len(train_raw)} | Dev: {len(dev_raw)} | Test: {len(test_raw)}')
print(f'Intents in corpus: {len(counts)} | singleton intents kept in train: {len(single)}')
print('\nExample:')
print(json.dumps(train_raw[0], indent=1))


class Lang:
    def __init__(self, corpus, cutoff=0):
        self.word2id   = self._w2id([x['utterance'] for x in corpus], cutoff)
        self.slot2id   = self._lab2id(sum([x['slots'].split() for x in corpus], []))
        self.intent2id = self._lab2id([x['intent'] for x in corpus], pad=False)
        self.id2word   = {v: k for k, v in self.word2id.items()}
        self.id2slot   = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    def _w2id(self, utterances, cutoff=0):
        # cls goes on the end of every utterance and its hidden state is what the
        # intent head reads
        vocab = {'pad': PAD_TOKEN, 'unk': 1, 'cls': 2}
        for word, freq in Counter(sum([u.split() for u in utterances], [])).items():
            if freq > cutoff and word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def _lab2id(self, elements, pad=True):
        # some intents only turn up in test (day_name for one), so they need
        # somewhere to go instead of blowing up with a KeyError
        vocab = {'pad': PAD_TOKEN, 'unk': 1} if pad else {'unk': 0}
        for e in sorted(set(elements)):
            if e not in vocab:
                vocab[e] = len(vocab)
        return vocab


lang = Lang(train_raw, cutoff=0)

table1 = pd.DataFrame([
    {'Split': 'Train', 'Utterances': len(train_raw), 'Purpose': 'Parameter learning'},
    {'Split': 'Dev',   'Utterances': len(dev_raw),   'Purpose': 'Model selection'},
    {'Split': 'Test',  'Utterances': len(test_raw),  'Purpose': 'Final held-out evaluation'},
])
display(table1)
print(f'Vocabulary: {len(lang.word2id)} words | Slots: {len(lang.slot2id)} | '
      f'Intents: {len(lang.intent2id)}')


class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        self.lang = lang
        self.unk_id = lang.word2id[unk]
        self.utt_ids, self.slot_ids, self.intent_ids = [], [], []
        self.words, self.gold_slots = [], []      # raw strings, kept for scoring
        for x in dataset:
            words = x['utterance'].split()
            slots = x['slots'].split()
            assert len(words) == len(slots), 'word/slot length mismatch'
            # cls at the end, not the start - attention is causal so the last
            # position is the only one that has seen everything
            self.utt_ids.append([lang.word2id.get(w, self.unk_id) for w in words]
                                + [lang.word2id['cls']])
            # unseen slots and the cls position both get PAD so they stay out of the loss
            self.slot_ids.append([lang.slot2id.get(s, PAD_TOKEN) for s in slots]
                                 + [PAD_TOKEN])
            self.intent_ids.append(lang.intent2id.get(x['intent'], lang.intent2id['unk']))
            self.words.append(words)
            self.gold_slots.append(slots)

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        return {'utterance': torch.tensor(self.utt_ids[idx]),
                'slots':     torch.tensor(self.slot_ids[idx]),
                'intent':    self.intent_ids[idx],
                'words':     self.words[idx],
                'gold':      self.gold_slots[idx]}


def collate_fn(batch):
    def pad(seqs):
        lengths = [len(s) for s in seqs]
        out = torch.full((len(seqs), max(lengths)), PAD_TOKEN, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = s
        return out, torch.tensor(lengths)

    batch.sort(key=lambda x: len(x['utterance']), reverse=True)
    utts,  lengths = pad([b['utterance'] for b in batch])
    slots, _       = pad([b['slots'] for b in batch])
    return {'utterances': utts.to(DEVICE),
            'y_slots':    slots.to(DEVICE),
            'intents':    torch.tensor([b['intent'] for b in batch]).to(DEVICE),
            'lengths':    lengths.to(DEVICE),
            'words':      [b['words'] for b in batch],
            'gold':       [b['gold'] for b in batch]}


train_loader = DataLoader(IntentsAndSlots(train_raw, lang), batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, shuffle=True)
dev_loader   = DataLoader(IntentsAndSlots(dev_raw, lang), batch_size=64,
                          collate_fn=collate_fn)
test_loader  = DataLoader(IntentsAndSlots(test_raw, lang), batch_size=64,
                          collate_fn=collate_fn)

print(f'Batches -- train: {len(train_loader)} | dev: {len(dev_loader)} | '
      f'test: {len(test_loader)}')
