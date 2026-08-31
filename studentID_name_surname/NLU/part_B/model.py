# Model architecture.
import torch
import torch.nn as nn
from transformers import AutoModel


class JointBERT(nn.Module):
    # encoder model - [CLS] is already a sentence summary, so use it
    def __init__(self, name='google-bert/bert-base-uncased', dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.drop        = nn.Dropout(dropout)
        self.slot_head   = nn.Linear(h, len(slot2id))
        self.intent_head = nn.Linear(h, len(intent2id))

    def forward(self, input_ids, attention_mask):
        seq = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        seq = self.drop(seq)
        return self.slot_head(seq), self.intent_head(seq[:, 0])   # [CLS] is position 0


class JointGPT2(nn.Module):
    # decoder model - only the last real token has seen the whole utterance
    def __init__(self, name='openai-community/gpt2', dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.drop        = nn.Dropout(dropout)
        self.slot_head   = nn.Linear(h, len(slot2id))
        self.intent_head = nn.Linear(h, len(intent2id))

    def forward(self, input_ids, attention_mask):
        seq = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state
        seq = self.drop(seq)
        last = attention_mask.sum(1) - 1                 # index of the final real token
        pooled = seq[torch.arange(seq.size(0), device=seq.device), last]
        return self.slot_head(seq), self.intent_head(pooled)


def build_model(kind):
    return JointBERT() if kind == 'bert' else JointGPT2()


print('Models defined.')
