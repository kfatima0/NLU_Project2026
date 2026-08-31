# Model architecture.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_TOKEN = 0
MAX_LEN = 64


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.n_heads, self.h_dim = n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, d = x.size()
        q = self.w_q(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        sim  = (q @ k.transpose(-2, -1)) / math.sqrt(self.h_dim)
        sim  = sim.masked_fill(mask == 0, float('-inf'))
        attn = self.attn_dropout(F.softmax(sim, dim=-1))
        y = (attn @ v).transpose(1, 2).contiguous().view(B, L, d)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.act     = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.ff_dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.ff_dropout(self.linear2(self.act(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


MAX_LEN = 64


class JointGPT2(nn.Module):
    def __init__(self, vocab_size, n_slots, n_intents, max_len=MAX_LEN, d_model=128,
                 n_heads=2, num_layers=2, ff_dim=512, dropout=0.0):
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.pos_embed   = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)

        # Step 2: dropout right before the two output layers
        self.out_dropout = nn.Dropout(dropout)
        self.slot_head   = nn.Linear(d_model, n_slots)
        self.intent_head = nn.Linear(d_model, n_intents)

        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('mask', mask)

    def forward(self, utterances, lengths):
        B, L = utterances.shape
        pos = torch.arange(L, device=utterances.device)
        x = self.emb_dropout(self.token_embed(utterances) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x, self.mask[:, :, :L, :L])
        x = self.ln_f(x)

        slot_logits = self.slot_head(self.out_dropout(x))          # (B, L, n_slots)

        # last real token of each utterance -- the only position that saw everything
        last = x[torch.arange(B, device=x.device), lengths - 1]
        intent_logits = self.intent_head(self.out_dropout(last))   # (B, n_intents)
        return slot_logits, intent_logits


def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


print('Model defined.')
