# Model architecture.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 128


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.n_heads = n_heads
        self.h_dim   = d_model // n_heads   # dimensions per head

        self.w_q      = nn.Linear(d_model, d_model)
        self.w_k      = nn.Linear(d_model, d_model)
        self.w_v      = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout  = nn.Dropout(dropout)   # (2) after the attention softmax
        self.resid_dropout = nn.Dropout(dropout)   # (3) after the output projection

    def forward(self, x, mask):
        B, L, d_model = x.size()

        # project, then split across the heads -> (B, n_heads, L, h_dim)
        q = self.w_q(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        # scaled dot product attention, masked so nothing can look ahead
        sim  = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.h_dim))
        sim  = sim.masked_fill(mask == 0, float('-inf'))
        attn = self.attn_dropout(F.softmax(sim, dim=-1))

        # stick the heads back together
        y = (attn @ v).transpose(1, 2).contiguous().view(B, L, d_model)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1    = nn.Linear(d_model, hidden_dim)
        self.act        = nn.GELU()
        self.linear2    = nn.Linear(hidden_dim, d_model)
        self.ff_dropout = nn.Dropout(dropout)      # (4) after the second linear

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return self.ff_dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.0):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)   # norm -> attention -> residual
        x = x + self.ff(self.ln2(x))           # norm -> feed forward -> residual
        return x


class GPT2(nn.Module):
    def __init__(self, vocab_size, pos_emb_size=BLOCK_SIZE, d_model=128, n_heads=2,
                 num_layers=2, ff_dim=512, dropout=0.0, weight_tying=False):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed   = nn.Embedding(pos_emb_size, d_model)
        self.emb_dropout = nn.Dropout(dropout)     # (1) after the embedding sum

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if weight_tying:
            # both are (vocab_size x d_model) so they can literally be one tensor -
            # whatever vector means 'cat' going in is what we look for coming out
            self.lm_head.weight = self.token_embed.weight

        # lower triangular mask so position i only sees 0..i
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('mask', mask)

    def forward(self, idx):
        B, L = idx.shape
        assert L <= self.pos_emb_size, f'sequence length {L} exceeds {self.pos_emb_size}'
        pos = torch.arange(L, device=idx.device)
        x   = self.emb_dropout(self.token_embed(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x, self.mask[:, :, :L, :L])
        return self.lm_head(self.ln_f(x))          # (B, L, vocab_size)


print('Model defined.')
