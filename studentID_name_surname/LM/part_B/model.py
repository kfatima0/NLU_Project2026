# Model architecture.
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

LORA_DROPOUT = 0.05


class LoRAQKV(nn.Module):
    # GPT2 keeps q, k and v in one fused Conv1D, so I wrap it and add a separate
    # adapter for each of the three. The original layer stays frozen.

    def __init__(self, base, rank, alpha, dropout=LORA_DROPOUT):
        super().__init__()
        self.base = base                        # the original layer, frozen
        d_in, d_out = base.weight.shape         # (n_embd, 3 * n_embd)
        self.d       = d_out // 3
        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank
        self.drop    = nn.Dropout(dropout)

        # 0 is query, 1 is key, 2 is value
        self.lora_A = nn.Parameter(torch.zeros(3, rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(3, self.d, rank))
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)   # random init for A
        # B is left at zero so the adapter starts off doing nothing

    def forward(self, x):
        out = self.base(x)                      # original output
        xd  = self.drop(x)
        delta = torch.cat(
            [(xd @ self.lora_A[i].T) @ self.lora_B[i].T for i in range(3)], dim=-1)
        return out + self.scaling * delta


class GPT2_LoRA(GPT2LMHeadModel):
    # GPT2 with my own LoRA adapters on q, k and v

    @classmethod
    def from_pretrained(cls, *args, rank=8, alpha=16, lora_dropout=LORA_DROPOUT, **kwargs):
        model = super().from_pretrained(*args, **kwargs)
        model.inject_lora(rank, alpha, lora_dropout)
        return model

    def inject_lora(self, rank, alpha, lora_dropout=LORA_DROPOUT):
        for p in self.parameters():                     # freeze everything first
            p.requires_grad = False
        for block in self.transformer.h:                # then swap in the adapters
            block.attn.c_attn = LoRAQKV(block.attn.c_attn, rank, alpha, lora_dropout)
        for name, p in self.named_parameters():         # and only let those train
            if 'lora_' in name:
                p.requires_grad = True
        self.lora_rank, self.lora_alpha = rank, alpha
        return self


def param_stats(model, verbose=True):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f'  total params     : {total:,}')
        print(f'  trainable params : {trainable:,} ({100 * trainable / total:.3f}%)')
        print(f'  frozen params    : {total - trainable:,}')
    return total, trainable


print('LoRA implementation defined.')
