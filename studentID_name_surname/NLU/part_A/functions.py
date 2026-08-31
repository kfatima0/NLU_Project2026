# Training and evaluation functions.
from utils import *
from model import *
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from conll import evaluate


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


def train_loop(loader, optimizer, criterion_slots, criterion_intents, model, clip=CLIP):
    model.train()
    losses = []
    for sample in loader:
        optimizer.zero_grad()
        slots, intent = model(sample['utterances'], sample['lengths'])
        loss = (criterion_intents(intent, sample['intents'])
                + criterion_slots(slots.permute(0, 2, 1), sample['y_slots']))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def eval_loop(loader, criterion_slots, criterion_intents, model, lang):
    model.eval()
    losses = []
    ref_slots, hyp_slots = [], []
    ref_intents, hyp_intents = [], []

    with torch.no_grad():
        for sample in loader:
            slots, intent = model(sample['utterances'], sample['lengths'])
            loss = (criterion_intents(intent, sample['intents'])
                    + criterion_slots(slots.permute(0, 2, 1), sample['y_slots']))
            losses.append(loss.item())

            ref_intents.extend(sample['intents'].tolist())
            hyp_intents.extend(torch.argmax(intent, dim=1).tolist())

            pred = torch.argmax(slots, dim=2)
            for i, seq in enumerate(pred):
                n = sample['lengths'][i].item() - 1      # cut the cls back off
                # use the original label strings here, otherwise a slot missing from the
                # train vocab would quietly be scored as padding
                words = sample['words'][i]
                gold  = sample['gold'][i]
                got   = [lang.id2slot[s] for s in seq[:n].tolist()]
                ref_slots.append(list(zip(words, gold)))
                hyp_slots.append(list(zip(words, got)))

    try:
        slot_f1 = evaluate(ref_slots, hyp_slots)['total']['f']
    except Exception as e:      # a degenerate run can predict no spans at all
        print('  conll evaluate failed:', e)
        slot_f1 = 0.0

    acc = sum(int(a == b) for a, b in zip(ref_intents, hyp_intents)) / len(ref_intents)
    joint = 0.5 * slot_f1 + 0.5 * acc        # the joint score used for picking models
    return slot_f1, acc, joint, sum(losses) / len(losses)


results = []


def train_one(seed, lr, d_model, n_heads, num_layers, ff_dim, dropout,
              n_epochs=N_EPOCHS, patience=PATIENCE):
    # run a single seed, return the scores of the best dev epoch
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    model = JointGPT2(len(lang.word2id), len(lang.slot2id), len(lang.intent2id),
                      d_model=d_model, n_heads=n_heads, num_layers=num_layers,
                      ff_dim=ff_dim, dropout=dropout).to(DEVICE)
    model.apply(init_weights)
    n_params = sum(p.numel() for p in model.parameters())

    criterion_slots   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best = {'joint': -1.0}
    best_state, best_epoch, pat = None, -1, patience
    hist = []

    for epoch in range(1, n_epochs + 1):
        train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
        f1, acc, joint, _ = eval_loop(dev_loader, criterion_slots, criterion_intents,
                                      model, lang)
        hist.append(joint)
        if joint > best['joint']:
            best = {'joint': joint, 'f1': f1, 'acc': acc}
            best_state, best_epoch, pat = copy.deepcopy(model.state_dict()), epoch, patience
        else:
            pat -= 1
            if pat <= 0:
                break

    model.load_state_dict(best_state)
    t_f1, t_acc, t_joint, _ = eval_loop(test_loader, criterion_slots, criterion_intents,
                                        model, lang)
    del model
    torch.cuda.empty_cache()
    return {'best_epoch': best_epoch, 'n_params': n_params, 'hist': hist,
            'dev_f1': best['f1'], 'dev_acc': best['acc'], 'dev_joint': best['joint'],
            'test_f1': t_f1, 'test_acc': t_acc, 'test_joint': t_joint}


def ms(values):
    # mean/std, in percent
    a = np.array(values) * 100
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def run_experiment(name, lr, d_model=128, n_heads=2, num_layers=1, ff_dim=256,
                   dropout=0.0, seeds=SEEDS):
    print(f'\n{"=" * 74}')
    print(f'Experiment : {name}')
    print(f'  lr={lr}  d_model={d_model}  n_heads={n_heads}  num_layers={num_layers}  '
          f'ff_dim={ff_dim}  dropout={dropout}')
    print('=' * 74)

    runs = []
    for seed in tqdm(seeds, desc=name, leave=False):
        r = train_one(seed, lr, d_model, n_heads, num_layers, ff_dim, dropout)
        runs.append(r)
        print(f'  seed {seed}: epoch {r["best_epoch"]:2d} | dev J {100*r["dev_joint"]:.2f} '
              f'| test F1 {100*r["test_f1"]:.2f} | test acc {100*r["test_acc"]:.2f}')

    agg = {'name': name, 'lr': lr, 'd_model': d_model, 'n_heads': n_heads,
           'num_layers': num_layers, 'ff_dim': ff_dim, 'dropout': dropout,
           'params': runs[0]['n_params'],
           'best_epoch_mean': float(np.mean([r['best_epoch'] for r in runs])),
           'best_epoch_std': float(np.std([r['best_epoch'] for r in runs], ddof=1)),
           'hists': [r['hist'] for r in runs]}
    for key in ['dev_f1', 'dev_acc', 'dev_joint', 'test_f1', 'test_acc', 'test_joint']:
        m, s = ms([r[key] for r in runs])
        agg[key], agg[key + '_std'] = round(m, 2), round(s, 2)

    print(f'  -> dev J {agg["dev_joint"]:.2f} +/- {agg["dev_joint_std"]:.2f} | '
          f'test F1 {agg["test_f1"]:.2f} +/- {agg["test_f1_std"]:.2f} | '
          f'test acc {agg["test_acc"]:.2f} +/- {agg["test_acc_std"]:.2f} | '
          f'{agg["params"]/1e6:.2f}M params')
    results.append(agg)
    return agg
