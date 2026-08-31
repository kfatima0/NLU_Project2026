# Training and evaluation functions.
from utils import *
from model import *
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_loop(data, optimizer, criterion, model, clip=CLIP):
    model.train()
    loss_sum, n_total = 0.0, 0
    pbar = tqdm(data, desc='Training', leave=False)
    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(input_ids)
        loss   = criterion(output.permute(0, 2, 1), labels)   # CrossEntropy wants (B, C, L)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        loss_sum += loss.item() * n_tokens
        n_total  += n_tokens
        if i % 100 == 0:
            pbar.set_postfix(loss=loss_sum / n_total)

    avg_loss = loss_sum / n_total
    return math.exp(avg_loss), avg_loss


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_sum, n_total = 0.0, 0
    with torch.no_grad():
        for input_ids, labels, n_tokens in tqdm(data, desc='Evaluating', leave=False):
            output = model(input_ids)
            loss   = eval_criterion(output.permute(0, 2, 1), labels)
            loss_sum += loss.item() * n_tokens
            n_total  += n_tokens

    avg_loss = loss_sum / n_total
    return math.exp(avg_loss), avg_loss


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


def run_experiment(name, lr, d_model=128, n_heads=2, num_layers=2, ff_dim=512,
                   dropout=0.0, weight_tying=False, n_epochs=N_EPOCHS, patience=PATIENCE):
    print(f'\n{"=" * 68}')
    print(f'Experiment : {name}')
    print(f'  lr={lr}  d_model={d_model}  n_heads={n_heads}  '
          f'num_layers={num_layers}  ff_dim={ff_dim}')
    print(f'  dropout={dropout}  weight_tying={weight_tying}')
    print('=' * 68)

    torch.manual_seed(SEED)   # same starting point for every config so the comparison is fair
    model = GPT2(vocab_len, pos_emb_size=BLOCK_SIZE, d_model=d_model, n_heads=n_heads,
                 num_layers=num_layers, ff_dim=ff_dim, dropout=dropout,
                 weight_tying=weight_tying).to(DEVICE)
    model.apply(init_weights)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters : {n_params:,} ({n_params / 1e6:.2f}M)')

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    train_hist, dev_hist = [], []
    best_ppl, best_model, best_epoch, pat = math.inf, None, -1, patience

    pbar = tqdm(range(1, n_epochs + 1), desc='Epochs')
    for epoch in pbar:
        train_ppl, _ = train_loop(train_loader, optimizer, criterion, model)
        dev_ppl,   _ = eval_loop(dev_loader, criterion, model)
        train_hist.append(train_ppl)
        dev_hist.append(dev_ppl)
        pbar.set_description(f'epoch {epoch}: train {train_ppl:.2f} / dev {dev_ppl:.2f}')
        print(f'  epoch {epoch}: train PPL {train_ppl:7.2f} | dev PPL {dev_ppl:7.2f}')

        if dev_ppl < best_ppl:
            best_ppl, best_epoch = dev_ppl, epoch
            best_model = copy.deepcopy(model).to('cpu')
            pat = patience
        else:
            pat -= 1
            if pat <= 0:
                print(f'  early stop at epoch {epoch}')
                break

    best_model.to(DEVICE)
    test_ppl, _ = eval_loop(test_loader, criterion, best_model)
    best_model.to('cpu')

    print(f'  -> best epoch {best_epoch} | dev PPL {best_ppl:.2f} | test PPL {test_ppl:.2f}')

    results.append({
        'name': name, 'lr': lr, 'd_model': d_model, 'n_heads': n_heads,
        'num_layers': num_layers, 'ff_dim': ff_dim, 'dropout': dropout,
        'weight_tying': weight_tying, 'params': n_params,
        'best_epoch': best_epoch,
        'dev_ppl': round(best_ppl, 2), 'test_ppl': round(test_ppl, 2),
        'train_hist': [round(v, 2) for v in train_hist],
        'dev_hist': [round(v, 2) for v in dev_hist],
    })
    return best_model
