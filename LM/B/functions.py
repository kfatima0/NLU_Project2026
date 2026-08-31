# Training and evaluation functions.
from utils import *
from model import *
import math
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_loop(loader, optimizer, model, scaler=None, clip=CLIP):
    model.train()
    loss_sum, n_total = 0.0, 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    for batch in tqdm(loader, desc='Training', leave=False):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        if scaler is not None:
            # fp16 forward/backward, fp32 master weights - the T4 has tensor cores
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(input_ids=batch, labels=batch)
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)              # unscale before clipping
            torch.nn.utils.clip_grad_norm_(trainable, clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(input_ids=batch, labels=batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, clip)
            optimizer.step()

        n = batch.shape[0] * (batch.shape[1] - 1)
        loss_sum += out.loss.item() * n
        n_total  += n
    avg = loss_sum / n_total
    return math.exp(avg), avg


def eval_loop(loader, model):
    model.eval()
    loss_sum, n_total = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            batch = batch.to(DEVICE)
            with torch.autocast('cuda', enabled=AMP, dtype=torch.float16):
                out = model(input_ids=batch, labels=batch)
            n = batch.shape[0] * (batch.shape[1] - 1)
            loss_sum += out.loss.item() * n
            n_total  += n
    avg = loss_sum / n_total
    return math.exp(avg), avg


results = []


def run_experiment(name, rank, alpha, lr, lora_dropout=LORA_DROPOUT,
                   n_epochs=N_EPOCHS, patience=PATIENCE):
    print(f'\n{"=" * 68}')
    print(f'Experiment : {name}')
    print(f'  rank={rank}  alpha={alpha}  lr={lr}  lora_dropout={lora_dropout}')
    print('=' * 68)

    torch.manual_seed(SEED)
    model = GPT2_LoRA.from_pretrained('openai-community/gpt2', rank=rank, alpha=alpha,
                                      lora_dropout=lora_dropout).to(DEVICE)
    total, trainable = param_stats(model)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda') if AMP else None

    train_hist, dev_hist = [], []
    best_ppl, best_state, best_epoch, pat = math.inf, None, -1, patience

    pbar = tqdm(range(1, n_epochs + 1), desc='Epochs')
    for epoch in pbar:
        train_ppl, _ = train_loop(train_loader, optimizer, model, scaler)
        dev_ppl,   _ = eval_loop(dev_loader, model)
        train_hist.append(train_ppl)
        dev_hist.append(dev_ppl)
        pbar.set_description(f'epoch {epoch}: train {train_ppl:.2f} / dev {dev_ppl:.2f}')
        print(f'  epoch {epoch}: train PPL {train_ppl:7.2f} | dev PPL {dev_ppl:7.2f}')

        if dev_ppl < best_ppl:
            best_ppl, best_epoch = dev_ppl, epoch
            # no point saving the frozen backbone, just keep the adapters
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items() if 'lora_' in k}
            pat = patience
        else:
            pat -= 1
            if pat <= 0:
                print(f'  early stop at epoch {epoch}')
                break

    model.load_state_dict(best_state, strict=False)
    test_ppl, _ = eval_loop(test_loader, model)
    print(f'  -> best epoch {best_epoch} | dev PPL {best_ppl:.2f} | test PPL {test_ppl:.2f}')

    results.append({
        'name': name, 'rank': rank, 'alpha': alpha, 'lr': lr,
        'lora_dropout': lora_dropout, 'trainable': trainable,
        'best_epoch': best_epoch,
        'dev_ppl': round(best_ppl, 2), 'test_ppl': round(test_ppl, 2),
        'train_hist': [round(v, 2) for v in train_hist],
        'dev_hist': [round(v, 2) for v in dev_hist],
    })
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return best_ppl, test_ppl
