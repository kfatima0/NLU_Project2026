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


def run_epoch(loader, model, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    slot_loss = nn.CrossEntropyLoss(ignore_index=IGNORE)
    intent_loss = nn.CrossEntropyLoss()
    ref, hyp, gold_i, pred_i, losses = [], [], [], [], []

    for b in loader:
        with torch.set_grad_enabled(train):
            with torch.autocast('cuda', enabled=AMP, dtype=torch.float16):
                slots, intent = model(b['input_ids'], b['attention_mask'])
                loss = (intent_loss(intent, b['intents'])
                        + slot_loss(slots.reshape(-1, slots.size(-1)),
                                    b['labels'].reshape(-1)))
        if train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()

        losses.append(loss.item())
        gold_i.extend(b['intents'].tolist())
        pred_i.extend(intent.argmax(1).tolist())

        pred_slots = slots.argmax(-1)
        for i, pos in enumerate(b['first_pos']):
            # read predictions back only at the first piece of each word
            tags = [id2slot.get(int(pred_slots[i, p]), 'O') for p in pos]
            ref.append(list(zip(b['words'][i], b['gold'][i])))
            hyp.append(list(zip(b['words'][i], tags)))

    try:
        f1 = evaluate(ref, hyp)['total']['f']
    except Exception as e:
        print('  conll failed:', e)
        f1 = 0.0
    acc = sum(int(a == c) for a, c in zip(gold_i, pred_i)) / len(gold_i)
    return f1, acc, 0.5 * f1 + 0.5 * acc, float(np.mean(losses))


results = []


def train_one(seed, kind, model_name, lr):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    _, (tr, dv, te) = build_loaders(model_name)

    model = build_model(kind).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda') if AMP else None

    best, best_state, best_epoch, pat, hist = {'joint': -1.0}, None, -1, PATIENCE, []
    for epoch in range(1, N_EPOCHS + 1):
        run_epoch(tr, model, optimizer, scaler)
        f1, acc, joint, _ = run_epoch(dv, model)
        hist.append(joint)
        if joint > best['joint']:
            best = {'joint': joint, 'f1': f1, 'acc': acc}
            best_state, best_epoch, pat = copy.deepcopy(model.state_dict()), epoch, PATIENCE
        else:
            pat -= 1
            if pat <= 0:
                break

    model.load_state_dict(best_state)
    t_f1, t_acc, t_joint, _ = run_epoch(te, model)
    del model, optimizer
    torch.cuda.empty_cache()
    return {'best_epoch': best_epoch, 'n_params': n_params, 'hist': hist,
            'dev_f1': best['f1'], 'dev_acc': best['acc'], 'dev_joint': best['joint'],
            'test_f1': t_f1, 'test_acc': t_acc, 'test_joint': t_joint}


def ms(v):
    a = np.array(v) * 100
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def run_experiment(name, kind, model_name, lr, seeds=SEEDS):
    print(f'\n{"=" * 74}')
    print(f'{name}  |  {model_name}  |  lr={lr}')
    print('=' * 74)
    runs = []
    for s in tqdm(seeds, desc=name, leave=False):
        r = train_one(s, kind, model_name, lr)
        runs.append(r)
        print(f'  seed {s}: epoch {r["best_epoch"]:2d} | dev J {100*r["dev_joint"]:.2f} '
              f'| test F1 {100*r["test_f1"]:.2f} | test acc {100*r["test_acc"]:.2f}')

    agg = {'name': name, 'kind': kind, 'model': model_name, 'lr': lr,
           'params': runs[0]['n_params'],
           'best_epoch_mean': float(np.mean([r['best_epoch'] for r in runs])),
           'best_epoch_std': float(np.std([r['best_epoch'] for r in runs], ddof=1)),
           'hists': [r['hist'] for r in runs]}
    for k in ['dev_f1', 'dev_acc', 'dev_joint', 'test_f1', 'test_acc', 'test_joint']:
        m, s = ms([r[k] for r in runs])
        agg[k], agg[k + '_std'] = round(m, 2), round(s, 2)
    print(f'  -> dev J {agg["dev_joint"]:.2f} +/- {agg["dev_joint_std"]:.2f} | '
          f'test F1 {agg["test_f1"]:.2f} +/- {agg["test_f1_std"]:.2f} | '
          f'test acc {agg["test_acc"]:.2f} +/- {agg["test_acc_std"]:.2f}')
    results.append(agg)
    return agg
