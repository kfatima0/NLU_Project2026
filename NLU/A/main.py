# Runs the experiments and prints the results.
from functions import *


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


for lr, tag in [(1e-3, '1e-3'), (5e-4, '5e-4'), (3e-4, '3e-4')]:
    run_experiment(name=f'base_lr{tag}', lr=lr)


best_lr = 1e-3  # selected by dev PPL (Table 2)

INCREMENTAL = [
    ('step0_baseline',    dict(d_model=128, n_heads=2, num_layers=1, ff_dim=256, dropout=0.0)),
    ('step1_width192',      dict(d_model=192, n_heads=2, num_layers=1, ff_dim=256, dropout=0.0)),
    ('step2_heads4',        dict(d_model=192, n_heads=4, num_layers=1, ff_dim=256, dropout=0.0)),
    ('step3_depth2',     dict(d_model=192, n_heads=4, num_layers=2, ff_dim=256, dropout=0.0)),
    ('step4_ffn512',       dict(d_model=192, n_heads=4, num_layers=2, ff_dim=512, dropout=0.0)),
    ('step5_dropout01',  dict(d_model=192, n_heads=4, num_layers=2, ff_dim=512, dropout=0.1)),
]

for exp_name, cfg in INCREMENTAL:
    run_experiment(name=exp_name, lr=best_lr, **cfg)
