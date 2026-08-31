# Figures. Run after main.py so `results` is populated.
import numpy as np
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))
palette = {'gpt2': '#3b6ea5', 'bert': '#6aa84f'}

# (a) dev joint score per epoch, averaged over the three seeds
for r in results:
    n = min(len(h) for h in r['hists'])
    mean = np.mean([h[:n] for h in r['hists']], axis=0) * 100
    axes[0].plot(np.arange(1, n + 1), mean, marker='o', ms=3,
                 color=palette[r['kind']],
                 ls='-' if str(r['lr']) == '5e-05' else '--', label=r['name'])
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Dev joint score (%)')
axes[0].set_title('Development joint score across epochs')
axes[0].legend(frameon=False, fontsize=8)

# (b) dev joint with seed spread
order = sorted(results, key=lambda r: -r['dev_joint'])
x = np.arange(len(order))
b = axes[1].bar(x, [r['dev_joint'] for r in order],
                yerr=[r['dev_joint_std'] for r in order], capsize=4,
                color=[palette[r['kind']] for r in order], width=0.55)
axes[1].bar_label(b, fmt='%.2f', fontsize=8, padding=3)
axes[1].set_xticks(x, [r['name'] for r in order], rotation=15, fontsize=8)
axes[1].set_ylabel('Dev joint score (%)'); axes[1].set_title('Development comparison')
axes[1].set_ylim(min(r['dev_joint'] for r in order) - 2, 100)

# (c) held-out test performance
w = 0.38
b1 = axes[2].bar(x - w / 2, [r['test_f1'] for r in order], w,
                 yerr=[r['test_f1_std'] for r in order], capsize=3,
                 label='Test Slot F1', color='#3b6ea5')
b2 = axes[2].bar(x + w / 2, [r['test_acc'] for r in order], w,
                 yerr=[r['test_acc_std'] for r in order], capsize=3,
                 label='Test Intent Accuracy', color='#e69138')
for bars in (b1, b2):
    axes[2].bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
axes[2].set_xticks(x, [r['name'] for r in order], rotation=15, fontsize=8)
axes[2].set_ylabel('Score (%)'); axes[2].set_title('Held-out test performance')
axes[2].set_ylim(min(r['test_f1'] for r in order) - 4, 100)
axes[2].legend(frameon=False, fontsize=8)

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()
