# Figures. Run after main.py so `results` is populated.
import numpy as np
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0))

# (a) learning-rate search, mean dev joint with std error bars
x = np.arange(len(lr_rows))
vals = [r['dev_joint'] for r in lr_rows]
errs = [r['dev_joint_std'] for r in lr_rows]
sel = int(np.argmax(vals))
colors = ['#3b6ea5'] * len(lr_rows); colors[sel] = '#2e9e5b'
b = axes[0].bar(x, vals, yerr=errs, capsize=4, color=colors, width=0.55)
axes[0].bar_label(b, fmt='%.2f', fontsize=8, padding=3)
axes[0].set_xticks(x, [r['name'].replace('base_lr', '') for r in lr_rows])
axes[0].set_xlabel('Learning rate'); axes[0].set_ylabel('Development joint score (%)')
axes[0].set_title('Baseline Learning-Rate Search')
axes[0].set_ylim(min(vals) - 1.2, max(vals) + 0.8)

# (b) development metrics across the incremental changes
xi = np.arange(len(inc_rows))
axes[1].errorbar(xi, [r['dev_f1'] for r in inc_rows],
                 yerr=[r['dev_f1_std'] for r in inc_rows], marker='o',
                 capsize=3, color='#3b6ea5', label='Dev Slot F1')
axes[1].errorbar(xi, [r['dev_acc'] for r in inc_rows],
                 yerr=[r['dev_acc_std'] for r in inc_rows], marker='s',
                 capsize=3, color='#e69138', label='Dev Intent Accuracy')
axes[1].errorbar(xi, [r['dev_joint'] for r in inc_rows],
                 yerr=[r['dev_joint_std'] for r in inc_rows], marker='^',
                 capsize=3, color='#2e9e5b', label='Dev Joint Score')
axes[1].set_xticks(xi, [r['name'] for r in inc_rows], rotation=30, ha='right', fontsize=7)
axes[1].set_ylabel('Score (%)'); axes[1].set_title('Development Performance Across Incremental Experiments')
axes[1].legend(frameon=False, fontsize=8)

# (c) held-out test metrics per configuration
w = 0.38
b1 = axes[2].bar(xi - w / 2, [r['test_f1'] for r in inc_rows], w,
                 yerr=[r['test_f1_std'] for r in inc_rows], capsize=3,
                 label='Test Slot F1', color='#3b6ea5')
b2 = axes[2].bar(xi + w / 2, [r['test_acc'] for r in inc_rows], w,
                 yerr=[r['test_acc_std'] for r in inc_rows], capsize=3,
                 label='Test Intent Accuracy', color='#e69138')
axes[2].set_xticks(xi, [r['name'] for r in inc_rows], rotation=30, ha='right', fontsize=7)
axes[2].set_ylabel('Score (%)'); axes[2].set_title('Test Performance Across Incremental Experiments')
axes[2].set_ylim(min(r['test_f1'] for r in inc_rows) - 4, 100)
axes[2].legend(frameon=False, fontsize=8)

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()
