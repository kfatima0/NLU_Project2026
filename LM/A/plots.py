# Figures. Run after main.py so `results` is populated.
import numpy as np
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(6.2, 3.6))

labels = [r['name'].replace('base_lr', '') for r in lr_rows]
x = np.arange(len(lr_rows))
w = 0.38
best_i = int(np.argmin([r['dev_ppl'] for r in lr_rows]))

b1 = ax.bar(x - w / 2, [r['dev_ppl'] for r in lr_rows], w,
            label='Best dev PPL', color='#3b6ea5')
b2 = ax.bar(x + w / 2, [r['test_ppl'] for r in lr_rows], w,
            label='Test PPL', color='#6aa84f')
b1[best_i].set_color('#e69138')          # highlight the selected rate

for bars in (b1, b2):
    ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=1)

ax.set_xticks(x, labels)
ax.set_xlabel('Learning rate')
ax.set_ylabel('Perplexity')
ax.set_title('Baseline learning-rate search')
ax.legend(frameon=False, fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()


sel = best_inc
epochs = np.arange(1, len(sel['dev_hist']) + 1)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(epochs, sel['train_hist'], marker='o', color='#3b6ea5', label='Train PPL')
ax.plot(epochs, sel['dev_hist'],   marker='o', color='#e69138', label='Dev PPL')
ax.scatter([sel['best_epoch']], [sel['dev_ppl']], color='#cc0000', zorder=5,
           s=60, label='Best dev epoch')
ax.annotate(f"Best epoch {sel['best_epoch']}\nDev PPL {sel['dev_ppl']:.1f}",
            xy=(sel['best_epoch'], sel['dev_ppl']),
            xytext=(-10, 18), textcoords='offset points', fontsize=8, ha='right')

ax.set_xlabel('Epoch')
ax.set_ylabel('Perplexity')
ax.set_title(f"Train vs development PPL: {sel['name']}")
ax.set_xticks(epochs)
ax.legend(frameon=False, fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()


names = [r['name'] for r in inc_rows]
dev   = [r['dev_ppl'] for r in inc_rows]
test  = [r['test_ppl'] for r in inc_rows]
x     = np.arange(len(inc_rows))
sel_i = int(np.argmin(dev))

fig, ax = plt.subplots(figsize=(8.4, 3.8))
ax.plot(x, dev,  marker='o', color='#3b6ea5', label='Dev PPL')
ax.plot(x, test, marker='s', color='#e69138', label='Test PPL')
ax.scatter([x[sel_i]], [dev[sel_i]], color='#cc0000', zorder=5, s=70,
           label='Selected by dev')

for xi, (d, t) in enumerate(zip(dev, test)):
    ax.annotate(f'{d:.1f}', (xi, d), textcoords='offset points', xytext=(0, 7),
                ha='center', fontsize=7)
    ax.annotate(f'{t:.1f}', (xi, t), textcoords='offset points', xytext=(0, -12),
                ha='center', fontsize=7)

ax.set_xticks(x, [n.replace('_', ' ') for n in names], rotation=20, ha='right', fontsize=8)
ax.set_ylabel('Perplexity')
ax.set_title('Incremental Part 1.A experiment comparison')
ax.legend(frameon=False, fontsize=8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()


metrics = ['dev_ppl', 'test_ppl', 'best_epoch']
titles  = ['Dev PPL', 'Test PPL', 'Best epoch']

M = np.array([[r[m] for m in metrics] for r in inc_rows], dtype=float)
rng = M.max(axis=0) - M.min(axis=0)
norm = np.where(rng == 0, 0.5, (M - M.min(axis=0)) / np.where(rng == 0, 1, rng))

fig, ax = plt.subplots(figsize=(5.6, 4.2))
im = ax.imshow(norm, cmap='viridis_r', aspect='auto', vmin=0, vmax=1)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        val = f'{M[i, j]:.1f}' if metrics[j] != 'best_epoch' else f'{int(M[i, j])}'
        ax.text(j, i, val, ha='center', va='center', fontsize=8,
                color='white' if norm[i, j] > 0.55 else 'black')

ax.set_xticks(range(len(metrics)), titles)
ax.set_yticks(range(len(inc_rows)), [r['name'].split('_')[0] for r in inc_rows])
ax.set_title('Normalized summary of incremental experiments')
fig.colorbar(im, ax=ax, label='Normalized value', fraction=0.046)
plt.tight_layout()
plt.show()
