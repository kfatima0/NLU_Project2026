# Figures. Run after main.py so `results` is populated.
import numpy as np
import matplotlib.pyplot as plt


colors = ['#3b6ea5', '#6aa84f', '#e69138']
fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8))

# (a) development perplexity by epoch
for r, c in zip(results, colors):
    ep = np.arange(1, len(r['dev_hist']) + 1)
    axes[0].plot(ep, r['dev_hist'], marker='o', color=c,
                 label=r['name'].split('_')[1])
axes[0].scatter([best['best_epoch']], [best['dev_ppl']], color='#cc0000', zorder=5, s=70)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Development perplexity')
axes[0].set_title('Development PPL by epoch')
axes[0].set_xticks(np.arange(1, max(len(r['dev_hist']) for r in results) + 1))
axes[0].legend(frameon=False, fontsize=8)

# (b) dev / test per configuration
order = sorted(results, key=lambda r: r['dev_ppl'])
x = np.arange(len(order)); w = 0.38
b1 = axes[1].bar(x - w / 2, [r['dev_ppl'] for r in order], w, label='Best dev PPL',
                 color='#3b6ea5')
b2 = axes[1].bar(x + w / 2, [r['test_ppl'] for r in order], w, label='Test PPL',
                 color='#e69138')
for bars in (b1, b2):
    axes[1].bar_label(bars, fmt='%.2f', fontsize=8, padding=1)
axes[1].set_xticks(x, [r['name'].split('_')[1] for r in order], fontsize=8)
axes[1].set_ylabel('Perplexity'); axes[1].set_title('Development and test perplexity')
axes[1].legend(frameon=False, fontsize=8)

# (c) trainable parameters vs test perplexity
for r, c in zip(results, colors):
    axes[2].scatter(r['trainable'] / 1e6, r['test_ppl'], s=80, color=c)
    axes[2].annotate(r['name'].split('_')[1], (r['trainable'] / 1e6, r['test_ppl']),
                     textcoords='offset points', xytext=(6, 5), fontsize=8)
axes[2].set_xlabel('Trainable parameters (millions)')
axes[2].set_ylabel('Test perplexity')
axes[2].set_title('Parameter efficiency')

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()


PART_1A = {'name': 'step5_dropout01', 'dev_ppl': 40.49, 'test_ppl': 36.44, 'params': 20658240}   # best result from part1A.ipynb

table3 = pd.DataFrame([
    {'Part': 'Part 1.A', 'Model family': 'Scratch GPT-style Transformer',
     'Selected configuration': PART_1A['name'], 'Training method': 'From scratch',
     'Dev PPL': PART_1A['dev_ppl'], 'Test PPL': PART_1A['test_ppl'],
     'Trainable params': f"{PART_1A['params'] / 1e6:.2f}M"},
    {'Part': 'Part 1.B', 'Model family': 'GPT2 with LoRA',
     'Selected configuration': best['name'],
     'Training method': 'Parameter-efficient fine-tuning',
     'Dev PPL': best['dev_ppl'], 'Test PPL': best['test_ppl'],
     'Trainable params': f"{best['trainable'] / 1e6:.2f}M"},
])
display(table3)

d_red = 100 * (PART_1A['dev_ppl']  - best['dev_ppl'])  / PART_1A['dev_ppl']
t_red = 100 * (PART_1A['test_ppl'] - best['test_ppl']) / PART_1A['test_ppl']
print(f'Relative reduction -- dev: {d_red:.2f}%  |  test: {t_red:.2f}%')

fig, ax = plt.subplots(figsize=(5.6, 3.8))
vals = [PART_1A['test_ppl'], best['test_ppl']]
bars = ax.bar(['Part 1.A\nscratch Transformer', 'Part 1.B\nGPT2 + LoRA'], vals,
              color=['#7f8c8d', '#6aa84f'], width=0.55)
ax.bar_label(bars, fmt='%.2f', fontsize=10, padding=2)
if vals[1] < vals[0]:
    # sit the label above the shorter bar, clear of the bar-value text
    ax.annotate(f'{t_red:.1f}% lower PPL', xy=(1, vals[1] + max(vals) * 0.20),
                ha='center', fontsize=10, color='#2e7d32', fontweight='bold')
ax.set_ylabel('Test perplexity')
ax.set_title('Final Language Modeling Comparison')
ax.set_ylim(0, max(vals) * 1.30)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()
