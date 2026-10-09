# -*- coding: utf-8 -*-
"""Regenerate Fig_AM_anisotropy_factors.png from the FINAL 9/9 validation JSON,
so the figure and the manuscript text both use experiment_v2_results.json values.
Panel (a): calibrated orientation multiplier f_AM_aniso (normalized to 1 at 0 deg).
Panel (b): total AM correction factor f_AM per validation case.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(HERE, 'output', 'experiment_v6_results.json')))

MATS = ['Ti64', '316L', 'AlSi10Mg']
NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
ORIS = [0, 45, 90]
COLORS = {'Ti64': '#d62728', '316L': '#1f77b4', 'AlSi10Mg': '#2ca02c'}

_cases = {m: {} for m in MATS}
for c in res['cases']:
    _cases[c['material']][c['orientation']] = c
aniso = {m: [_cases[m][o]['a_aniso'] for o in ORIS] for m in MATS}
f_am = {m: [_cases[m][o]['f_AM'] for o in ORIS] for m in MATS}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(ORIS))
w = 0.26
xlab = [u'0°', u'45°', u'90°']

# ---- (a) orientation multiplier ----
ax = axes[0]
for i, m in enumerate(MATS):
    xs = x + (i - 1) * w
    ax.bar(xs, aniso[m], w, color=COLORS[m], edgecolor='black', linewidth=0.5, label=NAMES[m])
    for xx, v in zip(xs, aniso[m]):
        ax.text(xx, v + 0.03, f'{v:g}', ha='center', fontsize=8)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(xlab)
ax.set_ylabel(r'$f_{\mathrm{AM}}^{\mathrm{aniso}}$ (normalized to $0^{\circ}$)')
ax.set_title('(a) Calibrated orientation multiplier')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 4.8)

# ---- (b) total f_AM ----
ax = axes[1]
for i, m in enumerate(MATS):
    xs = x + (i - 1) * w
    ax.bar(xs, f_am[m], w, color=COLORS[m], edgecolor='black', linewidth=0.5, label=NAMES[m])
    for xx, v in zip(xs, f_am[m]):
        ax.text(xx, v + 0.2, f'{v:.2f}', ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(xlab)
ax.set_ylabel(r'Total $f_{\mathrm{AM}}$')
ax.set_title('(b) Total AM correction factor per case')
ax.grid(alpha=0.3, axis='y')

fig.suptitle('AM correction factors (v6 calibrated run)', fontweight='bold')
fig.tight_layout()
out = os.path.join(HERE, 'figures', 'Fig_AM_anisotropy_factors.png')
fig.savefig(out, dpi=300)
print('Wrote:', out)

# print summary for the report
for m in MATS:
    print(m, 'aniso:', [f'{v:g}' for v in aniso[m]],
          'f_AM total:', [f'{v:.2f}' for v in f_am[m]],
          f'ratio(90/0)={f_am[m][2]/f_am[m][0]:.2f}')
