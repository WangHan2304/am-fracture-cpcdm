# -*- coding: utf-8 -*-
"""v7 版 Fig_AM_anisotropy_factors.png 重绘 — update_fig_AM_v7.py
数据源：experiment_v7_final_results.json（params + cases 的 a_aniso）
f_AM 总因子：实例化 TaylorCPCDM 读取 model.f_AM（factors_v4 × aniso_multiplier）
Panel (a)：校准取向乘数 f_AM_aniso（0°≡1）
Panel (b)：总 AM 修正因子 f_AM（基础因子 × 投影 × 取向乘数）
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'output')
FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

res = json.load(open(os.path.join(OUT_DIR, 'experiment_v7_final_results.json'),
                     encoding='utf-8'))
params = res['params']

MATS = ['Ti64', '316L', 'AlSi10Mg']
NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
ORIS = [0, 45, 90]
COLORS = {'Ti64': '#d62728', '316L': '#1f77b4', 'AlSi10Mg': '#2ca02c'}

EPS_STEP = {'Ti64': 5e-4, '316L': 1e-3, 'AlSi10Mg': 5e-4}

aniso = {m: [] for m in MATS}
f_am = {m: [] for m in MATS}
for m in MATS:
    p = params[m]
    gs = p['gs_mult']
    S0 = p['S0']
    for psi in ORIS:
        a = 1.0 if psi == 0 else (p['a45'] if psi == 45 else p['a90'])
        ov = {}
        if m == 'Ti64':
            ov = {'gs_basal': 700e6 * gs, 'gs_prism': 720e6 * gs,
                  'gs_pyr': 800e6 * gs}
        else:
            ov = {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[m] * gs}
        ov['aniso_multiplier'] = {0: 1.0, 45: p['a45'], 90: p['a90']}
        model = TaylorCPCDM(
            m, psi, S0_override=S0, n_grains=20,
            eps_step=EPS_STEP[m], n_sub=30, strain_rate=1e-3,
            model_version='v4', force_homogeneous=False,
            param_overrides=ov, seed=42,
        )
        aniso[m].append(a)
        f_am[m].append(float(model.f_AM))
        print(f'{m} {psi}° a={a:.4f} f_AM={model.f_AM:.4f}')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(ORIS))
w = 0.26
xlab = ['0°', '45°', '90°']

ax = axes[0]
for i, m in enumerate(MATS):
    xs = x + (i - 1) * w
    ax.bar(xs, aniso[m], w, color=COLORS[m], edgecolor='black',
           linewidth=0.5, label=NAMES[m])
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

ax = axes[1]
for i, m in enumerate(MATS):
    xs = x + (i - 1) * w
    ax.bar(xs, f_am[m], w, color=COLORS[m], edgecolor='black',
           linewidth=0.5, label=NAMES[m])
    for xx, v in zip(xs, f_am[m]):
        ax.text(xx, v + 0.2, f'{v:.2f}', ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(xlab)
ax.set_ylabel(r'Total $f_{\mathrm{AM}}$')
ax.set_title('(b) Total AM correction factor per case')
ax.grid(alpha=0.3, axis='y')

fig.suptitle('AM correction factors (v7 calibrated run)', fontweight='bold')
fig.tight_layout()
out = os.path.join(FIG_DIR, 'Fig_AM_anisotropy_factors.png')
fig.savefig(out, dpi=300)
print('Wrote:', out)

for m in MATS:
    print(m, 'aniso:', [f'{v:g}' for v in aniso[m]],
          'f_AM total:', [f'{v:.2f}' for v in f_am[m]],
          f'ratio(90/0)={f_am[m][2]/f_am[m][0]:.2f}')
