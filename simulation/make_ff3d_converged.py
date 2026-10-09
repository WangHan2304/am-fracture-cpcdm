# -*- coding: utf-8 -*-
"""Regenerate the two 3D full-field-vs-Taylor figures from the CONVERGED-increment
(Δε=1e-3) case JSONs, replacing the hard-coded Δε=2e-3 / +24.3% version.

Outputs:
  figures/ff3d_cpfem_vs_taylor.png        : Ti-6Al-4V 0° full-field vs Taylor bars
  figures/ff3d_r13_4orientation_bias.png  : orientation-resolved bias band (0/45/90)
All values are read from output/nonlocal_cases/*.json (local mode, eps_step=1e-3).
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, 'output', 'nonlocal_cases')
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)


def load(cid):
    with open(os.path.join(CASE, cid + '.json'), encoding='utf-8') as f:
        return json.load(f)


# --- orientation-resolved converged (local, Δε=1e-3) cases --------------------
ang_cases = {'$0^\\circ$': 'ng3_local_e1e3',
             '$45^\\circ$': 'ng3_ang45_local_e1e3',
             '$90^\\circ$': 'ng3_ang90_local_e1e3'}
eff, efs, biases = [], [], []
for lbl, cid in ang_cases.items():
    d = load(cid)
    efs.append(float(d['ef_taylor_eps_matched']))
    eff.append(float(d['fracture_strain']))
    biases.append(float(d['bias_vs_taylor_eps_matched_pct']))

# --- ng3 -> ng4 mesh convergence at 0°, fixed aggregate-D_c nonlocal branch ----
conv = {}
for lc, tag in [(0.25, 'lc=0.25'), (0.3333333, 'lc=1 grain')]:
    n3 = load('ng3_lc%s_e1e3' % ('0250' if lc == 0.25 else '0333'))
    n4 = load('ng4_lc%s_e1e3' % ('0250' if lc == 0.25 else '0333'))
    conv[tag] = (float(n3['bias_vs_taylor_eps_matched_pct']),
                 float(n4['bias_vs_taylor_eps_matched_pct']))

# --- Figure 1: single-orientation bars ----------------------------------------
fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=200)
labels = ['3D CPFEM\n(full-field)', 'Taylor\n(iso-strain)']
vals = [eff[0], efs[0]]
colors = ['#4C78A8', '#F58518']
bars = ax.bar(labels, vals, width=0.55, color=colors, edgecolor='k', linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f'{v:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel(r'Fracture strain $\varepsilon_f$ (-)', fontsize=11)
ax.set_title('Ti-6Al-4V, $0^\\circ$ (27 grains, seed 42)\n'
             f'converged increment $\\Delta\\varepsilon=10^{{-3}}$; Taylor bias {biases[0]:+.1f}\\%',
             fontsize=10)
ax.set_ylim(0, max(vals) * 1.28)
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.annotate('local-damage, aggregate-$D_c$ criterion\nng3$\\to$ng4 bias shift '
            f'{conv["lc=0.25"][1]-conv["lc=0.25"][0]:+.2f} pp (<1\\%)',
            xy=(0, eff[0]), xytext=(0.03, max(vals) * 0.55),
            fontsize=8, color='#444',
            arrowprops=dict(arrowstyle='->', color='#444', lw=0.8))
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ff3d_cpfem_vs_taylor.png'), bbox_inches='tight')
plt.close(fig)
print(f'OK ff3d_cpfem_vs_taylor.png  ff={eff[0]:.4f} taylor={efs[0]:.4f} bias={biases[0]:+.1f}%')

# --- Figure 2: orientation-resolved bias band ---------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
xs = list(ang_cases.keys())
cols = ['#4C78A8' if b >= 0 else '#E45756' for b in biases]
b2 = ax.bar(xs, biases, width=0.55, color=cols, edgecolor='k', linewidth=0.6)
for bar, v in zip(b2, biases):
    ax.text(bar.get_x() + bar.get_width() / 2,
            v + (2 if v >= 0 else -6), f'{v:+.1f}\\%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.axhline(0, color='k', lw=0.8)
ax.set_ylabel('Taylor bias $(\\varepsilon_f^{ff}-\\varepsilon_f^{T})/\\varepsilon_f^{T}$ [\\%]',
              fontsize=10)
ax.set_title('3D coupled-damage bias, converged increment $\\Delta\\varepsilon=10^{-3}$\n'
             'Ti-6Al-4V (27 grains, local damage, $D_c$ criterion)', fontsize=10)
ax.set_ylim(min(0, min(biases)) - 15, max(biases) * 1.25)
ax.grid(axis='y', linestyle=':', alpha=0.5)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'ff3d_r13_4orientation_bias.png'), bbox_inches='tight')
plt.close(fig)
print('OK ff3d_r13_4orientation_bias.png biases=' +
      ', '.join(f'{k}:{v:+.1f}' for k, v in zip(xs, biases)))
