# -*- coding: utf-8 -*-
"""
v7 版 Fig1 重绘 — regen_fig1_v7.py
====================================
用 v7 正典参数重跑九工况，重绘 Fig1_AM_microstructure_damage.png
（损伤演化 + 损伤率 3×2，去除硬编码 'Fig.1:' 标题，LaTeX 统一编号）。
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

MAT_NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
ORIENT_LABELS = {0: '0° (BD)', 45: '45°', 90: '90° (TD)'}
COLORS = {'Ti64': '#4C72B0', '316L': '#55A868', 'AlSi10Mg': '#C44E52'}
LINESTYLES = {0: '-', 45: '--', 90: '-.'}
MATS = ['Ti64', '316L', 'AlSi10Mg']
ORIS = [0, 45, 90]
EPS_STEP_316L = 1e-3


def _ov(m, a, gs):
    if m == 'Ti64':
        ov = {'gs_basal': 700e6 * gs, 'gs_prism': 720e6 * gs, 'gs_pyr': 800e6 * gs}
    else:
        ov = {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[m] * gs}
    ov['aniso_multiplier'] = {0: 1.0, 45: a[45], 90: a[90]}
    return ov


def main():
    v7 = json.load(open(os.path.join(OUT_DIR, 'experiment_v7_final_results.json'),
                        encoding='utf-8'))
    params = v7['params']  # {mat: {gs_mult, S0, a45, a90}}

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for j, m in enumerate(MATS):
        p = params[m]
        a = {0: 1.0, 45: p['a45'], 90: p['a90']}
        for psi in ORIS:
            ov = _ov(m, a, p['gs_mult'])
            model = TaylorCPCDM(
                m, psi, S0_override=p['S0'], n_grains=20,
                eps_step=EPS_STEP_316L if m == '316L' else 5e-4,
                n_sub=30, strain_rate=1e-3, model_version='v4',
                force_homogeneous=False, param_overrides=ov, seed=42,
            )
            max_strain = (EXPERIMENTAL[m]['ef'][psi] * 1.6 + 0.02
                          if m == '316L' else EXPERIMENTAL[m]['ef'][psi] * 2.0 + 0.01)
            r = model.run_uniaxial(max_strain)
            strain = np.array(r['strain'])
            damage = np.array(r['damage'])

            ax = axes[0, j]
            ax.plot(strain, damage, linestyle=LINESTYLES[psi], color=COLORS[m],
                    alpha=0.8 if psi == 0 else 0.5, label=ORIENT_LABELS[psi])
            ax.set_xlabel('True Strain')
            ax.set_ylabel('Damage D')
            ax.set_title(MAT_NAMES[m], fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, None)
            ax.set_ylim(0, 0.5)
            if j == 0:
                ax.legend(fontsize=7, loc='upper left')

            dD_de = np.gradient(damage, strain)
            ax2 = axes[1, j]
            ax2.plot(strain, dD_de, linestyle=LINESTYLES[psi], color=COLORS[m],
                     alpha=0.8 if psi == 0 else 0.5)
            ax2.set_xlabel('True Strain')
            ax2.set_ylabel('dD/dε')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, None)

            print(f"  {m} {psi}° done (ef={r['fracture_strain']:.4f})", flush=True)

    fig.suptitle('Damage evolution and rate in AM alloys under different loading orientations',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'Fig1_AM_microstructure_damage.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print('saved Fig1_AM_microstructure_damage.png')


if __name__ == '__main__':
    main()
