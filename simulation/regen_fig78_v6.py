"""
v6 版 Fig7/8 重绘 — regen_fig78_v6.py
======================================
用 v6 正典参数（S0/a45/a90，来自 experiment_v6_results.json）重跑九工况，
重绘：
  Fig7_stress_strain_v2.png  （应力-应变曲线 3×3，手稿引用名）
  Fig8_damage_v2.png         （损伤演化曲线 3×3）
覆盖 figures/ 下同名文件（内容为 v6 公式结构的结果）。
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
ORIENT_LABELS = {0: '0°', 45: '45°', 90: '90°'}
COLORS = {'Ti64': '#4C72B0', '316L': '#55A868', 'AlSi10Mg': '#C44E52'}
EPS_STEP_316L = 1e-3


def main():
    res = json.load(open(os.path.join(OUT_DIR, 'experiment_v6_results.json'),
                         encoding='utf-8'))
    S0 = {c['material']: c['S0'] for c in res['cases']}
    a = {}
    for c in res['cases']:
        a.setdefault(c['material'], {})[c['orientation']] = c['a_aniso']

    mats = ['Ti64', '316L', 'AlSi10Mg']
    oris = [0, 45, 90]

    # ---- 重跑九工况，收集曲线 ----
    results = {}
    for m in mats:
        results[m] = {}
        for psi in oris:
            eps_step = EPS_STEP_316L if m == '316L' else 5e-4
            max_strain = (EXPERIMENTAL[m]['ef'][psi] * 1.6 + 0.02
                          if m == '316L' else EXPERIMENTAL[m]['ef'][psi] * 2.0 + 0.01)
            model = TaylorCPCDM(
                m, psi, S0_override=S0[m], n_grains=20,
                eps_step=eps_step, n_sub=30, strain_rate=1e-3,
                model_version='v4', force_homogeneous=False,
                param_overrides={'aniso_multiplier': {0: 1.0, 45: a[m][45], 90: a[m][90]}},
                seed=42,
            )
            r = model.run_uniaxial(max_strain)
            results[m][psi] = r
            print(f'{m} {psi}deg done (ef={r["fracture_strain"]:.4f})')

    # ---- Fig 7: 应力应变曲线 ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for i, m in enumerate(mats):
        for j, psi in enumerate(oris):
            ax = axes[i, j]
            r = results[m][psi]
            ax.plot(r['strain'], np.array(r['stress']) / 1e6, '-',
                    color=COLORS[m], linewidth=2.2)
            ef_exp = EXPERIMENTAL[m]['ef'][psi]
            ef_pred = r['fracture_strain']
            ax.axvline(x=ef_exp, color='black', linestyle='-.', alpha=0.7,
                       linewidth=1.2, label=f"Exp εf={ef_exp:.3f}")
            ax.axvline(x=ef_pred, color=COLORS[m], linestyle=':', alpha=0.7,
                       linewidth=1.2, label=f"Pred εf={ef_pred:.3f}")
            ax.set_xlabel('True Strain')
            ax.set_ylabel('True Stress [MPa]')
            ax.set_title(f"{MAT_NAMES[m]} — {ORIENT_LABELS[psi]}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(ef_exp * 1.3, ef_pred * 1.3))
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Fig. 7: Stress–strain curves — modified CDM model (v6)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'Fig7_stress_strain_v2.png'), dpi=300)
    plt.close(fig)
    print('saved Fig7_stress_strain_v2.png')

    # ---- Fig 8: 损伤演化 ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for i, m in enumerate(mats):
        for j, psi in enumerate(oris):
            ax = axes[i, j]
            r = results[m][psi]
            ax.plot(r['strain'], r['damage'], '-', color=COLORS[m], linewidth=2.2)
            Dc = r.get('Dc', 0.02)
            ax.axhline(y=Dc, color='red', linestyle='--', alpha=0.7, linewidth=1,
                       label=f"Dc={Dc:.2f}")
            ax.set_xlabel('True Strain')
            ax.set_ylabel('Damage D')
            ax.set_title(f"{MAT_NAMES[m]} — {ORIENT_LABELS[psi]}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, EXPERIMENTAL[m]['ef'][psi] * 1.3)
            ax.set_ylim(0, max(Dc * 1.3, 0.5))
            if i == 0 and j == 2:
                ax.legend(fontsize=7)
    fig.suptitle('Fig. 8: Damage evolution — modified CDM model (v6)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'Fig8_damage_v2.png'), dpi=300)
    plt.close(fig)
    print('saved Fig8_damage_v2.png')


if __name__ == '__main__':
    main()
