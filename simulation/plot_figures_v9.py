"""
v9 图件重绘 — plot_figures_v9.py
===============================
基于 plot_data_v9.json 重绘论文核心图：
- Fig7_stress_strain_v2.png  （9 case 应力应变 + 断裂点）
- Fig8_damage_v2.png         （9 case 损伤演化）
- Fig9_errors_compare.png    （MOD vs LEM 标定误差，v9 纯投影律 vs v9b 公平 LEM）
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), 'output')
FIG = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG, exist_ok=True)

COLORS = {'Ti64': '#E63946', '316L': '#457B9D', 'AlSi10Mg': '#2A9D8F'}
NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
ORIS = [0, 45, 90]
LS = {0: '-', 45: '--', 90: ':'}


def load_v9():
    with open(os.path.join(OUT, 'plot_data_v9.json'), encoding='utf-8') as f:
        return json.load(f)


def fig7(data):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for i, mat in enumerate(['Ti64', '316L', 'AlSi10Mg']):
        for j, ori in enumerate(ORIS):
            ax = axes[i, j]
            d = data[mat][f'{ori}deg']
            strain = np.array(d['strain'])
            stress = np.array(d['stress'])
            damage = np.array(d['damage'])
            # 断裂点（损伤达到 Dc 处）— 用应力归零起点
            nz = np.where(stress > 1.0)[0]
            ef_pred = d['ef_pred']
            ef_exp = d['ef_exp']
            ax.plot(strain, stress, '-', color=COLORS[mat], lw=2.0,
                    label='Modified CDM (pure projection)')
            ax.axvline(ef_pred, color=COLORS[mat], ls=':', alpha=0.7, lw=1.2,
                       label=f'Pred. $\\varepsilon_f$={ef_pred:.3f}')
            ax.axvline(ef_exp, color='black', ls='-.', alpha=0.6, lw=1.2,
                       label=f'Exp. $\\varepsilon_f$={ef_exp:.3f}')
            ax.set_xlabel('True Strain')
            ax.set_ylabel('True Stress [MPa]')
            ax.set_title(f'{NAMES[mat]} - {ori}°')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')
            ax.set_xlim(0, max(ef_exp, ef_pred) * 1.25)
            ax.set_ylim(0, max(stress.max(), 100) * 1.1)
    fig.suptitle('v9 pure-projection-law calibration: simulated vs. experimental '
                 'stress–strain curves (literature g$_s$, single $S_0$ per material)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(FIG, 'Fig7_stress_strain_v2.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print('saved', p)


def fig8(data):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for i, mat in enumerate(['Ti64', '316L', 'AlSi10Mg']):
        for j, ori in enumerate(ORIS):
            ax = axes[i, j]
            d = data[mat][f'{ori}deg']
            strain = np.array(d['strain'])
            damage = np.array(d['damage'])
            ef_exp = d['ef_exp']
            ax.plot(strain, damage, '-', color=COLORS[mat], lw=2.0)
            ax.axvline(ef_exp, color='black', ls='-.', alpha=0.5, lw=1.0)
            ax.set_xlabel('True Strain')
            ax.set_ylabel('Damage $D$')
            ax.set_title(f'{NAMES[mat]} - {ori}°')
            ax.grid(alpha=0.3)
            ax.set_xlim(0, ef_exp * 1.3)
            ax.set_ylim(0, 0.5)
    fig.suptitle('v9 pure-projection-law: damage evolution (single $S_0$, '
                 'literature g$_s$; vertical dash-dot = experimental $\\varepsilon_f$)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(FIG, 'Fig8_damage_v2.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print('saved', p)


def fig9():
    """MOD (v9 纯投影律) vs LEM (v9b 公平同聚合) 标定误差分组柱状图。"""
    with open(os.path.join(OUT, 'experiment_v9_pure_projection_results.json'),
              encoding='utf-8') as f:
        r = json.load(f)
    mm = r['main_calibration']['modified_model']['materials']
    lm = r['main_calibration']['lemaitre_baseline']['materials']
    labels, e_mod, e_lem = [], [], []
    for mat in ['Ti64', '316L', 'AlSi10Mg']:
        for c in mm[mat]['cases']:
            labels.append(f'{NAMES[mat]}\n{c["orientation"]}°')
            e_mod.append(c['error_pct'])
            lem_case = next(x for x in lm[mat]['cases']
                            if x['orientation'] == c['orientation'])
            e_lem.append(lem_case['error_pct'])
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.bar(x - w/2, e_mod, w, label='Modified CDM (pure projection, v9)',
           color='#457B9D')
    ax.bar(x + w/2, e_lem, w, label='Lemaitre (identical aggregate, f$_{AM}\\equiv$1, v9c)',
           color='#E63946', alpha=0.75)
    ax.axhline(10, color='black', ls='--', lw=1.2, label='10% target')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Calibration $\\varepsilon_f$ error [%]')
    ax.set_title('v9 unified parameterization: calibration errors (single $S_0$ '
                 'per material, literature g$_s$; Lemaitre with identical '
                 '20-grain Taylor aggregate and f$_{AM}\\equiv$1)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIG, 'Fig9_errors_compare.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    print('saved', p)
    print('MOD avg:', np.mean(e_mod), 'LEM avg:', np.mean(e_lem))


if __name__ == '__main__':
    data = load_v9()
    fig7(data)
    fig8(data)
    fig9()
