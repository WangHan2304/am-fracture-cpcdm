"""重新生成 Fig9_errors_compare.png（修正模型 vs Lemaitre 基线分组柱状图）

作用：在 experiment_v2_results.json（修正模型正典结果）更新后，直接用两个
JSON 重绘对比图，无需重跑耗时的基线仿真（基线结果确定性不变，直接读取已有
lemaitre_baseline_results.json）。绘图样式与 lemaitre_baseline.py 保持一致。

用法：
    python regenerate_fig9.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
FIG = os.path.join(HERE, 'figures')

MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]
MAT_NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
COLORS = {'Ti64': '#E63946', '316L': '#457B9D', 'AlSi10Mg': '#2A9D8F'}

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def main():
    with open(os.path.join(OUT, 'experiment_v2_results.json'),
              'r', encoding='utf-8') as f:
        mod = json.load(f)
    with open(os.path.join(OUT, 'lemaitre_baseline_results.json'),
              'r', encoding='utf-8') as f:
        base = json.load(f)

    labels, errs_mod, errs_lem, bar_colors = [], [], [], []
    for mat in MATERIALS:
        for orient in ORIENTS:
            labels.append(f"{MAT_NAMES[mat]}\n{orient}°")
            errs_mod.append(mod['cases'][mat][str(orient)]['error_pct'])
            errs_lem.append(base['cases'][mat][str(orient)]['error_pct'])
            bar_colors.append(COLORS[mat])

    max_err = max(errs_lem)
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, errs_mod, w, label='Modified CDM (this work)',
                color='#457B9D', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w / 2, errs_lem, w, label='Conventional Lemaitre',
                color='#E63946', edgecolor='black', linewidth=0.5)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='10% target')

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.6,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Fracture strain prediction error [%]')
    ax.set_ylim(0, max_err * 1.25)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'Fig9_errors_compare.png'))
    plt.close(fig)

    print(f"已重绘: {FIG}/Fig9_errors_compare.png")
    print(f"  修正模型 9 工况平均误差: {np.mean(errs_mod):.2f}%")
    print(f"  基线 9 工况平均误差:     {np.mean(errs_lem):.2f}%")


if __name__ == '__main__':
    main()