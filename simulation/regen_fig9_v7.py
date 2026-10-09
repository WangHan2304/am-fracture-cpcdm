"""
v7 版 Fig9 重绘 — regen_fig9_v7.py
====================================
用 experiment_v7_results.json（修正模型）与 lemaitre_baseline_results.json
（经典基线）重绘九案例断裂应变预测误差对比柱状图，输出
figures/Fig9_errors_compare.png（手稿引用文件名）。
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

MAT_NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
ORIENTS = [0, 45, 90]
COLORS = {'Ti64': '#4C72B0', '316L': '#55A868', 'AlSi10Mg': '#C44E52'}


def main():
    v6 = json.load(open(os.path.join(OUT_DIR, 'experiment_v7_final_results.json'),
                        encoding='utf-8'))
    base = json.load(open(os.path.join(OUT_DIR, 'lemaitre_baseline_results.json'),
                          encoding='utf-8'))

    err_v6 = {}
    for c in v6['cases']:
        err_v6[(c['material'], c['orientation'])] = c['error_pct']
    err_base = {}
    for m in ['Ti64', '316L', 'AlSi10Mg']:
        for o in ['0', '45', '90']:
            err_base[(m, int(o))] = base['cases'][m][o]['error_pct']

    labels, e6, eb, colors = [], [], [], []
    for m in ['Ti64', '316L', 'AlSi10Mg']:
        for o in ORIENTS:
            labels.append(f"{MAT_NAMES[m]}\n{o}°")
            e6.append(err_v6[(m, o)])
            eb.append(err_base[(m, o)])
            colors.append(COLORS[m])

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w / 2, e6, w, color='#4C72B0', edgecolor='black',
                linewidth=0.6, label='Modified CDM model')
    b2 = ax.bar(x + w / 2, eb, w, color='#C44E52', edgecolor='black',
                linewidth=0.6, label='Lemaitre baseline')
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='10% threshold')
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.3,
                f'{b.get_height():.1f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#2c3e6b')
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.3,
                f'{b.get_height():.1f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#7d2a2a')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Fracture Strain Calibration Error [%]')
    ax.set_title('Calibration accuracy across the nine cases (v7) — '
                 'modified CDM model vs. Lemaitre baseline')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(eb) * 1.18, 12))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'Fig9_errors_compare.png'), dpi=300)
    plt.close(fig)
    print('saved figures/Fig9_errors_compare.png')


if __name__ == '__main__':
    main()
