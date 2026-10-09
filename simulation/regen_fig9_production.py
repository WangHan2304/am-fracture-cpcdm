# -*- coding: utf-8 -*-
"""
regen_fig9_production.py
========================
Regenerate the main-text Fig9_errors_compare.png from the leakage-free
PRODUCTION calibration (experiment_v12_fxi_transverse_results.json):
9 per-case fracture-strain calibration errors, modified CDM model vs the
equal-parameter Lemaitre baseline. Uses the corrected grain-morphology
projection xi_eff, so it must be re-run after that correction.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'output')
FIG_DIR = os.path.join(HERE, 'figures')

MAT_NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
MATS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]


def case_err_map(node):
    out = {}
    for m in MATS:
        for c in node['materials'][m]['cases']:
            out[(m, c['orientation'])] = c['error_pct']
    return out


def main():
    j = json.load(open(os.path.join(OUT_DIR, 'experiment_v12_fxi_transverse_results.json'), encoding='utf-8'))
    mc = j['main_calibration']
    e_mod = case_err_map(mc['modified_model'])
    e_lem = case_err_map(mc['lemaitre_baseline'])

    labels, emod, elem = [], [], []
    for m in MATS:
        for o in ORIENTS:
            labels.append('%s\n%d\u00b0' % (MAT_NAMES[m], o))
            emod.append(e_mod[(m, o)])
            elem.append(e_lem[(m, o)])

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    b1 = ax.bar(x - w / 2, emod, w, color='#4C72B0', edgecolor='black', linewidth=0.6, label='Modified CDM model')
    b2 = ax.bar(x + w / 2, elem, w, color='#C44E52', edgecolor='black', linewidth=0.6, label='Lemaitre baseline')
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='10% threshold')
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2., b.get_height() + 0.3, '%.1f' % b.get_height(),
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Fracture Strain Calibration Error [%]')
    ax.set_title('Calibration accuracy across the nine cases (leakage-free production) - '
                 'modified CDM model vs. Lemaitre baseline')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(emod), max(elem)) * 1.18)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'Fig9_errors_compare.png')
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print('saved', out)
    print('mod cal avg = %.2f  lem cal avg = %.2f' % (np.mean(emod), np.mean(elem)))


if __name__ == '__main__':
    main()
