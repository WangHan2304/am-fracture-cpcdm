# -*- coding: utf-8 -*-
"""
regen_fig78_production.py
=========================
Regenerate the main-text Fig7_stress_strain_v2.png and Fig8_damage_v2.png from
the leakage-free PRODUCTION calibration.

These plots are the modified-CDM-model stress-strain and damage trajectories for
the nine calibration cases (3 materials x 3 build orientations). They depend on
the corrected grain-morphology projection xi_eff (through _gs_directional at the
45 deg orientation) and on the frozen single-S0-per-material production
calibration, so they must be re-run after the xi_eff correction.

To guarantee an exact match with the reported numbers we reuse the production
solver entry point experiment_v12_fxi_transverse.run_case (corrected xi_eff,
literature g_s, single S0, no orientation multipliers) and read the frozen S0
from experiment_v12_fxi_transverse_results.json.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from experiment_v12_fxi_transverse import (  # noqa: E402
    run_case, MATERIALS, ORIENTS, PATH as V12_PATH,
)

FIG_DIR = os.path.join(HERE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

COLORS = {'Ti64': '#E63946', '316L': '#457B9D', 'AlSi10Mg': '#2A9D8F'}
NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}


def production_S0():
    j = json.load(open(V12_PATH, encoding='utf-8'))
    mm = j['main_calibration']['modified_model']['materials']
    return {m: mm[m]['S0'] for m in MATERIALS}


def collect_curves(S0):
    curves = {}
    for mat in MATERIALS:
        for ori in ORIENTS:
            r = run_case(mat, ori, S0[mat], homogeneous=False)
            curves[(mat, ori)] = {
                'strain': np.asarray(r['strain']),
                'stress': np.asarray(r['stress']),
                'damage': np.asarray(r['damage']),
                'ef_pred': float(r['fracture_strain']),
                'ef_exp': float(EXPERIMENTAL[mat]['ef'][ori]),
            }
    return curves


def fig7(curves):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for i, mat in enumerate(MATERIALS):
        for j, ori in enumerate(ORIENTS):
            ax = axes[i, j]
            d = curves[(mat, ori)]
            strain, stress = d['strain'], d['stress']
            ef_pred, ef_exp = d['ef_pred'], d['ef_exp']
            ax.plot(strain, stress, '-', color=COLORS[mat], lw=2.0,
                    label='Modified CDM (production)')
            ax.axvline(ef_pred, color=COLORS[mat], ls=':', alpha=0.8, lw=1.4,
                       label=f'Pred. $\\varepsilon_f$={ef_pred:.3f}')
            ax.axvline(ef_exp, color='black', ls='-.', alpha=0.6, lw=1.2,
                       label=f'Exp. $\\varepsilon_f$={ef_exp:.3f}')
            ax.set_xlabel('True Strain')
            ax.set_ylabel('True Stress [MPa]')
            ax.set_title(f'{NAMES[mat]} - {ori}\u00b0')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')
            ax.set_xlim(0, max(ef_exp, ef_pred) * 1.25)
            ax.set_ylim(0, max(stress.max(), 100) * 1.1)
    fig.suptitle('Leakage-free production calibration: simulated vs. experimental '
                 'stress\u2013strain curves (corrected $\\xi_{{eff}}$, literature $g_s$, '
                 'single $S_0$ per material)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(FIG_DIR, 'Fig7_stress_strain_v2.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', p)


def fig8(curves):
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for i, mat in enumerate(MATERIALS):
        for j, ori in enumerate(ORIENTS):
            ax = axes[i, j]
            d = curves[(mat, ori)]
            strain, damage = d['strain'], d['damage']
            ef_exp, ef_pred = d['ef_exp'], d['ef_pred']
            ax.plot(strain, damage, '-', color=COLORS[mat], lw=2.0)
            ax.axvline(ef_exp, color='black', ls='-.', alpha=0.5, lw=1.0)
            ax.axvline(ef_pred, color=COLORS[mat], ls=':', alpha=0.7, lw=1.0)
            ax.set_xlabel('True Strain')
            ax.set_ylabel('Damage $D$')
            ax.set_title(f'{NAMES[mat]} - {ori}\u00b0')
            ax.grid(alpha=0.3)
            ax.set_xlim(0, ef_exp * 1.3)
            ax.set_ylim(0, 0.5)
    fig.suptitle('Leakage-free production: damage evolution (corrected $\\xi_{{eff}}$, '
                 'single $S_0$, literature $g_s$; dash-dot = experimental '
                 '$\\varepsilon_f$, dotted = predicted)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    p = os.path.join(FIG_DIR, 'Fig8_damage_v2.png')
    fig.savefig(p, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', p)


def main():
    S0 = production_S0()
    print('production S0:', S0)
    curves = collect_curves(S0)
    fig7(curves)
    fig8(curves)
    print('per-case predicted ef:')
    for mat in MATERIALS:
        print('  ', mat, {o: round(curves[(mat, o)]['ef_pred'], 4) for o in ORIENTS},
              'exp', {o: EXPERIMENTAL[mat]['ef'][o] for o in ORIENTS})


if __name__ == '__main__':
    main()
