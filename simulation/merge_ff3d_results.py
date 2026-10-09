# -*- coding: utf-8 -*-
"""
R13: Merge 3D CPFEM results (Ti64 0deg existing + Ti64 45/90, AlSi 0deg expansion)
and generate 4-orientation bias table + updated figure.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = r'D:\20260618断裂模型论文\simulation\output'
FIG = r'D:\20260618断裂模型论文\simulation\figures'

# Existing Ti64 0deg result
existing_path = os.path.join(OUT, 'fullfield_3d_cpfem_results.json')
# Expansion results (Ti64 45/90, AlSi 0deg)
expansion_path = os.path.join(OUT, 'fullfield_3d_cpfem_expansion_results.json')
merged_path = os.path.join(OUT, 'fullfield_3d_cpfem_r13_merged.json')


def load_json(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def main():
    existing = load_json(existing_path)
    expansion = load_json(expansion_path)

    all_results = []
    if existing:
        if isinstance(existing, list):
            all_results.extend(existing)
        else:
            all_results.append(existing)
    if expansion:
        all_results.extend(expansion)

    # Deduplicate by (material, orientation)
    seen = set()
    unique = []
    for r in all_results:
        key = (r.get('material'), r.get('orientation'))
        if key not in seen and 'ef_fullfield' in r:
            seen.add(key)
            unique.append(r)

    # Sort: Ti64 0/45/90, then AlSi10Mg 0
    order = {'Ti64': [0, 45, 90], 'AlSi10Mg': [0, 45, 90], '316L': [0, 45, 90]}
    unique.sort(key=lambda r: (list(order.keys()).index(r['material'])
                               if r['material'] in order else 99,
                               order.get(r['material'], [0, 45, 90]).index(r['orientation'])
                               if r['orientation'] in order.get(r['material'], []) else 99))

    merged = {
        'description': 'R13 merged 3D CPFEM results: Ti64 0/45/90deg + AlSi10Mg 0deg',
        'protocol': {
            'mesh': '3x3x3 hexa8, 27 grains, 2x2x2 Gauss',
            'n_sub': 30, 'eps_step': 0.002, 'seed': 42,
            'fracture_criterion': 'D_agg >= Dc or localization onset',
            'note': 'Same constitutive law as TaylorCPCDM; only difference is full-field equilibrium vs iso-strain',
        },
        'cases': unique,
        'bias_summary': {},
    }

    # Compute bias summary
    for r in unique:
        mat = r['material']
        ori = r['orientation']
        key = f'{mat}_{ori}'
        if r.get('ef_taylor') and r['ef_taylor'] > 0:
            bias = (r['ef_fullfield'] - r['ef_taylor']) / r['ef_taylor'] * 100.0
            merged['bias_summary'][key] = {
                'ef_fullfield': r['ef_fullfield'],
                'ef_taylor': r['ef_taylor'],
                'bias_pct': round(bias, 2),
                'Dc': r.get('Dc'),
                'fracture_type': r.get('fracture_type', 'N/A'),
            }

    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=1)
    print(f'Merged {len(unique)} cases -> {merged_path}')

    # Print bias table
    print('\n=== 3D CPFEM 4-Orientation Bias Table ===')
    print(f'{"Material":<12} {"Ori":>4} {"ef_ff":>8} {"ef_Taylor":>10} {"Bias%":>8} {"Dc":>6} {"Type":<20}')
    print('-' * 75)
    for r in unique:
        if 'ef_fullfield' in r:
            bias = r.get('bias_pct', 'N/A')
            print(f'{r["material"]:<12} {r["orientation"]:>4}° {r["ef_fullfield"]:>8.4f} '
                  f'{r.get("ef_taylor", 0):>10.4f} {str(bias):>8} '
                  f'{r.get("Dc", "N/A"):>6} {r.get("fracture_type", "N/A"):<20}')

    # Generate figure if we have at least 2 cases
    if len(unique) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)

        # Left: bar chart of ef_ff vs ef_Taylor
        labels = []
        ff_vals = []
        t_vals = []
        for r in unique:
            if 'ef_fullfield' in r and r.get('ef_taylor'):
                labels.append(f"{r['material'].replace('AlSi10Mg','AlSi')}\n{r['orientation']}°")
                ff_vals.append(r['ef_fullfield'])
                t_vals.append(r['ef_taylor'])

        x = np.arange(len(labels))
        w = 0.35
        ax = axes[0]
        ax.bar(x - w/2, ff_vals, w, label='3D CPFEM (full-field)', color='#4C78A8', edgecolor='k', linewidth=0.5)
        ax.bar(x + w/2, t_vals, w, label='TaylorCPCDM (iso-strain)', color='#F58518', edgecolor='k', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Fracture strain $\\varepsilon_f$ (-)', fontsize=10)
        ax.set_title('3D Full-Field vs Taylor: Fracture Strain', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

        # Right: bias percentage
        ax2 = axes[1]
        biases = []
        blabels = []
        for r in unique:
            if 'ef_fullfield' in r and r.get('ef_taylor') and r['ef_taylor'] > 0:
                b = (r['ef_fullfield'] - r['ef_taylor']) / r['ef_taylor'] * 100.0
                biases.append(b)
                blabels.append(f"{r['material'].replace('AlSi10Mg','AlSi')}\n{r['orientation']}°")
        colors = ['#54A24B' if b >= 0 else '#E45756' for b in biases]
        bars = ax2.bar(range(len(biases)), biases, color=colors, edgecolor='k', linewidth=0.5, width=0.5)
        ax2.axhline(y=0, color='k', linewidth=0.8)
        ax2.set_xticks(range(len(blabels)))
        ax2.set_xticklabels(blabels, fontsize=9)
        ax2.set_ylabel('Taylor bias (\\%)', fontsize=10)
        ax2.set_title('Taylor Iso-Strain Bias\n(positive = Taylor under-predicts)', fontsize=10)
        ax2.grid(axis='y', linestyle=':', alpha=0.5)
        for bar, b in zip(bars, biases):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     b + (1.5 if b >= 0 else -2.5),
                     f'{b:+.1f}%', ha='center', va='bottom' if b >= 0 else 'top',
                     fontsize=9, fontweight='bold')
        for s in ('top', 'right'):
            ax2.spines[s].set_visible(False)

        plt.tight_layout()
        fig_path = os.path.join(FIG, 'ff3d_r13_4orientation_bias.png')
        plt.savefig(fig_path, bbox_inches='tight')
        print(f'\nFigure saved: {fig_path}')

    # Experimental comparison (if available in EXPERIMENTAL)
    try:
        sys_path = os.path.dirname(__file__)
        import sys
        sys.path.insert(0, os.path.join(sys_path, 'src'))
        from materials import EXPERIMENTAL
        print('\n=== Comparison with Experimental ef ===')
        print(f'{"Material":<12} {"Ori":>4} {"ef_exp":>8} {"ef_ff":>8} {"ff_err%":>8} {"ef_Taylor":>10} {"T_err%":>8}')
        print('-' * 70)
        for r in unique:
            mat = r['material']
            ori = r['orientation']
            if mat in EXPERIMENTAL and ori in EXPERIMENTAL[mat]['ef']:
                ef_exp = EXPERIMENTAL[mat]['ef'][ori]
                ff_err = abs(r['ef_fullfield'] - ef_exp) / ef_exp * 100
                t_err = abs(r.get('ef_taylor', 0) - ef_exp) / ef_exp * 100 if r.get('ef_taylor') else None
                print(f'{mat:<12} {ori:>4}° {ef_exp:>8.4f} {r["ef_fullfield"]:>8.4f} '
                      f'{ff_err:>7.1f}% {r.get("ef_taylor", 0):>10.4f} '
                      f'{t_err:>7.1f}%' if t_err else f'{mat:<12} {ori:>4}° {ef_exp:>8.4f} {r["ef_fullfield"]:>8.4f} {ff_err:>7.1f}%')
    except Exception as e:
        print(f'Experimental comparison skipped: {e}')


if __name__ == '__main__':
    main()
