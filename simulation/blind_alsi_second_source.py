# -*- coding: utf-8 -*-
"""
blind_alsi_second_source.py — AlSi10Mg 第二源盲测（MDPI Metals 2018, 8, 825）
================================================================================
标定集 AlSi10Mg 源为 wu2023sci_alsi（ef 0.056/0.047/0.037，单调 0>45>90）。
此独立源（Awd 2018 as-built，已换算到模型约定 0°=∥BD）ef 0°/45°/90° = 8.3%/5.7%/8.3%（V形、45°凹陷）：
  0°与接近各向同性(90°≈ 0°)，与标定集单调下降形状不同，但非方向极性的简单反转。
S0=0.20（标定值）直接外推，无重标定。
分辨率：eps_step=5e-4, n_sub=30（AlSi10Mg 主标定档）。
输出: output/blind_alsi_second_source_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'output')
K_G = 0.15
XI0 = 1.5
EXP = {0: 0.083, 45: 0.057, 90: 0.081}  # Awd 2018, 模型约定 0°=∥BD(printed 90°)


def gs_directional(psi):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    h = 1.0 - K_G * (1.0 - xi_eff(XI0, psi) / XI0)   # xi_eff(psi)/XI0
    return {'gs': 280e6 * h}


def main():
    res = {'material': 'AlSi10Mg (LPBF as-built)',
           'dataset': 'Metals 2018, 8, 825 (doi:10.3390/met8100825)',
           'S0_used': 0.20, 'resolution': 'eps_step=5e-4, n_sub=30'}
    rows = []
    for psi in (0, 45, 90):
        ov = gs_directional(psi)
        m = TaylorCPCDM('AlSi10Mg', psi, S0_override=0.20, n_grains=20,
                        eps_step=5e-4, n_sub=30, strain_rate=1e-3,
                        model_version='v4', force_homogeneous=False,
                        param_overrides=ov, seed=42)
        efp = float(m.run_uniaxial(EXP[psi] * 2.0 + 0.01)['fracture_strain'])
        err = abs(efp - EXP[psi]) / EXP[psi] * 100.0
        rows.append({'orientation_deg': psi, 'ef_exp': EXP[psi],
                     'ef_pred': round(efp, 4), 'error_pct': round(err, 2)})
        print(f"psi={psi}: pred {efp:.4f} exp {EXP[psi]:.3f} err {err:.1f}%", flush=True)
    res['cases'] = rows
    res['avg_error_pct'] = round(float(np.mean([r['error_pct'] for r in rows])), 2)
    res['ratio_90_over_0'] = {'pred': round(rows[2]['ef_pred'] / rows[0]['ef_pred'], 2),
                              'exp': round(EXP[90] / EXP[0], 2)}
    with open(os.path.join(OUT, 'blind_alsi_second_source_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print('DONE avg err', res['avg_error_pct'], 'ratio', res['ratio_90_over_0'])


if __name__ == '__main__':
    main()
