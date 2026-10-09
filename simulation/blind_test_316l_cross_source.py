# -*- coding: utf-8 -*-
"""
blind_test_316l_cross_source.py — 316L 跨源盲测（R8 C 组，改用 Hitzler 2017 源）
============================================================
独立源：Hitzler et al. 2017, Materials 10(10):1136（SLM 316L, as-built, 已机加工表面）。
已换算到模型约定(0°=载荷∥BD): ef 0°/45°/90° = 0.1176 / 0.3256 / 0.3324。
该源为真·反向趋势(0° 最脆、90° 最韧, 与标定集 0.37/0.31/0.255 的 0° 最韧相反)。
(原 pmc11721696 源因同一标签的正文值与重数字化值不可调和而被数据审计排除, 故弃用。)
协议：慢分辨率（n_sub=30, eps_step=1e-3）; S0 网格 {0.5,1,2,3,4} 标定 0° 折 → 预测 45°/90°。
描述符沿用 v12 316L、kg=0.15、新 f_xi。
输出: output/blind_test_316l_cross_source_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'output')
K_G = 0.15
XI0 = 2.5  # 316L

EXP_XS = {0: 0.1176, 45: 0.3256, 90: 0.3324}  # Hitzler 2017 源, 模型约定 0°=∥BD


def gs_directional(psi):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(XI0, psi) / XI0   # xi_eff(psi)/XI0
    h = 1.0 - K_G * (1.0 - ratio)
    return {'gs': 500e6 * h}


def run_case(psi, S0, max_strain=0.55):
    ov = gs_directional(psi)
    m = TaylorCPCDM('316L', psi, S0_override=S0, n_grains=20,
                    eps_step=1e-3, n_sub=30, strain_rate=1e-3,
                    model_version='v4', force_homogeneous=False,
                    param_overrides=ov, seed=42)
    return m.run_uniaxial(max_strain)


def main():
    results = {'material': '316L SS (SLM, as-built milled surfaces; Hitzler 2017 Materials 10:1136)',
               'note': ('Cross-source blind test: independent literature source, genuine reversed '
                        'direction trend vs calibration set (0 deg least ductile here, opposite to the '
                        'columnar-dominated calibration trend); protocol: slow resolution (n_sub=30, '
                        'eps_step=1e-3), S0 grid-calibrated on the 0 deg fold only, then blind 45/90 deg '
                        'predictions.'),
               'exp_ef': EXP_XS, 'grid': {}, 'blind': {}}
    grid = [0.5, 1.0, 2.0, 3.0, 4.0]
    best = None
    for s0 in grid:
        r = run_case(0, s0)
        efp = float(r['fracture_strain'])
        err = abs(efp - EXP_XS[0]) / EXP_XS[0] * 100.0
        results['grid'][str(s0)] = {'ef_pred_0deg': efp, 'err_pct_0deg': float(err)}
        print(f"grid S0={s0:.1f} 0deg ef={efp:.4f} err={err:.1f}%", flush=True)
        if best is None or err < best[1]:
            best = (s0, err)
    s0_star = best[0]
    results['S0_cal_on_0_MPa'] = s0_star
    results['train_err_pct'] = float(best[1])
    errs = []
    for psi in (45, 90):
        r = run_case(psi, s0_star)
        efp = float(r['fracture_strain'])
        err = abs(efp - EXP_XS[psi]) / EXP_XS[psi] * 100.0
        errs.append(err)
        results['blind'][f'{psi}deg'] = {'ef_exp': EXP_XS[psi], 'ef_pred': efp,
                                          'err_pct': float(err)}
        print(f"blind {psi}deg ef={efp:.4f} (exp {EXP_XS[psi]}) err={err:.1f}%", flush=True)
    results['avg_blind_ef_error_pct'] = float(np.mean(errs))
    results['direction_ratio_90_over_0'] = {
        'pred': float(results['blind']['90deg']['ef_pred'] /
                      results['grid'][str(s0_star)]['ef_pred_0deg']),
        'exp': EXP_XS[90] / EXP_XS[0]}
    with open(os.path.join(OUT, 'blind_test_316l_cross_source_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1)
    print(f"S0*={s0_star:.1f} train={best[1]:.1f}% avg_blind={results['avg_blind_ef_error_pct']:.1f}% "
          f"ratio90/0 pred={results['direction_ratio_90_over_0']['pred']:.2f} "
          f"exp={results['direction_ratio_90_over_0']['exp']:.2f}")
    print('DONE')


if __name__ == '__main__':
    main()
