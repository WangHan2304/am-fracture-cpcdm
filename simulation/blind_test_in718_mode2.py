# -*- coding: utf-8 -*-
"""
IN718 材料内盲测（Mode 2）— blind_test_in718_mode2.py
=====================================================
与主协议一致：每材料只标定 S0（一个可调参数），留出方向盲预测。
- 用 IN718 0° 数据（ef=0.30, Hovig&Azar Fig.15 读值, 已换算到模型约定 0°=载荷∥BD）二分标定 S0
- 用标定的 S0 盲预测 45° 与 90°（未参与标定）
- 同时报告与主 LOOCV 协议完全同构的"材料内留方向预测"

输出: output/blind_test_in718_mode2_results.json
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
XI0_IN718 = 3.0

IN718 = {
    'C11': 239e9, 'C12': 145e9, 'C44': 112e9,
    'g0': 165e6, 'gs': 280e6, 'h0': 1200e6, 'a': 2.2,
    'n_rate': 25.0, 'gamma_dot_0': 0.001, 'q_lat': 1.4,
    's_damage': 1.0, 'p_D': 0.03,
    'D0': 0.0003, 'phi': 0.0003, 'phi_crit': 0.02,
    'lambda_mp': 40.0, 'xi_grain': 3.0, 'theta_tex': 0.60,
    'beta1': 2.0, 'beta2': 1.5, 'beta3': 10.0, 'beta4': 0.8, 'beta5': 0.2,
    'm1': 2.0, 'm2': 1.5, 'm3': 1.0,
}

EXP = {0: 0.30, 45: 0.25, 90: 0.20}  # Hovig 印刷角相对建造板, 已换算到模型约定(0°=载荷∥BD), 与 fast 脚本一致


def gs_h(psi, kg=K_G):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(XI0_IN718, psi) / XI0_IN718   # xi_eff(psi)/XI0_IN718
    return 1.0 - kg * (1.0 - ratio)


def run_case(psi, S0, max_strain=0.40):
    ov = dict(IN718)
    ov['S0'] = S0
    ov['gs'] = ov['gs'] * gs_h(psi)
    model = TaylorCPCDM(
        '316L', psi, S0_override=S0, n_grains=20,
        eps_step=2e-3, n_sub=12, strain_rate=1e-3,
        model_version='v4', force_homogeneous=False,
        param_overrides=ov, seed=42,
    )
    return model.run_uniaxial(max_strain)


def calibrate_s0(psi, target_ef, grid=(2.0, 4.0, 6.0, 8.0)):
    """与 blind_test_in718_fast.py mode2 完全同构：在粗网格上选使 0° 误差最小的 S0（非二分）。"""
    best_s0, best_err, best_ef = grid[0], float('inf'), None
    for s0 in grid:
        r = run_case(psi, s0)
        ef = float(r['fracture_strain'])
        err = abs(ef - target_ef) / target_ef
        if err < best_err:
            best_s0, best_err, best_ef = s0, err, ef
    return best_s0, best_ef


def main():
    # 标定：IN718 0° 数据 → S0*
    s0_star, ef_cal = calibrate_s0(0, EXP[0])
    print(f"calibrated S0* = {s0_star:.3f} MPa (0° ef={ef_cal:.3f} vs target {EXP[0]})", flush=True)
    results = {'material': 'Inconel 718 (mode 2: material-internal LOOCV, S0 calibrated on 0° only)',
               'S0_calibrated_MPa': s0_star,
               'protocol_note': ('Identical to main protocol: single tunable S0 per material, '
                                 'leave-one-direction-out blind prediction; IN718 participates only '
                                 'through its 0° datum; 45°/90° are genuinely blind.'),
               'cases': []}
    for psi in (0, 45, 90):
        r = run_case(psi, s0_star)
        ef_p = float(r['fracture_strain'])
        uts_p = float(r['uts'] / 1e6)
        blind = psi != 0
        results['cases'].append({
            'psi': psi, 'blind': blind,
            'ef_exp': EXP[psi], 'ef_pred': ef_p,
            'ef_error_pct': abs(ef_p - EXP[psi]) / EXP[psi] * 100.0,
            'uts_pred_MPa': uts_p,
        })
        print(f"psi={psi:>2d} {'blind' if blind else 'calib'} ef_pred={ef_p:.3f} "
              f"(exp {EXP[psi]}) UTS_pred={uts_p:.0f}", flush=True)

    blind_cases = [c for c in results['cases'] if c['blind']]
    results['blind_avg_ef_error_pct'] = float(np.mean([c['ef_error_pct'] for c in blind_cases]))
    results['ef_direction_ratio_90_over_0_pred'] = results['cases'][2]['ef_pred'] / results['cases'][0]['ef_pred']
    results['ef_direction_ratio_90_over_0_exp'] = EXP[90] / EXP[0]
    with open(os.path.join(OUT, 'blind_test_in718_mode2_results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print(f"blind(45/90) avg_ef_error={results['blind_avg_ef_error_pct']:.1f}%  "
          f"ratio90/0 pred={results['ef_direction_ratio_90_over_0_pred']:.2f} exp={results['ef_direction_ratio_90_over_0_exp']:.2f}")
    print('DONE saved blind_test_in718_mode2_results.json')


if __name__ == '__main__':
    main()
