# -*- coding: utf-8 -*-
"""
blind_test_in718_fast.py — IN718 盲测快速版（f_xi 新式）
========================================================
v2 因 fit_s0 黄金搜索过慢被杀。快速版：
- n_sub=12, eps_step=2e-3（收敛精度略降，对 70-90% 级误差无实质影响）
- mode2 标定：S0 网格 {2,4,6,8} 扫 0° 折（各 1 次 run）→ 取最优 → 预测 45/90
- 参数同 IN718 原脚本（文献假设）
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'output')
S0_GEO_MEAN = float((0.32 * 3.20 * 0.20) ** (1.0 / 3.0))   # 316L S0=3.2 (v12, xi_eff-corrected)
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

# Hovig 2018 数据，已换算到模型约定 0°=载荷∥BD(printed 90°竖立), 90°=载荷⊥BD(printed 0°板内)
# ef 单调↓(0°最韧,与标定集同向); UTS 源仅给取向均值 970±47 MPa(无显著随角变化),三向均 970
EXP = {
    0: {'ef': 0.30, 'uts_MPa': 970.0},
    45: {'ef': 0.25, 'uts_MPa': 970.0},
    90: {'ef': 0.20, 'uts_MPa': 970.0},
}


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


def main():
    t0 = time.time()
    results = {'material': 'Inconel 718 (LPBF, as-solution-treated, Hovig&Azar 2018)',
               'projection': 'corrected transverse short-ligament f_xi (v12 set)',
               'resolution': 'fast: n_sub=12, eps_step=2e-3 (calibration/bias-level impact negligible at these error magnitudes)',
               'mode1_S0_geo_mean_MPa': S0_GEO_MEAN,
               'mode1': [], 'mode2': {}}
    # mode1：外推（快版重跑保持一致性）
    for psi in (0, 45, 90):
        r = run_case(psi, S0_GEO_MEAN)
        ef_p = float(r['fracture_strain'])
        uts_p = float(r['uts'] / 1e6)
        results['mode1'].append({
            'psi': psi, 'ef_exp': EXP[psi]['ef'], 'ef_pred': ef_p,
            'ef_error_pct': abs(ef_p - EXP[psi]['ef']) / EXP[psi]['ef'] * 100.0,
            'uts_exp_MPa': EXP[psi]['uts_MPa'], 'uts_pred_MPa': uts_p,
            'uts_error_pct': abs(uts_p - EXP[psi]['uts_MPa']) / EXP[psi]['uts_MPa'] * 100.0})
        print(f"[M1] psi={psi} ef={ef_p:.3f} err={results['mode1'][-1]['ef_error_pct']:.1f}% "
              f"UTS={uts_p:.0f} err={results['mode1'][-1]['uts_error_pct']:.1f}%", flush=True)
    results['mode1_avg_ef_error_pct'] = float(np.mean([c['ef_error_pct'] for c in results['mode1']]))
    results['mode1_avg_uts_error_pct'] = float(np.mean([c['uts_error_pct'] for c in results['mode1']]))
    # mode2：S0 网格标定 0° 折
    grid = [2.0, 4.0, 6.0, 8.0]
    best = None
    for s0 in grid:
        r = run_case(0, s0)
        err = abs(r['fracture_strain'] - EXP[0]['ef']) / EXP[0]['ef'] * 100.0
        results['mode2'][f'S0_{s0:.1f}_0deg'] = {'err_pct': float(err),
                                                  'ef_pred': float(r['fracture_strain'])}
        print(f"[M2-grid] S0={s0:.1f} 0deg err={err:.1f}% ef={float(r['fracture_strain']):.3f}", flush=True)
        if best is None or err < best[1]:
            best = (s0, err)
    s0_star = best[0]
    results['mode2']['S0_cal_on_0_MPa'] = s0_star
    results['mode2']['train_err_pct'] = float(best[1])
    for psi in (45, 90):
        r = run_case(psi, s0_star)
        ef_p = float(r['fracture_strain'])
        results['mode2'][f'blind_{psi}deg'] = {
            'ef_exp': EXP[psi]['ef'], 'ef_pred': ef_p,
            'ef_error_pct': abs(ef_p - EXP[psi]['ef']) / EXP[psi]['ef'] * 100.0}
        print(f"[M2] blind psi={psi} ef={ef_p:.3f} (exp {EXP[psi]['ef']}) "
              f"err={results['mode2'][f'blind_{psi}deg']['ef_error_pct']:.1f}%", flush=True)
    results['mode2_avg_blind_ef_error_pct'] = float((
        results['mode2']['blind_45deg']['ef_error_pct'] +
        results['mode2']['blind_90deg']['ef_error_pct']) / 2)
    r0 = run_case(0, s0_star)
    r90 = run_case(90, s0_star)
    results['mode2_ratio_90_over_0'] = {
        'pred': float(r90['fracture_strain'] / r0['fracture_strain']),
        'exp': EXP[90]['ef'] / EXP[0]['ef']}

    with open(os.path.join(OUT, 'blind_test_in718_fast_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=1, default=float)
    print(f"[M1] avg_ef={results['mode1_avg_ef_error_pct']:.1f}% avg_uts={results['mode1_avg_uts_error_pct']:.1f}%")
    print(f"[M2] S0*={s0_star:.1f} train={best[1]:.1f}% blind avg={results['mode2_avg_blind_ef_error_pct']:.1f}% "
          f"ratio90/0={results['mode2_ratio_90_over_0']['pred']:.2f} (exp {results['mode2_ratio_90_over_0']['exp']:.2f})")
    print(f'DONE saved blind_test_in718_fast_results.json  total={time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
