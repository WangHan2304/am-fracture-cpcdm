# -*- coding: utf-8 -*-
"""
第五轮⑤：Inconel 718 独立文献盲测 — blind_test_in718.py
========================================================
响应审稿"补充独立实验或公开数据集盲测"。选 Inconel 718（LPBF，固溶态 set 1，
Hovig & Azar, Adv. Mater. Sci. Eng. 2018, Hindawi 7650303）：
- 未参与任何校准（新材料、新数据源、FCC）
- 全部材料参数来自文献假设/估算（弹性、Voce、微观描述符），无一根拟合
- S0 不标定：取 3 个校准材料 S0 的跨材料几何平均（0.32*3.20*0.20)^(1/3)=0.585 MPa
  （明确标注为外推假设；同时报告 S0 无关的方向比预测）
- 实验值（图 15 读值，标注不确定度）：ef ~0.20/0.25/0.30（0/45/90°），
  UTS ~950/970/990 MPa（set 1, as-solution-treated）
- 用 316L 的 FCC 骨架（滑移系统/聚合），param_overrides 覆盖全部 IN718 参数

输出: output/blind_test_in718_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'output')
S0_GEO_MEAN = float((0.32 * 3.20 * 0.20) ** (1.0 / 3.0))
K_G = 0.15
XI0_IN718 = 3.0

# 文献/估算参数（全部 assumed，无拟合）——来源见脚本头注释
IN718 = {
    'C11': 239e9, 'C12': 145e9, 'C44': 112e9,      # 单晶弹性（文献 235-242/139-153/104-125 GPa）
    'g0': 165e6, 'gs': 280e6, 'h0': 1200e6, 'a': 2.2,   # Voce（固溶态 YS~504 MPa, UTS~970 MPa）
    'n_rate': 25.0, 'gamma_dot_0': 0.001, 'q_lat': 1.4,
    's_damage': 1.0, 'p_D': 0.03,                  # 损伤（与校准材料同构）
    'D0': 0.0003, 'phi': 0.0003, 'phi_crit': 0.02,  # 相对密度 99.97-99.98%（Hindawi）
    'lambda_mp': 40.0, 'xi_grain': 3.0, 'theta_tex': 0.60,  # 柱状晶 <100>||BD 强织构（EBSD 报告）
    'beta1': 2.0, 'beta2': 1.5, 'beta3': 10.0, 'beta4': 0.8, 'beta5': 0.2,
    'm1': 2.0, 'm2': 1.5, 'm3': 1.0,
}

# 实验（Hindawi set 1, Fig. 15 图读值；elongation at break 工程应变）
EXP = {
    0: {'ef': 0.20, 'uts_MPa': 950.0},
    45: {'ef': 0.25, 'uts_MPa': 970.0},
    90: {'ef': 0.30, 'uts_MPa': 990.0},
}


def gs_h(psi, kg=K_G):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(XI0_IN718, psi) / XI0_IN718   # xi_eff(psi)/XI0_IN718
    return 1.0 - kg * (1.0 - ratio)


def run_case(psi, S0, max_strain):
    ov = dict(IN718)
    ov['S0'] = S0
    ov['gs'] = ov['gs'] * gs_h(psi)
    model = TaylorCPCDM(
        '316L', psi, S0_override=S0, n_grains=20,
        eps_step=1e-3, n_sub=30, strain_rate=1e-3,
        model_version='v4', force_homogeneous=False,
        param_overrides=ov, seed=42,
    )
    return model.run_uniaxial(max_strain)


def main():
    results = {'material': 'Inconel 718 (LPBF, as-solution-treated, Hovig&Azar 2018)',
               'S0_geo_mean_MPa': S0_GEO_MEAN,
               'k_g': K_G,
               'note': ('All material parameters are literature-assumed/estimated; S0 is the '
                        'cross-material geometric mean of the three calibrated S0 values '
                        '(0.32/3.20/0.20 MPa), i.e. a genuine out-of-sample extrapolation; '
                        'experimental values are read from Fig. 15 of the source (uncertainty ~±10%).'),
               'cases': []}
    for psi in (0, 45, 90):
        r = run_case(psi, S0_GEO_MEAN, max_strain=0.40)
        ef_p = float(r['fracture_strain'])
        uts_p = float(r['uts'] / 1e6)
        results['cases'].append({
            'psi': psi,
            'ef_exp': EXP[psi]['ef'],
            'ef_pred': ef_p,
            'ef_error_pct': abs(ef_p - EXP[psi]['ef']) / EXP[psi]['ef'] * 100.0,
            'uts_exp_MPa': EXP[psi]['uts_MPa'],
            'uts_pred_MPa': uts_p,
            'uts_error_pct': abs(uts_p - EXP[psi]['uts_MPa']) / EXP[psi]['uts_MPa'] * 100.0,
        })
        print(f"psi={psi:>2d}  ef_pred={ef_p:.3f} (exp {EXP[psi]['ef']})  "
              f"UTS_pred={uts_p:.0f} (exp {EXP[psi]['uts_MPa']})", flush=True)

    ef_pred = [c['ef_pred'] for c in results['cases']]
    ef_exp = [c['ef_exp'] for c in results['cases']]
    ratio_pred = ef_pred[2] / ef_pred[0]
    ratio_exp = ef_exp[2] / ef_exp[0]
    results['direction_ratio_90_over_0'] = {'pred': ratio_pred, 'exp': ratio_exp}
    results['avg_ef_error_pct'] = float(np.mean([c['ef_error_pct'] for c in results['cases']]))
    results['avg_uts_error_pct'] = float(np.mean([c['uts_error_pct'] for c in results['cases']]))
    with open(os.path.join(OUT, 'blind_test_in718_results.json'), 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print(f"avg_ef_error={results['avg_ef_error_pct']:.1f}%  avg_uts_error={results['avg_uts_error_pct']:.1f}%  "
          f"ratio90/0 pred={ratio_pred:.2f} exp={ratio_exp:.2f}")
    print('DONE saved blind_test_in718_results.json')


if __name__ == '__main__':
    main()
