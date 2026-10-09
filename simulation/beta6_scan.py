# -*- coding: utf-8 -*-
"""
beta6_scan.py — 几何屏蔽项 β6 扫描（盲测材料方向趋势修复）
============================================================
目标：f_shield(ψ)=1−β6·sin²ψ 使三个"缺陷屏蔽型"盲测材料的 90° 欠预测修复：
  IN718 mode2（S0*=6.0）90°: 0.029→0.30（+10 倍，β6 需大）
  Ti64-2（S0=0.32）90°: 0.0439→0.0934（+2.1 倍）
  316L-2（S0=2.0）90°: 0.1047→0.20（+1.9 倍，慢版后台另跑）
fast（n_sub=12, eps_step=2e-3）；IN718/Ti64-2 的 ef 与慢版一致（已验证）。
输出: output/beta6_scan_results.json
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
TI64_XI0 = 3.5


def gs_h(mat, psi):
    xi0 = {'IN718': 3.0, 'Ti64': TI64_XI0}[mat]
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    return 1.0 - K_G * (1.0 - xi_eff(xi0, psi) / xi0)   # xi_eff(psi)/xi0


def run(mat, psi, S0, b6, max_strain):
    if mat == 'IN718':
        ov = dict(IN718)
        ov['gs'] = ov['gs'] * gs_h('IN718', psi)
        skeleton = '316L'
    else:
        h = gs_h('Ti64', psi)
        ov = {'beta6': b6,
              'gs_basal': 700e6 * h, 'gs_prism': 720e6 * h, 'gs_pyr': 800e6 * h}
        skeleton = 'Ti64'
    ov['S0'] = S0
    ov['beta6'] = b6
    m = TaylorCPCDM(skeleton, psi, S0_override=S0, n_grains=20,
                    eps_step=2e-3, n_sub=12, strain_rate=1e-3,
                    model_version='v4', force_homogeneous=False,
                    param_overrides=ov, seed=42)
    return m.run_uniaxial(max_strain)['fracture_strain']


def main():
    res = {}
    # IN718 mode2（S0*=6.0）：0° 命中 11.9% 不变；扫 45/90
    exp = {45: 0.25, 90: 0.30}
    res['IN718_mode2'] = {}
    for b6 in (0.0, 0.5, 0.7, 0.85, 0.95):
        row = {}
        for psi in (45, 90):
            efp = run('IN718', psi, 6.0, b6, 0.40)
            row[f'{psi}deg'] = {'ef_pred': float(efp),
                                'err_pct': float(abs(efp - exp[psi]) / exp[psi] * 100)}
        res['IN718_mode2'][str(b6)] = row
        print(f"IN718 b6={b6}: 45d={row['45deg']['ef_pred']:.4f} "
              f"({row['45deg']['err_pct']:.0f}%) 90d={row['90deg']['ef_pred']:.4f} "
              f"({row['90deg']['err_pct']:.0f}%)", flush=True)
    # Ti64-2（S0=0.32）
    exp_t = {0: 0.0789, 45: 0.0533, 90: 0.0934}
    res['Ti64_cross'] = {}
    for b6 in (0.0, 0.3, 0.5, 0.7):
        row = {}
        for psi in (0, 45, 90):
            efp = run('Ti64', psi, 0.32, b6, 0.15)
            row[f'{psi}deg'] = {'ef_pred': float(efp),
                                'err_pct': float(abs(efp - exp_t[psi]) / exp_t[psi] * 100)}
        res['Ti64_cross'][str(b6)] = row
        print(f"Ti64 b6={b6}: 0d={row['0deg']['ef_pred']:.4f} "
              f"({row['0deg']['err_pct']:.0f}%) 45d={row['45deg']['ef_pred']:.4f} "
              f"({row['45deg']['err_pct']:.0f}%) 90d={row['90deg']['ef_pred']:.4f} "
              f"({row['90deg']['err_pct']:.0f}%)", flush=True)
    with open(os.path.join(OUT, 'beta6_scan_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
