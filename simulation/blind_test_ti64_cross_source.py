# -*- coding: utf-8 -*-
"""
blind_test_ti64_cross_source.py — Ti64 跨文献源盲测（R7⑦）
=========================================================
独立数据集：Sun et al. 2024, Adv. Eng. Mater. (ADEM 10.1002/adem.202400942)
LPBF Ti6Al4V 三构建取向拉伸，按模型约定(0°=载荷∥建造方向Z, 90°=载荷⊥建造方向X):
Z(0°)=9.34%、Z45(45°)=5.33%、X(90°)=7.89% (工程断裂总应变)。
方向趋势: 0°最韧、45°为V形极小值, 与标定集同侧(非简单反转) → 检验投影律适用域。

协议：S0 取 v12 主标定 Ti64 值（不重新标定）；微结构描述符沿用论文值
（φ=0.003, λ=50, ξ=3.5, θ=0.42, β 基值）；gs(ψ) kg=0.15。
输出: output/blind_test_ti64_cross_source_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

# Sun 2024 ADEM 数据（工程断裂应变 %），已换算到模型约定 0°=∥BD(Z), 90°=⊥BD(X)
CROSS_DATA = {
    0:  {'ef': 0.0934, 'ef_std': 0.0067},   # Z（垂直建造，载荷∥BD）
    45: {'ef': 0.0533, 'ef_std': 0.0074},   # Z45（ZX 旋转 45°）
    90: {'ef': 0.0789, 'ef_std': 0.0056},   # X（面内水平，载荷⊥BD）
}
XI0 = 3.5
K_G = 0.15


def gs_directional(psi):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(XI0, psi) / XI0   # xi_eff(psi)/XI0
    h = 1.0 - K_G * (1.0 - ratio)
    return {'gs_basal': 700e6 * h, 'gs_prism': 720e6 * h, 'gs_pyr': 800e6 * h}


def main():
    import sys as _sys
    s0 = float(_sys.argv[1]) if len(_sys.argv) > 1 else 0.32
    print(f'Ti64 cross-source blind: S0={s0} (from v12 calibration), Sun2024 ADEM data', flush=True)
    rows = []
    for psi in (0, 45, 90):
        exp = CROSS_DATA[psi]['ef']
        ov = gs_directional(psi)
        m = TaylorCPCDM('Ti64', psi, S0_override=s0, n_grains=20,
                        eps_step=5e-4, n_sub=30, strain_rate=1e-3,
                        model_version='v4', force_homogeneous=False,
                        param_overrides=ov, seed=42)
        r = m.run_uniaxial(exp * 2.0 + 0.01)
        pred = float(r['fracture_strain'])
        err = abs(pred - exp) / exp * 100.0
        rows.append({'orientation_deg': psi, 'ef_exp': exp,
                     'ef_pred': round(pred, 4), 'error_pct': round(err, 2),
                     'uts_pred_MPa': round(float(r['uts'] / 1e6), 0)})
        print(f"  {psi:>2d}deg: exp={exp:.4f} pred={pred:.4f} err={err:5.2f}%", flush=True)
    avg = float(np.mean([x['error_pct'] for x in rows]))
    out = {'material': 'Ti64', 'dataset': 'Sun2024_ADEM_adem.202400942',
           'S0_used': s0, 'cases': rows, 'avg_error_pct': round(avg, 2)}
    path = os.path.join(os.path.dirname(__file__), 'output',
                        'blind_test_ti64_cross_source_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'avg={avg:.2f}% -> {path}')


if __name__ == '__main__':
    main()
