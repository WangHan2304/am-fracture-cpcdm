# -*- coding: utf-8 -*-
"""
blind_316l_beta6_slow.py — 316L(Hitzler 真·反转)跨源 90° β6 屏蔽扫描（慢版）
====================================================================
唯一具真正 90° 反转的缺陷屏蔽型源(Hitzler 316L): 实测 0°=0.1176 最脆、90°=0.3324 最韧。
S0=3.2（v12 xi_eff 修正后标定值，0° 折 f_shield=1 不变）；目标：90° ef 从 β6=0 欠预测向实测 0.3324 上修。
输出: output/blind_316l_beta6_slow_results.json
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
XI0 = 2.5
EXP_90 = 0.3324  # Hitzler 316L 实测 90°（真·反转方向）


def gs_h(psi):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    return 1.0 - K_G * (1.0 - xi_eff(XI0, psi) / XI0)   # xi_eff(psi)/XI0


def main():
    res = {}
    for b6 in (0.0, 0.3, 0.5, 0.7):
        ov = {'gs': 500e6 * gs_h(90), 'beta6': b6}
        m = TaylorCPCDM('316L', 90, S0_override=3.2, n_grains=20,
                        eps_step=1e-3, n_sub=30, strain_rate=1e-3,
                        model_version='v4', force_homogeneous=False,
                        param_overrides=ov, seed=42)
        efp = float(m.run_uniaxial(0.40)['fracture_strain'])
        err = abs(efp - EXP_90) / EXP_90 * 100.0
        res[str(b6)] = {'ef_pred_90deg': efp, 'err_pct': float(err)}
        print(f"316L-2 b6={b6}: 90d ef={efp:.4f} err={err:.1f}%", flush=True)
    with open(os.path.join(OUT, 'blind_316l_beta6_slow_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
