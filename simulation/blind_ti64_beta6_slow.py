# -*- coding: utf-8 -*-
"""
blind_ti64_beta6_slow.py — Ti64-2 跨源 β6 扫描（1e-3 慢版，后台）
====================================================================
fast 对 Ti64 不可靠（0° 0.129 vs 慢版 0.072）→ 1e-3/n_sub30 重跑。
S0=0.32（v12 标定值），Sun2024 实验 0.0789/0.0533/0.0934。
输出: output/blind_ti64_beta6_slow_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'output')
XI0 = 3.5
K_G = 0.15
EXP = {0: 0.0789, 45: 0.0533, 90: 0.0934}


def gs_directional(psi):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    h = 1.0 - K_G * (1.0 - xi_eff(XI0, psi) / XI0)   # xi_eff(psi)/XI0
    return {'gs_basal': 700e6 * h, 'gs_prism': 720e6 * h, 'gs_pyr': 800e6 * h}


def main():
    res = {}
    for b6 in (0.0, 0.2, 0.3, 0.5):
        row = {}
        for psi in (0, 45, 90):
            ov = gs_directional(psi)
            ov['beta6'] = b6
            m = TaylorCPCDM('Ti64', psi, S0_override=0.32, n_grains=20,
                            eps_step=1e-3, n_sub=30, strain_rate=1e-3,
                            model_version='v4', force_homogeneous=False,
                            param_overrides=ov, seed=42)
            efp = float(m.run_uniaxial(EXP[psi] * 2.0 + 0.01)['fracture_strain'])
            row[f'{psi}deg'] = {'ef_pred': efp,
                                'err_pct': float(abs(efp - EXP[psi]) / EXP[psi] * 100)}
            print(f"Ti64 b6={b6} psi={psi}: ef={efp:.4f} err={row[f'{psi}deg']['err_pct']:.1f}%",
                  flush=True)
        res[str(b6)] = row
        errs = [row[f'{psi}deg']['err_pct'] for psi in (0, 45, 90)]
        print(f"  -> mean err={np.mean(errs):.1f}%", flush=True)
    with open(os.path.join(OUT, 'blind_ti64_beta6_slow_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
