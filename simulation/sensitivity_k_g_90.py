# -*- coding: utf-8 -*-
"""k_g 方向敏感性补充：90° 下 k_g 的影响（0° 下 gs_h(0)=1 天然零影响）"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

MAT = 'Ti64'
PSI = 90
S0 = 0.32
XI0 = 3.5
OUT = os.path.join(os.path.dirname(__file__), 'output')


def gs_h(psi, kg):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(XI0, psi) / XI0   # xi_eff(psi)/XI0
    return 1.0 - kg * (1.0 - ratio)


def run(kg):
    max_strain = 0.05 * 2.0 + 0.01  # Ti64 90° ef=0.048
    model = TaylorCPCDM(
        MAT, PSI, S0_override=S0, n_grains=20,
        eps_step=5e-4, n_sub=30, strain_rate=1e-3,
        model_version='v4', force_homogeneous=False,
        param_overrides={
            'gs_basal': 700e6 * gs_h(PSI, kg),
            'gs_prism': 720e6 * gs_h(PSI, kg),
            'gs_pyr': 800e6 * gs_h(PSI, kg),
        }, seed=42,
    )
    return float(model.run_uniaxial(max_strain)['fracture_strain'])


def main():
    rows = []
    for kg in (0.0, 0.05, 0.15, 0.25, 0.4):
        ef = run(kg)
        rows.append({'k_g': kg, 'ef': ef})
        print(f"k_g={kg:.2f}  ef={ef:.5f}", flush=True)
    base = rows[0]['ef']
    res = {
        'material': MAT, 'psi': PSI, 'S0': S0, 'k_g_default': 0.15,
        'rows': rows,
        'span_rel_pct': (max(r['ef'] for r in rows) - min(r['ef'] for r in rows)) / base * 100.0,
        'note': '90-degree k_g sweep: k_g enters only through directional gs modification gs_h(psi).',
    }
    with open(os.path.join(OUT, 'sensitivity_k_g_90.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f"span_rel={res['span_rel_pct']:.1f}%  DONE saved sensitivity_k_g_90.json")


if __name__ == '__main__':
    main()
