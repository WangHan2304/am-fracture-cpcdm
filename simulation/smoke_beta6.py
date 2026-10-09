# -*- coding: utf-8 -*-
"""冒烟：β6=0 不变 + β6 生效（IN718 fast 0°）"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

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
K_G = 0.15
XI0 = 3.0


def gs_h(psi):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    return 1.0 - K_G * (1.0 - xi_eff(XI0, psi) / XI0)   # xi_eff(psi)/XI0


def run(psi, S0, b6):
    ov = dict(IN718)
    ov['S0'] = S0
    ov['gs'] = ov['gs'] * gs_h(psi)
    ov['beta6'] = b6
    m = TaylorCPCDM('316L', psi, S0_override=S0, n_grains=20,
                    eps_step=2e-3, n_sub=12, strain_rate=1e-3,
                    model_version='v4', force_homogeneous=False,
                    param_overrides=ov, seed=42)
    return m.run_uniaxial(0.40)['fracture_strain']


S0 = (0.32 * 3.20 * 0.20) ** (1.0 / 3.0)
for psi in (0, 45, 90):
    e0 = run(psi, S0, 0.0)
    e1 = run(psi, S0, 0.5)
    print(f"psi={psi:>2d}  beta6=0: {e0:.4f}   beta6=0.5: {e1:.4f}")
