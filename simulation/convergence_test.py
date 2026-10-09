"""
数值收敛性测试 — convergence_test.py
=====================================
P2-4 响应：网格（晶粒数）收敛与时间步收敛，Ti-6Al-4V 0° 为代表工况：
- n_grains ∈ {10, 20, 30, 50}，eps_step=5e-4 固定
- eps_step ∈ {2.5e-4, 5e-4, 1e-3}，n_grains=20 固定
输出 ef 随离散度的变化 → 论文 sec:convergence。
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

MAT = 'Ti64'
PSI = 0
# v9 参数（与主模型一致）：单 S0=0.32，gs 用文献值（无 gs_mult），无取向乘子
S0 = 0.32
GS_MULT = 1.0
MAX_STRAIN = EXPERIMENTAL[MAT]['ef'][PSI] * 2.0 + 0.01


def run(n_grains, eps_step, seed=42):
    ov = {'gs_basal': 700e6 * GS_MULT,
          'gs_prism': 720e6 * GS_MULT,
          'gs_pyr': 800e6 * GS_MULT}
    m = TaylorCPCDM(MAT, PSI, S0_override=S0, n_grains=n_grains,
                    eps_step=eps_step, n_sub=30, strain_rate=1e-3,
                    model_version='v4', seed=seed, param_overrides=ov)
    r = m.run_uniaxial(MAX_STRAIN)
    return r['fracture_strain'], r['uts'] / 1e6


def main():
    out = {'material': MAT, 'orientation': PSI, 'S0': S0}
    grains, steps = [], []
    for ng in [10, 20, 30, 50]:
        t0 = time.time()
        ef, uts = run(ng, 5e-4)
        grains.append({'n_grains': ng, 'ef': ef, 'uts_MPa': uts})
        print(f"n_grains={ng:>3d}  ef={ef:.5f}  uts={uts:.1f}  ({time.time()-t0:.0f}s)")
    for es in [2.5e-4, 5e-4, 1e-3]:
        t0 = time.time()
        ef, uts = run(20, es)
        steps.append({'eps_step': es, 'ef': ef, 'uts_MPa': uts})
        print(f"eps_step={es:g}  ef={ef:.5f}  uts={uts:.1f}  ({time.time()-t0:.0f}s)")

    out['n_grains_convergence'] = grains
    out['eps_step_convergence'] = steps
    base_ef = grains[[i for i, g in enumerate(grains) if g['n_grains'] == 20][0]]['ef']
    out['n_grains_relative_dev'] = {g['n_grains']: round(abs(g['ef'] - base_ef) / base_ef * 100, 3)
                                    for g in grains}
    path = os.path.join(os.path.dirname(__file__), 'output', 'convergence_test.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Saved:", path)


if __name__ == '__main__':
    main()
