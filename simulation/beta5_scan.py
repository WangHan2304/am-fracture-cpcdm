# -*- coding: utf-8 -*-
"""
beta5_scan.py — f_xi 新式下 β5 幅度快速扫描
============================================
问题：新式 f_xi=1+β5(1/ξ_eff−1) 下，316L(β5=0.15)/AlSi(β5=0.1) 过冲。
β5 是 assumed 参数（非拟合）→ 按方向性强度重新假设合法。
扫描：固定 S0=v12 主标定值，扫 β5 → 3 方向 ef 标定误差（cal 代理）。
Ti64 β5=0.2 已验证最优，保持。
输出: output/beta5_scan_results.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

SCAN = {
    '316L': {'S0': 3.2, 'beta5s': [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
             'eps': 1e-3},
    'AlSi10Mg': {'S0': 0.2, 'beta5s': [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10],
                 'eps': 5e-4},
}
XI0 = {'Ti64': 3.5, '316L': 2.5, 'AlSi10Mg': 1.5}
K_G = 0.15


def gs_directional(mat, psi):
    xi0 = XI0[mat]
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(xi0, psi) / xi0   # xi_eff(psi)/xi0 (am_correction_v4)
    h = 1.0 - K_G * (1.0 - ratio)
    base = {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6} \
        if mat == 'Ti64' else {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}
    return {k: v * h for k, v in base.items()}


def _worker(t):
    mat, psi, s0, b5 = t
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    ov = gs_directional(mat, psi)
    ov['beta5'] = b5
    m = TaylorCPCDM(mat, psi, S0_override=s0, n_grains=20,
                    eps_step=SCAN[mat]['eps'], n_sub=30, strain_rate=1e-3,
                    model_version='v4', force_homogeneous=False,
                    param_overrides=ov, seed=42)
    efp = m.run_uniaxial(max_strain)['fracture_strain']
    err = abs(efp - EXPERIMENTAL[mat]['ef'][psi]) / EXPERIMENTAL[mat]['ef'][psi] * 100.0
    return (mat, psi, b5, float(efp), float(err))


def main():
    tasks = []
    for mat in ('316L', 'AlSi10Mg'):
        for b5 in SCAN[mat]['beta5s']:
            for psi in (0, 45, 90):
                tasks.append((mat, psi, SCAN[mat]['S0'], b5))
    print(f'beta5 scan: {len(tasks)} runs', flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_worker, tasks))
    out = {}
    for mat in ('316L', 'AlSi10Mg'):
        rows = {}
        for b5 in SCAN[mat]['beta5s']:
            sel = [r for r in results if r[0] == mat and r[2] == b5]
            errs = [r[4] for r in sel]
            efp = {r[1]: r[3] for r in sel}
            rows[str(b5)] = {'ef_pred': efp, 'per_case_err': errs,
                             'mean_err': round(float(np.mean(errs)), 2)}
        out[mat] = rows
        print(f'== {mat} (S0={SCAN[mat]["S0"]}) ==')
        for b5, row in rows.items():
            print(f"  beta5={b5}: mean_err={row['mean_err']:6.2f}% "
                  f"ef={ {k: round(v,4) for k,v in row['ef_pred'].items()} }")
    path = os.path.join(os.path.dirname(__file__), 'output', 'beta5_scan_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'DONE -> {path}')


if __name__ == '__main__':
    main()
