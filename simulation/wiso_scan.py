# -*- coding: utf-8 -*-
"""
wiso_scan.py — λ_eff 各向同性权重 w_iso 扫描（新式 f_xi 下）
============================================================
假设：v12 316L/AlSi 恶化主因是 λ_eff 投影 90°/0° 比值过大（w_iso=0.25
→ λ_eff(90°)/λ_eff(0°)=4），新式 f_xi 放大该失衡。
扫描 w_iso ∈ [0.25, 0.35, 0.45, 0.55, 0.70]（S0 固定 v12 主标定值）
输出: output/wiso_scan_results.json
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
    '316L': {'S0': 3.2, 'eps': 1e-3},
    'AlSi10Mg': {'S0': 0.2, 'eps': 5e-4},
}
WISOS = [0.25, 0.35, 0.45, 0.55, 0.70]
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
    mat, psi, s0, w = t
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    ov = gs_directional(mat, psi)
    ov['_w_iso'] = w
    m = TaylorCPCDM(mat, psi, S0_override=s0, n_grains=20,
                    eps_step=SCAN[mat]['eps'], n_sub=30, strain_rate=1e-3,
                    model_version='v4', force_homogeneous=False,
                    param_overrides=ov, seed=42)
    efp = m.run_uniaxial(max_strain)['fracture_strain']
    err = abs(efp - EXPERIMENTAL[mat]['ef'][psi]) / EXPERIMENTAL[mat]['ef'][psi] * 100.0
    return (mat, psi, w, float(efp), float(err))


def main():
    tasks = [(m, p, SCAN[m]['S0'], w) for m in ('316L', 'AlSi10Mg')
             for w in WISOS for p in (0, 45, 90)]
    print(f'w_iso scan: {len(tasks)} runs', flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_worker, tasks))
    out = {}
    for mat in ('316L', 'AlSi10Mg'):
        rows = {}
        for w in WISOS:
            sel = [r for r in results if r[0] == mat and r[2] == w]
            errs = [r[4] for r in sel]
            efp = {r[1]: r[3] for r in sel}
            rows[str(w)] = {'ef_pred': efp, 'per_case_err': errs,
                            'mean_err': round(float(np.mean(errs)), 2)}
        out[mat] = rows
        print(f'== {mat} (S0={SCAN[mat]["S0"]}) ==')
        for w, row in rows.items():
            print(f"  w_iso={w}: mean_err={row['mean_err']:6.2f}% "
                  f"ef={ {k: round(v,4) for k,v in row['ef_pred'].items()} }")
    path = os.path.join(os.path.dirname(__file__), 'output', 'wiso_scan_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'DONE -> {path}')


if __name__ == '__main__':
    main()
