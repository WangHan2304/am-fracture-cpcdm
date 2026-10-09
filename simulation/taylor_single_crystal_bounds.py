# -*- coding: utf-8 -*-
"""
Taylor 假设量化影响：单晶极限分析
====================================
DAMASK 原生 FFT/CPFEM 在 Windows 不可行（已确认），用单晶极限替代：
对每个材料 0 度工况，跑 12 个 n_grains=1 的随机取向单晶（v7 全局参数），
统计单晶断裂应变散布；对比 20 晶粒 Taylor 多晶结果（v7 正典），
量化"取向平均/应力重分布被 Taylor 假设排除"导致的断裂应变偏差带。

输出: simulation/output/taylor_single_crystal_bounds.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

N_MC = 12          # 单晶随机取向数
N_GRAINS_TAYLOR = 20

V9 = {
    'Ti64':      {'S0': 0.32,  'gs_mult': 1.0},
    '316L':      {'S0': 3.20,  'gs_mult': 1.0},
    'AlSi10Mg':  {'S0': 0.20,  'gs_mult': 1.0},
}
EPS_STEP = {'Ti64': 5e-4, '316L': 1e-3, 'AlSi10Mg': 5e-4}


def _gs_override(mat, gs_mult):
    if mat == 'Ti64':
        return {'gs_basal': 700e6 * gs_mult,
                'gs_prism': 720e6 * gs_mult,
                'gs_pyr': 800e6 * gs_mult}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat] * gs_mult}


def run_ef(mat, psi, n_grains, seed):
    p = V9[mat]
    ov = _gs_override(mat, p['gs_mult'])
    model = TaylorCPCDM(
        mat, psi, S0_override=p['S0'], n_grains=n_grains,
        eps_step=EPS_STEP[mat], n_sub=30, strain_rate=1e-3,
        model_version='v4', force_homogeneous=False,
        param_overrides=ov, seed=seed,
    )
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    r = model.run_uniaxial(max_strain)
    return float(r['fracture_strain']), float(r['uts'] / 1e6)


def main():
    t0 = time.time()
    out = {}
    for mat in V9:
        # Taylor 多晶（v9 正典，seed=42 与主实验一致）
        ef_taylor, uts_taylor = run_ef(mat, 0, N_GRAINS_TAYLOR, 42)
        # 单晶极限散布
        efs, utss = [], []
        for k in range(N_MC):
            ef, u = run_ef(mat, 0, 1, seed=1000 + k)
            efs.append(ef)
            utss.append(u)
        efs = np.array(efs)
        utss = np.array(utss)
        dev = (efs - ef_taylor) / ef_taylor * 100.0
        out[mat] = {
            'ef_taylor_20grain': ef_taylor,
            'uts_taylor_20grain_MPa': uts_taylor,
            'ef_exp_0deg': EXPERIMENTAL[mat]['ef'][0],
            'ef_single_mean': float(efs.mean()),
            'ef_single_std': float(efs.std()),
            'ef_single_min': float(efs.min()),
            'ef_single_max': float(efs.max()),
            'dev_pct_mean': float(dev.mean()),
            'dev_pct_std': float(dev.std()),
            'dev_pct_min': float(dev.min()),
            'dev_pct_max': float(dev.max()),
            'uts_single_mean_MPa': float(utss.mean()),
            'uts_single_std_MPa': float(utss.std()),
            'n_single': N_MC,
        }
        print(f"[{mat}] Taylor ef={ef_taylor:.5f} | single ef "
              f"{efs.min():.5f}..{efs.max():.5f} (mean {efs.mean():.5f} "
              f"std {efs.std():.5f}) | dev {dev.min():+.2f}..{dev.max():+.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    out_path = os.path.join(os.path.dirname(__file__), 'output',
                            'taylor_single_crystal_bounds.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path} ({time.time()-t0:.0f}s total)")


if __name__ == '__main__':
    main()
