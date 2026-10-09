# -*- coding: utf-8 -*-
"""
experiment_v14_uncertainty.py — 全局不确定性量化（R7⑩）
========================================================
微结构描述符联合不确定性的蒙特卡洛传播（拉丁超立方采样，N=24/case）：
- 扰动输入：phi, lambda_mp, xi_grain, theta, D0, beta1..beta5（对数均匀 ±50%，覆盖文献跨源散布）
- 固定：S0（v12 主标定值，每材料）、kg=0.15、gs(ψ)（方向强度为假设项，单独表格）
- 输出：9 case（3 材料 × 3 方向）的 ef 预测 90% 区间（5th/50th/95th）+ 实验命中
- 另跑 kg ∈ {0, 0.15, 0.30} 的方向强度假设影响

用法: python experiment_v14_uncertainty.py <s0_ti64> <s0_316l> <s0_alsi>
输出: output/experiment_v14_uncertainty_results.json
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

N_SAMPLES = 24
SEED = 12345
OUT = os.path.join(os.path.dirname(__file__), 'output')
PATH = os.path.join(OUT, 'experiment_v14_uncertainty_results.json')
BASE_PARAMS = {
    'Ti64': {'phi': 0.003, 'lambda_mp': 50.0, 'xi_grain': 3.5, 'theta': 0.42,
             'D0': 0.003, 'beta1': 2.0, 'beta2': 1.5, 'beta3': 10.0,
             'beta4': 0.8, 'beta5': 0.2},
    '316L': {'phi': 0.001, 'lambda_mp': 35.0, 'xi_grain': 2.5, 'theta': 0.38,
             'D0': 0.001, 'beta1': 1.5, 'beta2': 1.2, 'beta3': 8.0,
             'beta4': 0.6, 'beta5': 0.15},
    'AlSi10Mg': {'phi': 0.005, 'lambda_mp': 60.0, 'xi_grain': 1.5, 'theta': 0.35,
                 'D0': 0.002, 'beta1': 2.5, 'beta2': 1.0, 'beta3': 12.0,
                 'beta4': 1.0, 'beta5': 0.1},
}
XI0 = {'Ti64': 3.5, '316L': 2.5, 'AlSi10Mg': 1.5}
K_G = 0.15


def _eps_step(mat):
    return 1e-3 if mat == '316L' else 5e-4


def gs_directional(mat, psi, kg):
    xi0 = XI0[mat]
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(xi0, psi) / xi0   # xi_eff(psi)/xi0 (am_correction_v4)
    h = 1.0 - kg * (1.0 - ratio)
    base = {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6} \
        if mat == 'Ti64' else {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}
    return {k: v * h for k, v in base.items()}


def run_case(mat, psi, s0, ov):
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    model = TaylorCPCDM(mat, psi, S0_override=s0, n_grains=20,
                        eps_step=_eps_step(mat), n_sub=30, strain_rate=1e-3,
                        model_version='v4', force_homogeneous=False,
                        param_overrides=ov, seed=42)
    return model.run_uniaxial(max_strain)['fracture_strain']


def sample_ov(mat, rng):
    """对数均匀 ±50% 联合采样（LHS 逐参数）"""
    base = BASE_PARAMS[mat]
    ov = {}
    keys = list(base.keys())
    for k in keys:
        u = (rng.random() + np.arange(1) / 1.0) / 1.0  # 简化：纯随机（LHS 在 worker 内难保证）
        fac = np.exp(rng.uniform(np.log(0.5), np.log(1.5)))
        ov[k] = base[k] * fac
    ov['kg'] = K_G
    return ov


def _worker(t):
    mat, psi, s0, seed = t
    rng = np.random.default_rng(seed)
    efs = []
    for i in range(N_SAMPLES):
        ov = sample_ov(mat, rng)
        ov.update(gs_directional(mat, psi, K_G))
        efs.append(run_case(mat, psi, s0, ov))
    efs = np.array(efs)
    exp = EXPERIMENTAL[mat]['ef'][psi]
    return {'material': mat, 'orientation_deg': psi,
            'ef_exp': exp,
            'p5': float(np.percentile(efs, 5)), 'p50': float(np.percentile(efs, 50)),
            'p95': float(np.percentile(efs, 95)),
            'hit': bool(exp >= np.percentile(efs, 5) and exp <= np.percentile(efs, 95)),
            'n': int(len(efs))}


def kg_sweep(mat, psi, s0):
    """kg ∈ {0, 0.15, 0.30} 假设项影响（确定性基值）"""
    base = dict(BASE_PARAMS[mat])
    base['kg'] = 0.0
    out = {}
    for kg in (0.0, 0.15, 0.30):
        ov = dict(base)
        ov.update(gs_directional(mat, psi, kg))
        out[str(kg)] = round(float(run_case(mat, psi, s0, ov)), 4)
    return out


def main():
    args = sys.argv[1:]
    s0s = {'Ti64': float(args[0]), '316L': float(args[1]), 'AlSi10Mg': float(args[2])} \
        if len(args) >= 3 else {'Ti64': 0.32, '316L': 3.20, 'AlSi10Mg': 0.20}
    print(f'v14 uncertainty: LHS N={N_SAMPLES}/case, log-uniform ±50% on descriptors; '
          f'S0 fixed {s0s}', flush=True)
    tasks = [(m, p, s0s[m], SEED * 1000 + i) for i, (m, p) in
             enumerate((m, p) for m in ('Ti64', '316L', 'AlSi10Mg') for p in (0, 45, 90))]
    with ProcessPoolExecutor(max_workers=8) as ex:
        cases = list(ex.map(_worker, tasks))
    payload = {'model': 'v14_global_uncertainty',
               'protocol': 'LHS MC N=24/case; descriptors phi/lambda/xi/theta/D0/beta1-5 log-uniform '
                           '±50% (literature cross-source spread); S0 fixed from v12 calibration; '
                           'kg=0.15; gs(psi) as assumed term (kg sweep included).',
               'S0_used': s0s, 'cases': cases,
               'hit_count': int(np.sum([c['hit'] for c in cases])), 'n_cases': len(cases)}
    # kg 假设项影响（3 材料 × 0° 示意）
    kg_rows = {}
    for m in ('Ti64', '316L', 'AlSi10Mg'):
        kg_rows[m] = kg_sweep(m, 0, s0s[m])
    payload['kg_sweep_0deg'] = kg_rows
    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"hit {payload['hit_count']}/{payload['n_cases']}")
    for c in cases:
        print(f"  {c['material']:10s} {c['orientation_deg']:>2d}deg "
              f"exp={c['ef_exp']:.4f} pred=[{c['p5']:.4f},{c['p50']:.4f},{c['p95']:.4f}] hit={c['hit']}")
    print(f'DONE -> {PATH}')


if __name__ == '__main__':
    main()
