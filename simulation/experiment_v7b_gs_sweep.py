"""
v7b gs 扫描 — experiment_v7b_gs_sweep.py
=========================================
对每材料在 gs_mult ∈ {2.2, 2.8, 3.4, 4.0} 下：
  1. S0 插值匹配 0° 实验 ef（每个 gs 候选独立重标）
  2. 记录 0° UTS、0° ef
目的：确定 Voce 饱和应力乘子的饱和点，判断 UTS 误差能否进入 5–8%。
输出：experiment_v7b_gs_sweep.json
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

N_GRAINS = 20
STRAIN_RATE = 1e-3
EPS_STEP_DEFAULT = 5e-4
EPS_STEP_316L = 1e-3
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
GS_CANDIDATES = [2.2, 2.8, 3.4, 4.0]

S0_GRID = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0, 200.0]


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def _gs_override(mat, gs_mult):
    if mat == 'Ti64':
        return {'gs_basal': 700e6 * gs_mult,
                'gs_prism': 720e6 * gs_mult,
                'gs_pyr': 800e6 * gs_mult}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat] * gs_mult}


def run_case(mat, psi, S0, gs_mult):
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    model = TaylorCPCDM(
        mat, psi, S0_override=S0, n_grains=N_GRAINS,
        eps_step=_eps_step(mat), n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=False,
        param_overrides=_gs_override(mat, gs_mult), seed=42,
    )
    return model.run_uniaxial(max_strain)


def ef_at(mat, S0, gs):
    return run_case(mat, 0, S0, gs)['fracture_strain']


def interp_S0(mat, ef_target, gs):
    pts = [(s0, ef_at(mat, s0, gs)) for s0 in S0_GRID]
    for i in range(len(pts) - 1):
        (s0a, efa), (s0b, efb) = pts[i], pts[i + 1]
        if (efa - ef_target) * (efb - ef_target) <= 0:
            s0_est = s0a + (ef_target - efa) / (efb - efa) * (s0b - s0a)
            break
    else:
        (s0a, efa), (s0b, efb) = pts[0], pts[1]
        k = (efb - efa) / (s0b - s0a)
        s0_est = s0a + (ef_target - efa) / k if k != 0 else 3.2
    for _ in range(1):  # 1 轮精化
        a0, b0 = s0_est / 1.5, s0_est * 1.5
        p2 = [(x, ef_at(mat, x, gs)) for x in (a0, s0_est, b0)]
        for i in range(len(p2) - 1):
            (x1, y1), (x2, y2) = p2[i], p2[i + 1]
            if (y1 - ef_target) * (y2 - ef_target) <= 0:
                s0_est = x1 + (ef_target - y1) / (y2 - y1) * (x2 - x1)
                break
    return s0_est


def sweep_material(mat):
    t0 = time.time()
    ef0 = EXPERIMENTAL[mat]['ef'][0]
    uts0 = EXPERIMENTAL[mat]['uts'][0]
    rows = []
    for gs in GS_CANDIDATES:
        S0 = interp_S0(mat, ef0, gs)
        r = run_case(mat, 0, S0, gs)
        rows.append({
            'gs_mult': gs, 'S0': float(S0),
            'ef_pred': float(r['fracture_strain']),
            'uts_pred_MPa': float(r['uts'] / 1e6),
            'uts_error_pct': float(abs(r['uts'] / 1e6 - uts0) / uts0 * 100.0),
        })
        print(f"  {mat:9s} gs={gs:.1f} S0={S0:6.3f} | "
              f"ef {ef0:.4f}->{r['fracture_strain']:.4f} | "
              f"UTS {uts0:.0f}->{r['uts']/1e6:.0f} MPa "
              f"err={abs(r['uts']/1e6-uts0)/uts0*100:.1f}%",
              flush=True)
    print(f"[done] {mat} sweep ({time.time()-t0:.0f}s)", flush=True)
    return mat, rows


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 80)
    print("v7b GS SWEEP (UTS saturation check)")
    print("=" * 80)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(sweep_material, m): m for m in MATERIALS}
        for fut in as_completed(futs):
            m, rows = fut.result()
            results[m] = rows

    payload = {'model': 'v7b_gs_sweep', 'gs_candidates': GS_CANDIDATES,
               'results': results}
    path = os.path.join(out_dir, 'experiment_v7b_gs_sweep.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
