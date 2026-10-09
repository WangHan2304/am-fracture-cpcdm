"""
v7e 316L 快速定案 — experiment_v7e_316L_fast.py
===============================================
316L 模拟单次 ~6 分钟（ef=0.37 工况 610 宏观步 × 30 子步），完整标定
（gs 8 次二分 + a 重标）需 8h+，不可行。改用：

1. gs 二分从 [1.0, 2.0] 只做 3 次（已测 gs=1.5 → UTS 731）：
   gs=1.25 → UTS?  → 取跨越 680 的区间中点
2. 最终 gs 处 S0 网格插值（7 点，无精化）匹配 0° ef
3. a45/a90：先复用 v6 值（1.2412/1.5603）试跑 45°/90°，看 ef 误差；
   若任一 >10% 再用快速二分重标（每取向 1 进程并行，~1.5h）
输出：experiment_v7e_316L.json
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
EPS_STEP_316L = 1e-3
A_MULT_LO, A_MULT_HI = 0.4, 8.0
MAX_BISECT = 25
S0_GRID = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0, 200.0]
MAT = '316L'
ORIENTS = [0, 45, 90]
V6_A = {45: 1.2412, 90: 1.5603}


def _gs_override(gs_mult):
    return {'gs': 500e6 * gs_mult}


def run_case(psi, S0, a_multi, gs_mult):
    max_strain = EXPERIMENTAL[MAT]['ef'][psi] * 1.6 + 0.02
    ov = _gs_override(gs_mult)
    if a_multi != 1.0:
        ov['aniso_multiplier'] = {0: 1.0, 45: a_multi, 90: a_multi}
    model = TaylorCPCDM(
        MAT, psi, S0_override=S0, n_grains=N_GRAINS,
        eps_step=EPS_STEP_316L, n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=False,
        param_overrides=ov, seed=42,
    )
    return model.run_uniaxial(max_strain)


def interp_S0_grid(ef_target, gs):
    pts = [(s0, run_case(0, s0, 1.0, gs)['fracture_strain']) for s0 in S0_GRID]
    for i in range(len(pts) - 1):
        (s0a, efa), (s0b, efb) = pts[i], pts[i + 1]
        if (efa - ef_target) * (efb - ef_target) <= 0:
            return s0a + (ef_target - efa) / (efb - efa) * (s0b - s0a), pts
    (s0a, efa), (s0b, efb) = pts[0], pts[1]
    k = (efb - efa) / (s0b - s0a)
    out = s0a + (ef_target - efa) / k if k != 0 else 3.2
    return out, pts


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 80)
    print("v7e 316L FAST (gs 3 bisections + v6-a probe)")
    print("=" * 80)
    t0 = time.time()
    ef0 = EXPERIMENTAL[MAT]['ef'][0]
    uts0 = EXPERIMENTAL[MAT]['uts'][0]

    # --- gs 二分 3 次（1.5 已测 731）---
    # 第 1 次已测：gs=1.5000 UTS=731（v7d 输出）
    measured = {1.5: 731.0}
    lo, hi = 1.0, 1.5
    for it in range(2):
        gs = 0.5 * (lo + hi)
        S0, _ = interp_S0_grid(ef0, gs)
        u = run_case(0, S0, 1.0, gs)['uts'] / 1e6
        measured[gs] = u
        print(f"  [316L] gs={gs:.4f} S0={S0:.3f} UTS={u:.0f}/{uts0} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if u < uts0:
            lo = gs
        else:
            hi = gs

    # 在 680 附近的插值区间做最终 gs（线性插值跨越 680）
    gs_pts = sorted(measured)
    gs_final = None
    for i in range(len(gs_pts) - 1):
        g1, g2 = gs_pts[i], gs_pts[i + 1]
        u1, u2 = measured[g1], measured[g2]
        if (u1 - uts0) * (u2 - uts0) <= 0:
            gs_final = g1 + (uts0 - u1) / (u2 - u1) * (g2 - g1)
            break
    if gs_final is None:
        gs_final = min(gs_pts, key=lambda g: abs(measured[g] - uts0))
    S0, _ = interp_S0_grid(ef0, gs_final)
    u0 = run_case(0, S0, 1.0, gs_final)['uts'] / 1e6
    print(f"  [316L] FINAL gs={gs_final:.4f} S0={S0:.3f} UTS={u0:.0f}/{uts0} "
          f"err={abs(u0-uts0)/uts0*100:.1f}%", flush=True)

    # --- a 试跑（v6 值）---
    a45, a90 = V6_A[45], V6_A[90]
    probes = {}
    for psi, a in ((45, a45), (90, a90)):
        r = run_case(psi, S0, a, gs_final)
        efp = r['fracture_strain']
        efe = EXPERIMENTAL[MAT]['ef'][psi]
        err = abs(efp - efe) / efe * 100.0
        probes[psi] = {'a': a, 'ef_pred': float(efp), 'ef_exp': float(efe),
                       'err_pct': float(err),
                       'uts_pred': float(r['uts'] / 1e6),
                       'uts_exp': float(EXPERIMENTAL[MAT]['uts'][psi]),
                       'uts_err_pct': float(
                           abs(r['uts']/1e6 - EXPERIMENTAL[MAT]['uts'][psi])
                           / EXPERIMENTAL[MAT]['uts'][psi] * 100.0)}
        print(f"  [316L] {psi}° a={a:.4f} ef {efe:.4f}->{efp:.4f} "
              f"err={err:.1f}% (v6-a probe)", flush=True)

    out = {
        'model': 'v7e_316L_fast',
        'gs_mult': float(gs_final), 'S0': float(S0),
        'uts0_0deg_MPa': float(u0), 'uts0_target_MPa': float(uts0),
        'uts0_err_pct': float(abs(u0 - uts0) / uts0 * 100.0),
        'a45_probe_v6': probes.get(45), 'a90_probe_v6': probes.get(90),
        'needs_a_recal': any(p['err_pct'] > 10.0 for p in probes.values()),
    }
    path = os.path.join(out_dir, 'experiment_v7e_316L.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}  (needs_a_recal={out['needs_a_recal']})")


if __name__ == '__main__':
    main()
