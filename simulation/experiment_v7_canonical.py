"""
v7 正典校准 — experiment_v7_canonical.py
========================================
响应第二轮审稿意见：
- 审稿 2a：f_λ = 1 + β4·(λ̄−1)，参考态（λ̄=1）f_AM→1（am_correction_v4 已改）
- 审稿 4：UTS 误差降到 5–8% —— 重新校准 Voce 饱和应力 g_s（每材料 1 个乘子，
  以 0° 实验 UTS 为目标），S0 同步重标（0° ef），a45/a90 重标（45°/90° ef）。
- 其余与 v6 相同：S≡S0 常数、f_AM 五因子物理投影 × 取向乘数 f_aniso(ψ)。
- 输出：experiment_v7_results.json（9 工况 ef/UTS 误差、S0/gs_mult/a45/a90）

标定流程（每材料）：
  1. gs_mult 二分（匹配 0° UTS），内层每次候选 gs_mult 用 S0 插值匹配 0° ef
     —— 迭代至 UTS 误差 < 0.5%（gs↑→应力↑→UTS↑ 且 ef↓，联合可逆）
  2. a45/a90 二分（匹配 45°/90° ef；gs 已定，a 仅经 f_AM 影响损伤）
  3. 跑 9 工况 → 误差表
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
MAX_BISECT = 45
A_MULT_LO, A_MULT_HI = 0.4, 8.0
GS_LO, GS_HI = 0.5, 2.2
GS_RANGE = {'Ti64': (2.5, 5.0), '316L': (1.0, 1.8), 'AlSi10Mg': (1.5, 3.0)}
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]

S0_GRID = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0, 200.0]


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def _gs_override(mat, gs_mult):
    """Voce 饱和应力乘子 → param_overrides（HCP 三族 / FCC 标量）。"""
    if mat == 'Ti64':
        return {'gs_basal': 700e6 * gs_mult,
                'gs_prism': 720e6 * gs_mult,
                'gs_pyr': 800e6 * gs_mult}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat] * gs_mult}


def run_case(mat, psi, S0, a_multi, gs_mult):
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    ov = _gs_override(mat, gs_mult)
    if a_multi != 1.0:
        ov['aniso_multiplier'] = {0: 1.0, 45: a_multi, 90: a_multi}
    model = TaylorCPCDM(
        mat, psi, S0_override=S0, n_grains=N_GRAINS,
        eps_step=_eps_step(mat), n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=False,
        param_overrides=ov, seed=42,
    )
    return model.run_uniaxial(max_strain)


def ef_at(mat, psi, S0, a, gs):
    return run_case(mat, psi, S0, a, gs)['fracture_strain']


def uts_at(mat, psi, S0, a, gs):
    return run_case(mat, psi, S0, a, gs)['uts'] / 1e6


def _interp_S0(mat, psi_ref, ef_target, a, gs):
    """S0 插值标定（对数网格 + 线性插值 + 精化；S0↑→ef↑ 单调）。"""
    def _solve(pts):
        for i in range(len(pts) - 1):
            (s0a, efa), (s0b, efb) = pts[i], pts[i + 1]
            if (efa - ef_target) * (efb - ef_target) <= 0:
                return s0a + (ef_target - efa) / (efb - efa) * (s0b - s0a)
        (s0a, efa), (s0b, efb) = pts[0], pts[1]
        k = (efb - efa) / (s0b - s0a)
        out = s0a + (ef_target - efa) / k if k != 0 else s0a
        (s0a, efa), (s0b, efb) = pts[-2], pts[-1]
        k2 = (efb - efa) / (s0b - s0a)
        out2 = s0b + (ef_target - efb) / k2 if k2 != 0 else s0b
        for v in (out, out2):
            if 0.02 <= v <= 400.0:
                return v
        return 3.2

    pts = [(s0, ef_at(mat, psi_ref, s0, a, gs)) for s0 in S0_GRID]
    s0_est = _solve(pts)
    for _ in range(2):
        a0, b0 = s0_est / 1.6, s0_est * 1.6
        pts2 = [(x, ef_at(mat, psi_ref, x, a, gs)) for x in (a0, s0_est, b0)]
        s0_new = _solve(pts2)
        if abs(s0_new - s0_est) < 1e-3 * s0_est:
            break
        s0_est = s0_new
    return s0_est


def _bisect_a(mat, psi, S0, target, gs, lo=A_MULT_LO, hi=A_MULT_HI):
    """a 二分匹配 ef@psi（a↑→f_AM↑→ef↓）。"""
    for _ in range(MAX_BISECT):
        mid = 0.5 * (lo + hi)
        ef = ef_at(mat, psi, S0, mid, gs)
        if ef < target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 5e-4 * max(1.0, mid):
            break
    return 0.5 * (lo + hi)


def calibrate_material(mat, max_gs_iter=14):
    """(gs_mult, S0, a45, a90) 联合标定。gs 区间按材料（扫描确定 UTS 敏感段）。"""
    t0 = time.time()
    ef0 = EXPERIMENTAL[mat]['ef'][0]
    uts0 = EXPERIMENTAL[mat]['uts'][0]
    lo0, hi0 = GS_RANGE[mat]

    def s0_for_gs(gs):
        return _interp_S0(mat, 0, ef0, 1.0, gs)

    # 外层二分 gs_mult 匹配 0° UTS（内层 S0 匹配 ef）
    lo, hi = lo0, hi0
    gs = None
    for _ in range(max_gs_iter):
        gs = 0.5 * (lo + hi)
        s0 = s0_for_gs(gs)
        u = uts_at(mat, 0, s0, 1.0, gs)
        if u < uts0:
            lo = gs          # 应力偏低 → 增大 gs
        else:
            hi = gs
        if hi - lo < 5e-4 * max(1.0, gs):
            break
    gs = 0.5 * (lo + hi)
    S0 = s0_for_gs(gs)
    u0 = uts_at(mat, 0, S0, 1.0, gs)

    # a45/a90（gs 已定）
    a45 = _bisect_a(mat, 45, S0, EXPERIMENTAL[mat]['ef'][45], gs)
    a90 = _bisect_a(mat, 90, S0, EXPERIMENTAL[mat]['ef'][90], gs)

    print(f"[calibrate] {mat}: gs_mult={gs:.4f}  S0={S0:.4f}  "
          f"a45={a45:.4f}  a90={a90:.4f}  (0° UTS {u0:.0f}/{uts0:.0f} MPa, "
          f"{time.time()-t0:.0f}s)")
    return gs, S0, a45, a90


def calibrate_and_run(m):
    """单材料：标定 + 9 工况运行（顶层函数，供 ProcessPoolExecutor）。"""
    gs, S0, a45, a90 = calibrate_material(m)
    am = {0: 1.0, 45: a45, 90: a90}
    out_cases = []
    for psi in ORIENTS:
        r = run_case(m, psi, S0, am[psi], gs)
        ef_exp = EXPERIMENTAL[m]['ef'][psi]
        uts_exp = EXPERIMENTAL[m]['uts'][psi]
        ef_err = abs(r['fracture_strain'] - ef_exp) / ef_exp * 100.0
        uts_err = abs(r['uts'] / 1e6 - uts_exp) / uts_exp * 100.0
        out_cases.append({
            'material': m, 'orientation': psi,
            'S0': float(S0), 'gs_mult': float(gs), 'a_aniso': float(am[psi]),
            'ef_exp': float(ef_exp), 'ef_pred': float(r['fracture_strain']),
            'error_pct': float(ef_err),
            'uts_exp_MPa': float(uts_exp), 'uts_pred_MPa': float(r['uts'] / 1e6),
            'uts_error_pct': float(uts_err),
        })
        print(f"  {m:9s} {psi:>3d}° | gs×{gs:5.3f} S0={S0:7.4f} a={am[psi]:6.3f} "
              f"| ef {ef_exp:.4f}->{r['fracture_strain']:.4f} err={ef_err:5.2f}% "
              f"| UTS {uts_exp:.0f}->{r['uts']/1e6:.0f} err={uts_err:5.2f}%",
              flush=True)
    return m, {'gs_mult': float(gs), 'S0': float(S0),
               'a45': float(a45), 'a90': float(a90)}, out_cases


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 88)
    print("CANONICAL v7 CALIBRATION (f_lambda fixed; Voce g_s recalibrated)")
    print("=" * 88)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    cases = []
    params = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(calibrate_and_run, m): m for m in MATERIALS}
        for fut in as_completed(futs):
            m, p, cs = fut.result()
            params[m] = p
            cases += cs
            print(f"[done] {m}", flush=True)

    errs = [c['error_pct'] for c in cases]
    uts = [c['uts_error_pct'] for c in cases]
    print("\n" + "=" * 88)
    print(f"SUMMARY: avg ef error = {np.mean(errs):.2f}%  |  "
          f"avg UTS error = {np.mean(uts):.2f}%  |  pass <10%: "
          f"{int(np.sum(np.array(errs) < 10.0))}/9")
    print("ef errors per case:", [round(e, 2) for e in errs])
    print("UTS errors per case:", [round(e, 2) for e in uts])
    print("=" * 88)

    payload = {
        'model': 'v7_canonical',
        'model_form': 'S=S0; f_AM=f_phi*f_theta*f_D0*f_lambda(psi)*f_xi(psi)'
                      '*f_aniso(psi); f_lambda=1+beta4*(lambda_bar-1); '
                      'Voce g_s recalibrated per material (0-deg UTS target)',
        'params': params,
        'eps_step': {'Ti64': EPS_STEP_DEFAULT, '316L': EPS_STEP_316L,
                     'AlSi10Mg': EPS_STEP_DEFAULT},
        'avg_error_pct': float(np.mean(errs)),
        'avg_uts_error_pct': float(np.mean(uts)),
        'pass_count': int(np.sum(np.array(errs) < 10.0)),
        'cases': cases,
    }
    path = os.path.join(out_dir, 'experiment_v7_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
