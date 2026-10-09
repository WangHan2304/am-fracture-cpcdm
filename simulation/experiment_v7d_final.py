"""
v7d 最终合并标定 — experiment_v7d_final.py
==========================================
v7 任务被终止（316L 超时），json 未写出；本脚本基于 stdout 已确认的
Ti64/AlSi10Mg 参数（gs=2.1996 处 UTS 饱和，扫描确认继续增大 gs 边际收益
<1%）完成最终 v7 正典结果：

- Ti64：gs=2.1996, S0=0.7838, a45=1.2455, a90=2.7402（v7 已完成，直接复用）
- AlSi10Mg：gs=2.1996, S0=0.1848；a45/a90 用 a 下限 0.4 重标
  （v7 中卡在 a=1.0 导致 45°/90° ef 误差 27.6%/24.6%）
- 316L：gs 扫描显示 UTS 对 gs 敏感（gs=2.2→850MPa 超调 +25%），快速标定
  （gs 二分 8 次 + S0 插值 + a 下限 0.4）定位 UTS≈680 的解
- 输出 experiment_v7_final_results.json（手稿/图件/LOOCV 唯一数据源）
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
A_MULT_LO, A_MULT_HI = 0.4, 8.0
MAX_BISECT = 40
S0_GRID = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0, 200.0]
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]

# v7 stdout 已确认（gs 二分在上界收敛）
V7_PARAMS = {
    'Ti64': {'gs_mult': 2.1996, 'S0': 0.7838, 'a45': 1.2455, 'a90': 2.7402},
    'AlSi10Mg': {'gs_mult': 2.1996, 'S0': 0.1848, 'a45': 1.0002, 'a90': 1.0002},
}


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def _gs_override(mat, gs_mult):
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


def interp_S0(mat, ef_target, gs, refine_rounds=1):
    pts = [(s0, ef_at(mat, 0, s0, 1.0, gs)) for s0 in S0_GRID]
    ok = None
    for i in range(len(pts) - 1):
        (s0a, efa), (s0b, efb) = pts[i], pts[i + 1]
        if (efa - ef_target) * (efb - ef_target) <= 0:
            ok = s0a + (ef_target - efa) / (efb - efa) * (s0b - s0a)
            break
    if ok is None:
        (s0a, efa), (s0b, efb) = pts[0], pts[1]
        k = (efb - efa) / (s0b - s0a)
        ok = s0a + (ef_target - efa) / k if k != 0 else 3.2
    s0_est = ok
    for _ in range(refine_rounds):
        a0, b0 = s0_est / 1.5, s0_est * 1.5
        p2 = [(x, ef_at(mat, 0, x, 1.0, gs)) for x in (a0, s0_est, b0)]
        for i in range(len(p2) - 1):
            (x1, y1), (x2, y2) = p2[i], p2[i + 1]
            if (y1 - ef_target) * (y2 - ef_target) <= 0:
                s0_est = x1 + (ef_target - y1) / (y2 - y1) * (x2 - x1)
                break
    return s0_est


def bisect_a(mat, psi, S0, target, gs):
    lo, hi = A_MULT_LO, A_MULT_HI
    for _ in range(MAX_BISECT):
        mid = 0.5 * (lo + hi)
        ef = ef_at(mat, psi, S0, mid, gs)
        if ef < target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 5e-4 * max(1.0, mid):
            break
    a = 0.5 * (lo + hi)
    ef = ef_at(mat, psi, S0, a, gs)
    return a, ef


def calibrate_316L():
    """316L：gs 二分定位 0° UTS≈680（8 次），内层 S0 匹配 0° ef，最后 a。"""
    t0 = time.time()
    ef0 = EXPERIMENTAL['316L']['ef'][0]
    uts0 = EXPERIMENTAL['316L']['uts'][0]

    def s0_for_gs(gs):
        return interp_S0('316L', ef0, gs, refine_rounds=1)

    lo, hi = 1.0, 2.0
    for _ in range(8):
        gs = 0.5 * (lo + hi)
        s0 = s0_for_gs(gs)
        u = run_case('316L', 0, s0, 1.0, gs)['uts'] / 1e6
        if u < uts0:
            lo = gs
        else:
            hi = gs
        print(f"  [316L] gs={gs:.4f} UTS={u:.0f}/{uts0} S0={s0:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    gs = 0.5 * (lo + hi)
    S0 = s0_for_gs(gs)
    u0 = run_case('316L', 0, S0, 1.0, gs)['uts'] / 1e6
    a45, ef45 = bisect_a('316L', 45, S0, EXPERIMENTAL['316L']['ef'][45], gs)
    a90, ef90 = bisect_a('316L', 90, S0, EXPERIMENTAL['316L']['ef'][90], gs)
    print(f"[calibrate] 316L: gs_mult={gs:.4f} S0={S0:.4f} "
          f"a45={a45:.4f} a90={a90:.4f} (0° UTS {u0:.0f}/{uts0}, "
          f"{time.time()-t0:.0f}s)", flush=True)
    return {'gs_mult': float(gs), 'S0': float(S0),
            'a45': float(a45), 'a90': float(a90)}


def recal_a_for(mat, gs, S0):
    """固定 gs/S0 下重标 a45/a90（a 下限 0.4）。"""
    t0 = time.time()
    a45, ef45 = bisect_a(mat, 45, S0, EXPERIMENTAL[mat]['ef'][45], gs)
    a90, ef90 = bisect_a(mat, 90, S0, EXPERIMENTAL[mat]['ef'][90], gs)
    print(f"[recal] {mat}: gs={gs:.4f} S0={S0:.4f} | "
          f"a45={a45:.4f} (ef {ef45:.4f}) a90={a90:.4f} (ef {ef90:.4f}) "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {'gs_mult': float(gs), 'S0': float(S0),
            'a45': float(a45), 'a90': float(a90)}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 80)
    print("v7d FINAL: Ti64 reuse + AlSi10Mg recal-a + 316L fast calibration")
    print("=" * 80)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    params_new = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {}
        futs[ex.submit(recal_a_for, 'AlSi10Mg',
                       V7_PARAMS['AlSi10Mg']['gs_mult'],
                       V7_PARAMS['AlSi10Mg']['S0'])] = 'AlSi10Mg'
        futs[ex.submit(calibrate_316L)] = '316L'
        for fut in as_completed(futs):
            key = futs[fut]
            if key == '316L':
                params_new['316L'] = fut.result()
            else:
                params_new['AlSi10Mg'] = fut.result()

    # Ti64 直接复用 v7 已定参数
    params_new['Ti64'] = dict(V7_PARAMS['Ti64'])
    print(f"[reuse] Ti64: gs={V7_PARAMS['Ti64']['gs_mult']:.4f} "
          f"S0={V7_PARAMS['Ti64']['S0']:.4f} "
          f"a45={V7_PARAMS['Ti64']['a45']:.4f} a90={V7_PARAMS['Ti64']['a90']:.4f}")

    # 9 工况
    cases = []
    for m in MATERIALS:
        p = params_new[m]
        am = {0: 1.0, 45: p['a45'], 90: p['a90']}
        for psi in ORIENTS:
            r = run_case(m, psi, p['S0'], am[psi], p['gs_mult'])
            ef_exp = EXPERIMENTAL[m]['ef'][psi]
            uts_exp = EXPERIMENTAL[m]['uts'][psi]
            ef_err = abs(r['fracture_strain'] - ef_exp) / ef_exp * 100.0
            uts_err = abs(r['uts'] / 1e6 - uts_exp) / uts_exp * 100.0
            cases.append({
                'material': m, 'orientation': psi,
                'S0': float(p['S0']), 'gs_mult': float(p['gs_mult']),
                'a_aniso': float(am[psi]),
                'ef_exp': float(ef_exp), 'ef_pred': float(r['fracture_strain']),
                'error_pct': float(ef_err),
                'uts_exp_MPa': float(uts_exp), 'uts_pred_MPa': float(r['uts'] / 1e6),
                'uts_error_pct': float(uts_err),
            })
            print(f"  {m:9s} {psi:>3d}° | gs×{p['gs_mult']:5.3f} "
                  f"S0={p['S0']:7.4f} a={am[psi]:6.3f} "
                  f"| ef {ef_exp:.4f}->{r['fracture_strain']:.4f} "
                  f"err={ef_err:5.2f}% | UTS {uts_exp:.0f}->{r['uts']/1e6:.0f} "
                  f"err={uts_err:5.2f}%", flush=True)

    errs = [c['error_pct'] for c in cases]
    uts = [c['uts_error_pct'] for c in cases]
    print("\n" + "=" * 80)
    print(f"SUMMARY: avg ef error = {np.mean(errs):.2f}%  |  "
          f"avg UTS error = {np.mean(uts):.2f}%  |  pass <10%: "
          f"{int(np.sum(np.array(errs) < 10.0))}/9")
    print("ef errors per case:", [round(e, 2) for e in errs])
    print("UTS errors per case:", [round(e, 2) for e in uts])
    print("=" * 80)

    payload = {
        'model': 'v7_final',
        'model_form': ('S=S0; f_AM=f_phi*f_theta*f_D0*f_lambda(psi)*f_xi(psi)'
                       '*f_aniso(psi); f_lambda=1+beta4*(lambda_bar-1); '
                       'Voce g_s recalibrated per material (0-deg UTS target); '
                       'f_aniso lower bound 0.4 (residual multiplier)'),
        'params': params_new,
        'avg_error_pct': float(np.mean(errs)),
        'avg_uts_error_pct': float(np.mean(uts)),
        'pass_count': int(np.sum(np.array(errs) < 10.0)),
        'cases': cases,
    }
    path = os.path.join(out_dir, 'experiment_v7_final_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
