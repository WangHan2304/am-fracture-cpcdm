"""
v7c 局部重标 — experiment_v7c_recal_a.py
========================================
在 v7 已完成的 gs_mult/S0 基础上，重标 a45/a90（a 下限放宽到 0.4），
修复 gs 抬高后 45°/90° 预测 ef 偏低的问题（AlSi10Mg 卡 a=1.0 场景）。

输入：experiment_v7_results.json（若存在则更新；否则按 gs=2.2 重跑标定）
输出：experiment_v7_final_results.json（最终正典结果，供手稿/图件/LOOCV 使用）
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
MAX_BISECT = 45
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]


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


def bisect_a(mat, psi, S0, target, gs, lo=A_MULT_LO, hi=A_MULT_HI):
    """a 二分匹配 ef@psi（a↑→f_AM↑→ef↓）。返回 a 与最终 ef。"""
    for _ in range(MAX_BISECT):
        mid = 0.5 * (lo + hi)
        ef = run_case(mat, psi, S0, mid, gs)['fracture_strain']
        if ef < target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 5e-4 * max(1.0, mid):
            break
    a = 0.5 * (lo + hi)
    ef = run_case(mat, psi, S0, a, gs)['fracture_strain']
    return a, ef


def recalibrate_material(mat, p):
    """重标 a45/a90（a 下限 0.4），保持 gs_mult/S0 不变。"""
    t0 = time.time()
    gs, S0 = p['gs_mult'], p['S0']
    a45, ef45 = bisect_a(mat, 45, S0, EXPERIMENTAL[mat]['ef'][45], gs)
    a90, ef90 = bisect_a(mat, 90, S0, EXPERIMENTAL[mat]['ef'][90], gs)
    print(f"[recal] {mat}: gs={gs:.4f} S0={S0:.4f} | "
          f"a45={a45:.4f} (ef {ef45:.4f})  a90={a90:.4f} (ef {ef90:.4f}) "
          f"({time.time()-t0:.0f}s)", flush=True)
    return mat, {'gs_mult': float(gs), 'S0': float(S0),
                 'a45': float(a45), 'a90': float(a90)}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(out_dir, 'experiment_v7_results.json')
    if not os.path.exists(src):
        print(f"ERROR: {src} not found — run experiment_v7_canonical.py first")
        sys.exit(1)
    v7 = json.load(open(src, encoding='utf-8'))
    print("=" * 80)
    print("v7c a45/a90 RECALIBRATION (a lower bound 0.4)")
    print("=" * 80)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    params_new = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(recalibrate_material, m, v7['params'][m]): m
                for m in MATERIALS}
        for fut in as_completed(futs):
            m, p = fut.result()
            params_new[m] = p

    # 用新 a 重跑 9 工况
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
        'model_form': v7.get('model_form', ''),
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
