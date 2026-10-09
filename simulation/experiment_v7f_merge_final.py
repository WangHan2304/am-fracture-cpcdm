"""
v7f 合并 — experiment_v7f_merge_final.py
========================================
合并 Ti64/AlSi10Mg（v7/v7d 已定参数）与 316L（v7e 快速标定）为
experiment_v7_final_results.json，并重跑全部 9 工况生成最终误差表。

316L 的 a45/a90：v7e 先以 v6 值试跑；若 needs_a_recal=true 且误差超限，
本脚本对 316L 的 a45/a90 做快速重标（a 下限 0.4，每取向一个进程）。
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
MAX_BISECT = 25
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]

# v7 stdout 已确认参数（gs 二分在上界收敛；扫描确认 UTS 饱和）
V7_PARAMS = {
    'Ti64': {'gs_mult': 2.1996, 'S0': 0.7838, 'a45': 1.2455, 'a90': 2.7402},
    'AlSi10Mg': {'gs_mult': 2.1996, 'S0': 0.1848, 'a45': 0.5584, 'a90': 0.5630},
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
    return model


def recal_a(mat, psi, S0, target, gs):
    """快速 a 二分（每取向独立进程）。"""
    t0 = time.time()
    lo, hi = A_MULT_LO, A_MULT_HI
    for _ in range(MAX_BISECT):
        mid = 0.5 * (lo + hi)
        r = run_case(mat, psi, S0, mid, gs).run_uniaxial(
            EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
            if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
        ef = r['fracture_strain']
        if ef < target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 5e-4 * max(1.0, mid):
            break
    a = 0.5 * (lo + hi)
    print(f"[recal-a] {mat} {psi}° a={a:.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    return a


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # ---- 读取 v7e 316L ----
    e = json.load(open(os.path.join(out_dir, 'experiment_v7e_316L.json'),
                       encoding='utf-8'))
    gs316, S0316 = e['gs_mult'], e['S0']
    a45 = e['a45_probe_v6']['a'] if e['a45_probe_v6'] else 1.2412
    a90 = e['a90_probe_v6']['a'] if e['a90_probe_v6'] else 1.5603
    need = e.get('needs_a_recal', False)
    print(f"[merge] 316L gs={gs316:.4f} S0={S0316:.4f} "
          f"a45={a45:.4f} a90={a90:.4f} needs_a_recal={need}", flush=True)

    if need:
        print("--- 316L a recalibration (parallel) ---", flush=True)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=2) as ex:
            f1 = ex.submit(recal_a, '316L', 45, S0316,
                           EXPERIMENTAL['316L']['ef'][45], gs316)
            f2 = ex.submit(recal_a, '316L', 90, S0316,
                           EXPERIMENTAL['316L']['ef'][90], gs316)
            a45, a90 = f1.result(), f2.result()

    params = dict(V7_PARAMS)
    params['316L'] = {'gs_mult': float(gs316), 'S0': float(S0316),
                      'a45': float(a45), 'a90': float(a90)}
    print("final params:", json.dumps(params, indent=2), flush=True)

    # ---- 9 工况 ----
    cases = []
    for m in MATERIALS:
        p = params[m]
        am = {0: 1.0, 45: p['a45'], 90: p['a90']}
        for psi in ORIENTS:
            model = run_case(m, psi, p['S0'], am[psi], p['gs_mult'])
            r = model.run_uniaxial(
                EXPERIMENTAL[m]['ef'][psi] * 1.6 + 0.02
                if m == '316L' else EXPERIMENTAL[m]['ef'][psi] * 2.0 + 0.01)
            ef_exp = EXPERIMENTAL[m]['ef'][psi]
            uts_exp = EXPERIMENTAL[m]['uts'][psi]
            ef_err = abs(r['fracture_strain'] - ef_exp) / ef_exp * 100.0
            uts_err = abs(r['uts'] / 1e6 - uts_exp) / uts_exp * 100.0
            cases.append({
                'material': m, 'orientation': psi,
                'S0': float(p['S0']), 'gs_mult': float(p['gs_mult']),
                'a_aniso': float(am[psi]), 'f_AM': float(model.f_AM),
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
                       'f_aniso in [0.4,8] (residual multiplier)'),
        'params': params,
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
