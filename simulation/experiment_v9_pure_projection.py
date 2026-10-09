"""
v9 无泄漏 LOOCV — experiment_v9_pure_projection.py
====================================================
响应审稿第三轮：消除 a45/a90 残余取向乘数的信息泄漏。

协议（最严格）：
- 取消 a45/a90：f_AM 不含 aniso_multiplier，方向效应完全由物理投影律
  lambda_eff(psi), xi_eff(psi), theta_bar(psi) 承载（v4 原始哲学）；
- g_s 固定为文献值（不参与任何拟合）——杜绝"gs 提前用留出方向数据"；
- 主模型：每材料仅标定 1 个可调参数 S0（3 方向联合最小化平均 ef 误差）；
- LOOCV：每折只用训练方向（2 个）标定 S0，留出方向盲预测；
- Lemaitre 基线：同协议（f_AM≡1，每材料仅标 S0）——公平比较（1 参数 vs 1 参数）。

输出：experiment_v9_pure_projection_results.json
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
ORIENTS = [0, 45, 90]
S0_GRID = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0]
REFINE_ROUNDS = 1

OUT = os.path.join(os.path.dirname(__file__), 'output')


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def _gs_literature(mat):
    """g_s 文献值（不拟合，杜绝泄漏）。"""
    if mat == 'Ti64':
        return {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}


def run_case(mat, psi, S0, homogeneous):
    """固定文献 g_s、无 aniso_multiplier 的运行。"""
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    ov = _gs_literature(mat)   # 文献 g_s；无 aniso_multiplier → 纯投影律
    model = TaylorCPCDM(
        mat, psi, S0_override=S0, n_grains=N_GRAINS,
        eps_step=_eps_step(mat), n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=homogeneous,
        param_overrides=ov, seed=42,
    )
    return model.run_uniaxial(max_strain)


def score_S0(mat, oris, S0, homogeneous):
    errs = []
    for ps in oris:
        efp = run_case(mat, ps, S0, homogeneous)['fracture_strain']
        errs.append(abs(efp - EXPERIMENTAL[mat]['ef'][ps])
                    / EXPERIMENTAL[mat]['ef'][ps])
    return float(np.mean(errs))


def solve_S0(mat, oris, homogeneous):
    """网格 + 局部精化，最小化 oris 平均相对 ef 误差。"""
    best_s0, best_e = None, None
    for s0 in S0_GRID:
        e = score_S0(mat, oris, s0, homogeneous)
        if best_e is None or e < best_e:
            best_s0, best_e = s0, e
    for _ in range(REFINE_ROUNDS):
        lo, hi = best_s0 / 1.6, best_s0 * 1.6
        for s0 in (lo, hi):
            e = score_S0(mat, oris, s0, homogeneous)
            if e < best_e:
                best_s0, best_e = s0, e
    return best_s0, best_e


def calibrate_material(mat, homogeneous):
    """主模型标定：3 方向联合，只标 S0。"""
    t0 = time.time()
    S0, cal_err = solve_S0(mat, ORIENTS, homogeneous)
    cases = []
    for ps in ORIENTS:
        r = run_case(mat, ps, S0, homogeneous)
        cases.append({
            'orientation': ps,
            'ef_exp': EXPERIMENTAL[mat]['ef'][ps],
            'ef_pred': float(r['fracture_strain']),
            'error_pct': abs(r['fracture_strain']
                             - EXPERIMENTAL[mat]['ef'][ps])
                          / EXPERIMENTAL[mat]['ef'][ps] * 100.0,
            'uts_exp_MPa': EXPERIMENTAL[mat]['uts'][ps],
            'uts_pred_MPa': float(r['uts'] / 1e6),
            'uts_error_pct': abs(r['uts'] / 1e6
                                 - EXPERIMENTAL[mat]['uts'][ps])
                              / EXPERIMENTAL[mat]['uts'][ps] * 100.0,
        })
    out = {
        'material': mat, 'S0': float(S0),
        'calibration_avg_err_pct': float(cal_err * 100.0),
        'cases': cases, 'walltime_s': round(time.time() - t0, 1),
    }
    print(f"  [CAL-{'MODEL' if not homogeneous else 'LEM'}] {mat:10s} "
          f"S0={S0:8.4f} cal_err={cal_err*100:6.2f}% | "
          f"ef {[round(c['error_pct'],2) for c in cases]} | "
          f"UTS {[round(c['uts_error_pct'],1) for c in cases]} | "
          f"{out['walltime_s']}s", flush=True)
    return out


def fold_one(mat, held, homogeneous):
    train = [o for o in ORIENTS if o != held]
    t0 = time.time()
    S0, train_err = solve_S0(mat, train, homogeneous)
    r = run_case(mat, held, S0, homogeneous)
    ef_exp = EXPERIMENTAL[mat]['ef'][held]
    ef_pred = float(r['fracture_strain'])
    err = abs(ef_pred - ef_exp) / ef_exp * 100.0
    uts_err = abs(r['uts'] / 1e6 - EXPERIMENTAL[mat]['uts'][held]) \
        / EXPERIMENTAL[mat]['uts'][held] * 100.0
    fold = {
        'material': mat, 'held_orientation': held,
        'train_orientations': train,
        'S0': float(S0), 'train_avg_err_pct': float(train_err * 100.0),
        'ef_exp': float(ef_exp), 'ef_pred': ef_pred,
        'error_pct': float(err),
        'uts_exp_MPa': float(EXPERIMENTAL[mat]['uts'][held]),
        'uts_pred_MPa': float(r['uts'] / 1e6),
        'uts_error_pct': float(uts_err),
        'walltime_s': round(time.time() - t0, 1),
    }
    tag = 'MODEL' if not homogeneous else 'LEMAITRE'
    print(f"  [{tag}] {mat:10s} hold={held:>3d}deg | S0={S0:8.4f} "
          f"train_err={train_err*100:5.2f}% | ef {ef_exp:.4f}->{ef_pred:.4f} "
          f"err={err:6.2f}% | UTS err={uts_err:5.2f}% | {fold['walltime_s']}s",
          flush=True)
    return fold


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'experiment_v9_pure_projection_results.json')

    # ---- 主模型标定（MODEL 与 LEM） ----
    cal_model = {m: calibrate_material(m, False) for m in MATERIALS}
    cal_lem = {m: calibrate_material(m, True) for m in MATERIALS}
    payload = {
        'model': 'v9_pure_projection',
        'notes': {
            'protocol': ('NO a45/a90 residual multipliers; orientation '
                         'effect carried entirely by projection laws '
                         'lambda_eff(xi)/xi_eff(psi)/theta_bar(psi); g_s fixed '
                         'to literature values (never fitted); main-model '
                         'calibration = single S0 per material over all 3 '
                         'orientations; LOOCV = per-fold single S0 on training '
                         'orientations only; Lemaitre baseline identical '
                         'protocol (1 parameter per material).'),
        },
        'main_calibration': {
            'modified_model': {
                'materials': cal_model,
                'avg_error_pct': float(np.mean(
                    [c['error_pct'] for m in cal_model.values()
                     for c in m['cases']])),
                'avg_uts_error_pct': float(np.mean(
                    [c['uts_error_pct'] for m in cal_model.values()
                     for c in m['cases']])),
                'pass_count': int(np.sum(
                    [c['error_pct'] < 10.0 for m in cal_model.values()
                     for c in m['cases']])),
            },
            'lemaitre_baseline': {
                'materials': cal_lem,
                'avg_error_pct': float(np.mean(
                    [c['error_pct'] for m in cal_lem.values()
                     for c in m['cases']])),
                'avg_uts_error_pct': float(np.mean(
                    [c['uts_error_pct'] for m in cal_lem.values()
                     for c in m['cases']])),
            },
        },
        'loocv': {'modified_model': {'folds': []},
                  'lemaitre_baseline': {'folds': []}},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---- LOOCV ----
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = [(m, h, False) for m in MATERIALS for h in ORIENTS]
    tasks += [(m, h, True) for m in MATERIALS for h in ORIENTS]
    print(f"[parallel] {len(tasks)} LOOCV folds (8 workers)", flush=True)

    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fold_one, m, h, hm): (m, h, hm)
                for m, h, hm in tasks}
        done = 0
        for fut in as_completed(futs):
            m, h, hm = futs[fut]
            fold = fut.result()
            key = 'lemaitre_baseline' if hm else 'modified_model'
            payload['loocv'][key]['folds'].append(fold)
            done += 1
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f"[progress] {done}/{len(tasks)}", flush=True)

    for key in ('modified_model', 'lemaitre_baseline'):
        folds = payload['loocv'][key]['folds']
        payload['loocv'][key]['avg_ef_error_pct'] = float(
            np.mean([f['error_pct'] for f in folds]))
        payload['loocv'][key]['avg_uts_error_pct'] = float(
            np.mean([f['uts_error_pct'] for f in folds]))
        payload['loocv'][key]['pass_10pct'] = int(np.sum(
            [f['error_pct'] < 10.0 for f in folds]))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    mm = payload['loocv']['modified_model']
    lm = payload['loocv']['lemaitre_baseline']
    print("\n" + "=" * 88)
    print(f"MAIN CAL  modified avg={payload['main_calibration']['modified_model']['avg_error_pct']:.2f}% "
          f"UTS={payload['main_calibration']['modified_model']['avg_uts_error_pct']:.2f}%")
    print(f"MAIN CAL  lemaitre avg={payload['main_calibration']['lemaitre_baseline']['avg_error_pct']:.2f}% "
          f"UTS={payload['main_calibration']['lemaitre_baseline']['avg_uts_error_pct']:.2f}%")
    print(f"LOOCV     modified avg={mm['avg_ef_error_pct']:.2f}% "
          f"UTS={mm['avg_uts_error_pct']:.2f}% pass={mm['pass_10pct']}/9")
    print(f"LOOCV     lemaitre avg={lm['avg_ef_error_pct']:.2f}% "
          f"UTS={lm['avg_uts_error_pct']:.2f}% pass={lm['pass_10pct']}/9")
    print("=" * 88)


if __name__ == '__main__':
    main()
