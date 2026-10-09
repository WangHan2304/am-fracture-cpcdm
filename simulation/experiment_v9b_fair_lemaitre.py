"""
v9b 公平基线 — experiment_v9b_fair_lemaitre.py
=============================================
修正 v9 的公平性缺陷：Lemaitre 基线必须与主模型使用【相同】聚合方式
（非齐次 Taylor、相同 30 晶粒纹理、相同硬化），仅损伤律不同
（f_AM≡1）。MODEL 结果直接复用 v9 已完成的 S0 与 case 数据；
LEM 用 force_homogeneous=False 重跑主标定 + 9 折 LOOCV。
输出合并 JSON：experiment_v9_pure_projection_results.json（覆盖更新 LEM 部分）
"""
import json
import os
import sys
import time

import numpy as np

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
PATH = os.path.join(OUT, 'experiment_v9_pure_projection_results.json')
MODEL_S0_DONE = {'Ti64': 0.32, '316L': 3.2, 'AlSi10Mg': 0.2}  # v9 已定 MODEL S0


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def _gs_literature(mat):
    if mat == 'Ti64':
        return {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}


def run_case(mat, psi, S0, homogeneous):
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    ov = _gs_literature(mat)
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


def material_block_lem(mat, S0):
    """LEM（非齐次 Taylor，同聚合）3 方向完整 case。"""
    t0 = time.time()
    cases = []
    for ps in ORIENTS:
        r = run_case(mat, ps, S0, False)
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
        'calibration_avg_err_pct': float(np.mean(
            [c['error_pct'] for c in cases])),
        'cases': cases, 'walltime_s': round(time.time() - t0, 1),
    }
    print(f"  [BLK-LEM-fair] {mat:10s} S0={S0:8.4f} "
          f"cal_err={out['calibration_avg_err_pct']:6.2f}% | "
          f"ef {[round(c['error_pct'],2) for c in cases]} | "
          f"UTS {[round(c['uts_error_pct'],1) for c in cases]} | "
          f"{out['walltime_s']}s", flush=True)
    return out


def fold_one_lem(mat, held):
    train = [o for o in ORIENTS if o != held]
    t0 = time.time()
    S0, train_err = solve_S0(mat, train, False)  # 与主模型相同聚合
    r = run_case(mat, held, S0, False)
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
    print(f"  [LEMAITRE-fair] {mat:10s} hold={held:>3d}deg | S0={S0:8.4f} "
          f"train_err={train_err*100:5.2f}% | ef {ef_exp:.4f}->{ef_pred:.4f} "
          f"err={err:6.2f}% | UTS err={uts_err:5.2f}% | {fold['walltime_s']}s",
          flush=True)
    return fold


def aic_bic(model_folds, lem_folds, k_model, k_lem):
    def stats(folds):
        rs = np.array([abs(f['ef_pred'] - f['ef_exp']) for f in folds])
        s2 = float(np.mean(rs ** 2))
        lnL = -len(rs) / 2 * np.log(2 * np.pi * s2) - len(rs) / 2
        return rs, s2, lnL
    rs_m, s2_m, lnL_m = stats(model_folds)
    rs_l, s2_l, lnL_l = stats(lem_folds)
    aic_m, bic_m = 2 * k_model - 2 * lnL_m, k_model * np.log(len(rs_m)) - 2 * lnL_m
    aic_l, bic_l = 2 * k_lem - 2 * lnL_l, k_lem * np.log(len(rs_l)) - 2 * lnL_l
    return {
        'n': int(len(rs_m)),
        'k_modified': int(k_model), 'k_lemaitre': int(k_lem),
        'rms_modified': float(np.sqrt(s2_m)),
        'rms_lemaitre': float(np.sqrt(s2_l)),
        'lnL_modified': float(lnL_m), 'lnL_lemaitre': float(lnL_l),
        'AIC_modified': float(aic_m), 'AIC_lemaitre': float(aic_l),
        'BIC_modified': float(bic_m), 'BIC_lemaitre': float(bic_l),
        'dAIC': float(aic_m - aic_l), 'dBIC': float(bic_m - bic_l),
    }


def main():
    # 读取 v9 既有结果（MODEL 部分保留）
    with open(PATH, encoding='utf-8') as f:
        payload = json.load(f)

    # ---- LEM 公平版主标定 ----
    print("== LEM fair main calibration (same non-homogeneous Taylor) ==", flush=True)
    cal_lem = {}
    for m in MATERIALS:
        t0 = time.time()
        S0, err = solve_S0(m, ORIENTS, False)
        cal_lem[m] = material_block_lem(m, S0)

    payload['main_calibration']['lemaitre_baseline'] = {
        'materials': cal_lem,
        'avg_error_pct': float(np.mean(
            [c['error_pct'] for m in cal_lem.values() for c in m['cases']])),
        'avg_uts_error_pct': float(np.mean(
            [c['uts_error_pct'] for m in cal_lem.values() for c in m['cases']])),
        'note': 'v9b: same non-homogeneous Taylor aggregate, texture and '
                'hardening as modified model; only damage law differs (f_AM=1).',
    }
    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # ---- LEM 公平版 LOOCV 9 折 ----
    from concurrent.futures import ProcessPoolExecutor, as_completed
    tasks = [(m, h) for m in MATERIALS for h in ORIENTS]
    print(f"[parallel] {len(tasks)} LEM-fair LOOCV folds (8 workers)", flush=True)
    payload['loocv']['lemaitre_baseline'] = {'folds': []}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fold_one_lem, m, h): (m, h) for m, h in tasks}
        done = 0
        for fut in as_completed(futs):
            m, h = futs[fut]
            fold = fut.result()
            payload['loocv']['lemaitre_baseline']['folds'].append(fold)
            done += 1
            with open(PATH, 'w', encoding='utf-8') as f:
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

    mm_folds = payload['loocv']['modified_model']['folds']
    lm_folds = payload['loocv']['lemaitre_baseline']['folds']
    payload['statistics'] = {
        'AIC_BIC_LOOCV': aic_bic(mm_folds, lm_folds, k_model=3, k_lem=3),
        'note': 'v9b: Lemaitre uses identical aggregate (non-homogeneous '
                'Taylor, same texture/hardening); both k=3.',
    }
    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    mm = payload['loocv']['modified_model']
    lm = payload['loocv']['lemaitre_baseline']
    st = payload['statistics']['AIC_BIC_LOOCV']
    print("\n" + "=" * 92)
    print(f"MAIN CAL modified avg={payload['main_calibration']['modified_model']['avg_error_pct']:.2f}% "
          f"UTS={payload['main_calibration']['modified_model']['avg_uts_error_pct']:.2f}%")
    print(f"MAIN CAL lemaitre avg={payload['main_calibration']['lemaitre_baseline']['avg_error_pct']:.2f}% "
          f"UTS={payload['main_calibration']['lemaitre_baseline']['avg_uts_error_pct']:.2f}%")
    print(f"LOOCV     modified avg={mm['avg_ef_error_pct']:.2f}% "
          f"UTS={mm['avg_uts_error_pct']:.2f}% pass={mm['pass_10pct']}/9")
    print(f"LOOCV     lemaitre avg={lm['avg_ef_error_pct']:.2f}% "
          f"UTS={lm['avg_uts_error_pct']:.2f}% pass={lm['pass_10pct']}/9")
    print(f"AIC/BIC   modified {st['AIC_modified']:.1f}/{st['BIC_modified']:.1f} "
          f"lemaitre {st['AIC_lemaitre']:.1f}/{st['BIC_lemaitre']:.1f} "
          f"dAIC={st['dAIC']:.1f} dBIC={st['dBIC']:.1f}")
    print("=" * 92)


if __name__ == '__main__':
    main()
