# -*- coding: utf-8 -*-
"""
experiment_v13_gtn.py — GTN 公平基线
=====================================
R7⑧：CP+GTN 同协议公平比较。
- GTN 基线：Taylor 塑性 + GTN 形核驱动孔隙演化（e_N=0.15, s_N=0.10, f_c=0.15 文献固定；
  损伤不反馈应力，与主模型 one-way 同构）
- 每材料唯一拟合参数 = f_N（形核总量），与主模型单 S0 对称 → AIC/BIC k=3 公平
- 主标定（3 材料）+ LOOCV（9 折，模块级 worker 并行）+ AIC/BIC
输出: output/experiment_v13_gtn_results.json
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from materials import EXPERIMENTAL
from taylor_cpcdm import TaylorCPCDM

N_GRAINS = 20
STRAIN_RATE = 1e-3
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]
A_GRID = [0.1, 0.2, 0.4, 0.8, 1.5, 3.0, 6.0]
REFINE_ROUNDS = 1
OUT = os.path.join(os.path.dirname(__file__), 'output')
PATH = os.path.join(OUT, 'experiment_v13_gtn_results.json')


def _eps_step(mat):
    return 1e-3 if mat == '316L' else 5e-4


def run_gtn(mat, psi, A):
    max_strain = EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02 \
        if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01
    model = TaylorCPCDM(
        mat, psi, S0_override=0.32, n_grains=N_GRAINS,
        eps_step=_eps_step(mat), n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=False,
        param_overrides={'A': A}, seed=42, damage_model='gtn')
    return model.run_uniaxial(max_strain)


def score_A(mat, oris, A):
    errs = []
    for ps in oris:
        efp = run_gtn(mat, ps, A)['fracture_strain']
        errs.append(abs(efp - EXPERIMENTAL[mat]['ef'][ps])
                    / EXPERIMENTAL[mat]['ef'][ps])
    return float(np.mean(errs))


def solve_A(mat, oris):
    best, best_e = None, None
    for A in A_GRID:
        e = score_A(mat, oris, A)
        if best_e is None or e < best_e:
            best, best_e = A, e
    for _ in range(REFINE_ROUNDS):
        lo, hi = best / 1.6, best * 1.6
        for A in (lo, hi):
            e = score_A(mat, oris, A)
            if e < best_e:
                best, best_e = A, e
    return best, best_e


def calibrate_material(mat):
    t0 = time.time()
    A, cal_err = solve_A(mat, ORIENTS)
    cases = []
    for ps in ORIENTS:
        r = run_gtn(mat, ps, A)
        cases.append({'orientation': ps,
                      'ef_exp': EXPERIMENTAL[mat]['ef'][ps],
                      'ef_pred': float(r['fracture_strain']),
                      'error_pct': abs(r['fracture_strain'] - EXPERIMENTAL[mat]['ef'][ps])
                                   / EXPERIMENTAL[mat]['ef'][ps] * 100.0,
                      'uts_exp_MPa': EXPERIMENTAL[mat]['uts'][ps],
                      'uts_pred_MPa': float(r['uts'] / 1e6),
                      'uts_error_pct': abs(r['uts'] / 1e6 - EXPERIMENTAL[mat]['uts'][ps])
                                       / EXPERIMENTAL[mat]['uts'][ps] * 100.0})
    out = {'material': mat, 'A': float(A),
           'calibration_avg_err_pct': float(cal_err * 100.0),
           'cases': cases, 'walltime_s': round(time.time() - t0, 1)}
    print(f"  [GTN-CAL] {mat:10s} A={A:6.3f} cal_err={cal_err*100:6.2f}% | "
          f"ef {[round(c['error_pct'], 2) for c in cases]} | "
          f"UTS {[round(c['uts_error_pct'], 1) for c in cases]} | {out['walltime_s']}s",
          flush=True)
    return out


def fold_one(mat, held):
    train = [o for o in ORIENTS if o != held]
    t0 = time.time()
    A, train_err = solve_A(mat, train)
    r = run_gtn(mat, held, A)
    ef_exp = EXPERIMENTAL[mat]['ef'][held]
    err = abs(r['fracture_strain'] - ef_exp) / ef_exp * 100.0
    uts_err = abs(r['uts'] / 1e6 - EXPERIMENTAL[mat]['uts'][held]) \
        / EXPERIMENTAL[mat]['uts'][held] * 100.0
    print(f"  [GTN-LOOCV] {mat:10s} hold={held:>3d}deg | A={A:6.3f} "
          f"train_err={train_err*100:6.2f}% blind_ef={err:6.2f}% "
          f"blind_uts={uts_err:5.2f}% | {time.time()-t0:.0f}s", flush=True)
    return {'material': mat, 'held_orientation': held, 'A': float(A),
            'train_err_pct': float(train_err * 100.0),
            'blind_ef_error_pct': float(err),
            'blind_uts_error_pct': float(uts_err)}


def _worker(t):
    mat, held = t
    return fold_one(mat, held)


def main():
    print('v13 GTN baseline: f_N fitted per material (k=3), e_N=0.15 s_N=0.10 f_c=0.15 fixed', flush=True)
    payload = {
        'model': 'v13_gtn_baseline',
        'notes': {
            'protocol': 'GTN-type baseline: Taylor plasticity + strain-controlled constant '
                        'nucleation rate f_dot=A*p_dot (AM defect-driven; growth term vanishes under '
                        'incompressible Taylor plasticity); f_c=0.15 literature-fixed; damage non-interacting '
                        '(same one-way structure as modified model); single fitted A per material; '
                        'LOOCV per-fold A on training orientations only; AIC/BIC k=3.'
        },
        'main_calibration': {'materials': {}, 'avg_error_pct': None, 'avg_uts_error_pct': None, 'pass_count': None},
        'loocv': {'folds': []},
        'aic_bic': {},
    }
    mats, errs, utss, passes = {}, [], [], 0
    for mat in MATERIALS:
        c = calibrate_material(mat)
        mats[mat] = c
        errs.append(c['calibration_avg_err_pct'])
        utss.append(np.mean([x['uts_error_pct'] for x in c['cases']]))
        passes += sum(1 for x in c['cases'] if x['error_pct'] <= 10.0)
    payload['main_calibration']['materials'] = mats
    payload['main_calibration']['avg_error_pct'] = float(np.mean(errs))
    payload['main_calibration']['avg_uts_error_pct'] = float(np.mean(utss))
    payload['main_calibration']['pass_count'] = int(passes)
    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    tasks = [(m, h) for m in MATERIALS for h in ORIENTS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        payload['loocv']['folds'] = list(ex.map(_worker, tasks))
    folds = payload['loocv']['folds']
    payload['loocv']['avg_ef_error_pct'] = float(np.mean([f['blind_ef_error_pct'] for f in folds]))
    payload['loocv']['avg_uts_error_pct'] = float(np.mean([f['blind_uts_error_pct'] for f in folds]))
    payload['loocv']['pass_10pct'] = int(np.sum([f['blind_ef_error_pct'] <= 10.0 for f in folds]))

    rs = [f['blind_ef_error_pct'] / 100.0 * EXPERIMENTAL[f['material']]['ef'][f['held_orientation']]
          for f in folds]
    n, k = len(rs), 3
    sigma2 = float(np.mean(np.array(rs) ** 2))
    lnL = -n / 2.0 * np.log(2 * np.pi * sigma2) - n / 2.0
    payload['aic_bic'] = {
        'AIC': 2 * k - 2 * lnL, 'BIC': k * np.log(n) - 2 * lnL,
        'RMS': float(np.sqrt(sigma2)), 'lnL': lnL, 'k': k, 'n': n}
    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    mc = payload['main_calibration']
    lv = payload['loocv']
    st = payload['aic_bic']
    print(f"GTN MAIN  avg={mc['avg_error_pct']:.2f}% UTS={mc['avg_uts_error_pct']:.2f}% pass={mc['pass_count']}/9")
    print(f"GTN LOOCV avg={lv['avg_ef_error_pct']:.2f}% UTS={lv['avg_uts_error_pct']:.2f}% pass={lv['pass_10pct']}/9")
    print(f"GTN AIC={st['AIC']:.2f} BIC={st['BIC']:.2f} RMS={st['RMS']:.4f}")
    print(f'DONE saved {PATH}')


if __name__ == '__main__':
    main()
