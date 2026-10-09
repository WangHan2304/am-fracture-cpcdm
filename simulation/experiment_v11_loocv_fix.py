# -*- coding: utf-8 -*-
"""
v11 LOOCV 修复重跑 — experiment_v11_loocv_fix.py
================================================
修复 experiment_v11_gs_directional.py 的 LOOCV 崩溃
(ProcessPoolExecutor 内局部 lambda 不可 pickle)。
- 只重跑 LOOCV（9 折 × 2 模型，kg=0.15）；主标定结果从既有 JSON 读入合并
- 模块级 worker，无 lambda / 无局部函数 → 子进程可正常序列化

输出: simulation/output/experiment_v11_gs_directional_results.json（完整合并版）
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

N_GRAINS = 20
STRAIN_RATE = 1e-3
EPS_STEP_DEFAULT = 5e-4
EPS_STEP_316L = 1e-3
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]
S0_GRID = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0]
REFINE_ROUNDS = 1
K_G = 0.15
XI0 = {'Ti64': 3.5, '316L': 2.5, 'AlSi10Mg': 1.5}
OUT = os.path.join(os.path.dirname(__file__), 'output')
PATH = os.path.join(OUT, 'experiment_v11_gs_directional_results.json')


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def _gs_literature(mat):
    if mat == 'Ti64':
        return {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}


def _gs_directional(mat, psi):
    xi0 = XI0[mat]
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(xi0, psi) / xi0   # xi_eff(psi)/xi0 (am_correction_v4)
    h = 1.0 - K_G * (1.0 - ratio)
    base = _gs_literature(mat)
    return {k: v * h for k, v in base.items()}


def run_case(mat, psi, S0, homogeneous):
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    ov = _gs_directional(mat, psi)
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
    tag = 'MODEL' if not homogeneous else 'LEM'
    print(f"  [{tag}] {mat:10s} hold={held:>3d}deg | S0={S0:8.4f} "
          f"train_err={train_err*100:6.2f}% blind_ef={err:6.2f}% "
          f"blind_uts={uts_err:5.2f}% | {time.time()-t0:.0f}s", flush=True)
    return {'material': mat, 'held_orientation': held, 'S0': float(S0),
            'train_err_pct': float(train_err * 100.0),
            'blind_ef_error_pct': float(err),
            'blind_uts_error_pct': float(uts_err)}


def compute_aic_bic(model_folds, lemaitre_folds):
    n = len(model_folds)
    k = 3
    stats = {}
    for key, folds in (('modified_model', model_folds), ('lemaitre_baseline', lemaitre_folds)):
        rs = [f['blind_ef_error_pct'] / 100.0 * EXPERIMENTAL[f['material']]['ef'][f['held_orientation']]
              for f in folds]
        rms = float(np.sqrt(np.mean(np.array(rs) ** 2)))
        sigma2 = float(np.mean(np.array(rs) ** 2))
        lnL = -n / 2.0 * np.log(2 * np.pi * sigma2) - n / 2.0
        aic = 2 * k - 2 * lnL
        bic = k * np.log(n) - 2 * lnL
        stats[key] = {'AIC': aic, 'BIC': bic, 'RMS': rms, 'lnL': lnL, 'k': k, 'n': n}
    stats['dAIC'] = stats['modified_model']['AIC'] - stats['lemaitre_baseline']['AIC']
    stats['dBIC'] = stats['modified_model']['BIC'] - stats['lemaitre_baseline']['BIC']
    return stats


def _worker(t):
    """模块级 worker（避免局部 lambda 不可 pickle）"""
    mat, held, homogeneous = t
    return (homogeneous, fold_one(mat, held, homogeneous))


def main():
    print('v11 LOOCV fix: kg=0.15, 9 folds x 2 models', flush=True)
    with open(PATH, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    tasks = [(m, h, hm) for hm, _ in ((False, 'modified_model'), (True, 'lemaitre_baseline'))
             for m in MATERIALS for h in ORIENTS]
    labels = {False: 'modified_model', True: 'lemaitre_baseline'}
    payload['loocv'] = {'modified_model': {'folds': []}, 'lemaitre_baseline': {'folds': []}}
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_worker, tasks))
    for hm, fold in results:
        payload['loocv'][labels[hm]]['folds'].append(fold)
    for key in ('modified_model', 'lemaitre_baseline'):
        folds = payload['loocv'][key]['folds']
        payload['loocv'][key]['avg_ef_error_pct'] = float(
            np.mean([f['blind_ef_error_pct'] for f in folds]))
        payload['loocv'][key]['avg_uts_error_pct'] = float(
            np.mean([f['blind_uts_error_pct'] for f in folds]))
        payload['loocv'][key]['pass_10pct'] = int(np.sum(
            [f['blind_ef_error_pct'] <= 10.0 for f in folds]))
    payload['aic_bic'] = compute_aic_bic(
        payload['loocv']['modified_model']['folds'],
        payload['loocv']['lemaitre_baseline']['folds'])
    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    mm, lm = payload['loocv']['modified_model'], payload['loocv']['lemaitre_baseline']
    mc, lc = payload['main_calibration']['modified_model'], payload['main_calibration']['lemaitre_baseline']
    st = payload['aic_bic']
    print(f"MAIN CAL  modified avg={mc['avg_error_pct']:.2f}% UTS={mc['avg_uts_error_pct']:.2f}% pass={mc['pass_count']}/9")
    print(f"MAIN CAL  lemaitre avg={lc['avg_error_pct']:.2f}% UTS={lc['avg_uts_error_pct']:.2f}% pass={lc['pass_count']}/9")
    print(f"LOOCV     modified avg={mm['avg_ef_error_pct']:.2f}% UTS={mm['avg_uts_error_pct']:.2f}% pass={mm['pass_10pct']}/9")
    print(f"LOOCV     lemaitre avg={lm['avg_ef_error_pct']:.2f}% UTS={lm['avg_uts_error_pct']:.2f}% pass={lm['pass_10pct']}/9")
    print(f"AIC/BIC   MOD AIC={st['modified_model']['AIC']:.2f} BIC={st['modified_model']['BIC']:.2f} RMS={st['modified_model']['RMS']:.4f} | "
          f"LEM AIC={st['lemaitre_baseline']['AIC']:.2f} BIC={st['lemaitre_baseline']['BIC']:.2f} RMS={st['lemaitre_baseline']['RMS']:.4f} | "
          f"dAIC={st['dAIC']:.2f} dBIC={st['dBIC']:.2f}")
    print(f'DONE saved {PATH}')


if __name__ == '__main__':
    main()
