# -*- coding: utf-8 -*-
"""
方案A 相对归一化投影律（优化版）— experiment_v15_lambda_rel.py
=============================================================
与 experiment_v12_fxi_transverse.py 同协议，唯一区别：修正模型的熔池边界投影采用
**相对归一化** λ̄_rel(ψ)=λ_eff(ψ)/λ_eff(0°)（am_correction_v4 的 _lambda_norm='rel'）。

双轨并列：仅重算 rel 修正模型（校准 + LOOCV），Lemaitre 基线复用 v12 折
（force_homogeneous → f_AM=1，λ 归一化模式无关 → 与 v12 完全一致）。
主校准的 S0 网格按 (S0, orientation) 并行（数学不变，逐任务确定性）。

输出: simulation/output/experiment_v15_lambda_rel_results.json
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
PATH = os.path.join(OUT, 'experiment_v15_lambda_rel_results.json')
V12PATH = os.path.join(OUT, 'experiment_v12_fxi_transverse_results.json')
NPROC = int(os.cpu_count() or 8)


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
    ov['_lambda_norm'] = 'rel'          # 方案A：相对归一化（LEM 时 f_AM=1 该项无效）
    model = TaylorCPCDM(
        mat, psi, S0_override=S0, n_grains=N_GRAINS,
        eps_step=_eps_step(mat), n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=homogeneous,
        param_overrides=ov, seed=42,
    )
    return model.run_uniaxial(max_strain)


# ---- parallel grid evaluation for MAIN calibration (rel modified model) ----
def _sim_task(t):
    mat, psi, S0 = t
    r = run_case(mat, psi, S0, homogeneous=False)
    ef = float(r['fracture_strain'])
    err = abs(ef - EXPERIMENTAL[mat]['ef'][psi]) / EXPERIMENTAL[mat]['ef'][psi]
    return (S0, psi, ef, err, float(r['uts'] / 1e6))


def solve_S0_parallel(ex, mat, oris):
    jobs = [(mat, psi, S0) for S0 in S0_GRID for psi in oris]
    res = list(ex.map(_sim_task, jobs))
    agg = {}
    for S0, psi, ef, err, uts in res:
        agg.setdefault(S0, []).append(err)
    means = {S0: float(np.mean(v)) for S0, v in agg.items()}
    best_s0 = min(means, key=means.get)
    best_e = means[best_s0]
    for _ in range(REFINE_ROUNDS):
        lo, hi = best_s0 / 1.6, best_s0 * 1.6
        rj = [(mat, psi, S0) for S0 in (lo, hi) for psi in oris]
        rr = list(ex.map(_sim_task, rj))
        aggr = {}
        for S0, psi, ef, err, uts in rr:
            aggr.setdefault(S0, []).append(err)
        for S0, v in aggr.items():
            m = float(np.mean(v))
            if m < best_e:
                best_e, best_s0 = m, S0
    return best_s0, best_e


def calibrate_material(ex, mat):
    t0 = time.time()
    S0, cal_err = solve_S0_parallel(ex, mat, ORIENTS)
    cases = []
    for psi in ORIENTS:
        ef_pred = run_case(mat, psi, S0, False)['fracture_strain']
        uts_pred = run_case(mat, psi, S0, False)['uts'] / 1e6
        cases.append({
            'orientation': psi, 'ef_exp': EXPERIMENTAL[mat]['ef'][psi],
            'ef_pred': float(ef_pred),
            'error_pct': abs(ef_pred - EXPERIMENTAL[mat]['ef'][psi]) / EXPERIMENTAL[mat]['ef'][psi] * 100.0,
            'uts_exp_MPa': EXPERIMENTAL[mat]['uts'][psi], 'uts_pred_MPa': float(uts_pred),
            'uts_error_pct': abs(uts_pred - EXPERIMENTAL[mat]['uts'][psi]) / EXPERIMENTAL[mat]['uts'][psi] * 100.0,
        })
    ratio_pred = cases[0]['ef_pred'] / cases[2]['ef_pred'] if cases[2]['ef_pred'] > 0 else float('nan')
    ratio_exp = cases[0]['ef_exp'] / cases[2]['ef_exp']
    out = {'material': mat, 'S0': float(S0), 'calibration_avg_err_pct': float(cal_err * 100.0),
           'ratio_0_90_pred': float(ratio_pred), 'ratio_0_90_exp': float(ratio_exp),
           'cases': cases, 'walltime_s': round(time.time() - t0, 1)}
    print(f"  [CAL-REL] {mat:10s} S0={S0:8.4f} cal_err={cal_err*100:6.2f}% | "
          f"ef {[round(c['error_pct'], 2) for c in cases]} | ratio0/90 pred={ratio_pred:.2f} exp={ratio_exp:.2f} | "
          f"{out['walltime_s']}s", flush=True)
    return out


# ---- LOOCV folds (serial solve inside each parallel worker) ----
def score_S0(mat, oris, S0):
    errs = [abs(run_case(mat, ps, S0, False)['fracture_strain'] - EXPERIMENTAL[mat]['ef'][ps])
            / EXPERIMENTAL[mat]['ef'][ps] for ps in oris]
    return float(np.mean(errs))


def solve_S0_serial(mat, oris):
    best_s0, best_e = None, None
    for s0 in S0_GRID:
        e = score_S0(mat, oris, s0)
        if best_e is None or e < best_e:
            best_s0, best_e = s0, e
    for _ in range(REFINE_ROUNDS):
        lo, hi = best_s0 / 1.6, best_s0 * 1.6
        for s0 in (lo, hi):
            e = score_S0(mat, oris, s0)
            if e < best_e:
                best_s0, best_e = s0, e
    return best_s0, best_e


def fold_one(mat, held):
    train = [o for o in ORIENTS if o != held]
    t0 = time.time()
    S0, train_err = solve_S0_serial(mat, train)
    r = run_case(mat, held, S0, False)
    ef_exp = EXPERIMENTAL[mat]['ef'][held]
    ef_pred = float(r['fracture_strain'])
    err = abs(ef_pred - ef_exp) / ef_exp * 100.0
    uts_err = abs(r['uts'] / 1e6 - EXPERIMENTAL[mat]['uts'][held]) / EXPERIMENTAL[mat]['uts'][held] * 100.0
    print(f"  [REL-LOOCV] {mat:10s} hold={held:>3d}deg | S0={S0:8.4f} train_err={train_err*100:6.2f}% "
          f"blind_ef={err:6.2f}% blind_uts={uts_err:5.2f}% | {time.time()-t0:.0f}s", flush=True)
    return {'material': mat, 'held_orientation': held, 'S0': float(S0),
            'train_err_pct': float(train_err * 100.0), 'blind_ef_error_pct': float(err),
            'blind_uts_error_pct': float(uts_err)}


def _fold_worker(t):
    mat, held = t
    return fold_one(mat, held)


def compute_aic_bic(model_folds, lem_folds):
    n = len(model_folds)
    k = 3
    stats = {}
    for key, folds in (('modified_model_rel', model_folds), ('lemaitre_baseline', lem_folds)):
        rs = [f['blind_ef_error_pct'] / 100.0 * EXPERIMENTAL[f['material']]['ef'][f['held_orientation']] for f in folds]
        sigma2 = float(np.mean(np.array(rs) ** 2))
        lnL = -n / 2.0 * np.log(2 * np.pi * sigma2) - n / 2.0
        stats[key] = {'AIC': 2 * k - 2 * lnL, 'BIC': k * np.log(n) - 2 * lnL,
                      'RMS': float(np.sqrt(sigma2)), 'lnL': lnL, 'k': k, 'n': n}
    stats['dAIC'] = stats['modified_model_rel']['AIC'] - stats['lemaitre_baseline']['AIC']
    stats['dBIC'] = stats['modified_model_rel']['BIC'] - stats['lemaitre_baseline']['BIC']
    return stats


def paired_bootstrap(model_folds, lem_folds, n_boot=100000, seed=42):
    # align by (material, held orientation)
    lm = {(f['material'], f['held_orientation']): f['blind_ef_error_pct'] for f in lem_folds}
    md = np.array([f['blind_ef_error_pct'] - lm[(f['material'], f['held_orientation'])] for f in model_folds])
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(md, size=len(md), replace=True).mean() for _ in range(n_boot)])
    # wilcoxon + sign test
    nz = md[md != 0]
    ranks = np.argsort(np.argsort(np.abs(nz))) + 1
    w_pos = ranks[nz > 0].sum()
    n = len(md)
    sign_p = float(min(1.0, 2 * sum(__import__('math').comb(n, i) for i in range(min((md > 0).sum(), (md < 0).sum()), n + 1)) / (2 ** n)))
    return {'mean_diff_pp': float(md.mean()),
            'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            'n_mod_worse': int((md > 0).sum()), 'n_mod_better': int((md < 0).sum()),
            'sign_test_p': sign_p, 'wilcoxon_Wpos': float(w_pos)}


def main():
    print(f'v15 rel-normalization: parallel calib grid (NPROC={NPROC}); Lemaitre reused from v12', flush=True)
    v12 = json.load(open(V12PATH, encoding='utf-8'))
    lem_folds = v12['loocv']['lemaitre_baseline']['folds']
    lem_cal = v12['main_calibration']['lemaitre_baseline']

    payload = {'model': 'v15_lambda_rel',
               'notes': {'protocol': 'Scheme B; RELATIVE MPB normalization lambda_eff(psi)/lambda_eff(0); '
                                     'f_xi transverse-short-ligament; gs(psi) kg=0.15; single S0/material; '
                                     'LOOCV per-fold S0; Lemaitre baseline reused from v12 (f_AM=1, '
                                     'normalization-independent); AIC/BIC blind folds, k=3 both.'},
               'main_calibration': {'modified_model_rel': {'materials': {}},
                                    'lemaitre_baseline_reused': lem_cal},
               'loocv': {'modified_model_rel': {'folds': []}, 'lemaitre_baseline': {'folds': lem_folds}},
               'aic_bic': {}, 'bootstrap': {}}

    # ---- rel main calibration (parallel grid) ----
    t_start = time.time()
    mats, errs, utss, passes = {}, [], [], 0
    with ProcessPoolExecutor(max_workers=NPROC) as ex:
        for mat in MATERIALS:
            c = calibrate_material(ex, mat)
            mats[mat] = c
            errs.append(c['calibration_avg_err_pct'])
            utss.append(np.mean([x['uts_error_pct'] for x in c['cases']]))
            passes += sum(1 for x in c['cases'] if x['error_pct'] <= 10.0)
        payload['main_calibration']['modified_model_rel'] = {
            'materials': mats, 'avg_error_pct': float(np.mean(errs)),
            'avg_uts_error_pct': float(np.mean(utss)), 'pass_count': int(passes)}
        with open(PATH, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        # ---- rel LOOCV (18 folds parallel) ----
        tasks = [(m, h) for m in MATERIALS for h in ORIENTS]
        folds = list(ex.map(_fold_worker, tasks))
    payload['loocv']['modified_model_rel']['folds'] = folds
    # checkpoint folds immediately so downstream aggregation survives interruption
    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    payload['loocv']['modified_model_rel']['avg_ef_error_pct'] = float(np.mean([f['blind_ef_error_pct'] for f in folds]))
    payload['loocv']['modified_model_rel']['avg_uts_error_pct'] = float(np.mean([f['blind_uts_error_pct'] for f in folds]))
    payload['loocv']['modified_model_rel']['pass_10pct'] = int(np.sum([f['blind_ef_error_pct'] <= 10.0 for f in folds]))
    payload['loocv']['lemaitre_baseline']['avg_ef_error_pct'] = v12['loocv']['lemaitre_baseline']['avg_ef_error_pct']
    payload['aic_bic'] = compute_aic_bic(folds, lem_folds)
    payload['bootstrap'] = paired_bootstrap(folds, lem_folds)
    payload['total_walltime_s'] = round(time.time() - t_start, 1)
    with open(PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    mc = payload['main_calibration']['modified_model_rel']
    mm = payload['loocv']['modified_model_rel']
    st = payload['aic_bic']
    print(f"CAL  rel avg={mc['avg_error_pct']:.2f}% UTS={mc['avg_uts_error_pct']:.2f}% pass={mc['pass_count']}/9 "
          f"per-mat={ {m: round(c['calibration_avg_err_pct'],2) for m,c in mc['materials'].items()} }")
    print(f"RATIO0/90 rel={ {m: round(c['ratio_0_90_pred'],2) for m,c in mc['materials'].items()} } "
          f"exp={ {m: round(c['ratio_0_90_exp'],2) for m,c in mc['materials'].items()} }")
    print(f"LOOCV rel avg={mm['avg_ef_error_pct']:.2f}% (lem {mm.get('avg_ef_error_pct')} vs reused lem={payload['loocv']['lemaitre_baseline']['avg_ef_error_pct']:.2f}%) pass={mm['pass_10pct']}/9")
    print(f"AIC rel={st['modified_model_rel']['AIC']:.2f} RMS={st['modified_model_rel']['RMS']:.4f} | lem={st['lemaitre_baseline']['AIC']:.2f} RMS={st['lemaitre_baseline']['RMS']:.4f} | dAIC={st['dAIC']:.2f} dBIC={st['dBIC']:.2f}")
    print(f"BOOT {payload['bootstrap']}")
    print(f"DONE saved {PATH} total={payload['total_walltime_s']}s")


if __name__ == '__main__':
    main()
