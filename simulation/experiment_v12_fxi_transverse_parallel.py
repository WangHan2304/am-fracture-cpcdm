# -*- coding: utf-8 -*-
"""
experiment_v12_fxi_transverse_parallel.py
=========================================
Parallel re-implementation of experiment_v12_fxi_transverse.py under the CORRECTED
grain-morphology projection xi_eff (src/am_correction_v4.py). It reproduces the EXACT
serial selection logic (6-point S0 grid + one in-place refinement round; LOOCV per-fold
S0 on the two training orientations; AIC/BIC with k=3) but decouples the expensive
TaylorCPCDM runs from the argmin bookkeeping:

  1. Pre-warm every (material, orientation, S0, model) prediction that solve_S0 could
     ever query, in a ProcessPoolExecutor. The queried S0 set is
        S0_ALL = grid U {g/1.6 : g in grid} U {g*1.6 : g in grid}
     which is closed under a single refinement round (REFINE_ROUNDS == 1), so the
     returned best_s0 always lies in S0_ALL and the cached re-runs never miss.
  2. Re-run score_S0 / solve_S0 / calibrate / fold / aic_bic as pure arithmetic over
     the cached predictions -> numerically identical to the serial driver.

Writes the same output file (experiment_v12_fxi_transverse_results.json) so the
downstream cross-validation / blind drivers read the refreshed S0 unchanged in schema.
"""
import os
import sys
import time
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from experiment_v12_fxi_transverse import (  # noqa: E402
    run_case, MATERIALS, ORIENTS, S0_GRID, REFINE_ROUNDS, compute_aic_bic,
)

OUT = os.path.join(HERE, 'output')
PATH = os.path.join(OUT, 'experiment_v12_fxi_transverse_results.json')
LOG = os.path.join(OUT, 'rerun_v12_parallel.log')
N_WORKERS = 14

# S0 values that solve_S0 may query: the grid plus one closed refinement round.
S0_ALL = sorted(
    set(S0_GRID) | {g / 1.6 for g in S0_GRID} | {g * 1.6 for g in S0_GRID}
)


def _job(t):
    mat, psi, s0, hom = t
    r = run_case(mat, psi, s0, hom)
    return (t, float(r['fracture_strain']), float(r['uts']))


def prewarm():
    jobs = [(m, p, s0, h) for m in MATERIALS for p in ORIENTS
            for s0 in S0_ALL for h in (False, True)]
    t0 = time.time()
    pred = {}
    print(f'[prewarm] {len(jobs)} run_cases over {N_WORKERS} workers ...', flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for i, (key, ef, uts) in enumerate(ex.map(_job, jobs, chunksize=1)):
            pred[key] = (ef, uts)
            if (i + 1) % 40 == 0:
                print(f'  prewarm {i+1}/{len(jobs)}  ({time.time()-t0:.0f}s)', flush=True)
    print(f'[prewarm] done in {time.time()-t0:.0f}s', flush=True)
    return pred


# ---- cached re-implementations of the serial selection ----
def rc(pred, mat, psi, s0, hom):
    ef, uts = pred[(mat, psi, s0, hom)]
    return {'fracture_strain': ef, 'uts': uts}


def score_S0(pred, mat, oris, s0, hom):
    errs = []
    for p in oris:
        efp = rc(pred, mat, p, s0, hom)['fracture_strain']
        errs.append(abs(efp - EXPERIMENTAL[mat]['ef'][p]) / EXPERIMENTAL[mat]['ef'][p])
    return float(np.mean(errs))


def solve_S0(pred, mat, oris, hom):
    best_s0, best_e = None, None
    for s0 in S0_GRID:
        e = score_S0(pred, mat, oris, s0, hom)
        if best_e is None or e < best_e:
            best_s0, best_e = s0, e
    for _ in range(REFINE_ROUNDS):
        lo, hi = best_s0 / 1.6, best_s0 * 1.6
        for s0 in (lo, hi):
            e = score_S0(pred, mat, oris, s0, hom)
            if e < best_e:
                best_s0, best_e = s0, e
    return best_s0, best_e


def calibrate_material(pred, mat, hom):
    s0, cal_err = solve_S0(pred, mat, ORIENTS, hom)
    cases = []
    for p in ORIENTS:
        r = rc(pred, mat, p, s0, hom)
        cases.append({
            'orientation': p,
            'ef_exp': EXPERIMENTAL[mat]['ef'][p],
            'ef_pred': float(r['fracture_strain']),
            'error_pct': abs(r['fracture_strain'] - EXPERIMENTAL[mat]['ef'][p])
            / EXPERIMENTAL[mat]['ef'][p] * 100.0,
            'uts_exp_MPa': EXPERIMENTAL[mat]['uts'][p],
            'uts_pred_MPa': float(r['uts'] / 1e6),
            'uts_error_pct': abs(r['uts'] / 1e6 - EXPERIMENTAL[mat]['uts'][p])
            / EXPERIMENTAL[mat]['uts'][p] * 100.0,
        })
    out = {'material': mat, 'S0': float(s0),
           'calibration_avg_err_pct': float(cal_err * 100.0), 'cases': cases}
    tag = 'LEM' if hom else 'MODEL'
    print(f'  [CAL-{tag}] {mat:10s} S0={s0:8.4f} cal_err={cal_err*100:6.2f}% | '
          f'ef {[round(c["error_pct"],2) for c in cases]} | '
          f'UTS {[round(c["uts_error_pct"],1) for c in cases]}', flush=True)
    return out


def fold_one(pred, mat, held, hom):
    train = [o for o in ORIENTS if o != held]
    s0, train_err = solve_S0(pred, mat, train, hom)
    r = rc(pred, mat, held, s0, hom)
    ef_exp = EXPERIMENTAL[mat]['ef'][held]
    err = abs(r['fracture_strain'] - ef_exp) / ef_exp * 100.0
    uts_err = abs(r['uts'] / 1e6 - EXPERIMENTAL[mat]['uts'][held]) \
        / EXPERIMENTAL[mat]['uts'][held] * 100.0
    tag = 'LEM' if hom else 'MODEL'
    print(f'  [{tag}] {mat:10s} hold={held:>3d}deg | S0={s0:8.4f} '
          f'train_err={train_err*100:6.2f}% blind_ef={err:6.2f}% blind_uts={uts_err:5.2f}%',
          flush=True)
    return {'material': mat, 'held_orientation': held, 'S0': float(s0),
            'train_err_pct': float(train_err * 100.0),
            'blind_ef_error_pct': float(err),
            'blind_uts_error_pct': float(uts_err)}


def main():
    fh = open(LOG, 'w', encoding='utf-8')

    def emit(*a):
        s = ' '.join(str(x) for x in a)
        print(s, flush=True)
        fh.write(s + '\n'); fh.flush()

    emit('v12 PARALLEL rerun under CORRECTED xi_eff; kg=0.15; single S0/material')
    pred = prewarm()

    payload = {
        'model': 'v12_fxi_transverse',
        'notes': {
            'protocol': 'Scheme B; f_xi transverse-short-ligament 1+beta5*(1/xi_eff-1) '
                        'with CORRECTED radial conjugate-diameter xi_eff (eq:xi_eff); '
                        'gs(psi) kg=0.15 assumed; single S0 per material; LOOCV per-fold '
                        'S0 on training orientations only; Lemaitre baseline identical '
                        '(f_AM=1, same gs(psi)); AIC/BIC on blind folds, k=3 both models.',
            'xi_eff_fix': 'grain-morphology projection corrected (45deg aspect = 1); '
                          'this run replaces the pre-fix parametric-angle form.',
        },
        'main_calibration': {
            'modified_model': {'materials': {}, 'avg_error_pct': None,
                               'avg_uts_error_pct': None, 'pass_count': None},
            'lemaitre_baseline': {'materials': {}, 'avg_error_pct': None,
                                  'avg_uts_error_pct': None, 'pass_count': None}},
        'loocv': {'modified_model': {'folds': []}, 'lemaitre_baseline': {'folds': []}},
        'aic_bic': {},
    }

    for hom, key in ((False, 'modified_model'), (True, 'lemaitre_baseline')):
        mats, errs, utss, passes = {}, [], [], 0
        for mat in MATERIALS:
            c = calibrate_material(pred, mat, hom)
            mats[mat] = c
            errs.append(c['calibration_avg_err_pct'])
            utss.append(np.mean([x['uts_error_pct'] for x in c['cases']]))
            passes += sum(1 for x in c['cases'] if x['error_pct'] <= 10.0)
        payload['main_calibration'][key]['materials'] = mats
        payload['main_calibration'][key]['avg_error_pct'] = float(np.mean(errs))
        payload['main_calibration'][key]['avg_uts_error_pct'] = float(np.mean(utss))
        payload['main_calibration'][key]['pass_count'] = int(passes)
        emit(f'  CAL {key}: avg={np.mean(errs):.2f}% UTS={np.mean(utss):.2f}% pass={passes}/9')

    labels = {False: 'modified_model', True: 'lemaitre_baseline'}
    for hom in (False, True):
        for mat in MATERIALS:
            for held in ORIENTS:
                payload['loocv'][labels[hom]]['folds'].append(
                    fold_one(pred, mat, held, hom))
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

    mc = payload['main_calibration']['modified_model']
    lc = payload['main_calibration']['lemaitre_baseline']
    mm = payload['loocv']['modified_model']
    lm = payload['loocv']['lemaitre_baseline']
    st = payload['aic_bic']
    emit('S0 per material: ' + str({m: round(mc['materials'][m]['S0'], 4) for m in MATERIALS}))
    emit(f"MAIN CAL  modified avg={mc['avg_error_pct']:.2f}% UTS={mc['avg_uts_error_pct']:.2f}% pass={mc['pass_count']}/9")
    emit(f"MAIN CAL  lemaitre avg={lc['avg_error_pct']:.2f}% UTS={lc['avg_uts_error_pct']:.2f}% pass={lc['pass_count']}/9")
    emit(f"LOOCV     modified avg={mm['avg_ef_error_pct']:.2f}% UTS={mm['avg_uts_error_pct']:.2f}% pass={mm['pass_10pct']}/9")
    emit(f"LOOCV     lemaitre avg={lm['avg_ef_error_pct']:.2f}% UTS={lm['avg_uts_error_pct']:.2f}% pass={lm['pass_10pct']}/9")
    emit(f"AIC/BIC   MOD AIC={st['modified_model']['AIC']:.2f} BIC={st['modified_model']['BIC']:.2f} RMS={st['modified_model']['RMS']:.4f} | "
         f"LEM AIC={st['lemaitre_baseline']['AIC']:.2f} BIC={st['lemaitre_baseline']['BIC']:.2f} RMS={st['lemaitre_baseline']['RMS']:.4f} | "
         f"dAIC={st['dAIC']:.2f} dBIC={st['dBIC']:.2f}")
    emit(f'DONE saved {PATH}')
    fh.close()


if __name__ == '__main__':
    main()
