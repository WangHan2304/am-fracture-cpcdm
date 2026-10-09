# -*- coding: utf-8 -*-
"""
v10 有效应力驱动协议 — experiment_v10_effective_stress.py
============================================================
响应审稿第四轮第 3 条（CP-CDM 耦合一致性）：
- 代码：流动规则用有效剪应力 tau~ = tau/(1-D)，滑移阻力 g 不随损伤缩放
  → 损伤促进屈服（不再相消），弹性刚度保持不退化的简化（与代码一致）
- 其余协议与 v9 完全相同：无 a45/a90 乘数、文献 g_s、单 S0 标定、
  LOOCV 每折仅用训练方向标定、Lemaitre 基线同协议（f_AM≡1）。

输出: simulation/output/experiment_v10_effective_stress_results.json
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


def calibrate_material(mat, homogeneous):
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
    tag = 'MODEL' if not homogeneous else 'LEM'
    print(f"  [CAL-{tag}] {mat:10s} S0={S0:8.4f} cal_err={cal_err*100:6.2f}% | "
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


def compute_aic_bic(model_folds, lemaitre_folds):
    """LOOCV 残差 r = |Δεf|（绝对值，单位应变）；σ²=mean(r²)；k=3 vs k=3；n=9。
    与论文已透明化的口径一致（AIC=2k−2lnL, BIC=k·ln n−2lnL）。"""
    stats = {}
    for key, folds in (('modified_model', model_folds),
                       ('lemaitre_baseline', lemaitre_folds)):
        r = np.array([f['error_pct'] / 100.0 for f in folds])
        n = len(r)
        k = 3  # 每材料一个 S0（三材料共 3 个拟合参数）
        sigma2 = float(np.mean(r ** 2))
        lnL = -n / 2.0 * (np.log(2.0 * np.pi * sigma2) + 1.0)
        aic = 2.0 * k - 2.0 * lnL
        bic = k * np.log(n) - 2.0 * lnL
        rms = float(np.sqrt(sigma2))
        stats[key] = {
            'n': n, 'k': k, 'sigma2': sigma2, 'lnL': float(lnL),
            'AIC': float(aic), 'BIC': float(bic), 'RMS': rms,
        }
    stats['dAIC'] = stats['modified_model']['AIC'] - stats['lemaitre_baseline']['AIC']
    stats['dBIC'] = stats['modified_model']['BIC'] - stats['lemaitre_baseline']['BIC']
    return stats


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'experiment_v10_effective_stress_results.json')

    cal_model = {m: calibrate_material(m, False) for m in MATERIALS}
    cal_lem = {m: calibrate_material(m, True) for m in MATERIALS}

    def _avg(cals, key):
        return float(np.mean([c[key] for m in cals.values()
                             for c in m['cases']]))

    payload = {
        'model': 'v10_effective_stress',
        'notes': {
            'protocol': ('v10: flow rule driven by effective resolved shear '
                         'tau~ = tau/(1-D) with slip resistance g NOT scaled by '
                         'damage (damage promotes yielding); elastic stiffness '
                         'remains undegraded (explicit simplification, matches '
                         'code); damage evolution Y = sigma^2/(2E), fracture at '
                         'D=D_c. All other v9 protocol identical: no orientation '
                         'multipliers, literature g_s, single S0 per material, '
                         'LOOCV per-fold single S0 on training orientations, '
                         'Lemaitre baseline identical (f_AM=1).'),
        },
        'main_calibration': {
            'modified_model': {
                'materials': cal_model,
                'avg_error_pct': _avg(cal_model, 'error_pct'),
                'avg_uts_error_pct': _avg(cal_model, 'uts_error_pct'),
                'pass_count': int(np.sum(
                    [c['error_pct'] < 10.0 for m in cal_model.values()
                     for c in m['cases']])),
            },
            'lemaitre_baseline': {
                'materials': cal_lem,
                'avg_error_pct': _avg(cal_lem, 'error_pct'),
                'avg_uts_error_pct': _avg(cal_lem, 'uts_error_pct'),
            },
        },
        'loocv': {'modified_model': {'folds': []},
                  'lemaitre_baseline': {'folds': []}},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

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

    payload['statistics'] = compute_aic_bic(
        payload['loocv']['modified_model']['folds'],
        payload['loocv']['lemaitre_baseline']['folds'])

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    mm = payload['loocv']['modified_model']
    lm = payload['loocv']['lemaitre_baseline']
    st = payload['statistics']
    print("\n" + "=" * 90)
    print(f"MAIN CAL  modified avg={payload['main_calibration']['modified_model']['avg_error_pct']:.2f}% "
          f"UTS={payload['main_calibration']['modified_model']['avg_uts_error_pct']:.2f}%")
    print(f"MAIN CAL  lemaitre avg={payload['main_calibration']['lemaitre_baseline']['avg_error_pct']:.2f}% "
          f"UTS={payload['main_calibration']['lemaitre_baseline']['avg_uts_error_pct']:.2f}%")
    print(f"LOOCV     modified avg={mm['avg_ef_error_pct']:.2f}% "
          f"UTS={mm['avg_uts_error_pct']:.2f}% pass={mm['pass_10pct']}/9")
    print(f"LOOCV     lemaitre avg={lm['avg_ef_error_pct']:.2f}% "
          f"UTS={lm['avg_uts_error_pct']:.2f}% pass={lm['pass_10pct']}/9")
    print(f"AIC/BIC   MOD AIC={st['modified_model']['AIC']:.2f} "
          f"BIC={st['modified_model']['BIC']:.2f} | "
          f"LEM AIC={st['lemaitre_baseline']['AIC']:.2f} "
          f"BIC={st['lemaitre_baseline']['BIC']:.2f} | "
          f"dAIC={st['dAIC']:.2f} dBIC={st['dBIC']:.2f} | "
          f"RMS {st['modified_model']['RMS']:.4f} vs {st['lemaitre_baseline']['RMS']:.4f}")
    print("=" * 90)


if __name__ == '__main__':
    main()
