"""
LOOCV-2：每折只标定 S0 — experiment_v8_loocv_S0only.py
=====================================================
响应第二轮审稿 3a："每折只标定 S0，固定所有微结构系数"。

协议：
- 微结构系数（β_{1-5}、m_{1,2}、φ、φ_crit、λ、ξ、θ、D_0、p_D、D_c）与取向乘数
  （a45/a90 分段 f_aniso）以及 Voce 饱和应力 g_s 均取 v7 全局校准值，全程固定；
- 每折（留出一个取向）仅重新标定损伤能量强度 S0（1 参数），
  目标 = 最小化两个训练取向断裂应变的平均相对误差（网格 + 局部精化）；
- 预测留出取向的断裂应变与 UTS —— 盲测；
- 同时跑 Lemaitre 基线（f_AM≡1，同样仅标定 S0），用于公平对比。

并行：ProcessPoolExecutor(8)，每折实时写盘 + 断点续跑。
输出：experiment_v8_loocv_results.json
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
S0_GRID = [0.2, 0.8, 3.2, 12.8, 50.0]

OUT = os.path.join(os.path.dirname(__file__), 'output')


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def _gs_override(mat, gs_mult):
    if mat == 'Ti64':
        return {'gs_basal': 700e6 * gs_mult,
                'gs_prism': 720e6 * gs_mult,
                'gs_pyr': 800e6 * gs_mult}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat] * gs_mult}


def _fixed_overrides(mat, v7p, homogeneous):
    """v7 全局固定参数（gs_mult + a45/a90 分段乘数）。"""
    pm = v7p[mat]
    if homogeneous:
        return _gs_override(mat, pm['gs_mult'])
    ov = _gs_override(mat, pm['gs_mult'])
    ov['aniso_multiplier'] = {0: 1.0, 45: pm['a45'], 90: pm['a90']}
    return ov


def run_case(mat, psi, S0, overrides, homogeneous):
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    model = TaylorCPCDM(
        mat, psi, S0_override=S0, n_grains=N_GRAINS,
        eps_step=_eps_step(mat), n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=homogeneous,
        param_overrides=overrides, seed=42,
    )
    return model.run_uniaxial(max_strain)


def _solve_S0(mat, train_oris, overrides, homogeneous):
    """只标定 S0：网格 + 局部精化，最小化训练平均相对 ef 误差。"""
    def score(s0):
        errs = []
        for ps in train_oris:
            efp = run_case(mat, ps, s0, overrides, homogeneous)['fracture_strain']
            errs.append(abs(efp - EXPERIMENTAL[mat]['ef'][ps])
                        / EXPERIMENTAL[mat]['ef'][ps])
        return float(np.mean(errs))

    best_s0, best_e = None, None
    for s0 in S0_GRID:
        e = score(s0)
        if best_e is None or e < best_e:
            best_s0, best_e = s0, e
    # 局部精化 1 轮（±40% 邻域三分采样）
    for _ in range(1):
        a0, b0 = best_s0 / 1.6, best_s0 * 1.6
        for s0 in (a0, best_s0, b0):
            e = score(s0)
            if e < best_e:
                best_s0, best_e = s0, e
    return best_s0, best_e


def fold_one(mat, held, homogeneous, v7p):
    train = [o for o in ORIENTS if o != held]
    ov = _fixed_overrides(mat, v7p, homogeneous)
    t0 = time.time()
    S0, train_err = _solve_S0(mat, train, ov, homogeneous)

    ef_exp = EXPERIMENTAL[mat]['ef'][held]
    r_pred = run_case(mat, held, S0, ov, homogeneous)
    ef_pred = r_pred['fracture_strain']
    err = abs(ef_pred - ef_exp) / ef_exp * 100.0
    uts_err = abs(r_pred['uts'] / 1e6 - EXPERIMENTAL[mat]['uts'][held]) \
        / EXPERIMENTAL[mat]['uts'][held] * 100.0
    fold = {
        'material': mat, 'held_orientation': held,
        'train_orientations': train,
        'S0': float(S0), 'train_avg_err_pct': float(train_err * 100.0),
        'ef_exp': float(ef_exp), 'ef_pred': float(ef_pred),
        'error_pct': float(err),
        'uts_exp_MPa': float(EXPERIMENTAL[mat]['uts'][held]),
        'uts_pred_MPa': float(r_pred['uts'] / 1e6),
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
    v7 = json.load(open(os.path.join(OUT, 'experiment_v7_final_results.json'),
                        encoding='utf-8'))
    v7p = v7['params']
    path = os.path.join(OUT, 'experiment_v8_loocv_results.json')

    if os.path.exists(path):
        old = json.load(open(path, encoding='utf-8'))
        done_keys = {(f['material'], f['held_orientation'], False)
                     for f in old.get('modified_model', {}).get('folds', [])}
        done_keys |= {(f['material'], f['held_orientation'], True)
                      for f in old.get('lemaitre_baseline', {}).get('folds', [])}
        print(f"[resume] {len(done_keys)} folds already done", flush=True)
        all_folds = old.get('modified_model', {}).get('folds', [])
        lem_folds = old.get('lemaitre_baseline', {}).get('folds', [])
    else:
        done_keys, old, all_folds, lem_folds = set(), None, [], []

    tasks = [(m, held, False) for m in MATERIALS for held in ORIENTS
             if (m, held, False) not in done_keys]
    tasks += [(m, held, True) for m in MATERIALS for held in ORIENTS
              if (m, held, True) not in done_keys]
    print(f"[parallel] {len(tasks)} folds to run (8 workers)", flush=True)

    def _save():
        payload = {
            'model': 'v8_loocv_S0only',
            'notes': {
                'protocol': 'leave-one-orientation-out; per fold ONLY S0 is '
                            're-calibrated (train-mean ef error minimised); '
                            'all microstructure coefficients, orientation '
                            'multipliers a45/a90, and Voce g_s fixed at v7 '
                            'global values',
                'lemaitre_baseline': 'same, f_AM=1',
            },
            'fixed_params_v7': v7p,
            'modified_model': {
                'folds': all_folds,
                'avg_ef_error_pct': float(np.mean([f['error_pct'] for f in all_folds])),
                'avg_uts_error_pct': float(np.mean([f['uts_error_pct'] for f in all_folds])),
                'pass_10pct': int(np.sum(np.array([f['error_pct'] for f in all_folds]) < 10.0)),
            },
            'lemaitre_baseline': {
                'folds': lem_folds,
                'avg_ef_error_pct': float(np.mean([f['error_pct'] for f in lem_folds])),
                'avg_uts_error_pct': float(np.mean([f['uts_error_pct'] for f in lem_folds])),
            },
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[saved] {len(all_folds)}+{len(lem_folds)} folds", flush=True)

    if tasks:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fold_one, m, h, hm, v7p): (m, h, hm)
                    for m, h, hm in tasks}
            done_ct = 0
            for fut in as_completed(futs):
                m, h, hm = futs[fut]
                fold = fut.result()
                (lem_folds if hm else all_folds).append(fold)
                done_ct += 1
                _save()
                print(f"[progress] {done_ct}/{len(tasks)}", flush=True)

    me = [f['error_pct'] for f in all_folds]
    mu = [f['uts_error_pct'] for f in all_folds]
    le = [f['error_pct'] for f in lem_folds]
    lu = [f['uts_error_pct'] for f in lem_folds]
    print("\n" + "=" * 88)
    print(f"Modified model  LOOCV ef avg = {np.mean(me):6.2f}%  "
          f"(per-case {[round(e,1) for e in me]})")
    print(f"Modified model  LOOCV UTS avg = {np.mean(mu):6.2f}%")
    print(f"Lemaitre base   LOOCV ef avg = {np.mean(le):6.2f}%  "
          f"(per-case {[round(e,1) for e in le]})")
    print(f"Lemaitre base   LOOCV UTS avg = {np.mean(lu):6.2f}%")
    print("=" * 88)


if __name__ == '__main__':
    main()
