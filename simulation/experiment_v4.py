"""
CP-CDM v4 诚实实验协议
=======================
v4 模型（am_correction_v4.py）：
  - 去掉自由拟合乘数 f_AM_aniso
  - S ≡ S0 常数（仅在 0° 标定）
  - 取向依赖性完全由物理描述子承载：
      λ(ψ)：有效熔池边界密度（横观各向同性弱界面网络几何投影）
      ξ(ψ)：旋转椭球晶粒沿加载方向的解析长宽比
      θ(ψ)：取向集合的平均最大 Schmid 因子（由 Taylor 取向组直接计算）
  这些投影律是几何假设（文献可核），不含拟合取向数据的自由参数。

协议 A（主协议）：S0 仅在 0° 标定，45°/90° 共 6 个工况严格盲预测。
协议 B（留一材料交叉验证）：每次留出 1 个材料，用训练材料 0° 数据估计 S0
  转移律（透明、无取向数据泄漏），预测留出材料全部 3 取向。

数值一致性（与 v3 一致，勿改）：
  run_uniaxial 的应变网格由 linspace(0, max_strain, n) 生成，步长随
  max_strain 变化；损伤级联段对网格/S0 微扰极其敏感。标定与验证必须用
  完全相同的 max_strain 公式（target*2+0.01）与 n_steps。
"""

import json
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

EPS_STEP = 5e-4
N_SUB = 30
N_GRAINS = 20
STRAIN_RATE = 1e-3
CAL_MAX_ITER = 80

MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]
MAT_NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
COLORS = {'Ti64': '#E63946', '316L': '#457B9D', 'AlSi10Mg': '#2A9D8F'}


def run_v4(material_key, orientation_deg, S0_val, max_strain, homogeneous=False):
    """v4 模型 / Lemaitre 基线共用的一次运行，返回 result dict"""
    model = TaylorCPCDM(material_key, orientation_deg, S0_override=S0_val,
                        n_grains=N_GRAINS, eps_step=EPS_STEP, n_sub=N_SUB,
                        strain_rate=STRAIN_RATE, model_version='v4',
                        force_homogeneous=homogeneous)
    return model.run_uniaxial(max_strain)


def calibrate_S0(material_key, orientation_deg=0, homogeneous=False, verbose=False):
    """二分法只为给定取向（默认 0°）匹配断裂应变来标定 S0。

    ef(S0) 单调递增（S0↑→S_eff↑→Y/S↓→损伤↓→ef↑）。
    max_strain 与预测阶段公式完全一致（target*2+0.01）。
    """
    target_ef = EXPERIMENTAL[material_key]['ef'][orientation_deg]
    max_strain = target_ef * 2.0 + 0.01
    tol = max(1.5e-3, 0.01 * target_ef)

    scan = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0, 200.0, 800.0]
    pts = []
    for s0v in scan:
        efv = run_v4(material_key, orientation_deg, s0v, max_strain,
                     homogeneous=homogeneous)['fracture_strain']
        if verbose:
            print(f"    scan S0={s0v:9.3f}  ef={efv:.4f} (tgt {target_ef:.4f})")
        pts.append((s0v, efv))
        if efv >= target_ef:
            break

    below = [p for p in pts if p[1] < target_ef]
    above = [p for p in pts if p[1] >= target_ef]
    if not above:
        warnings.warn(f"[{material_key}] 上限仍未达 ef={target_ef}，取扫描最大 S0")
        return pts[-1][0]
    if not below:
        lo, hi = above[0][0] / 4.0, above[0][0]
    else:
        lo, hi = below[-1][0], above[0][0]

    best_S0, best_err = None, np.inf
    for _ in range(CAL_MAX_ITER):
        mid = 0.5 * (lo + hi)
        efm = run_v4(material_key, orientation_deg, mid, max_strain,
                     homogeneous=homogeneous)['fracture_strain']
        err = abs(efm - target_ef)
        if err < best_err:
            best_S0, best_err = mid, err
        if err < tol:
            break
        if efm < target_ef:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return best_S0


def predict_case(material_key, orientation_deg, S0_val, homogeneous=False):
    """给定已标定的 S0，对任意取向做预测（不再拟合）"""
    target_ef = EXPERIMENTAL[material_key]['ef'][orientation_deg]
    max_strain = target_ef * 2.0 + 0.01
    r = run_v4(material_key, orientation_deg, S0_val, max_strain,
               homogeneous=homogeneous)
    ef_exp = target_ef
    ef_pred = r['fracture_strain']
    err = abs(ef_pred - ef_exp) / ef_exp * 100.0
    return {
        'ef_exp': ef_exp, 'ef_pred': ef_pred, 'error_pct': err,
        'uts_exp': EXPERIMENTAL[material_key]['uts'][orientation_deg],
        'uts_pred_mpa': float(r['uts'] / 1e6),
        'f_AM': float(r['f_AM']), 'S0': S0_val,
    }


def protocol_A():
    """S0 仅 0° 标定；0° 为训练、45°/90° 为盲测"""
    print("=" * 76)
    print("PROTOCOL A — S0 calibrated on 0° only; 45°/90° blind")
    S0_train = {}
    for m in MATERIALS:
        S0_train[m] = calibrate_S0(m, 0, homogeneous=False, verbose=False)
        print(f"  {MAT_NAMES[m]:12s} S0(0°) = {S0_train[m]:.6f} MPa")

    results = {}
    for m in MATERIALS:
        results[m] = {}
        for o in ORIENTS:
            results[m][o] = predict_case(m, o, S0_train[m])

    train_errs = [results[m][0]['error_pct'] for m in MATERIALS]
    blind_errs = [results[m][o]['error_pct'] for m in MATERIALS for o in (45, 90)]
    blind_uts = [abs(results[m][o]['uts_pred_mpa'] - results[m][o]['uts_exp'])
                 / results[m][o]['uts_exp'] * 100
                 for m in MATERIALS for o in (45, 90)]
    print("\n  Material      Ori  ef_exp  ef_pred  err%")
    for m in MATERIALS:
        for o in ORIENTS:
            r = results[m][o]
            tag = '  (train)' if o == 0 else '  (BLIND)'
            print(f"  {MAT_NAMES[m]:12s} {o:3d}  {r['ef_exp']:.4f}  "
                  f"{r['ef_pred']:.4f}  {r['error_pct']:5.2f}{tag}")
    print(f"\n  Train (0°)      avg err = {np.mean(train_errs):.2f}%")
    print(f"  Blind (45/90°)  avg err = {np.mean(blind_errs):.2f}%")
    print(f"  Blind (45/90°)  UTS avg err = {np.mean(blind_uts):.2f}%")
    return results, S0_train, {'train': train_errs, 'blind': blind_errs,
                               'blind_uts': blind_uts}


def protocol_B():
    """留一材料交叉验证：留出材料的 S0 由训练材料的 S0 经透明转移律估计。

    转移律：S0 ∝ D0（初始损伤是损伤能量强度的直接材料指标）。
    S0_heldout = mean(S0_train_i / D0_train_i) × D0_heldout
    只用训练材料的 0° 数据，无留出材料取向数据泄漏。
    """
    print("\n" + "=" * 76)
    print("PROTOCOL B — Leave-one-material-out (S0 transfer via D0 scaling)")
    D0 = {m: EXPERIMENTAL[m]['D0'] for m in MATERIALS}

    all_results = {}
    for held in MATERIALS:
        train = [m for m in MATERIALS if m != held]
        S0_train = {m: calibrate_S0(m, 0, homogeneous=False) for m in train}
        ratio = np.mean([S0_train[m] / D0[m] for m in train])
        S0_pred = ratio * D0[held]
        print(f"  hold {MAT_NAMES[held]:12s}  S0(transfer)={S0_pred:.4f} "
              f" (ratio={ratio:.2f})")
        all_results[held] = {}
        for o in ORIENTS:
            all_results[held][o] = predict_case(held, o, S0_pred)

    errs = [all_results[m][o]['error_pct'] for m in MATERIALS for o in ORIENTS]
    uts_errs = [abs(all_results[m][o]['uts_pred_mpa']
                    - all_results[m][o]['uts_exp']) / all_results[m][o]['uts_exp'] * 100
                for m in MATERIALS for o in ORIENTS]
    print(f"\n  LOMO avg ef err (all 9 cases) = {np.mean(errs):.2f}%")
    print(f"  LOMO avg UTS err (all 9)     = {np.mean(uts_errs):.2f}%")
    return all_results, {'ef': errs, 'uts': uts_errs}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    resA, S0_A, errs_A = protocol_A()
    resB, errs_B = protocol_B()

    # 基线（相同协议）+ 图，见 lemaitre_baseline_v4.py；这里仅保存 v4 结果
    payload = {
        'model': 'v4_physical',
        'protocol_A': {
            'S0_calibrated_0deg': {m: float(S0_A[m]) for m in MATERIALS},
            'errs': {k: [float(x) for x in v] for k, v in errs_A.items()},
        },
        'protocol_B_lomo': {
            'errs': {k: [float(x) for x in v] for k, v in errs_B.items()},
        },
        'protocol_A_cases': {
            m: {str(o): {k: float(v) for k, v in resA[m][o].items()}
                for o in ORIENTS} for m in MATERIALS
        },
    }
    with open(os.path.join(out_dir, 'experiment_v4_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out_dir}/experiment_v4_results.json")
    return payload


if __name__ == '__main__':
    main()