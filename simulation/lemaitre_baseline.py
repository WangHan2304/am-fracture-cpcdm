# -*- coding: utf-8 -*-
"""
传统 Lemaitre 模型基线对照
==========================
与 experiment_v3.py 完全相同的 Taylor 多晶 CP 求解器、硬化参数与标定流程，
唯一区别：损伤模型退化为传统 Lemaitre 形式
  - f_AM ≡ 1（beta1..beta5 = 0）
  - S_eff = S0（alpha1 = alpha2 = 0）
  - D(0) = 0（D0 = 0，无初始损伤）
  - 无各向异性乘数（_f_AM_aniso = 1）
S0 仅在 0° 取向实验断裂应变上二分标定，随后直接预测 45°/90°。
注意：标定与预测必须使用相同的 max_strain 公式（ef_exp*2+0.01），
因为 run_uniaxial 的应变网格随 max_strain 变化，而延性材料在断裂
级联段对网格位置高度敏感（316L 上 0.1mm 的步长差可致 ~20% ef 偏差）。
网格不一致会让基线"看起来更好"（本次曾低估 316L 误差约 10 个百分点）。
输出：
  output/lemaitre_baseline_results.json
  figures/Fig9_errors_compare.png（修正模型 vs Lemaitre 基线 分组柱状图）
"""

import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

EPS_STEP = 5e-4
N_SUB = 30
N_GRAINS = 20
CAL_MAX_ITER = 80

MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]
COLORS = {'Ti64': '#E63946', '316L': '#457B9D', 'AlSi10Mg': '#2A9D8F'}
MAT_NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}

# 使 AM 修正全部失效的中性参数覆盖
NEUTRAL = {
    'beta1': 0.0, 'beta2': 0.0, 'beta3': 0.0, 'beta4': 0.0, 'beta5': 0.0,
    'alpha1': 0.0, 'alpha2': 0.0,
    'D0': 0.0,
    '_f_AM_aniso': 1.0,
}


def run_lemaitre(material_key, orientation_deg, S0_val, max_strain):
    """传统 Lemaitre 基线：同一 CP 求解器，关闭全部 AM 修正后运行一次"""
    model = TaylorCPCDM(material_key, orientation_deg, S0_override=S0_val,
                        n_grains=N_GRAINS, eps_step=EPS_STEP, n_sub=N_SUB,
                        aniso_override=1.0, param_overrides=dict(NEUTRAL))
    return model.run_uniaxial(max_strain)


def calibrate_S0_lemaitre(material_key, verbose=True):
    """二分法仅用 0° 取向实验断裂应变标定传统 Lemaitre 的 S0

    数值一致性说明：run_uniaxial 的应变网格由 linspace(0, max_strain, n) 生成，
    步长 = max_strain/(n-1) 随 max_strain 变化；在延性材料（316L）接近断裂的
    级联损伤段，断裂应变对网格位置敏感（Δ~1e-3 的步长摄动可致 ~20% 的 ef 波动）。
    因此标定必须使用与 Part 2 验证完全相同的 max_strain 公式（target*2+0.01），
    保证调用同一 ef(S0) 函数（experiment_v3.py 的"回验"步骤亦然）。
    """
    target_ef = EXPERIMENTAL[material_key]['ef'][0]
    max_strain = target_ef * 2.0 + 0.01  # 与 Part 2 验证网格一致，勿改
    tol = max(1.5e-3, 0.01 * target_ef)

    scan_grid = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0]
    pts = []
    for s0v in scan_grid:
        efv = run_lemaitre(material_key, 0, s0v, max_strain)['fracture_strain']
        if verbose:
            print(f"    scan S0={s0v:8.3f}  ef={efv:.4f} (target {target_ef:.4f})")
        pts.append((s0v, efv))
        if efv >= target_ef:
            break

    below = [p for p in pts if p[1] < target_ef]
    above = [p for p in pts if p[1] >= target_ef]
    if not above:
        raise RuntimeError(f"[{material_key}] S0=50 仍达不到 ef={target_ef}")
    if not below:
        lo, hi = above[0][0] / 4.0, above[0][0]
    else:
        lo, hi = below[-1][0], above[0][0]

    best_S0, best_err = None, np.inf
    for _ in range(CAL_MAX_ITER):
        mid = 0.5 * (lo + hi)
        ef_mid = run_lemaitre(material_key, 0, mid, max_strain)['fracture_strain']
        err = abs(ef_mid - target_ef)
        if err < best_err:
            best_S0, best_err = mid, err
        if err < tol:
            break
        if ef_mid < target_ef:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break

    if verbose:
        print(f"  -> S0_lemaitre={best_S0:.6f} MPa (best ef err={best_err:.4f})")
    return best_S0


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures')

    print("=" * 80)
    print("CONVENTIONAL LEMAITRE BASELINE (no AM corrections)")

    # ---- 0° 标定 ----
    print("\n[Part 1] S0 calibration on 0 deg only")
    S0_lem = {}
    for mat in MATERIALS:
        print(f"\n  {MAT_NAMES[mat]} (target ef={EXPERIMENTAL[mat]['ef'][0]:.3f}):")
        S0_lem[mat] = calibrate_S0_lemaitre(mat)

    # ---- 9 组预测 ----
    print("\n[Part 2] 9-case prediction with conventional Lemaitre model")
    cases = {}
    errors, errors_4590 = [], []
    print(f"\n{'Material':<14}{'Orient':<8}{'ef_exp':<10}{'ef_pred':<10}{'Err%':<8}")
    print("-" * 50)
    for mat in MATERIALS:
        cases[mat] = {}
        for orient in ORIENTS:
            ef_exp = EXPERIMENTAL[mat]['ef'][orient]
            max_strain = ef_exp * 2.0 + 0.01
            r = run_lemaitre(mat, orient, S0_lem[mat], max_strain)
            ef_pred = r['fracture_strain']
            err = abs(ef_pred - ef_exp) / ef_exp * 100
            cases[mat][str(orient)] = {
                'ef_exp': ef_exp, 'ef_pred': ef_pred,
                'error_pct': err,
                'uts_exp': EXPERIMENTAL[mat]['uts'][orient],
                'uts_pred_mpa': float(r['uts'] / 1e6),
            }
            errors.append(err)
            if orient != 0:
                errors_4590.append(err)
            print(f"{MAT_NAMES[mat]:<14}{orient:<8}{ef_exp:<10.4f}"
                  f"{ef_pred:<10.4f}{err:<8.2f}")
    avg_all = float(np.mean(errors))
    avg_4590 = float(np.mean(errors_4590))
    print("-" * 50)
    print(f"Average ef error (9 cases): {avg_all:.2f}%")
    print(f"Average ef error (45/90 deg only): {avg_4590:.2f}%")
    print(f"Max ef error: {max(errors):.2f}%")

    # ---- 与修正模型对比的分组柱状图 ----
    with open(os.path.join(out_dir, 'experiment_v2_results.json'),
              'r', encoding='utf-8') as f:
        mod = json.load(f)

    labels, errs_mod, errs_lem, bar_colors = [], [], [], []
    for mat in MATERIALS:
        for orient in ORIENTS:
            labels.append(f"{MAT_NAMES[mat]}\n{orient}°")
            errs_mod.append(mod['cases'][mat][str(orient)]['error_pct'])
            errs_lem.append(cases[mat][str(orient)]['error_pct'])
            bar_colors.append(COLORS[mat])

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, errs_mod, w, label='Modified CDM (this work)',
                color='#457B9D', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w / 2, errs_lem, w, label='Conventional Lemaitre',
                color='#E63946', edgecolor='black', linewidth=0.5)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='10% target')
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.6,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Fracture strain prediction error [%]')
    ax.set_ylim(0, max(errs_lem) * 1.25)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'Fig9_errors_compare.png'))
    plt.close(fig)

    # ---- 保存 JSON ----
    results = {
        'timestamp': 'lemaitre_baseline_v1',
        'description': 'Conventional Lemaitre baseline: same Taylor CP solver, '
                       'f_AM=1, S_eff=S0, D0=0, no aniso multiplier; '
                       'S0 calibrated on 0 deg only.',
        'EPS_STEP': EPS_STEP, 'N_SUB': N_SUB, 'N_GRAINS': N_GRAINS,
        'S0_lemaitre': {k: float(v) for k, v in S0_lem.items()},
        'avg_error_pct_all9': avg_all,
        'avg_error_pct_45_90': avg_4590,
        'max_error_pct': float(max(errors)),
        'cases': cases,
    }

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o

    with open(os.path.join(out_dir, 'lemaitre_baseline_results.json'), 'w',
              encoding='utf-8') as f:
        json.dump(clean(results), f, indent=2)

    print(f"\nSaved: {out_dir}/lemaitre_baseline_results.json")
    print(f"Saved: {fig_dir}/Fig9_errors_compare.png")
    print("DONE")


if __name__ == '__main__':
    main()
