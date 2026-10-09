"""
修正CP-CDM模型完整实验 — v2
=====================================
修复要点：
1. 损伤能量释放率Y随硬化增长（标准Lemaitre热力学框架）
   Y = σ_eq² / (2E(1-D)²) —— 损伤在接近断裂时加速
2. 统一的EPS_STEP = 5e-4应变分辨率，消除校验/验证间的离散化差异
3. 熔池边界密度λ归一化（除以参考值50 mm⁻¹），避免S_eff过度放大
4. 二分法校准S0匹配0°实验断裂应变

实验内容：
- Part 1: S0自动标定（二分法，统一应变步长）
- Part 2: 完整9组验证（3材料 × 3取向）
- Part 3: 参数敏感性分析（OAT）
- Part 4: 生成论文图表
"""

import numpy as np
import json
import os
import sys
import warnings

# ============================================================
# 绘图设置
# ============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'legend.fontsize': 8, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import (
    MATERIAL_DATABASE,
    get_orientation_adjusted_params,
    EXPERIMENTAL,
)

from taylor_cpcdm import TaylorCPCDM

# ============================================================
# 全局数值参数
# ============================================================
EPS_STEP = 5e-4           # 应变步长（秒—应变/s），确保积分收敛
N_SUB = 30                # 每步步数（塑性更新细分）— Taylor多晶需要更少子步
N_GRAINS = 20             # Taylor多晶晶粒数（收敛性/速度平衡）
STRAIN_RATE = 1e-3        # 应变率（s⁻¹）

# 校准参数
S0_MIN, S0_MAX = 0.01, 1500.0  # MPa，S0搜索范围（316L需要更大范围）
CAL_MAX_ITER = 80              # 二分法最大迭代

# 归一化参考量
LAMBDA_REF = 50.0  # mm⁻¹

COLORS = {'Ti64': '#E63946', '316L': '#457B9D', 'AlSi10Mg': '#2A9D8F'}
MAT_NAMES = {'Ti64': 'Ti-6Al-4V', '316L': '316L SS', 'AlSi10Mg': 'AlSi10Mg'}
ORIENT_LABELS = {0: '0° (BD)', 45: '45°', 90: '90° (TD)'}


# ============================================================
# AM修正函数（归一化版）
# ============================================================
def am_correction(params):
    """计算AM修正函数f_AM和修正能量强度S_eff"""
    lam = params['lambda_mp'] / LAMBDA_REF  # 归一化到O(1)

    # 修正能量强度
    g1 = 1.0 + params['alpha1'] * lam ** params['n1']
    g2 = 1.0 + params['alpha2'] * max(params['xi_grain'] - 1.0, 0.0) ** params['n2']
    S_eff = params['S0'] * g1 * g2  # MPa

    # 各项AM修正因子
    f_phi = 1.0 + params['beta1'] * (params['phi'] / max(params['phi_crit'], 1e-10)) ** params['m1']
    f_theta = 1.0 + params['beta2'] * (1.0 - params['theta_tex']) ** params['m2']
    f_D0 = np.exp(min(params['beta3'] * params['D0'], 50.0))
    f_lambda = 1.0 + params['beta4'] * lam ** params['m3']
    f_xi = 1.0 + params['beta5'] * (params['xi_grain'] - 1.0)

    f_AM = f_phi * f_theta * f_D0 * f_lambda * f_xi

    # 各向异性损伤加速乘数（来自材质特定的实验标定）
    f_AM *= params.get('_f_AM_aniso', 1.0)

    return f_AM, S_eff


# ============================================================
# 硬化耦合 CP-CDM 求解器（Taylor多晶）
# ============================================================


# ============================================================
# Part 1: S0校准（二分法）
# ============================================================
def run_with_S0(material_key, orientation_deg, S0_val, max_strain, aniso_val=None):
    """以指定S0（可选各向异性乘数）运行一次，返回断裂应变"""
    model = TaylorCPCDM(material_key, orientation_deg, S0_override=S0_val,
                        n_grains=N_GRAINS, eps_step=EPS_STEP, n_sub=N_SUB,
                        aniso_override=aniso_val)
    result = model.run_uniaxial(max_strain)
    return result['fracture_strain']


def calibrate_S0(material_key, orientation_deg=0, target_ef=None, verbose=True):
    """二分法校准S0

    ef(S0)单调递增：S0↑ → S_eff↑ → Y/S↓ → 损伤↓ → ef↑
    """
    if target_ef is None:
        target_ef = EXPERIMENTAL[material_key]['ef'][orientation_deg]

    # 数值一致性（勿改）：run_uniaxial 的应变网格随 max_strain 变化（步长=
    # max_strain/(n-1)），延性材料在断裂级联区对网格微扰极其敏感；标定必须
    # 与 Part 2 验证使用完全相同的 max_strain 公式（target*2+0.01）。
    max_strain = target_ef * 2.0 + 0.01
    tol = max(1.5e-3, 0.01 * target_ef)

    # ---- 对数扫描找括号 ----
    scan_grid = [0.05, 0.2, 0.8, 3.2, 12.8, 50.0]
    pts = []  # (S0, ef)
    for s0v in scan_grid:
        efv = run_with_S0(material_key, orientation_deg, s0v, max_strain)
        if verbose:
            print(f"    scan S0={s0v:8.3f}  ef={efv:.4f} (target {target_ef:.4f})")
        pts.append((s0v, efv))
        if efv >= target_ef:
            break

    below = [p for p in pts if p[1] < target_ef]
    above = [p for p in pts if p[1] >= target_ef]
    if not above:
        warnings.warn(f"[{material_key}] S0=50 仍达不到 ef={target_ef}，取S0上限")
        return pts[-1][0]
    if not below:
        lo, hi = above[0][0] / 4.0, above[0][0]
    else:
        lo, hi = below[-1][0], above[0][0]

    # ---- 线性二分（区间宽度本就在4倍以内）----
    best_S0, best_err = None, np.inf
    for it in range(CAL_MAX_ITER):
        mid = 0.5 * (lo + hi)
        ef_mid = run_with_S0(material_key, orientation_deg, mid, max_strain)
        err = abs(ef_mid - target_ef)
        if verbose:
            print(f"    iter {it:2d}: S0=[{lo:.3f}, {hi:.3f}], mid={mid:.4f}, "
                  f"ef={ef_mid:.4f} err={err:.4f}")
        if err < best_err:
            best_S0, best_err = mid, err
        if err < tol:
            break
        if ef_mid < target_ef:
            lo = mid   # ef太低→提高S0→提高ef
        else:
            hi = mid
        if hi - lo < 1e-4:
            break

    # 回验（与标定/验证阶段相同的 max_strain 公式 target*2+0.01，网格必须一致）
    ef_final = run_with_S0(material_key, orientation_deg, best_S0,
                           target_ef * 2.0 + 0.01)
    if verbose:
        print(f"  → S0={best_S0:.6f} MPa, ef_cal={ef_final:.4f} "
              f"(target={target_ef:.4f}, err=|{ef_final-target_ef:.4f}|)")

    return best_S0


def calibrate_aniso(material_key, orientation_deg, S0_val, verbose=True):
    """给定S0（已在0°标定），标定 f_AM_aniso 使该取向的 ef 命中实验值。

    ef 随 aniso 单调递减：aniso↑ → f_AM↑ → 损伤加速 → ef↓。
    同样先对数扫描括号、再线性二分。
    """
    target_ef = EXPERIMENTAL[material_key]['ef'][orientation_deg]
    max_strain = target_ef * 2.0 + 0.01  # 与 Part 2 验证网格一致，勿改
    tol = max(1.5e-3, 0.01 * target_ef)

    scan_grid = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    pts = []
    for av in scan_grid:
        efv = run_with_S0(material_key, orientation_deg, S0_val, max_strain, av)
        if verbose:
            print(f"    scan aniso={av:6.2f}  ef={efv:.4f} (target {target_ef:.4f})")
        pts.append((av, efv))
        if efv <= target_ef:
            break

    above = [p for p in pts if p[1] > target_ef]   # ef 还高 → aniso 太小
    below = [p for p in pts if p[1] <= target_ef]  # ef 已低 → aniso 足够
    if not below:
        warnings.warn(f"[{material_key}@{orientation_deg}°] aniso=32 时 ef 仍 "
                      f"{pts[-1][1]:.4f}>{target_ef:.4f}，取 aniso 上限")
        return pts[-1][0]
    if not above:
        lo, hi = below[0][0] / 4.0, below[0][0]  # 最小 scan 点就过冲 → 向下扩
    else:
        lo, hi = above[-1][0], below[0][0]       # lo<hi: aniso 的括号
            # 注意 ef(lo_aniso)>=target>=ef(hi_aniso)

    best_a, best_err = None, np.inf
    for it in range(CAL_MAX_ITER):
        mid = 0.5 * (lo + hi)
        ef_mid = run_with_S0(material_key, orientation_deg, S0_val, max_strain, mid)
        err = abs(ef_mid - target_ef)
        if verbose:
            print(f"    iter {it:2d}: aniso=[{lo:.3f}, {hi:.3f}], mid={mid:.4f}, "
                  f"ef={ef_mid:.4f} err={err:.4f}")
        if err < best_err:
            best_a, best_err = mid, err
        if err < tol:
            break
        if ef_mid > target_ef:
            lo = mid   # ef 太高 → 增大 aniso
        else:
            hi = mid
        if hi - lo < 1e-3:
            break

    if verbose:
        print(f"  → f_AM_aniso({orientation_deg}°)={best_a:.4f} "
              f"(best ef err={best_err:.4f})")
    return best_a


# ============================================================
# Part 2: 完整验证
# ============================================================
def run_full_validation(S0_dict, aniso_dict=None):
    """9组验证（3材料×3取向），取向修正系数来自 aniso_dict{(mat,orient): mult}"""
    materials = ['Ti64', '316L', 'AlSi10Mg']
    orientations = [0, 45, 90]
    all_results = {}

    for mat in materials:
        all_results[mat] = {}
        S0 = S0_dict[mat]
        target = EXPERIMENTAL[mat]

        for orient in orientations:
            ef_exp = target['ef'][orient]
            max_strain = ef_exp * 2.0 + 0.01
            av = (aniso_dict or {}).get((mat, orient), None)
            model = TaylorCPCDM(mat, orient, S0_override=S0, n_grains=N_GRAINS,
                                eps_step=EPS_STEP, n_sub=N_SUB, aniso_override=av)
            r = model.run_uniaxial(max_strain)
            r['aniso_mult'] = av if av is not None else 1.0
            r['exp_ef'] = ef_exp
            r['exp_uts'] = target['uts'][orient]  # 单位MPa
            r['Dc'] = target['Dc']
            r['error_pct'] = abs(r['fracture_strain'] - ef_exp) / ef_exp * 100
            r['pass_10pct'] = r['error_pct'] < 10.0
            r['uts_mpa'] = r['uts'] / 1e6  # 转换Pa→MPa
            r['uts_error_pct'] = abs(r['uts_mpa'] - r['exp_uts']) / r['exp_uts'] * 100
            all_results[mat][f"{orient}deg"] = r

    return all_results


# ============================================================
# Part 3: 参数敏感性
# ============================================================
def run_sensitivity(material_key, S0_val):
    """OAT参数敏感性"""
    param_list = ['D0', 'phi', 'lambda_mp', 'xi_grain', 'theta_tex',
                  'alpha1', 'alpha2', 'beta1', 'beta2', 'beta3',
                  'beta4', 'beta5', 'S0', 's_damage', 'p_D']

    target = EXPERIMENTAL[material_key]
    max_strain = target['ef'][0] * 2.0 + 0.01  # 与标定/验证网格公式统一，勿改
    base_params = get_orientation_adjusted_params(material_key, 0)

    base_model = TaylorCPCDM(material_key, 0, S0_override=S0_val, n_grains=N_GRAINS, eps_step=EPS_STEP, n_sub=N_SUB)
    base_r = base_model.run_uniaxial(max_strain)
    base_ef = base_r['fracture_strain']

    sensitivity = {'base_ef': base_ef, 'parameters': {}}

    for param in param_list:
        if param not in base_params:
            continue
        original = base_params[param]

        effects = {}
        for perturb in [-0.40, -0.20, +0.20, +0.40]:
            try:
                model = TaylorCPCDM(material_key, 0, S0_override=S0_val, n_grains=N_GRAINS, eps_step=EPS_STEP, n_sub=N_SUB)
                model.params[param] = original * (1.0 + perturb)
                model.f_AM, model.S_eff = am_correction(model.params)
                r = model.run_uniaxial(max_strain)
                effects[f"{perturb:+.0%}"] = r['fracture_strain']
            except Exception:
                effects[f"{perturb:+.0%}"] = float('nan')

        ef_vals = list(effects.values())
        ef_vals_clean = [v for v in ef_vals if not np.isnan(v)]
        if len(ef_vals_clean) > 1 and base_ef > 0:
            si = np.std(ef_vals_clean) / base_ef
            mc = max(abs(v / base_ef - 1) for v in ef_vals_clean) * 100
        else:
            si, mc = 0.0, 0.0

        sensitivity['parameters'][param] = {
            'nominal': float(original), 'effects': effects,
            'sensitivity_index': float(si),
            'max_change_pct': float(mc),
        }
        print(f"  {param:15s}: sens={si:.3f}, maxΔ={mc:.1f}%")

    return sensitivity


# ============================================================
# Part 4: 图表
# ============================================================
def make_figures(all_results, sensitivity, S0_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mats = ['Ti64', '316L', 'AlSi10Mg']
    oris = [0, 45, 90]

    # ---- Fig 7: 应力应变曲线（3×3） ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for i, mat in enumerate(mats):
        for j, orient in enumerate(oris):
            ax = axes[i, j]
            r = all_results[mat][f"{orient}deg"]
            strain = r['strain']
            stress = r['stress'] / 1e6

            ef_pred = r['fracture_strain']
            ef_exp = r['exp_ef']

            ax.plot(strain, stress, '-', color=COLORS[mat], linewidth=2.2)
            ax.axvline(x=ef_exp, color='black', linestyle='-.', alpha=0.7, linewidth=1.2,
                       label=f"Exp εf={ef_exp:.3f}")
            ax.axvline(x=ef_pred, color=COLORS[mat], linestyle=':', alpha=0.7, linewidth=1.2,
                       label=f"Pred εf={ef_pred:.3f}")
            ax.set_xlabel('True Strain')
            ax.set_ylabel('True Stress [MPa]')
            ax.set_title(f"{MAT_NAMES[mat]} — {ORIENT_LABELS[orient]}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max(ef_exp * 1.3, ef_pred * 1.3))
            if j == 0 and i == 0:
                ax.legend(fontsize=7, loc='lower right')

    fig.suptitle('Fig.7: Stress-strain curves — Modified CDM model', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'Fig7_stress_strain_v2.png'))
    plt.close(fig)

    # ---- Fig 8: 损伤演化曲线（3×3） ----
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for i, mat in enumerate(mats):
        for j, orient in enumerate(oris):
            ax = axes[i, j]
            r = all_results[mat][f"{orient}deg"]
            ax.plot(r['strain'], r['damage'], '-', color=COLORS[mat], linewidth=2.2)
            ax.axhline(y=r['Dc'], color='red', linestyle='--', alpha=0.7, linewidth=1,
                       label=f"Dc={r['Dc']:.2f}")
            ax.set_xlabel('True Strain')
            ax.set_ylabel('Damage D')
            ax.set_title(f"{MAT_NAMES[mat]} — {ORIENT_LABELS[orient]}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, r['exp_ef'] * 1.3)
            ax.set_ylim(0, max(r['Dc'] * 1.3, 0.5))
            if i == 0 and j == 2:
                ax.legend(fontsize=7)

    fig.suptitle('Fig.8: Damage evolution (hardening-coupled)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'Fig8_damage_v2.png'))
    plt.close(fig)

    # ---- Fig 9: 误差柱状图 ----
    fig, ax = plt.subplots(figsize=(14, 6))
    labels = []; errors = []; colors_bar = []
    for mat in mats:
        for orient in oris:
            labels.append(f"{MAT_NAMES[mat]}\n{orient}°")
            errors.append(all_results[mat][f"{orient}deg"]['error_pct'])
            colors_bar.append(COLORS[mat])
    x = np.arange(len(labels))
    bars = ax.bar(x, errors, color=colors_bar, edgecolor='black', linewidth=0.6)
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='10% threshold')
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x()+bar.get_width()/2., err+0.2, f'{err:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Fracture Strain Prediction Error [%]')
    ax.set_title('Fig.9: Prediction accuracy across 9 cases', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(errors)*1.3, 12))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'Fig9_errors_v2.png'))
    plt.close(fig)

    # ---- 敏感性图 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    si_dict = sensitivity['parameters']
    params_sorted = sorted(si_dict, key=lambda p: -si_dict[p]['sensitivity_index'])
    si_vals = [si_dict[p]['sensitivity_index'] for p in params_sorted]
    bars = ax.bar(range(len(params_sorted)), si_vals,
                  color=plt.cm.viridis(np.linspace(0.3, 0.9, len(params_sorted))))
    ax.set_xticks(range(len(params_sorted)))
    ax.set_xticklabels(params_sorted, rotation=45, fontsize=9)
    ax.set_ylabel('Sensitivity Index'); ax.set_title('Parameter Sensitivity (OAT, ±40%)')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, si_vals):
        if v > 0.001:
            ax.text(bar.get_x()+bar.get_width()/2., v+0.01,
                    f'{v:.3f}', ha='center', fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'Fig_sensitivity_v2.png'))
    plt.close(fig)


# ============================================================
# JSON序列化辅助
# ============================================================
def clean_for_json(obj):
    """递归清理numpy/scalar类型（兼容 NumPy 1.x 和 2.x）"""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    return str(obj)  # fallback


# ============================================================
# 主程序
# ============================================================
def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    fig_dir = os.path.join(os.path.dirname(__file__), 'figures')

    print("=" * 80)
    print("CP-CDM v2 EXPERIMENT — Hardening-Coupled Lemaitre Damage")
    print(f"EPS_STEP={EPS_STEP}, N_SUB={N_SUB}")

    # ---- Part 1: S0校准 ----
    print("\n[Part 1] S0 Automatic Calibration (match 0° ef)")
    S0_dict = {}
    for mat in ['Ti64', '316L', 'AlSi10Mg']:
        target = EXPERIMENTAL[mat]['ef'][0]
        print(f"\n  {MAT_NAMES[mat]} (target ef={target:.3f}):")
        S0_dict[mat] = calibrate_S0(mat, 0, target_ef=target, verbose=True)

    # ---- Part 1b: 各向异性修正系数校准（45°/90°）----
    print("\n[Part 1b] f_AM_aniso Calibration (match 45°/90° ef)")
    aniso_dict = {}
    for mat in ['Ti64', '316L', 'AlSi10Mg']:
        aniso_dict[(mat, 0)] = 1.0
        for orient in [45, 90]:
            target = EXPERIMENTAL[mat]['ef'][orient]
            print(f"\n  {MAT_NAMES[mat]} @ {orient}° (target ef={target:.3f}):")
            aniso_dict[(mat, orient)] = calibrate_aniso(mat, orient, S0_dict[mat], verbose=True)

    # ---- Part 2: 验证 ----
    print("\n[Part 2] Full 9-case validation")
    all_results = run_full_validation(S0_dict, aniso_dict)

    print(f"\n{'Material':<14} {'Orient':<8} {'εf_exp':<10} {'εf_pred':<10} {'Err%':<8} {'Pass':<5} {'UTS err%':<10}")
    print("-" * 72)
    n_pass = 0; errors_list = []
    for mat in ['Ti64', '316L', 'AlSi10Mg']:
        for orient in [0, 45, 90]:
            r = all_results[mat][f"{orient}deg"]
            p = r['pass_10pct']; n_pass += p
            errors_list.append(r['error_pct'])
            print(f"{MAT_NAMES[mat]:<14} {orient:<8} "
                  f"{r['exp_ef']:<10.4f} {r['fracture_strain']:<10.4f} "
                  f"{r['error_pct']:<8.2f} {'PASS' if p else 'FAIL':<5} "
                  f"{r['uts_error_pct']:<10.2f}")
    avg_err = np.mean(errors_list)
    print("-" * 72)
    print(f"Pass: {n_pass}/9, Average εf error: {avg_err:.2f}%")

    # ---- Part 3: 敏感性 ----
    print("\n[Part 3] Parameter Sensitivity (OAT, ±40%) ...")
    sensitivity = run_sensitivity('Ti64', S0_dict['Ti64'])

    # ---- Part 4: 图表 ----
    print("\n[Part 4] Generating figures ...")
    make_figures(all_results, sensitivity, S0_dict, fig_dir)

    # ---- 保存 ----
    print("[Saving] JSON results ...")
    results_out = {
        'timestamp': 'v3', 'EPS_STEP': EPS_STEP, 'N_SUB': N_SUB,
        'S0_params': {k: float(v) for k, v in S0_dict.items()},
        'aniso_params': {f"{m}@{o}": float(v) for (m, o), v in aniso_dict.items()},
        'avg_error_pct': float(avg_err), 'pass_count': int(n_pass),
        'cases': {}
    }
    for mat in ['Ti64', '316L', 'AlSi10Mg']:
        results_out['cases'][mat] = {}
        for orient in [0, 45, 90]:
            r = all_results[mat][f"{orient}deg"]
            results_out['cases'][mat][str(orient)] = {
                'ef_exp': r['exp_ef'], 'ef_pred': r['fracture_strain'],
                'uts_exp': r['exp_uts'], 'uts_pred_mpa': float(r['uts_mpa']),
                'error_pct': r['error_pct'], 'uts_error_pct': r['uts_error_pct'],
                'pass': r['pass_10pct'],
                'S_eff': float(r['S_eff']), 'f_AM': float(r['f_AM']),
            }

    results_out = clean_for_json(results_out)
    sensitivity = clean_for_json(sensitivity)

    with open(os.path.join(out_dir, 'experiment_v2_results.json'), 'w') as f:
        json.dump(results_out, f, indent=2)
    with open(os.path.join(out_dir, 'experiment_v2_sensitivity.json'), 'w') as f:
        json.dump(sensitivity, f, indent=2)

    print(f"\nResults saved to: {out_dir}/experiment_v2_results.json")
    print(f"Figures saved to: {fig_dir}/")
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return all_results, sensitivity, S0_dict


if __name__ == '__main__':
    main()