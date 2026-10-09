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
    get_slip_systems_FCC,
    get_slip_systems_HCP,
    build_stiffness_tensor,
    double_contraction,
    get_orientation_adjusted_params,
)

# ============================================================
# 全局数值参数
# ============================================================
EPS_STEP = 5e-4           # 应变步长（秒—应变/s），确保积分收敛
N_SUB = 50                # 每步步数（塑性更新细分）
STRAIN_RATE = 1e-3        # 应变率（s⁻¹）

# 校准参数
S0_MIN, S0_MAX = 0.01, 1500.0  # MPa，S0搜索范围（316L需要更大范围）
CAL_MAX_ITER = 80              # 二分法最大迭代

# 归一化参考量
LAMBDA_REF = 50.0  # mm⁻¹

# ============================================================
# 实验参考数据
# ============================================================
EXPERIMENTAL = {
    'Ti64': {
        'name': 'Ti-6Al-4V', 'E': 112e9,
        'ef': {0: 0.080, 45: 0.063, 90: 0.048},
        'uts': {0: 1150, 45: 1100, 90: 1050},
        'Dc': 0.42, 'D0': 0.003,
    },
    '316L': {
        'name': '316L SS', 'E': 192e9,
        'ef': {0: 0.370, 45: 0.310, 90: 0.255},
        'uts': {0: 680, 45: 640, 90: 595},
        'Dc': 0.48, 'D0': 0.001,
    },
    'AlSi10Mg': {
        'name': 'AlSi10Mg', 'E': 70e9,
        'ef': {0: 0.056, 45: 0.047, 90: 0.037},
        'uts': {0: 395, 45: 365, 90: 335},
        'Dc': 0.35, 'D0': 0.010,
    },
}

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
# 硬化耦合 CP-CDM 求解器
# ============================================================
class CPCDM_v2:
    """晶体塑性 + 修正CDM求解器（显式积分，硬化耦合损伤）"""

    def __init__(self, material_key, orientation_deg=0, S0_override=None):
        self.material_key = material_key
        self.orientation_deg = orientation_deg
        self.params = get_orientation_adjusted_params(material_key, orientation_deg)

        if S0_override is not None:
            self.params['S0'] = S0_override

        self.C = build_stiffness_tensor(self.params)
        self.E_eff = EXPERIMENTAL[material_key]['E']

        # 滑移系
        cs = self.params['crystal_structure']
        if cs == 'FCC':
            self.slip_systems = get_slip_systems_FCC()
        elif cs == 'HCP':
            self.slip_systems = get_slip_systems_HCP()
        else:
            raise ValueError(f"Unknown crystal structure: {cs}")

        self.N_slip = len(self.slip_systems)
        self.s0 = np.array([s[1] for s in self.slip_systems])
        self.m0 = np.array([s[0] for s in self.slip_systems])
        self.P0 = np.einsum('si,sj->sij', self.s0, self.m0)

        self.f_AM, self.S_eff = am_correction(self.params)
        self._init_state()

    @property
    def Dc(self):
        return EXPERIMENTAL[self.material_key]['Dc']

    def _init_state(self):
        self.Fp = np.eye(3)
        if 'g0' in self.params:
            self.g = np.ones(self.N_slip) * self.params['g0']
        else:
            self.g = np.zeros(self.N_slip)
            self.g[0:3] = self.params['g0_basal']
            self.g[3:6] = self.params['g0_prism']
            self.g[6:12] = self.params['g0_pyr']
        self.D = self.params['D0']
        self.p = 0.0
        self.gamma_cum = np.zeros(self.N_slip)

    def _resolved_shear(self, S):
        return np.array([np.sum(S * self.P0[a]) for a in range(self.N_slip)])

    def _flow_stress_mises(self, S):
        dev = S - np.trace(S) / 3.0 * np.eye(3)
        return np.sqrt(1.5 * np.sum(dev * dev))

    def _Y(self, sigma_mises):
        """损伤能量释放率（标准Lemaitre形式）"""
        D_safe = np.clip(self.D, 0.0, 0.95)
        return sigma_mises ** 2 / (2.0 * self.E_eff * (1.0 - D_safe) ** 2)

    def _hardening_rate(self, dgamma_dt):
        N = self.N_slip
        h0 = self.params['h0']
        a_exp = self.params['a']
        q_lat = self.params['q_lat']
        gs = self.params.get('gs', 700e6)

        q = q_lat * np.ones((N, N))
        np.fill_diagonal(q, 1.0)

        dg_dt = np.zeros(N)
        for a in range(N):
            for b in range(N):
                g_ratio = 1.0 - self.g[b] / gs
                if g_ratio > 1e-10:
                    dg_dt[a] += q[a, b] * h0 * g_ratio ** a_exp * abs(dgamma_dt[b])
        return dg_dt

    def integrate(self, strain_path, verbose=False):
        """沿应变路径积分（固定EPS_STEP保证收敛一致性）

        Args:
            strain_path: (n_steps, 6) Voigt应变
                           [ε11, ε22, ε33, γ12, γ13, γ23]

        Returns:
            stress_history, D_history, p_history
        """
        n_steps = len(strain_path)
        stress_history = np.zeros((n_steps, 6))
        D_history = np.zeros(n_steps)
        p_history = np.zeros(n_steps)

        D_history[0] = self.D
        p_history[0] = self.p

        for i in range(1, n_steps):
            eps_prev = strain_path[i - 1]
            eps_curr = strain_path[i]
            deps = eps_curr - eps_prev
            eps_norm = np.linalg.norm(deps)

            if eps_norm < 1e-12:
                stress_history[i] = stress_history[i - 1]
                D_history[i] = D_history[i - 1]
                p_history[i] = p_history[i - 1]
                continue

            # 全量变形梯度（当前步终点）
            F_total = np.eye(3)
            F_total[0, 0] = 1.0 + eps_curr[0]
            F_total[1, 1] = 1.0 + eps_curr[1]
            F_total[2, 2] = 1.0 + eps_curr[2]

            dt = eps_norm / STRAIN_RATE

            # 子步迭代（仅细分塑性更新）
            for _ in range(N_SUB):
                # 弹性试探（使用全量变形梯度）
                Fe = F_total @ np.linalg.inv(self.Fp)
                Ee = 0.5 * (Fe.T @ Fe - np.eye(3))
                S = double_contraction(self.C, Ee)

                tau = self._resolved_shear(S)

                # 剪切率（幂律）
                n_rate = self.params['n_rate']
                gdot0 = self.params['gamma_dot_0']
                tau_abs = np.abs(tau)
                ratio = np.where(self.g > 1e-6, tau_abs / self.g, 0.0)
                ratio = np.clip(ratio, 0.0, 10.0)
                gamma_dot = gdot0 * ratio ** n_rate * np.sign(tau)

                # 塑性增量
                dgamma = gamma_dot * (dt / N_SUB)
                dgamma = np.clip(dgamma, -0.01, 0.01)

                # 更新 Fp
                Lp = np.sum(dgamma[:, None, None] * self.P0, axis=0)
                self.Fp = (np.eye(3) + Lp) @ self.Fp

                # 硬化
                dg_dt = self._hardening_rate(gamma_dot)
                self.g += dg_dt * (dt / N_SUB)
                self.g = np.maximum(self.g, 1e-3 * abs(self.params.get('g0', 350e6)))

                # 累积塑性应变
                dp = np.sqrt(2.0 / 3.0) * np.sum(np.abs(dgamma))
                self.p += dp
                self.gamma_cum += np.abs(dgamma)

                # 损伤更新（硬化耦合）
                if self.p > self.params['p_D'] and dp > 0:
                    sigma_mises = self._flow_stress_mises(S)
                    Y = self._Y(sigma_mises)
                    S_pa = self.S_eff * 1e6
                    r = Y / max(S_pa, 1e-10)
                    dD = r ** self.params['s_damage'] * dp * self.f_AM
                    dD = np.clip(dD, 0.0, 0.05)
                    self.D = min(0.99, self.D + dD)

                if self.D >= 0.99:
                    break

            # 收敛应力记录（含损伤软化）
            Fe_final = F_total @ np.linalg.inv(self.Fp)
            Ee_final = 0.5 * (Fe_final.T @ Fe_final - np.eye(3))
            S_final = double_contraction(self.C, Ee_final)
            sigma_final = (1.0 - self.D) * S_final

            stress_history[i, 0] = sigma_final[0, 0]
            stress_history[i, 1] = sigma_final[1, 1]
            stress_history[i, 2] = sigma_final[2, 2]
            stress_history[i, 3] = sigma_final[0, 1]
            stress_history[i, 4] = sigma_final[0, 2]
            stress_history[i, 5] = sigma_final[1, 2]

            D_history[i] = self.D
            p_history[i] = self.p

        return stress_history, D_history, p_history

    def run_uniaxial(self, max_strain, verbose=False):
        """单轴拉伸模拟（恒定应变步长）"""
        n_steps = max(10, int(max_strain / EPS_STEP))
        nu = 0.30
        eps_axial = np.linspace(0, max_strain, n_steps + 1)
        strain_path = np.zeros((n_steps + 1, 6))
        for i in range(n_steps + 1):
            strain_path[i, 0] = eps_axial[i]
            strain_path[i, 1] = -nu * eps_axial[i]
            strain_path[i, 2] = -nu * eps_axial[i]

        stress_h, D_h, p_h = self.integrate(strain_path)

        # 断裂判定
        frac_idx = np.where(D_h >= self.Dc)[0]
        fracture_strain = eps_axial[frac_idx[0]] if len(frac_idx) > 0 else max_strain

        sigma_axial = stress_h[:, 0].copy()
        if len(frac_idx) > 0 and frac_idx[0] < len(sigma_axial):
            sigma_axial[frac_idx[0]:] = 0.0

        uts = np.max(sigma_axial[:frac_idx[0]+1] if len(frac_idx) > 0 else sigma_axial)

        if verbose:
            print(f"  ef={fracture_strain:.4f}, UTS={uts/1e6:.1f} MPa, "
                  f"D_final={D_h[-1]:.3f}")

        return {
            'strain': eps_axial, 'stress': sigma_axial,
            'damage': D_h, 'p_cum': p_h,
            'material': self.material_key,
            'orientation': self.orientation_deg,
            'fracture_strain': fracture_strain, 'uts': uts,
            'S0': self.params['S0'], 'f_AM': self.f_AM, 'S_eff': self.S_eff,
        }


# ============================================================
# Part 1: S0校准（二分法）
# ============================================================
def run_with_S0(material_key, orientation_deg, S0_val, max_strain):
    """以指定S0运行一次，返回断裂应变"""
    model = CPCDM_v2(material_key, orientation_deg, S0_override=S0_val)
    result = model.run_uniaxial(max_strain)
    return result['fracture_strain']


def calibrate_S0(material_key, orientation_deg=0, target_ef=None, verbose=True):
    """二分法校准S0

    ef(S0)单调递增：S0↑ → S_eff↑ → Y/S↓ → 损伤↓ → ef↑
    """
    if target_ef is None:
        target_ef = EXPERIMENTAL[material_key]['ef'][orientation_deg]

    max_strain = target_ef * 2.0 + 0.05

    lo, hi = S0_MIN, S0_MAX
    ef_lo = run_with_S0(material_key, orientation_deg, lo, max_strain)
    ef_hi = run_with_S0(material_key, orientation_deg, hi, max_strain)

    if ef_lo > target_ef or ef_hi < target_ef:
        warnings.warn(
            f"[{material_key}] ef范围[{ef_lo:.4f}, {ef_hi:.4f}]不含目标{target_ef:.4f}，"
            f"扩展S0搜索范围"
        )
    if ef_lo > target_ef:
        # ef_lo>target_ef说明S0_MIN太大，减小
        lo = max(1e-6, lo / 100.0)
        ef_lo = run_with_S0(material_key, orientation_deg, lo, max_strain)

    best_S0 = None
    for it in range(CAL_MAX_ITER):
        mid = (lo + hi) / 2.0
        ef_mid = run_with_S0(material_key, orientation_deg, mid, max_strain)
        if verbose and (it % 10 == 0 or it < 3):
            print(f"    iter {it:2d}: S0=[{lo:.4f}, {hi:.4f}], mid={mid:.4f}, "
                  f"ef_mid={ef_mid:.4f} (target={target_ef:.4f})")

        if abs(ef_mid - target_ef) < 1e-4:
            best_S0 = mid
            break

        if ef_mid < target_ef:
            lo = mid  # 需要更大损伤 → 更小S0... wait no...
            # ef↑ → S0↑. If ef_mid < target, ef_mid too LOW, need HIGHER ef → HIGHER S0
            # So lo = mid
            lo = mid
        else:
            hi = mid  # ef_mid > target, ef too HIGH, need LOWER ef → LOWER S0

        if hi - lo < 1e-6:
            best_S0 = mid
            break

    if best_S0 is None:
        best_S0 = (lo + hi) / 2.0

    ef_final = run_with_S0(material_key, orientation_deg, best_S0, max_strain)
    if verbose:
        print(f"  → S0={best_S0:.6f} MPa, ef_cal={ef_final:.4f} "
              f"(target={target_ef:.4f}, err=|{ef_final-target_ef:.4f}|)")

    return best_S0


# ============================================================
# Part 2: 完整验证
# ============================================================
def run_full_validation(S0_dict):
    """9组验证（3材料×3取向）"""
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
            model = CPCDM_v2(mat, orient, S0_override=S0)
            r = model.run_uniaxial(max_strain)
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
    max_strain = target['ef'][0] * 1.5 + 0.01
    base_params = get_orientation_adjusted_params(material_key, 0)

    base_model = CPCDM_v2(material_key, 0, S0_override=S0_val)
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
                model = CPCDM_v2(material_key, 0, S0_override=S0_val)
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

    # ---- Part 2: 验证 ----
    print("\n[Part 2] Full 9-case validation")
    all_results = run_full_validation(S0_dict)

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
        'timestamp': 'v2', 'EPS_STEP': EPS_STEP, 'N_SUB': N_SUB,
        'S0_params': {k: float(v) for k, v in S0_dict.items()},
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