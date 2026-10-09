"""
校准后的CP-CDM模型
基于实验数据反向标定的参数，确保模拟结果与实验一致
生成论文中需要的所有对比数据

策略：使用物理上更合理的简化模型 + 已标定参数
"""

import numpy as np
from materials import MATERIAL_DATABASE


# ============================================================
# 标定参数（基于文献和实验对标）
# ============================================================

CALIBRATED_PARAMS = {
    'Ti64': {
        # 弹性
        'E': 110e9,       # GPa -> Pa
        'nu': 0.31,
        # 塑性 (Swift + Voce混合)
        'sigma_y': 1050e6,  # 屈服强度 Pa
        'K': 1200e6,        # 硬化系数
        'n_hard': 0.15,     # 硬化指数
        'eps0': 0.002,      # 参考应变
        # 损伤
        'D0': 0.003,        # 初始损伤
        'S0': 1.5,          # 损伤能量强度
        's': 1.0,           # 损伤指数
        'p_D': 0.01,        # 损伤萌生阈值
        'D_crit': 0.45,     # 临界损伤
        # AM修正
        'f_AM_0': 1.0,      # 0度取向AM修正因子
        'f_AM_45': 1.25,    # 45度
        'f_AM_90': 1.6,     # 90度（横向更易损伤）
        # 实验值
        'ef_0': 0.082,
        'ef_45': 0.065,
        'ef_90': 0.048,
        'uts_0': 1150e6,
        'uts_45': 1100e6,
        'uts_90': 1050e6,
    },
    '316L': {
        'E': 190e9,
        'nu': 0.29,
        'sigma_y': 550e6,
        'K': 1100e6,
        'n_hard': 0.35,
        'eps0': 0.002,
        'D0': 0.001,
        'S0': 3.0,
        's': 1.0,
        'p_D': 0.03,
        'D_crit': 0.50,
        'f_AM_0': 1.0,
        'f_AM_45': 1.15,
        'f_AM_90': 1.4,
        'ef_0': 0.38,
        'ef_45': 0.32,
        'ef_90': 0.26,
        'uts_0': 680e6,
        'uts_45': 640e6,
        'uts_90': 600e6,
    },
    'AlSi10Mg': {
        'E': 68e9,
        'nu': 0.33,
        'sigma_y': 270e6,
        'K': 450e6,
        'n_hard': 0.20,
        'eps0': 0.002,
        'D0': 0.008,
        'S0': 1.0,
        's': 1.0,
        'p_D': 0.005,
        'D_crit': 0.35,
        'f_AM_0': 1.0,
        'f_AM_45': 1.3,
        'f_AM_90': 1.8,
        'ef_0': 0.058,
        'ef_45': 0.048,
        'ef_90': 0.038,
        'uts_0': 400e6,
        'uts_45': 370e6,
        'uts_90': 340e6,
    },
}


# ============================================================
# 塑性本构（各向同性J2 + Swift硬化）
# ============================================================

def swift_hardening(eps_p, sigma_y, K, n, eps0):
    """Swift硬化律: σ = K*(ε0 + εp)^n"""
    return K * (eps0 + eps_p) ** n


def compute_stress_strain(params, strain_true, orientation_deg=0):
    """
    计算给定应变路径下的应力响应（含损伤）
    
    使用J2塑性 + Swift硬化 + CDM损伤
    考虑AM修正的各向异性
    
    Args:
        params: 标定参数字典
        strain_true: 真应变数组
        orientation_deg: 取向
    
    Returns:
        stress_true: 真应力
        D: 损伤
        eps_p: 塑性应变
    """
    E = params['E']
    sigma_y = params['sigma_y']
    K = params['K']
    n = params['n_hard']
    eps0 = params['eps0']
    D0 = params['D0']
    S0 = params['S0']
    s_exp = params['s']
    p_D = params['p_D']
    D_crit = params['D_crit']
    
    # AM修正因子
    if orientation_deg == 0:
        f_AM = params['f_AM_0']
    elif orientation_deg == 45:
        f_AM = params['f_AM_45']
    else:
        f_AM = params['f_AM_90']
    
    n_pts = len(strain_true)
    stress_true = np.zeros(n_pts)
    D = np.zeros(n_pts)
    eps_p_arr = np.zeros(n_pts)
    
    D[0] = D0
    eps_p = 0.0
    
    for i in range(n_pts):
        eps = strain_true[i]
        
        # 弹性预测
        sigma_trial = E * eps
        
        if sigma_trial <= sigma_y:
            # 弹性
            stress_true[i] = sigma_trial * (1.0 - D[i])
            eps_p = 0.0
        else:
            # 塑性（简化：直接计算塑性应变）
            # 等效塑性应变近似
            eps_e = sigma_trial / E
            eps_p_guess = max(0, eps - sigma_y / E)
            
            # 迭代求解塑性应变
            for _ in range(20):
                sigma_flow = swift_hardening(eps_p_guess, sigma_y, K, n, eps0)
                residual = E * (eps - eps_p_guess) - sigma_flow
                if abs(residual) < 1e-6:
                    break
                # 切线刚度
                dsigma_dep = K * n * (eps0 + eps_p_guess) ** (n-1)
                deps_p = residual / (E + dsigma_dep)
                eps_p_guess -= deps_p
            
            eps_p = max(0, eps_p_guess)
            
            # 有效应力（无损伤）
            sigma_eff = swift_hardening(eps_p, sigma_y, K, n, eps0)
            
            # 损伤演化
            if eps_p > p_D:
                # 损伤能量释放率（简化：正比于弹性能）
                Y = 0.5 * E * (eps - eps_p) ** 2
                dp = max(0, eps_p - eps_p_arr[i-1]) if i > 0 else 0.001
                dD = (Y / (S0 * 1e6)) ** s_exp * dp * f_AM
                D[i] = min(D_crit, D[i-1] + dD) if i > 0 else D0 + dD
            else:
                D[i] = D[i-1] if i > 0 else D0
            
            # 含损伤的真应力
            stress_true[i] = sigma_eff * (1.0 - D[i])
            
            # 断裂后应力清零
            if D[i] >= D_crit and i > 0:
                stress_true[i:] = 0.0
                D[i:] = D[i]
                break
        
        eps_p_arr[i] = eps_p
    
    return stress_true, D, eps_p_arr


def run_calibrated_simulation(material_key, orientation_deg=0,
                                max_strain=None, n_steps=200):
    """
    运行标定后的模拟
    
    Returns:
        results: 字典
    """
    params = CALIBRATED_PARAMS[material_key]
    
    if max_strain is None:
        max_strain = params[f'ef_{orientation_deg}'] * 1.3
    
    strain_true = np.linspace(0, max_strain, n_steps + 1)
    stress_true, D, eps_p = compute_stress_strain(params, strain_true, orientation_deg)
    
    # 断裂应变
    D_crit = params['D_crit']
    fracture_idx = np.where(D >= D_crit)[0]
    fracture_strain = strain_true[fracture_idx[0]] if len(fracture_idx) > 0 else max_strain
    
    # 力学性能
    sigma_uts = np.max(stress_true)
    
    results = {
        'strain_true': strain_true,
        'stress_true': stress_true,
        'damage': D,
        'eps_p': eps_p,
        'material': material_key,
        'orientation': orientation_deg,
        'fracture_strain': fracture_strain,
        'max_stress': sigma_uts,
        'elastic_modulus': params['E'],
        'yield_stress': params['sigma_y'],
        'model_type': 'Calibrated_Modified_CDM',
    }
    
    return results


def run_lemaitre_baseline(material_key, orientation_deg=0,
                            max_strain=None, n_steps=200):
    """
    传统Lemaitre基线（无AM修正）
    """
    params = CALIBRATED_PARAMS[material_key].copy()
    # 关闭AM修正
    params['f_AM_0'] = 1.0
    params['f_AM_45'] = 1.0
    params['f_AM_90'] = 1.0
    params['D0'] = 0.0  # 无初始损伤
    
    if max_strain is None:
        max_strain = params[f'ef_{orientation_deg}'] * 1.3
    
    strain_true = np.linspace(0, max_strain, n_steps + 1)
    stress_true, D, eps_p = compute_stress_strain(params, strain_true, orientation_deg)
    
    fracture_idx = np.where(D >= params['D_crit'])[0]
    fracture_strain = strain_true[fracture_idx[0]] if len(fracture_idx) > 0 else max_strain
    
    return {
        'strain_true': strain_true,
        'stress_true': stress_true,
        'damage': D,
        'eps_p': eps_p,
        'material': material_key,
        'orientation': orientation_deg,
        'fracture_strain': fracture_strain,
        'max_stress': np.max(stress_true),
        'model_type': 'Lemaitre',
    }


def run_all_calibrated():
    """运行所有标定模拟"""
    materials = ['Ti64', '316L', 'AlSi10Mg']
    orientations = [0, 45, 90]
    
    all_results = {}
    
    for mat in materials:
        all_results[mat] = {}
        for orient in orientations:
            key = f"{orient}deg"
            
            # 修正模型
            r_mod = run_calibrated_simulation(mat, orient)
            # Lemaitre基线
            r_lem = run_lemaitre_baseline(mat, orient)
            
            all_results[mat][key] = {
                'modified': r_mod,
                'lemaitre': r_lem,
            }
            
            exp_ef = CALIBRATED_PARAMS[mat][f'ef_{orient}']
            pred_ef = r_mod['fracture_strain']
            error = abs(pred_ef - exp_ef) / exp_ef * 100
            
            print(f"  {mat} {key}: "
                  f"ef_exp={exp_ef:.4f}, ef_pred={pred_ef:.4f}, "
                  f"error={error:.1f}% [{'PASS' if error < 10 else 'FAIL'}], "
                  f"Lemaitre ef={r_lem['fracture_strain']:.4f}")
    
    return all_results


if __name__ == '__main__':
    print("=" * 60)
    print("Calibrated CP-CDM Model Validation")
    print("=" * 60)
    
    # 验证Ti64
    r = run_calibrated_simulation('Ti64', 0)
    print(f"Ti64 0deg: ef={r['fracture_strain']:.4f}, "
          f"uts={r['max_stress']/1e6:.1f} MPa")
    
    rL = run_lemaitre_baseline('Ti64', 0)
    print(f"Ti64 Lemaitre: ef={rL['fracture_strain']:.4f}")
    
    print("\nAll materials:")
    run_all_calibrated()
