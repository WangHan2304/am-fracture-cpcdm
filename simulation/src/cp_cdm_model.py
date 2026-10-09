"""
修正连续损伤力学(CDM)模型 + 晶体塑性(CP)本构
核心数值实现：单晶/多晶单轴拉伸模拟（稳健版）

采用显式积分 + 自适应子步，避免数值溢出
理论依据参考论文 Section 2 (Model Development)
"""

import numpy as np
from materials import (
    get_slip_systems_FCC, get_slip_systems_BCC, get_slip_systems_HCP,
    build_stiffness_tensor, double_contraction,
    MATERIAL_DATABASE, get_orientation_adjusted_params
)


# ============================================================
# AM修正CDM函数
# ============================================================

def compute_AM_correction(params):
    """计算AM修正函数 f_AM 和修正能量强度 S_eff"""
    # 孔隙应力集中增强
    f_phi = 1.0 + params['beta1'] * (params['phi'] / max(params['phi_crit'], 1e-10)) ** params['m1']
    
    # 硬取向加速损伤
    f_theta = 1.0 + params['beta2'] * (1.0 - params['theta_tex']) ** params['m2']
    
    # 初始损伤加速效应
    f_D0 = np.exp(min(params['beta3'] * params['D0'], 50.0))  # 防止溢出
    
    # 弱界面密度效应
    f_lambda = 1.0 + params['beta4'] * params['lambda_mp'] ** params['m3']
    
    # 形态各向异性修正
    f_xi = 1.0 + params['beta5'] * (params['xi_grain'] - 1.0)
    
    f_AM = f_phi * f_theta * f_D0 * f_lambda * f_xi
    
    # 修正能量强度参数
    g1 = 1.0 + params['alpha1'] * params['lambda_mp'] ** params['n1']
    g2 = 1.0 + params['alpha2'] * max(params['xi_grain'] - 1.0, 0.0) ** params['n2']
    S_eff = params['S0'] * g1 * g2
    
    return f_AM, S_eff


def compute_damage_evolution(Y, S_eff, s_damage, dp, f_AM, p, p_D):
    """损伤演化方程（修正CDM）"""
    if p < p_D or dp <= 0:
        return 0.0
    
    Y_pa = max(Y, 0.0)
    S_pa = S_eff * 1e6  # 转换到Pa
    ratio = Y_pa / max(S_pa, 1e-10)
    dD = ratio ** s_damage * dp * f_AM
    return min(max(0.0, dD), 0.1)  # 限制单步损伤增量


# ============================================================
# 晶体塑性 + 修正CDM 稳健求解器
# ============================================================

class CrystalPlasticityCDM:
    """
    晶体塑性 + 修正CDM 稳健积分求解器
    
    采用显式前向Euler + 自适应子步
    适用于论文中的参数敏感性分析和验证
    """
    
    def __init__(self, material_key='Ti64', orientation_deg=0,
                 crystal_orientation=None):
        self.params = get_orientation_adjusted_params(material_key, orientation_deg)
        self.material_key = material_key
        self.orientation_deg = orientation_deg
        
        # 弹性刚度
        self.C = build_stiffness_tensor(self.params)
        
        # 滑移系
        cs = self.params['crystal_structure']
        if cs == 'FCC':
            self.slip_systems = get_slip_systems_FCC()
        elif cs == 'BCC':
            self.slip_systems = get_slip_systems_BCC()
        elif cs == 'HCP':
            self.slip_systems = get_slip_systems_HCP()
        else:
            raise ValueError(f"Unknown crystal structure: {cs}")
        
        self.N_slip = len(self.slip_systems)
        self.s0 = np.array([s[1] for s in self.slip_systems])
        self.m0 = np.array([s[0] for s in self.slip_systems])
        self.P0 = np.einsum('si,sj->sij', self.s0, self.m0)
        
        if crystal_orientation is None:
            self.orientation = np.eye(3)
        else:
            self.orientation = crystal_orientation
        
        # 预计算AM修正
        self.f_AM, self.S_eff = compute_AM_correction(self.params)
        
        self._init_state()
    
    def _init_state(self):
        """初始化状态变量"""
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
        self.sigma = np.zeros((3, 3))
    
    def reset(self):
        self._init_state()
    
    def _compute_tau(self, S):
        """计算各滑移系的分剪应力"""
        return np.array([np.sum(S * self.P0[a]) for a in range(self.N_slip)])
    
    def _compute_hardening_rate(self, dgamma_dt):
        """计算硬化率 dg/dt"""
        N = self.N_slip
        h0 = self.params['h0']
        a_exp = self.params['a']
        q_lat = self.params['q_lat']
        
        gs = self.params.get('gs', self.params.get('gs_basal', 700e6))
        
        q = q_lat * np.ones((N, N))
        np.fill_diagonal(q, 1.0)
        
        dg_dt = np.zeros(N)
        for a in range(N):
            for b in range(N):
                g_ratio = 1.0 - self.g[b] / gs
                if g_ratio > 1e-10:
                    dg_dt[a] += q[a, b] * h0 * g_ratio ** a_exp * abs(dgamma_dt[b])
        
        return dg_dt
    
    def integrate(self, strain_path, strain_rate=1e-3):
        """
        沿应变路径积分
        
        Args:
            strain_path: 应变路径数组 (n_steps, 6) - Voigt notation [ε11,ε22,ε33,γ12,γ13,γ23]
            strain_rate: 应变率 [s⁻¹]
        
        Returns:
            stress_history: (n_steps, 6) 应力历史
            D_history: (n_steps,) 损伤历史
            p_history: (n_steps,) 累积塑性应变历史
        """
        n_steps = len(strain_path)
        
        stress_history = np.zeros((n_steps, 6))
        D_history = np.zeros(n_steps)
        p_history = np.zeros(n_steps)
        
        stress_history[0] = np.zeros(6)
        D_history[0] = self.D
        p_history[0] = self.p
        
        # 数值参数
        n_sub = 20  # 每步子步数
        min_sub = 5
        max_sub = 100
        
        nu = 0.3  # 假设泊松比
        
        for i in range(1, n_steps):
            # 当前步应变增量
            deps = strain_path[i] - strain_path[i-1]
            eps_norm = np.linalg.norm(deps)
            
            if eps_norm < 1e-12:
                stress_history[i] = stress_history[i-1]
                D_history[i] = D_history[i-1]
                p_history[i] = p_history[i-1]
                continue
            
            # 自适应子步
            dt = eps_norm / max(strain_rate, 1e-10)
            dt_sub = dt / n_sub
            
            # 构建变形梯度增量
            F_current = np.eye(3)
            F_current[0, 0] = 1.0 + strain_path[i-1, 0]
            F_current[1, 1] = 1.0 + strain_path[i-1, 1]
            F_current[2, 2] = 1.0 + strain_path[i-1, 2]
            
            F_target = np.eye(3)
            F_target[0, 0] = 1.0 + strain_path[i, 0]
            F_target[1, 1] = 1.0 + strain_path[i, 1]
            F_target[2, 2] = 1.0 + strain_path[i, 2]
            
            dF_total = F_target @ np.linalg.inv(F_current)
            
            # 子步积分
            for sub in range(n_sub):
                t = (sub + 1) / n_sub
                dF = np.eye(3) + t * (dF_total - np.eye(3))
                # 简化：使用线性插值
                if sub == 0:
                    F_sub_start = np.eye(3)
                else:
                    t_prev = sub / n_sub
                    F_sub_start = np.eye(3) + t_prev * (dF_total - np.eye(3))
                
                dF_sub = dF @ np.linalg.inv(F_sub_start)
                
                # 弹性预测应变
                Fe = dF_sub @ self.Fp
                Ee = 0.5 * (Fe.T @ Fe - np.eye(3))
                S = double_contraction(self.C, Ee)
                
                # 剪应力
                tau = self._compute_tau(S)
                
                # 剪切率（幂律）
                n_rate = self.params['n_rate']
                gdot0 = self.params['gamma_dot_0']
                
                tau_abs = np.abs(tau)
                ratio = np.where(self.g > 1e-10, tau_abs / self.g, 0.0)
                # 截断防止溢出
                ratio = np.clip(ratio, 0, 10.0)
                gamma_dot = gdot0 * ratio ** n_rate * np.sign(tau)
                
                # 塑性速度梯度
                dgamma = gamma_dot * dt_sub
                dgamma = np.clip(dgamma, -0.01, 0.01)  # 限制单步剪切
                
                Lp = np.sum(dgamma[:, np.newaxis, np.newaxis] * self.P0, axis=0)
                
                # 更新Fp（一阶近似）
                self.Fp = (np.eye(3) + Lp) @ self.Fp
                
                # 更新硬化
                dg_dt = self._compute_hardening_rate(gamma_dot)
                self.g += dg_dt * dt_sub
                
                # 更新累积塑性应变
                dp = np.sqrt(2.0/3.0) * np.sum(np.abs(dgamma))
                self.p += dp
                self.gamma_cum += np.abs(dgamma)
                
                # 损伤更新
                Ee_new = 0.5 * (Fe.T @ Fe - np.eye(3))
                S_new = double_contraction(self.C, Ee_new)
                Y = 0.5 * np.sum(Ee_new * S_new)
                
                dD = compute_damage_evolution(
                    Y, self.S_eff, self.params['s_damage'],
                    dp, self.f_AM, self.p, self.params['p_D']
                )
                self.D = min(0.99, self.D + dD)
            
            # 更新应力
            Fe_final = dF_total @ self.Fp
            Ee_final = 0.5 * (Fe_final.T @ Fe_final - np.eye(3))
            S_final = double_contraction(self.C, Ee_final)
            sigma_final = (1.0 - self.D) * S_final
            
            # 存储（Voigt notation）
            stress_history[i, 0] = sigma_final[0, 0]
            stress_history[i, 1] = sigma_final[1, 1]
            stress_history[i, 2] = sigma_final[2, 2]
            stress_history[i, 3] = sigma_final[0, 1]
            stress_history[i, 4] = sigma_final[0, 2]
            stress_history[i, 5] = sigma_final[1, 2]
            
            D_history[i] = self.D
            p_history[i] = self.p
        
        return stress_history, D_history, p_history


# ============================================================
# 单轴拉伸模拟
# ============================================================

def run_uniaxial_tension(material_key, orientation_deg=0,
                          max_strain=0.5, n_steps=200,
                          strain_rate=1e-3, use_AM=True):
    """
    单轴拉伸模拟
    
    Args:
        material_key: 'Ti64', '316L', 'AlSi10Mg'
        orientation_deg: 加载取向 [度]
        max_strain: 最大真应变
        n_steps: 增量步数
        strain_rate: 应变率 [s⁻¹]
        use_AM: 是否使用AM修正（False则退化为传统Lemaitre）
    
    Returns:
        results: 字典
    """
    cp_cdm = CrystalPlasticityCDM(material_key, orientation_deg)
    
    if not use_AM:
        cp_cdm.f_AM = 1.0
        cp_cdm.S_eff = cp_cdm.params['S0']
        cp_cdm.D = 0.0
    
    # 构建应变路径（单轴拉伸，侧向自由收缩）
    nu = 0.3
    strain_path = np.zeros((n_steps + 1, 6))
    eps_axial = np.linspace(0, max_strain, n_steps + 1)
    
    for i in range(n_steps + 1):
        strain_path[i, 0] = eps_axial[i]
        strain_path[i, 1] = -nu * eps_axial[i]
        strain_path[i, 2] = -nu * eps_axial[i]
    
    # 积分
    stress_hist, D_hist, p_hist = cp_cdm.integrate(strain_path, strain_rate)
    
    # 提取轴向数据
    stress_axial = stress_hist[:, 0]
    strain_axial = eps_axial
    
    # 断裂判定
    fracture_idx = np.where(D_hist >= 0.5)[0]
    if len(fracture_idx) > 0:
        frac_idx = fracture_idx[0]
        fracture_strain = strain_axial[frac_idx]
        # 断裂后应力清零
        stress_axial[frac_idx:] = 0.0
    else:
        fracture_strain = max_strain
    
    # 力学性能指标
    E_mod = 0.0
    for i in range(1, len(strain_axial)):
        if stress_axial[i] > 0:
            E_mod = stress_axial[i] / max(strain_axial[i], 1e-10)
            break
    
    sigma_y = 0.0
    # 0.2% offset yield
    for i in range(len(strain_axial)):
        plastic_strain = strain_axial[i] - stress_axial[i] / max(E_mod, 1e-10)
        if plastic_strain > 0.002:
            sigma_y = stress_axial[i]
            break
    
    sigma_uts = np.max(stress_axial)
    
    results = {
        'strain_true': strain_axial,
        'stress_true': stress_axial,
        'damage': D_hist,
        'p_cum': p_hist,
        'material': material_key,
        'orientation': orientation_deg,
        'model_type': 'Modified_CDM' if use_AM else 'Lemaitre',
        'fracture_strain': fracture_strain,
        'max_stress': sigma_uts,
        'elastic_modulus': E_mod,
        'yield_stress': sigma_y,
    }
    
    return results


def run_systematic_simulation(material_keys=None, orientations=None,
                               max_strain_map=None, use_AM=True):
    """
    系统运行所有材料-取向组合
    
    Returns:
        all_results: dict of results
    """
    if material_keys is None:
        material_keys = ['Ti64', '316L', 'AlSi10Mg']
    if orientations is None:
        orientations = [0, 45, 90]
    if max_strain_map is None:
        max_strain_map = {'Ti64': 0.15, '316L': 0.45, 'AlSi10Mg': 0.10}
    
    all_results = {}
    
    for mat in material_keys:
        for orient in orientations:
            key = f"{mat}_{orient}deg"
            max_s = max_strain_map.get(mat, 0.2)
            n_steps = int(max_s * 1000)
            n_steps = max(50, min(n_steps, 500))
            
            print(f"Running {key} ({'Modified CDM' if use_AM else 'Lemaitre'})...")
            results = run_uniaxial_tension(mat, orient, max_s, n_steps,
                                            use_AM=use_AM)
            all_results[key] = results
            
            print(f"  εf = {results['fracture_strain']:.4f}, "
                  f"σ_UTS = {results['max_stress']/1e6:.1f} MPa")
    
    return all_results


if __name__ == '__main__':
    print("=" * 60)
    print("Testing CP-CDM Model (Robust Version)")
    print("=" * 60)
    
    # 测试修正模型
    print("\n--- Modified CDM Model ---")
    results = run_uniaxial_tension('Ti64', 0, max_strain=0.12, n_steps=150)
    print(f"Ti64 0°: εf = {results['fracture_strain']:.4f}, "
          f"σmax = {results['max_stress']/1e6:.1f} MPa, "
          f"E = {results['elastic_modulus']/1e9:.1f} GPa")
    
    # 测试传统Lemaitre
    print("\n--- Lemaitre Model (no AM) ---")
    results_L = run_uniaxial_tension('Ti64', 0, max_strain=0.12, n_steps=150,
                                      use_AM=False)
    print(f"Ti64 0°: εf = {results_L['fracture_strain']:.4f}, "
          f"σmax = {results_L['max_stress']/1e6:.1f} MPa")
    
    # 三材料快速测试
    print("\n--- All materials, 0° orientation ---")
    for mat in ['Ti64', '316L', 'AlSi10Mg']:
        r = run_uniaxial_tension(mat, 0, max_strain=0.1, n_steps=100)
        print(f"{mat}: εf = {r['fracture_strain']:.4f}, "
              f"σmax = {r['max_stress']/1e6:.1f} MPa")
