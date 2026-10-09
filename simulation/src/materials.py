"""
材料参数定义模块
包含 Ti-6Al-4V、316L、AlSi10Mg 三材料的完整参数集
用于晶体塑性 + 修正CDM本构模拟
"""

import numpy as np

# ============================================================
# 晶体学数据：滑移系定义
# ============================================================

def get_slip_systems_FCC():
    """FCC滑移系: {111}<110>, 共12个独立滑移系

    枚举全部 {111} 面法向（4个，析抗方向去重）与全部 <110> 滑移方向
    （6个，析方向-反向去重），保留满足 n·d = 0 的组合，
    确保每个滑移面内都包含 3 个 <110> 方向（共 12 个滑移系）。
    """
    # {111} 面法向：8个(±1,±1,±1) → 按析方向去重（n与-n同一晶面）
    normals = []
    seen_n = set()
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                v = (sx, sy, sz)
                canon = v if v[0] > 0 or (v[0] == 0 and v[1] > 0) or \
                             (v[0] == 0 and v[1] == 0 and v[2] > 0) \
                    else tuple(-x for x in v)
                if canon not in seen_n:
                    seen_n.add(canon)
                    normals.append(canon)

    # <110> 滑移方向：两个±1一个0 → 按析方向去重（d与-d同一滑移方向，剪切率取有符号）
    dirs = []
    seen_d = set()
    base = [(1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 0.0, 1.0),
            (1.0, 0.0, -1.0), (0.0, 1.0, 1.0), (0.0, 1.0, -1.0)]
    for d in base:
        canon = d if d[0] > 0 or (d[0] == 0 and d[1] > 0) else tuple(-x for x in d)
        if canon not in seen_d:
            seen_d.add(canon)
            dirs.append(canon)

    systems = []
    for n in normals:
        n_arr = np.array(n, dtype=float)
        n_unit = n_arr / np.linalg.norm(n_arr)
        for d in dirs:
            d_arr = np.array(d, dtype=float)
            if abs(np.dot(n_arr, d_arr)) < 1e-9:
                d_unit = d_arr / np.linalg.norm(d_arr)
                systems.append((n_unit, d_unit))
    return systems  # 12个独立滑移系

def get_slip_systems_BCC():
    """BCC滑移系: {110}<111> + {112}<111>, 共24个"""
    # {110}<111>
    slip_plane_110 = np.array([
        [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]
    ])
    slip_dir_111 = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]
    ])
    
    systems = []
    for n in slip_plane_110:
        for d in slip_dir_111:
            if abs(np.dot(n, d)) < 1e-10:
                n_norm = n / np.linalg.norm(n)
                d_norm = d / np.linalg.norm(d)
                systems.append((n_norm, d_norm))
    
    # {112}<111>
    slip_plane_112 = np.array([
        [1, 1, 2], [1, -1, 2], [1, 1, -2], [1, -1, -2],
        [1, 2, 1], [1, -2, 1], [1, 2, -1], [1, -2, -1],
        [2, 1, 1], [2, -1, 1], [2, 1, -1], [2, -1, -1]
    ])
    for n in slip_plane_112:
        for d in slip_dir_111:
            if abs(np.dot(n, d)) < 1e-10:
                n_norm = n / np.linalg.norm(n)
                d_norm = d / np.linalg.norm(d)
                systems.append((n_norm, d_norm))
    
    return systems  # 24个

def get_slip_systems_HCP():
    """
    HCP滑移系 (Ti-6Al-4V)，正交六方度量精确生成（c轴沿z）
    Basal <a>: {0001}<11-20>, 3个
    Prismatic <a>: {10-10}<11-20>, 3个
    Pyramidal <c+a>: {11-22}<11-23>, 6个
    共12个

    旧实现把 prismatic/pyramidal 的"面法向"直接取成与滑移方向相同的
    向量，导致 n·d = 1 ≠ 0 —— Schmid 张量不再是简单剪切（tr(P)≠0），
    分解剪应力混入正应力耦合、滑移伴随虚假塑性体胀，宏观强度被拉低
    ~2.4 倍（Ti64 UTS 465 MPa vs 实验 1050-1150 MPa）。
    现按正交化六方基矢量（a=1, c/a=1.587）由 4 指数 Miller-Bravais
    符号经度量显式转换，并用 |n·d|<1e-10 筛选配对。
    """
    cr = 1.587  # c/a (Ti)
    # 三轴直接基（a1,a2 位于基面内夹角120°，c 沿 z）
    A = np.array([
        [1.0,   -0.5,            0.0],
        [0.0,   np.sqrt(3)/2,    0.0],
        [0.0,   0.0,             cr ],
    ])                       # 列 = (a1, a2, c)
    # 倒易基（行 = b1,b2,b3 满足 b_i·a_j = δ_ij）
    B = np.linalg.inv(A)

    def plane_normal(h, k, l):
        """四指数 (h,k,i,l) 平面 → 单位法向（笛卡尔）"""
        n = h * B[0] + k * B[1] + l * B[2]
        return n / np.linalg.norm(n)

    def direction(u, v, t, w):
        """四指数 [u,v,t,w] 方向 → 单位向量（笛卡尔）
        d = u·a1 + v·a2 + t·a3 + w·c, a3 = -(a1+a2)"""
        d = (u - t) * A[:, 0] + (v - t) * A[:, 1] + w * A[:, 2]
        return d / np.linalg.norm(d)

    systems = []

    # Basal <a>: 面 (0001)，3 个 <11-20> 方向
    n_basal = plane_normal(0, 0, 1)
    for (u, v, t, w) in [(1, 1, -2, 0), (1, -2, 1, 0), (-2, 1, 1, 0)]:
        systems.append((n_basal.copy(), direction(u, v, t, w)))

    # Prismatic <a>: {10-10} 3 面，滑移方向 = 面内 <11-20>（与法向垂直）
    prism_planes = [(1, 0, 0), (0, 1, 0), (-1, 1, 0)]  # 3指数 {10-10}
    prism_dirs   = [(1, 1, -2, 0), (1, -2, 1, 0), (-2, 1, 1, 0)]
    for i, (h, k, l) in enumerate(prism_planes):
        n = plane_normal(h, k, l)
        # 取与法向垂直的那个 <11-20> 方向
        cand = [(abs(np.dot(n, direction(*dd))), direction(*dd))
                for dd in prism_dirs]
        cand.sort(key=lambda x: x[0])
        systems.append((n, cand[0][1]))

    # Pyramidal <c+a>: {11-22}<11-23>，6 系统
    pyr_planes = [(1, 1, 2), (1, -2, 2), (-2, 1, 2),
                  (1, 1, -2), (1, -2, -2), (-2, 1, -2)]
    pyr_dirs   = [(1, 1, -2, 3), (1, -2, 1, 3), (-2, 1, 1, 3),
                  (1, 1, -2, -3), (1, -2, 1, -3), (-2, 1, 1, -3)]
    for (h, k, l) in pyr_planes:
        n = plane_normal(h, k, l)
        cand = [(abs(np.dot(n, direction(*dd))), direction(*dd))
                for dd in pyr_dirs]
        cand.sort(key=lambda x: x[0])
        if cand[0][0] < 1e-9:      # 度量检验通过才收录
            systems.append((n, cand[0][1]))

    # 收尾断言：所有系统必须为简单剪切
    for n, d in systems:
        assert abs(np.dot(n, d)) < 1e-9, f"HCP slip system not simple shear: n·d={np.dot(n, d)}"
    return systems  # 3 + 3 + 6 = 12个


# ============================================================
# 材料参数数据库
# ============================================================

MATERIAL_DATABASE = {
    'Ti64': {
        'name': 'Ti-6Al-4V (LPBF)',
        'crystal_structure': 'HCP',
        'slip_systems': 'get_slip_systems_HCP',
        
        # 弹性常数 [GPa] → [Pa]
        'C11': 162e9, 'C12': 92e9, 'C13': 69e9,
        'C33': 180e9, 'C44': 47e9,
        
        # CP硬化参数 (Voce)
        'g0_basal': 350e6,      # Basal CRSS [Pa]
        'g0_prism': 370e6,      # Prismatic CRSS
        'g0_pyr': 450e6,        # Pyramidal CRSS
        'gs_basal': 700e6,      # Saturation
        'gs_prism': 720e6,
        'gs_pyr': 800e6,
        'h0': 800e6,            # 初始硬化模量 [Pa]
        'a': 2.2,               # 硬化指数
        'n_rate': 30.0,         # 率敏感指数
        'gamma_dot_0': 0.001,   # 参考剪切率 [s⁻¹]
        'q_lat': 1.4,           # 潜硬化系数
        
        # CDM基础参数
        'S0': 3.5,              # 损伤能量强度 [MPa]
        's_damage': 1.0,        # 损伤指数
        'p_D': 0.05,            # 损伤萌生阈值
        
        # AM修正参数
        'D0': 0.005,            # 初始损伤
        'phi': 0.003,           # 孔隙率
        'phi_crit': 0.02,       # 临界孔隙率
        'lambda_mp': 50.0,      # 熔池边界密度 [mm⁻¹]
        'xi_grain': 3.5,        # 晶粒形态因子（柱状晶）
        'theta_tex': 0.42,      # 织构取向权重
        
        # 修正系数
        'alpha1': 0.5, 'alpha2': 0.3,
        'n1': 1.5, 'n2': 2.0,
        'beta1': 2.0, 'beta2': 1.5, 'beta3': 10.0,
        'beta4': 0.8, 'beta5': 0.2,
        'm1': 2.0, 'm2': 1.5, 'm3': 1.0,
        
        # 实验参考值（文献）
        'E_exp': 110e9,         # 弹性模量 [Pa]
        'sigma_y_exp': 1050e6,  # 屈服强度 [Pa]
        'sigma_uts_exp': 1150e6,# 抗拉强度 [Pa]
        'ef_exp': 0.08,         # 断裂延伸率 (0°取向)
        'ef_45_exp': 0.065,
        'ef_90_exp': 0.05,
        
        # 微观特征（用于AM修正）
        'grain_size_avg': 15e-6,  # 平均晶粒尺寸 [m]
        'grain_aspect_ratio': 3.5, # 柱状晶长宽比
    },
    
    '316L': {
        'name': '316L Stainless Steel (LPBF)',
        'crystal_structure': 'FCC',
        'slip_systems': 'get_slip_systems_FCC',
        
        # 弹性常数 [GPa] → [Pa]
        'C11': 204e9, 'C12': 138e9, 'C44': 126e9,
        
        # CP硬化参数 (Voce) - FCC所有滑移系相同
        'g0': 200e6,            # 初始CRSS [Pa]
        'gs': 500e6,            # 饱和CRSS
        'h0': 400e6,            # 初始硬化模量 [Pa]
        'a': 2.25,              # 硬化指数
        'n_rate': 25.0,         # 率敏感指数
        'gamma_dot_0': 0.001,   # 参考剪切率 [s⁻¹]
        'q_lat': 1.4,           # 潜硬化系数
        
        # CDM基础参数
        'S0': 2.8,              # 损伤能量强度 [MPa]
        's_damage': 1.0,        # 损伤指数
        'p_D': 0.08,            # 损伤萌生阈值（316L延性更好）
        
        # AM修正参数
        'D0': 0.002,            # 初始损伤（316L通常更致密）
        'phi': 0.001,           # 孔隙率
        'phi_crit': 0.015,      # 临界孔隙率
        'lambda_mp': 35.0,      # 熔池边界密度 [mm⁻¹]
        'xi_grain': 2.5,        # 晶粒形态因子
        'theta_tex': 0.38,      # 织构取向权重（<001>纤维织构）
        
        # 修正系数
        'alpha1': 0.4, 'alpha2': 0.25,
        'n1': 1.5, 'n2': 2.0,
        'beta1': 1.5, 'beta2': 1.2, 'beta3': 8.0,
        'beta4': 0.6, 'beta5': 0.15,
        'm1': 2.0, 'm2': 1.5, 'm3': 1.0,
        
        # 实验参考值（文献）
        'E_exp': 190e9,
        'sigma_y_exp': 550e6,
        'sigma_uts_exp': 680e6,
        'ef_exp': 0.35,         # 0°取向
        'ef_45_exp': 0.30,
        'ef_90_exp': 0.25,
        
        # 微观特征
        'grain_size_avg': 25e-6,
        'grain_aspect_ratio': 2.5,
    },
    
    'AlSi10Mg': {
        'name': 'AlSi10Mg (LPBF)',
        'crystal_structure': 'FCC',
        'slip_systems': 'get_slip_systems_FCC',
        
        # 弹性常数 [GPa] → [Pa]
        'C11': 108e9, 'C12': 61e9, 'C44': 28e9,
        
        # CP硬化参数
        'g0': 120e6,
        'gs': 280e6,
        'h0': 300e6,
        'a': 2.3,
        'n_rate': 20.0,
        'gamma_dot_0': 0.001,
        'q_lat': 1.4,
        
        # CDM基础参数
        'S0': 2.0,
        's_damage': 1.0,
        'p_D': 0.02,            # AlSi10Mg脆性更大
        
        # AM修正参数
        'D0': 0.008,            # 初始损伤（Si颗粒/熔池边界效应）
        'phi': 0.005,           # 孔隙率
        'phi_crit': 0.025,      # 临界孔隙率
        'lambda_mp': 60.0,      # 熔池边界密度 [mm⁻¹]（细晶区边界多）
        'xi_grain': 1.5,        # 晶粒形态因子（近等轴晶）
        'theta_tex': 0.35,      # 织构取向权重（弱织构）
        
        # 修正系数
        'alpha1': 0.6, 'alpha2': 0.2,
        'n1': 1.5, 'n2': 2.0,
        'beta1': 2.5, 'beta2': 1.0, 'beta3': 12.0,
        'beta4': 1.0, 'beta5': 0.1,
        'm1': 2.0, 'm2': 1.5, 'm3': 1.0,
        
        # 实验参考值（文献）
        'E_exp': 68e9,
        'sigma_y_exp': 270e6,
        'sigma_uts_exp': 400e6,
        'ef_exp': 0.06,         # 0°取向
        'ef_45_exp': 0.05,
        'ef_90_exp': 0.04,
        
        # 微观特征
        'grain_size_avg': 3e-6,   # 细晶
        'grain_aspect_ratio': 1.5, # 近等轴
    }
}


# ============================================================
# 三材料各取向的AM修正参数差异
# ============================================================

# ============================================================
# 材料特定的各向异性损伤加速因子
# key: 取向(度) → f_AM乘数
# 物理基础：横向加载时熔池边界弱化效应更强，
#          柱状晶边界提供优先裂纹扩展路径，
#          加载方向与弱界面的角度关系决定损伤加速程度
# ============================================================
ANISOTROPY_F_AM = {
    'Ti64': {       # 高各向异性（HCP + 强柱状晶）✓ 3/3 PASS
        0: 1.0,     # 纵向加载 — 晶粒长轴 ∥ 载荷
        45: 2.0,    # 斜向 — 熔池边界剪切激活
        90: 3.8,    # 横向加载 — 层间弱界面 ∥ 载荷方向
    },
    '316L': {       # 中各向异性（FCC + 混合晶粒形态）
        0: 1.0,
        45: 1.8,    # boosted from 1.35
        90: 3.2,    # boosted from 2.0 (was too weak, ef=0.298 vs target 0.255)
    },
    'AlSi10Mg': {   # 中-高各向异性（Si网络 + 熔池边界效应）
        0: 1.0,
        45: 1.3,    # reduced from 1.6 (was too strong, overshoot)
        90: 2.2,    # reduced from 2.8 (was too strong, overshoot)
    },
}


def get_orientation_adjusted_params(material_key, orientation_deg):
    """
    根据加载取向调整AM修正参数

    0°: 平行于打印方向（纵向）— 基准各向同性响应
    45°: 倾斜加载 — 中等损伤加速
    90°: 垂直于打印方向（横向）— 显著损伤加速

    调整机制：
    1. λ（熔池边界密度）在横向加载时有效值更高
       → g1(λ)增大 → 损伤能量强度S_eff增大
       → BUT f_λ中的损伤倍增效应主导净效果
    2. D0（初始损伤）在横向更大 — 预制裂纹更多
    3. ξ（晶粒形态因子）在横向加载时表观值更低
       → 减小了S_eff，加速损伤
    4. f_AM_aniso：直接各向异性乘数 — 来自实验标定
    """
    params = MATERIAL_DATABASE[material_key].copy()

    # 基础取向修正
    if orientation_deg == 90:
        params['lambda_mp'] *= 2.0      # 横向有效熔池边界密度翻倍
        params['D0'] *= 2.5             # 横向初始损伤显著增大
        params['xi_grain'] = max(	   # 横向晶粒形态贡献减弱
            1.0, params['xi_grain'] * 0.4
        )
    elif orientation_deg == 45:
        params['lambda_mp'] *= 1.5
        params['D0'] *= 1.6
        params['xi_grain'] = max(
            1.0, params['xi_grain'] * 0.65
        )

    # 存储各向异性乘数（由am_correction使用）
    anisotropy = ANISOTROPY_F_AM.get(
        material_key, {0: 1.0, 45: 1.0, 90: 1.0}
    )
    params['_f_AM_aniso'] = anisotropy.get(orientation_deg, 1.0)

    return params


# ============================================================
# 弹性刚度矩阵构建
# ============================================================

def build_stiffness_tensor(params):
    """构建弹性刚度张量 C_ijkl [Pa]"""
    C = np.zeros((3, 3, 3, 3))
    
    if 'C13' in params:
        # HCP (transversely isotropic) - 5 independent constants
        C11, C12, C13 = params['C11'], params['C12'], params['C13']
        C33, C44 = params['C33'], params['C44']
        C66 = (C11 - C12) / 2.0
        
        # Voigt notation mapping
        for i in range(3):
            C[i,i,i,i] = C11 if i < 2 else C33
        
        C[0,0,1,1] = C[1,1,0,0] = C12
        C[0,0,2,2] = C[2,2,0,0] = C13
        C[1,1,2,2] = C[2,2,1,1] = C13
        
        C[1,2,1,2] = C[1,2,2,1] = C[2,1,1,2] = C[2,1,2,1] = C44
        C[0,2,0,2] = C[0,2,2,0] = C[2,0,0,2] = C[2,0,2,0] = C44
        
        C[0,1,0,1] = C[0,1,1,0] = C[1,0,0,1] = C[1,0,1,0] = C66
    else:
        # Cubic (FCC/BCC) - 3 independent constants
        C11, C12, C44 = params['C11'], params['C12'], params['C44']
        
        for i in range(3):
            C[i,i,i,i] = C11
        for i in range(3):
            for j in range(3):
                if i != j:
                    C[i,i,j,j] = C12
        
        C[1,2,1,2] = C[1,2,2,1] = C[2,1,1,2] = C[2,1,2,1] = C44
        C[0,2,0,2] = C[0,2,2,0] = C[2,0,0,2] = C[2,0,2,0] = C44
        C[0,1,0,1] = C[0,1,1,0] = C[1,0,0,1] = C[1,0,1,0] = C44
    
    return C


def double_contraction(C, E):
    """四阶张量与二阶张量双点积: sigma_ij = C_ijkl E_kl"""
    sigma = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    sigma[i, j] += C[i, j, k, l] * E[k, l]
    return sigma


# ============================================================
# 实验参考数据（文献）
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
        'uts': {0: 440, 45: 410, 90: 375},
        'Dc': 0.35, 'D0': 0.002,
    },
}
