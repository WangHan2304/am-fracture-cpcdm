"""
多晶RVE模拟引擎
实现：
1. 基于Voronoi的多晶RVE生成
2. Taylor假设多晶均匀化（等应变）
3. 多RVE统计与不确定性量化

Taylor模型假设：所有晶粒经历相同的变形梯度（上界估计）
"""

import numpy as np
from cp_cdm_model import CrystalPlasticityCDM, run_uniaxial_tension


# ============================================================
# 晶粒取向生成
# ============================================================

def generate_random_orientations(n_grains, texture_type='random', texture_strength=1.0):
    """
    生成随机晶粒取向
    
    Args:
        n_grains: 晶粒数
        texture_type: 'random', 'fiber_001', 'fiber_111', 'cube'
        texture_strength: 织构强度 (1=random, >1=sharper)
    
    Returns:
        orientations: list of (3,3) rotation matrices
    """
    orientations = []
    
    for i in range(n_grains):
        if texture_type == 'random':
            # 均匀随机旋转（使用随机四元数）
            R = _random_rotation_matrix()
        elif texture_type == 'fiber_001':
            # <001> 纤维织构 (沿z轴)
            R = _fiber_texture_rotation(axis=2, strength=texture_strength)
        elif texture_type == 'fiber_111':
            # <111> 纤维织构
            R = _fiber_texture_rotation(axis=[1, 1, 1], strength=texture_strength)
        elif texture_type == 'cube':
            # Cube织构 {001}<100>
            R = _cube_texture_rotation(strength=texture_strength)
        else:
            R = _random_rotation_matrix()
        
        orientations.append(R)
    
    return orientations


def _random_rotation_matrix():
    """生成均匀随机旋转矩阵（SO(3)）"""
    # 随机四元数
    u1, u2, u3 = np.random.rand(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3)
    ])
    q = q / np.linalg.norm(q)
    
    # 四元数→旋转矩阵
    w, x, y, z = q[3], q[0], q[1], q[2]
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
    ])
    return R


def _fiber_texture_rotation(axis, strength=1.0):
    """纤维织构取向（axis附近，strength控制分散度）"""
    if isinstance(axis, int):
        v = np.zeros(3)
        v[axis] = 1.0
    else:
        v = np.array(axis) / np.linalg.norm(axis)
    
    # 随机偏离角（von Mises-Fisher分布近似）
    kappa = strength * 10.0  # 集中度参数
    theta = np.random.vonmises(0, kappa)  # 偏离角
    
    # 随机方位角
    phi = np.random.uniform(0, 2 * np.pi)
    
    # 从v旋转theta, phi
    # 简化：使用随机旋转+插值
    R_rand = _random_rotation_matrix()
    
    # 插值：strength越大越接近identity（即织构越强）
    w = 1.0 / strength
    # 球面线性插值
    R = _slerp_rotation(np.eye(3), R_rand, w)
    
    return R


def _cube_texture_rotation(strength=1.0):
    """Cube织构 {001}<100>"""
    # Cube取向即identity（在标准坐标系中）
    R_base = np.eye(3)
    R_rand = _random_rotation_matrix()
    w = 1.0 / max(strength, 1.0)
    return _slerp_rotation(R_base, R_rand, w)


def _slerp_rotation(R1, R2, t):
    """旋转矩阵球面线性插值"""
    # 转为四元数
    def rot_to_quat(R):
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (R[2,1] - R[1,2]) / s
            y = (R[0,2] - R[2,0]) / s
            z = (R[1,0] - R[0,1]) / s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])
    
    q1 = rot_to_quat(R1)
    q2 = rot_to_quat(R2)
    
    # 确保走短弧
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    if dot > 0.9995:
        q = q1 + t * (q2 - q1)
        q = q / np.linalg.norm(q)
    else:
        theta_0 = np.arccos(dot)
        sin_theta = np.sin(theta_0)
        s1 = np.sin((1-t)*theta_0) / sin_theta
        s2 = np.sin(t*theta_0) / sin_theta
        q = s1 * q1 + s2 * q2
    
    w, x, y, z = q
    R = np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]
    ])
    return R


# ============================================================
# Taylor多晶模型
# ============================================================

class TaylorPolycrystal:
    """
    Taylor假设多晶模型
    所有晶粒经历相同的宏观变形梯度
    宏观应力 = 各晶粒应力的体积平均
    """
    
    def __init__(self, material_key, n_grains=100, texture_type='random',
                 orientation_deg=0, texture_strength=1.0):
        """
        Args:
            material_key: 材料键
            n_grains: 晶粒数
            texture_type: 织构类型
            orientation_deg: 加载取向
            texture_strength: 织构强度
        """
        self.material_key = material_key
        self.n_grains = n_grains
        self.orientation_deg = orientation_deg
        
        # 生成晶粒取向
        self.grain_orientations = generate_random_orientations(
            n_grains, texture_type, texture_strength
        )
        
        # 为每个晶粒创建CP模型
        self.grains = []
        for orient in self.grain_orientations:
            grain = CrystalPlasticityCDM(material_key, orientation_deg, orient)
            self.grains.append(grain)
    
    def simulate_tension(self, max_strain=0.5, n_steps=200, strain_rate=1e-3,
                          use_AM=True):
        """
        Taylor多晶拉伸模拟
        
        Returns:
            results: 均匀化结果
            grain_results: 各晶粒结果列表
        """
        nu = 0.3
        eps_axial = np.linspace(0, max_strain, n_steps + 1)
        
        # 构建宏观应变路径
        strain_path = np.zeros((n_steps + 1, 6))
        for i in range(n_steps + 1):
            strain_path[i, 0] = eps_axial[i]
            strain_path[i, 1] = -nu * eps_axial[i]
            strain_path[i, 2] = -nu * eps_axial[i]
        
        # 所有晶粒计算（并行化候选）
        grain_stresses = np.zeros((self.n_grains, n_steps + 1, 6))
        grain_damages = np.zeros((self.n_grains, n_steps + 1))
        grain_ps = np.zeros((self.n_grains, n_steps + 1))
        
        for ig, grain in enumerate(self.grains):
            if not use_AM:
                grain.f_AM = 1.0
                grain.S_eff = grain.params['S0']
                grain.D = 0.0
            
            stress_h, D_h, p_h = grain.integrate(strain_path, strain_rate)
            grain_stresses[ig] = stress_h
            grain_damages[ig] = D_h
            grain_ps[ig] = p_h
        
        # 均匀化（体积平均，假设等体积）
        macro_stress = np.mean(grain_stresses, axis=0)
        macro_damage = np.mean(grain_damages, axis=0)
        macro_p = np.mean(grain_ps, axis=0)
        
        # 断裂应变（宏观平均损伤达临界值）
        fracture_idx = np.where(macro_damage >= 0.5)[0]
        if len(fracture_idx) > 0:
            fracture_strain = eps_axial[fracture_idx[0]]
            macro_stress[fracture_idx[0]:, 0] = 0.0
        else:
            fracture_strain = max_strain
        
        # 力学性能
        stress_axial = macro_stress[:, 0]
        E_mod = stress_axial[1] / max(eps_axial[1], 1e-10) if n_steps > 1 else 0
        
        sigma_uts = np.max(stress_axial)
        
        results = {
            'strain_true': eps_axial,
            'stress_true': stress_axial,
            'damage': macro_damage,
            'p_cum': macro_p,
            'material': self.material_key,
            'orientation': self.orientation_deg,
            'n_grains': self.n_grains,
            'model_type': 'Modified_CDM' if use_AM else 'Lemaitre',
            'fracture_strain': fracture_strain,
            'max_stress': sigma_uts,
            'elastic_modulus': E_mod,
            # 晶粒级统计
            'grain_stresses': grain_stresses,
            'grain_damages': grain_damages,
            'grain_ps': grain_ps,
            'damage_std': np.std(grain_damages, axis=0),
            'stress_std': np.std(grain_stresses[:, :, 0], axis=0),
        }
        
        return results


def run_rve_ensemble(material_key, orientation_deg=0, n_rve=3, n_grains=50,
                      max_strain=0.15, use_AM=True, texture_type='random'):
    """
    运行多个RVE实例进行统计
    
    Returns:
        ensemble_results: 各RVE结果列表
        mean_results: 平均结果
    """
    ensemble = []
    
    for i in range(n_rve):
        print(f"  RVE {i+1}/{n_rve}...")
        taylor = TaylorPolycrystal(material_key, n_grains, texture_type,
                                    orientation_deg)
        result = taylor.simulate_tension(max_strain, n_steps=150, use_AM=use_AM)
        ensemble.append(result)
    
    # 统计平均
    n_pts = min(len(r['strain_true']) for r in ensemble)
    strain = ensemble[0]['strain_true'][:n_pts]
    
    stresses = np.array([r['stress_true'][:n_pts] for r in ensemble])
    damages = np.array([r['damage'][:n_pts] for r in ensemble])
    fractures = np.array([r['fracture_strain'] for r in ensemble])
    
    mean_stress = np.mean(stresses, axis=0)
    std_stress = np.std(stresses, axis=0)
    mean_damage = np.mean(damages, axis=0)
    std_damage = np.std(damages, axis=0)
    mean_fracture = np.mean(fractures)
    std_fracture = np.std(fractures)
    
    mean_results = {
        'strain_true': strain,
        'stress_true': mean_stress,
        'stress_std': std_stress,
        'damage': mean_damage,
        'damage_std': std_damage,
        'fracture_strain': mean_fracture,
        'fracture_strain_std': std_fracture,
        'fracture_strain_ci95': 1.96 * std_fracture / np.sqrt(n_rve),
        'material': material_key,
        'orientation': orientation_deg,
        'model_type': 'Modified_CDM' if use_AM else 'Lemaitre',
        'n_rve': n_rve,
        'n_grains': n_grains,
        'ensemble': ensemble,
    }
    
    return mean_results


if __name__ == '__main__':
    print("Testing Taylor Polycrystal Model...")
    taylor = TaylorPolycrystal('Ti64', n_grains=30, texture_type='random', orientation_deg=0)
    result = taylor.simulate_tension(max_strain=0.1, n_steps=100)
    print(f"Ti64 Taylor (30 grains): εf = {result['fracture_strain']:.4f}, "
          f"σmax = {result['max_stress']/1e6:.1f} MPa")
