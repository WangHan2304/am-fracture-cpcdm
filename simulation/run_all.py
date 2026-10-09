"""
论文完整模拟运行脚本
生成所有对比数据和中间结果
输出到 simulation/output/
"""

import numpy as np
import json
import os
import sys
from datetime import datetime

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cp_cdm_model import run_uniaxial_tension, run_systematic_simulation
from rve_simulator import run_rve_ensemble
from materials import MATERIAL_DATABASE


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_all_simulations():
    """运行所有模拟并保存结果"""
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    ensure_dir(output_dir)
    
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'materials': {},
    }
    
    materials = ['Ti64', '316L', 'AlSi10Mg']
    orientations = [0, 45, 90]
    
    print("=" * 70)
    print("FULL SIMULATION RUN FOR PAPER")
    print("=" * 70)
    
    # ================================================
    # Part 1: 单晶级模拟（参数验证）
    # ================================================
    print("\n" + "=" * 70)
    print("PART 1: Single Crystal Simulations")
    print("=" * 70)
    
    for mat in materials:
        results_summary['materials'][mat] = {}
        for orient in orientations:
            key = f"{orient}deg"
            print(f"\n--- {mat} {key} (Modified CDM) ---")
            
            # 修正模型
            r_mod = run_uniaxial_tension(mat, orient, max_strain=0.12, 
                                          n_steps=150, use_AM=True)
            
            # 传统Lemaitre
            r_lem = run_uniaxial_tension(mat, orient, max_strain=0.12,
                                          n_steps=150, use_AM=False)
            
            results_summary['materials'][mat][key] = {
                'modified': {
                    'fracture_strain': float(r_mod['fracture_strain']),
                    'max_stress_MPa': float(r_mod['max_stress'] / 1e6),
                    'elastic_modulus_GPa': float(r_mod['elastic_modulus'] / 1e9),
                    'yield_stress_MPa': float(r_mod.get('yield_stress', 0) / 1e6),
                },
                'lemaitre': {
                    'fracture_strain': float(r_lem['fracture_strain']),
                    'max_stress_MPa': float(r_lem['max_stress'] / 1e6),
                }
            }
            
            # 与实验值对比
            exp_data = MATERIAL_DATABASE[mat]
            exp_ef = exp_data.get(f'ef_{orient}_exp', exp_data.get('ef_exp', 0.1))
            ef_pred = r_mod['fracture_strain']
            error = abs(ef_pred - exp_ef) / max(exp_ef, 0.001) * 100
            
            results_summary['materials'][mat][key]['experimental'] = {
                'ef_exp': float(exp_ef),
                'ef_pred': float(ef_pred),
                'error_pct': float(error),
                'pass_10pct': bool(error < 10.0),
            }
            
            status = "PASS" if error < 10 else "FAIL"
            print(f"  ef_exp={exp_ef:.4f}, ef_pred={ef_pred:.4f}, error={error:.1f}% [{status}]")
    
    # ================================================
    # Part 2: 多晶RVE模拟（Taylor模型）
    # ================================================
    print("\n" + "=" * 70)
    print("PART 2: Polycrystal RVE Simulations (Taylor Model)")
    print("=" * 70)
    
    for mat in materials:
        print(f"\n--- {mat} RVE Ensemble ---")
        for orient in [0, 90]:
            key = f"{orient}deg"
            
            # 修正模型 RVE
            rve_mod = run_rve_ensemble(mat, orient, n_rve=3, n_grains=30,
                                        max_strain=0.1, use_AM=True)
            
            # Lemaitre RVE
            rve_lem = run_rve_ensemble(mat, orient, n_rve=3, n_grains=30,
                                        max_strain=0.1, use_AM=False)
            
            results_summary['materials'][mat][f'{key}_rve'] = {
                'modified': {
                    'fracture_strain': float(rve_mod['fracture_strain']),
                    'fracture_strain_std': float(rve_mod['fracture_strain_std']),
                    'fracture_strain_ci95': float(rve_mod['fracture_strain_ci95']),
                },
                'lemaitre': {
                    'fracture_strain': float(rve_lem['fracture_strain']),
                    'fracture_strain_std': float(rve_lem['fracture_strain_std']),
                }
            }
    
    # ================================================
    # Part 3: 参数敏感性分析
    # ================================================
    print("\n" + "=" * 70)
    print("PART 3: Parameter Sensitivity Analysis")
    print("=" * 70)
    
    sensitivity = run_sensitivity_analysis('Ti64', orient=0)
    results_summary['sensitivity'] = sensitivity
    
    # ================================================
    # Part 4: AM修正贡献度分析
    # ================================================
    print("\n" + "=" * 70)
    print("PART 4: AM Correction Contribution Analysis")
    print("=" * 70)
    
    contribution = run_contribution_analysis('Ti64')
    results_summary['contribution'] = contribution
    
    # ================================================
    # 保存结果
    # ================================================
    summary_path = os.path.join(output_dir, 'simulation_summary.json')
    
    # 自定义JSON序列化
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {summary_path}")
    
    # 打印汇总表
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print("\nFracture Strain Prediction Errors:")
    print("-" * 60)
    print(f"{'Material':<12} {'Orient':<8} {'εf_exp':<10} {'εf_pred':<10} {'Error%':<8} {'Pass'}")
    print("-" * 60)
    
    total_pass = 0
    total_count = 0
    for mat in materials:
        for orient in orientations:
            key = f"{orient}deg"
            data = results_summary['materials'][mat][key]
            exp = data['experimental']
            total_count += 1
            if exp['pass_10pct']:
                total_pass += 1
            print(f"{mat:<12} {key:<8} {exp['ef_exp']:<10.4f} "
                  f"{exp['ef_pred']:<10.4f} {exp['error_pct']:<8.1f} "
                  f"{'PASS' if exp['pass_10pct'] else 'FAIL'}")
    
    print("-" * 60)
    print(f"Pass rate: {total_pass}/{total_count} ({total_pass/total_count*100:.0f}%)")
    
    return results_summary


def run_sensitivity_analysis(material_key, orient=0):
    """
    参数敏感性分析
    对关键AM修正参数进行±20%扰动，观察断裂应变变化
    """
    params_to_test = [
        'D0', 'phi', 'lambda_mp', 'xi_grain', 'theta_tex',
        'alpha1', 'alpha2', 'beta1', 'beta2', 'beta3',
        'beta4', 'beta5', 'S0', 's_damage', 'p_D'
    ]
    
    # 基准值
    from cp_cdm_model import CrystalPlasticityCDM
    base = CrystalPlasticityCDM(material_key, orient)
    
    base_results = run_uniaxial_tension(material_key, orient, 
                                         max_strain=0.1, n_steps=100)
    base_ef = base_results['fracture_strain']
    
    sensitivity = {'base_ef': float(base_ef), 'parameters': {}}
    
    for param in params_to_test:
        original = base.params.get(param, None)
        if original is None:
            continue
        
        effects = {}
        for perturb in [-0.2, -0.1, 0.1, 0.2]:
            # 临时修改参数
            saved_val = base.params[param]
            base.params[param] = original * (1.0 + perturb)
            base.reset()
            
            # 重新计算修正
            from cp_cdm_model import compute_AM_correction
            base.f_AM, base.S_eff = compute_AM_correction(base.params)
            
            r = run_uniaxial_tension(material_key, orient, 
                                      max_strain=0.1, n_steps=100)
            effects[f'{perturb:+.0%}'] = float(r['fracture_strain'])
            
            # 恢复
            base.params[param] = saved_val
        
        # 计算敏感度（归一化）
        ef_values = list(effects.values())
        sensitivity['parameters'][param] = {
            'nominal': float(original),
            'effects': effects,
            'sensitivity_index': float(np.std(ef_values) / max(base_ef, 0.001)),
            'max_change_pct': float(max(abs(v/base_ef - 1) for v in ef_values) * 100),
        }
        
        print(f"  {param}: sensitivity = {sensitivity['parameters'][param]['sensitivity_index']:.3f}")
    
    return sensitivity


def run_contribution_analysis(material_key):
    """
    AM修正各分项贡献度分析
    逐项关闭修正，看断裂应变变化
    """
    base = run_uniaxial_tension(material_key, 0, max_strain=0.1, n_steps=100, use_AM=True)
    base_ef = base['fracture_strain']
    
    # 传统Lemaitre基线
    lem = run_uniaxial_tension(material_key, 0, max_strain=0.1, n_steps=100, use_AM=False)
    lem_ef = lem['fracture_strain']
    
    contributions = {
        'base_modified_ef': float(base_ef),
        'lemaitre_ef': float(lem_ef),
        'total_effect_pct': float((base_ef - lem_ef) / max(lem_ef, 0.001) * 100),
        'components': {}
    }
    
    # 逐项关闭（设为中性值）
    components = {
        'porosity (f_phi)': {'phi': 0, 'D0': 0},
        'texture (f_theta)': {'theta_tex': 0.5},  # 中性织构
        'melt_pool (f_lambda)': {'lambda_mp': 0},
        'morphology (f_xi)': {'xi_grain': 1.0},
    }
    
    for name, param_changes in components.items():
        from cp_cdm_model import CrystalPlasticityCDM
        cp = CrystalPlasticityCDM(material_key, 0)
        
        for p, v in param_changes.items():
            cp.params[p] = v
        
        from cp_cdm_model import compute_AM_correction
        cp.f_AM, cp.S_eff = compute_AM_correction(cp.params)
        
        r = run_uniaxial_tension(material_key, 0, max_strain=0.1, n_steps=100)
        ef = r['fracture_strain']
        
        delta = base_ef - ef
        contributions['components'][name] = {
            'ef_without': float(ef),
            'delta_ef': float(delta),
            'contribution_pct': float(delta / max(base_ef - lem_ef, 0.001) * 100),
        }
        
        print(f"  Without {name}: εf = {ef:.4f} (Δ = {delta:.4f}, "
              f"{contributions['components'][name]['contribution_pct']:.1f}%)")
    
    return contributions


if __name__ == '__main__':
    run_all_simulations()
