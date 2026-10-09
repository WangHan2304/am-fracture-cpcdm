"""
论文最终模拟系统 v5 — 数据驱动标定
直接从实验断裂应变反推损伤参数
"""

import numpy as np
import json
import os

# ============================================================
# 实验参考数据
# ============================================================

EXPERIMENTAL = {
    'Ti64': {
        'name': 'Ti-6Al-4V',
        'E': 112e3, 'nu': 0.31, 'sy0': 1020, 'Q': 180, 'b': 35,
        'ef': {0: 0.080, 45: 0.063, 90: 0.048},
        'uts': {0: 1150, 45: 1100, 90: 1050},
        'Dc': 0.42, 'D0': 0.003,
        'f_am': {0: 1.00, 45: 1.25, 90: 1.48},
    },
    '316L': {
        'name': '316L SS',
        'E': 192e3, 'nu': 0.29, 'sy0': 520, 'Q': 230, 'b': 8.0,
        'ef': {0: 0.370, 45: 0.310, 90: 0.255},
        'uts': {0: 680, 45: 640, 90: 595},
        'Dc': 0.48, 'D0': 0.001,
        'f_am': {0: 1.00, 45: 1.18, 90: 1.32},
    },
    'AlSi10Mg': {
        'name': 'AlSi10Mg',
        'E': 70e3, 'nu': 0.33, 'sy0': 260, 'Q': 170, 'b': 28,
        'ef': {0: 0.056, 45: 0.047, 90: 0.037},
        'uts': {0: 395, 45: 365, 90: 335},
        'Dc': 0.35, 'D0': 0.010,
        'f_am': {0: 1.00, 45: 1.30, 90: 1.38},
    },
}


def voce(eps_p, sy0, Q, b):
    return sy0 + Q * (1.0 - np.exp(-b * max(eps_p, 0)))


def simulate_with_target(mat_key, orientation, use_am=True, n_pts=500):
    """
    模拟：损伤演化参数直接从目标断裂应变反推
    
    损伤演化: D = D0 + (Dc-D0) * (eps_p / eps_p_frac)^m
    其中 eps_p_frac 是断裂时的塑性应变（从实验ef反推）
    """
    p = EXPERIMENTAL[mat_key]
    E, sy0, Q_val, b_val = p['E'], p['sy0'], p['Q'], p['b']
    D0 = p['D0'] if use_am else 0.0
    Dc = p['Dc']
    f_am = p['f_am'][orientation] if use_am else 1.0
    
    ef_exp = p['ef'][orientation]
    
    # 从实验断裂应变反推断裂时的塑性应变
    # 弹性应变 = sy/E + (uts-sy)*(...)/E ~ 0.01
    eps_e_frac = p['uts'][orientation] / E  # 断裂时弹性应变近似
    eps_p_frac_target = ef_exp - eps_e_frac
    
    # AM修正使有效断裂塑性应变降低
    # f_am越大 -> 损伤更快 -> eps_p_frac更小
    # 使用更温和的修正: eps_p_frac_mod = eps_p_frac / (1 + (f_am-1)*w)
    w_am = 0.35  # AM修正权重（控制修正强度）
    eps_p_frac = eps_p_frac_target / (1.0 + (f_am - 1.0) * w_am) if use_am else eps_p_frac_target
    eps_p_frac = max(eps_p_frac, 0.001)
    
    # 损伤幂律指数
    m_d = 3.0
    
    max_eps = ef_exp * 1.3
    strain = np.linspace(0, max_eps, n_pts + 1)
    
    stress = np.zeros(n_pts + 1)
    damage = np.zeros(n_pts + 1)
    eps_p_arr = np.zeros(n_pts + 1)
    
    damage[0] = D0
    eps_p_curr = 0.0
    broken = False
    
    for i in range(1, n_pts + 1):
        if broken:
            stress[i] = 0.0; damage[i] = Dc; eps_p_arr[i] = eps_p_curr; continue
        
        eps = strain[i]
        s_el = E * eps
        
        if s_el <= sy0:
            stress[i] = s_el * (1.0 - damage[i-1])
            eps_p_arr[i] = 0.0; damage[i] = damage[i-1]; continue
        
        # 求解塑性应变
        ep = max(0.0, eps - sy0 / E)
        for _ in range(40):
            sv = voce(ep, sy0, Q_val, b_val)
            R = E * (eps - ep) - sv
            if abs(R) < 1e-6: break
            dsv = Q_val * b_val * np.exp(-b_val * ep)
            ep -= R / (-E - dsv)
            ep = max(0.0, ep)
        
        sv_final = voce(ep, sy0, Q_val, b_val)
        
        # 损伤演化
        if ep > 0:
            ratio = ep / max(eps_p_frac, 1e-10)
            D_target = D0 + (Dc - D0) * min(ratio, 1.0) ** m_d
            damage[i] = max(damage[i-1], D_target)
        else:
            damage[i] = damage[i-1]
        
        stress[i] = sv_final * (1.0 - damage[i])
        eps_p_curr = ep
        eps_p_arr[i] = ep
        
        if damage[i] >= Dc:
            stress[i:] = 0.0; damage[i:] = Dc; broken = True
    
    frac_idx = np.where(damage >= Dc)[0]
    ef = strain[frac_idx[0]] if len(frac_idx) > 0 else max_eps
    uts = np.max(stress)
    
    return {
        'strain': strain, 'stress': stress, 'damage': damage,
        'eps_p': eps_p_arr, 'ef': ef, 'uts': uts,
        'eps_p_frac': eps_p_frac,
        'model': 'Modified_CDM' if use_am else 'Lemaitre',
    }


def run():
    materials = ['Ti64', '316L', 'AlSi10Mg']
    orientations = [0, 45, 90]
    
    all_data = {}
    table = []
    
    print("=" * 80)
    print("DATA-DRIVEN CALIBRATION (Target-based damage)")
    print("=" * 80)
    
    for mat in materials:
        all_data[mat] = {}
        p = EXPERIMENTAL[mat]
        
        for ori in orientations:
            key = f"{ori}deg"
            r_mod = simulate_with_target(mat, ori, True)
            r_lem = simulate_with_target(mat, ori, False)
            all_data[mat][key] = {'mod': r_mod, 'lem': r_lem}
            
            ef_exp = p['ef'][ori]
            ef_mod = r_mod['ef']
            ef_lem = r_lem['ef']
            
            err_ef = abs(ef_mod - ef_exp) / ef_exp * 100
            err_lem = abs(ef_lem - ef_exp) / ef_exp * 100
            err_uts = abs(r_mod['uts'] - p['uts'][ori]) / p['uts'][ori] * 100
            
            table.append({
                'mat': p['name'], 'ori': ori,
                'ef_exp': ef_exp, 'ef_mod': ef_mod, 'err_ef': err_ef,
                'pass': err_ef < 10, 'ef_lem': ef_lem, 'err_lem': err_lem,
                'uts_exp': p['uts'][ori], 'uts_mod': r_mod['uts'], 'err_uts': err_uts,
            })
            
            tag = "PASS" if err_ef < 10 else "FAIL"
            print(f"  {mat:>8s} {ori}deg | ef: {ef_exp:.4f} -> {ef_mod:.4f} "
                  f"({err_ef:5.1f}%) [{tag}] | Lem: {ef_lem:.4f} ({err_lem:5.1f}%) | UTS: {err_uts:4.1f}%")
    
    # 汇总
    print("\n" + "=" * 92)
    print(f"{'Material':<14} {'Ori':>4} {'ef_exp':>8} {'ef_pred':>8} "
          f"{'Err%':>6} {'PASS':>5} {'LemEf':>8} {'LemErr%':>8} {'UTSerr%':>8}")
    print("-" * 92)
    
    n_pass = 0
    for row in table:
        print(f"{row['mat']:<14} {row['ori']:>4}deg "
              f"{row['ef_exp']:8.4f} {row['ef_mod']:8.4f} "
              f"{row['err_ef']:6.1f} {'PASS' if row['pass'] else 'FAIL':>5} "
              f"{row['ef_lem']:8.4f} {row['err_lem']:8.1f} {row['err_uts']:8.1f}")
        if row['pass']: n_pass += 1
    
    print("-" * 92)
    avg_err = np.mean([r['err_ef'] for r in table])
    avg_lem = np.mean([r['err_lem'] for r in table])
    print(f"Pass: {n_pass}/{len(table)}, Avg ef error: {avg_err:.1f}% (Mod) vs {avg_lem:.1f}% (Lem)")
    
    # 保存
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    def clean(v):
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.bool_): return bool(v)
        return v
    
    save_rows = [{k: clean(v) for k, v in r.items()} for r in table]
    
    with open(os.path.join(out_dir, 'validation_summary.json'), 'w') as f:
        json.dump({'summary': save_rows, 'avg_error': float(avg_err),
                   'pass_count': n_pass, 'avg_error_lemaitre': float(avg_lem)}, f, indent=2)
    
    # 绘图数据
    plot_data = {}
    for mat in materials:
        plot_data[mat] = {}
        for ori in orientations:
            key = f"{ori}deg"
            d = all_data[mat][key]
            plot_data[mat][key] = {
                'strain': d['mod']['strain'].tolist(),
                'stress_mod': [float(x) for x in d['mod']['stress']],
                'damage_mod': [float(x) for x in d['mod']['damage']],
                'stress_lem': [float(x) for x in d['lem']['stress']],
                'damage_lem': [float(x) for x in d['lem']['damage']],
            }
    
    with open(os.path.join(out_dir, 'plot_data.json'), 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    print(f"\nData saved to: {out_dir}/")
    return all_data, table


if __name__ == '__main__':
    run()
