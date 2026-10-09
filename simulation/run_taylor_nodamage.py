"""TaylorCPCDM 无损伤对比：与 DAMASK FFT 全场同参数、同 30 晶粒微结构、
同 0° 单轴拉伸，输出宏观 σ-ε 曲线与 UTS。
无损伤 = S0 设极大(1e12 MPa)使损伤不演化；D0 用实验数据库值。
"""
import sys, json
sys.path.insert(0, r'D:\20260618断裂模型论文\simulation\src')
import numpy as np
from taylor_cpcdm import TaylorCPCDM

out_dir = r'D:\20260618断裂模型论文\simulation\output'
results = {}

for mat, max_strain in [('316L', 0.30), ('Ti64', 0.20), ('AlSi10Mg', 0.15)]:
    model = TaylorCPCDM(
        mat, orientation_deg=0, n_grains=30, seed=42,
        model_version='v4',
        S0_override=1e12,          # 无损伤：有效应力 →0，损伤不演化
        param_overrides={'S0': 1e12},
    )
    res = model.run_uniaxial(max_strain=max_strain)
    strain = res['strain']
    stress = res['stress']
    uts = res['uts']
    # 0.2% 偏移屈服
    offset = 0.002
    sig_off = stress - model.exp['E'] * (strain - offset)
    yp_idx = np.where((strain > offset) & (sig_off > 0))[0]
    yp = float(strain[yp_idx[0]]) if len(yp_idx) else np.nan
    results[mat] = {
        'strain': strain.tolist(),
        'stress': stress.tolist(),
        'uts_MPa': float(uts),
        'yield_0.2pct_strain': yp,
        'E_used': model.exp['E'],
    }
    print(f"{mat}: UTS={uts:.1f} MPa, 0.2% yield at eps={yp:.4f}, n={len(strain)}")
    # 保存曲线 CSV
    csv = f"{out_dir}/taylor_nodamage_{mat}_0deg.csv"
    np.savetxt(csv, np.column_stack([strain, stress]),
               header='eps_11, sigma_11_MPa', fmt='%.6f', comments='')
    print("  saved", csv)

with open(f"{out_dir}/taylor_nodamage_results.json", 'w') as f:
    json.dump(results, f, indent=1)
print("DONE")
