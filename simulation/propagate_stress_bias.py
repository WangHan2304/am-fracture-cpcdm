"""应力偏差→断裂应变传播估计（第7条降级量化）。
用 Taylor 与 DAMASK-FFT 两条宏观应力路径驱动同一单点损伤演化，
比较断裂应变差异 = 全场重分布对断裂应变预测的影响（estimate）。
单点等效：p = eps - sigma/E, dD = (Y/S0)^s dp, Y = sigma^2/(2E), Dc=0.99。
"""
import os
import numpy as np

out_dir = r'D:\20260618断裂模型论文\simulation\output'
wsl_base = r'\\wsl$\Ubuntu-24.04\home\zj'

MATS = {
    '316L':    dict(S0=3.20, E=204e9, max_eps=0.30, tag='316L'),
    'AlSi10Mg': dict(S0=0.20, E=108e9, max_eps=0.15, tag='AlSi10Mg'),
    'Ti64':    dict(S0=0.32, E=162e9, max_eps=0.20, tag='Ti64'),
}
s_exp = 1.0
D0 = 0.0  # 传播估计中取干净起点，突出应力路径效应
Dc = 0.99
n_sub = 40000  # 子步插值

report = {}
for mat, cfg in MATS.items():
    S0, E, mtag = cfg['S0'] * 1e6, cfg['E'], cfg['tag']

    # Taylor 应力曲线（Pa）
    tp = os.path.join(out_dir, f'taylor_nodamage_{mtag}_0deg.csv')
    te, ts = np.loadtxt(tp, skiprows=1).T
    # FFT 应力曲线（MPa -> Pa）
    dp = os.path.join(wsl_base, f'damask_{mtag}_fft32', 'macro_curve.csv')
    de, ds = np.loadtxt(dp, skiprows=1).T
    ds = ds * 1e6

    # 统一网格：细插值（对数应变坐标）
    eps_grid = np.linspace(1e-4, min(te.max(), de.max(), cfg['max_eps']) - 1e-4, n_sub)
    sig_t = np.interp(eps_grid, te, ts)
    sig_f = np.interp(eps_grid, de, ds)

    def fracture_strain(sig):
        D = 0.0
        p_prev = 0.0
        ef = None
        for i in range(1, len(eps_grid)):
            eps = eps_grid[i]
            ee = sig[i] / E
            p = max(0.0, eps - ee)
            dp = p - p_prev
            Y = sig[i] ** 2 / (2.0 * E)
            dD = (Y / S0) ** s_exp * dp
            D += dD
            p_prev = p
            if D >= Dc:
                ef = float(eps)
                break
        if ef is None:
            ef = float(eps_grid[-1])
        return ef

    ef_t = fracture_strain(sig_t)
    ef_f = fracture_strain(sig_f)
    rel = (ef_f - ef_t) / ef_t * 100.0
    report[mat] = dict(ef_taylor=ef_t, ef_fftstress=ef_f, rel_pct=rel)
    print(f"{mat}: ef(Taylor stress)={ef_t:.4f}  ef(FFT stress)={ef_f:.4f}  rel={rel:+.1f}%")

with open(os.path.join(out_dir, 'stress_bias_propagation.json'), 'w') as f:
    import json
    json.dump(report, f, indent=1, default=float)
print('saved stress_bias_propagation.json')
