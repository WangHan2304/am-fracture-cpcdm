"""TaylorCPCDM 无损伤 vs DAMASK FFT 全场对比分析。
统一坐标：对数应变 ε_ln = ln(1+ε_eng)。输出对比表 + 曲线数据。"""
import json, os
import numpy as np

base_wsl = r'\\wsl$\Ubuntu-24.04\home\zj'
out_dir = r'D:\20260618断裂模型论文\simulation\output'

def load_curve(path):
    d = np.loadtxt(path, skiprows=1)
    return d[:, 0], d[:, 1]

materials = ['316L', 'Ti64', 'AlSi10Mg']
report = {}

for mat in materials:
    # Taylor（工程应变 → 对数应变；应力 MPa 已由 Pa/1e6）
    tay_path = os.path.join(out_dir, f'taylor_nodamage_{mat}_0deg.csv')
    te, ts_pa = load_curve(tay_path)
    te_ln = np.log(1.0 + te)
    ts = ts_pa / 1e6  # Pa → MPa

    # DAMASK FFT
    dam_path = os.path.join(base_wsl, f'damask_{mat}_fft32', 'macro_curve.csv')
    if not os.path.exists(dam_path):
        print(f"{mat}: DAMASK curve not ready, skip")
        continue
    de, ds = load_curve(dam_path)

    # 对齐：在 DAMASK 应变点上插值 Taylor 应力
    common_eps = np.linspace(0.001, min(de.max(), te_ln.max()) - 1e-6, 100)
    tay_interp = np.interp(common_eps, te_ln, ts)
    dam_interp = np.interp(common_eps, de, ds)

    rel = (tay_interp - dam_interp) / np.maximum(dam_interp, 1e-9) * 100

    # UTS
    tay_uts = ts.max()
    dam_uts = ds.max()
    uts_rel = (tay_uts - dam_uts) / dam_uts * 100

    # 0.2% 偏移屈服（应力-弹性线交点）粗略：取 σ(ε=0.005) 作为流动应力近似
    s_5m = float(np.interp(0.005, te_ln, ts))
    d_5m = float(np.interp(0.005, de, ds))

    report[mat] = {
        'taylor_uts_MPa': float(tay_uts),
        'damask_uts_MPa': float(dam_uts),
        'uts_rel_diff_pct': float(uts_rel),
        'flow_stress_0.005_taylor_MPa': s_5m,
        'flow_stress_0.005_damask_MPa': d_5m,
        'flow_0.005_rel_pct': float((s_5m - d_5m) / d_5m * 100),
        'mean_rel_diff_pct': float(np.mean(np.abs(rel))),
        'max_rel_diff_pct': float(np.max(np.abs(rel))),
        'eps_max_common': float(common_eps.max()),
        'curve': {'eps': common_eps.tolist(), 'taylor_MPa': tay_interp.tolist(),
                  'damask_MPa': dam_interp.tolist(), 'rel_pct': rel.tolist()},
    }
    print(f"{mat}: Taylor UTS={tay_uts:.1f} vs DAMASK UTS={dam_uts:.1f} MPa "
          f"(rel {uts_rel:+.1f}%) | flow@0.005: {s_5m:.1f} vs {d_5m:.1f} MPa "
          f"({(s_5m-d_5m)/d_5m*100:+.1f}%) | mean|rel|={np.mean(np.abs(rel)):.1f}%")

with open(os.path.join(out_dir, 'taylor_vs_damask_results.json'), 'w') as f:
    json.dump(report, f, indent=1)
print("saved taylor_vs_damask_results.json")
