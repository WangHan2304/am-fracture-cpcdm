# -*- coding: utf-8 -*-
"""
v9 泄漏无关协议的参数敏感性分析（OAT ±40%）
====================================================
与 v9 正典完全一致：
- 模型版本 v4（投影律承载取向依赖，无 a45/a90 乘数）
- g_s 固定文献值（Ti64: 700/720/800 MPa）
- S0 = 0.32 MPa（v9 权威标定值，Ti64）
- 工况：Ti-6Al-4V, 0°取向, 20 晶粒, eps_step=5e-4, n_sub=30

输出: simulation/output/sensitivity_v9.json
      simulation/figures/Fig_sensitivity_v2.png (覆盖为 v9 版)
"""
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

MAT = 'Ti64'
ORIENT = 0
S0 = 0.32          # v9 权威值
N_GRAINS = 20
EPS_STEP = 5e-4
N_SUB = 30
RATE = 1e-3
PERTURB = [-0.40, -0.20, +0.20, +0.40]

GS_LIT = {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}

PARAM_LIST = ['D0', 'phi', 'lambda_mp', 'xi_grain', 'theta_tex',
              'alpha1', 'alpha2', 'beta1', 'beta2', 'beta3',
              'beta4', 'beta5', 'S0', 's_damage', 'p_D']

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')


def run_ef(overrides):
    """单次运行返回断裂应变（v9 协议，S0 权威值 + 文献 g_s）。"""
    base = dict(GS_LIT)
    base['S0'] = S0
    base.update(overrides)
    max_strain = EXPERIMENTAL[MAT]['ef'][ORIENT] * 2.0 + 0.01
    model = TaylorCPCDM(
        MAT, ORIENT, S0_override=S0, n_grains=N_GRAINS,
        eps_step=EPS_STEP, n_sub=N_SUB, strain_rate=RATE,
        model_version='v4', force_homogeneous=False,
        param_overrides=base, seed=42,
    )
    r = model.run_uniaxial(max_strain)
    return float(r['fracture_strain'])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()

    base_ef = run_ef({})
    print(f"base ef = {base_ef:.5f}")

    sensitivity = {'base_ef': base_ef, 'material': MAT,
                   'orientation': ORIENT, 'S0': S0,
                   'protocol': 'v9 leakage-free',
                   'parameters': {}}

    for param in PARAM_LIST:
        # 获取标称值（从默认 model.params）
        m0 = TaylorCPCDM(
            MAT, ORIENT, S0_override=S0, n_grains=N_GRAINS,
            eps_step=EPS_STEP, n_sub=N_SUB, strain_rate=RATE,
            model_version='v4', force_homogeneous=False,
            param_overrides=GS_LIT, seed=42,
        )
        original = m0.params[param]

        effects = {}
        for p in PERTURB:
            try:
                ef = run_ef({param: original * (1.0 + p)})
                effects[f"{p:+.0%}"] = ef
            except Exception:
                effects[f"{p:+.0%}"] = float('nan')

        ef_vals = list(effects.values())
        ef_clean = [v for v in ef_vals if not np.isnan(v)]
        if len(ef_clean) > 1 and base_ef > 0:
            si = float(np.std(ef_clean) / base_ef)
            mc = float(max(abs(v / base_ef - 1) for v in ef_clean) * 100.0)
        else:
            si, mc = 0.0, 0.0

        sensitivity['parameters'][param] = {
            'nominal': float(original), 'effects': effects,
            'sensitivity_index': si, 'max_change_pct': mc,
        }
        print(f"  {param:12s}: nominal={original:8.4g}  sens={si:.3f}  maxΔ={mc:.1f}%",
              flush=True)

    # 保存 JSON
    out_path = os.path.join(OUT_DIR, 'sensitivity_v9.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(sensitivity, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")

    # 绘图（覆盖 Fig_sensitivity_v2.png 为 v9 版）
    si_dict = sensitivity['parameters']
    params_sorted = sorted(si_dict, key=lambda p: -si_dict[p]['sensitivity_index'])
    si_vals = [si_dict[p]['sensitivity_index'] for p in params_sorted]
    mc_vals = [si_dict[p]['max_change_pct'] for p in params_sorted]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(range(len(params_sorted)), si_vals,
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(params_sorted))))
    ax1.set_xticks(range(len(params_sorted)))
    ax1.set_xticklabels(params_sorted, rotation=45, fontsize=9)
    ax1.set_ylabel('Sensitivity Index', color='#1f77b4')
    ax1.grid(True, alpha=0.3, axis='y')
    ax2 = ax1.twinx()
    ax2.plot(range(len(params_sorted)), mc_vals, 'r--o', linewidth=1.5, markersize=5,
             label='max |Δεf| [%]')
    ax2.set_ylabel('Max |Δεf| [%]', color='red')
    for bar, v in zip(bars, si_vals):
        if v > 0.001:
            ax1.text(bar.get_x() + bar.get_width() / 2., v + 0.01,
                     f'{v:.3f}', ha='center', fontsize=7)
    ax1.set_title('Parameter Sensitivity (OAT, ±40%, v9 leakage-free protocol, Ti-6Al-4V 0°)')
    ax2.legend(loc='upper right', framealpha=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'Fig_sensitivity_v2.png'), dpi=200)
    plt.close(fig)
    print(f"Wrote {os.path.join(FIG_DIR, 'Fig_sensitivity_v2.png')} "
          f"({time.time()-t0:.0f}s total)")


if __name__ == '__main__':
    main()
