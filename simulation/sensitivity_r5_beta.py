# -*- coding: utf-8 -*-
"""
第五轮④：β 系数扩展多因素敏感性（响应审稿"对 β 做全局敏感性"）
============================================================
有限设计（multi-factor factorial，非完整 Sobol——Taylor 显式求解成本限制下
的诚实近似）：
- 参数：beta3、beta4、beta5（Ti64 基值 10/0.8/0.2），每参数 3 水平（-40%/0/+40%）
  全因子 3^3 = 27 组合 + 基线 = 28 次 run_case（Ti64 0°、S0=0.32、v11 gs(psi) 生产参数）
- 另加 k_g（方向强度系数）单因素 3 水平（0.05/0.15/0.25）
- 输出：每组合 ef；主效应（边际均值变化范围）、双因素交互最大 |Δ|、一阶指数（OAT 兼容口径）

结果写入 output/sensitivity_r5_beta.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

MAT = 'Ti64'
PSI = 0
S0 = 0.32
XI0 = 3.5
K_G = 0.15
BASE = {'beta3': 10.0, 'beta4': 0.8, 'beta5': 0.2}
OUT = os.path.join(os.path.dirname(__file__), 'output')


def gs_h(psi, kg):
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(XI0, psi) / XI0   # xi_eff(psi)/XI0
    return 1.0 - kg * (1.0 - ratio)


def run(ov, kg):
    max_strain = 0.08 * 2.0 + 0.01  # Ti64 0° ef=0.08
    model = TaylorCPCDM(
        MAT, PSI, S0_override=S0, n_grains=20,
        eps_step=5e-4, n_sub=30, strain_rate=1e-3,
        model_version='v4', force_homogeneous=False,
        param_overrides={
            'gs_basal': 700e6 * gs_h(PSI, kg),
            'gs_prism': 720e6 * gs_h(PSI, kg),
            'gs_pyr': 800e6 * gs_h(PSI, kg),
            **ov,
        }, seed=42,
    )
    return model.run_uniaxial(max_strain)['fracture_strain']


def main():
    levels = {'beta3': [6.0, 10.0, 14.0], 'beta4': [0.48, 0.8, 1.12], 'beta5': [0.12, 0.2, 0.28]}
    combos = [(b3, b4, b5) for b3 in levels['beta3']
              for b4 in levels['beta4'] for b5 in levels['beta5']]
    print(f"beta factorial: {len(combos)} runs + baseline + k_g sweep", flush=True)

    t0 = time.time()
    ef_base = run({}, K_G)
    print(f"baseline ef={ef_base:.5f} ({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for i, (b3, b4, b5) in enumerate(combos):
        ef = run({'beta3': b3, 'beta4': b4, 'beta5': b5}, K_G)
        rows.append({'beta3': b3, 'beta4': b4, 'beta5': b5, 'ef': float(ef)})
        if (i + 1) % 9 == 0:
            print(f"  {i+1}/{len(combos)} done ({time.time()-t0:.0f}s)", flush=True)

    kg_rows = []
    for kg in (0.05, 0.15, 0.25):
        ef = run({}, kg)
        kg_rows.append({'k_g': kg, 'ef': float(ef)})

    # ---- 主效应（边际均值）：参数在其 3 水平下 ef 的均值，报告 (max-min)/base ----
    main_effects = {}
    for p in ('beta3', 'beta4', 'beta5'):
        means = {}
        for lv in levels[p]:
            vals = [r['ef'] for r in rows if abs(r[p] - lv) < 1e-9]
            means[lv] = float(np.mean(vals))
        span = max(means.values()) - min(means.values())
        main_effects[p] = {
            'marginal_means': means,
            'span_abs': span,
            'span_rel_pct': span / ef_base * 100.0,
        }

    # ---- 双因素交互：固定参数 p 时，q 的边际均值范围随 p 水平变化的最大漂移 ----
    interactions = {}
    for p, q in (('beta3', 'beta4'), ('beta3', 'beta5'), ('beta4', 'beta5')):
        drift = 0.0
        for lv_p in levels[p]:
            sub = [r for r in rows if abs(r[p] - lv_p) < 1e-9]
            means_q = {}
            for lv_q in levels[q]:
                vals = [r['ef'] for r in sub if abs(r[q] - lv_q) < 1e-9]
                means_q[lv_q] = float(np.mean(vals))
            span_q = max(means_q.values()) - min(means_q.values())
            drift = max(drift, span_q)
        interactions[f'{p}x{q}'] = {'max_span_abs': drift, 'max_span_rel_pct': drift / ef_base * 100.0}

    kg_eff = {}
    if kg_rows:
        kg_vals = [r['ef'] for r in kg_rows]
        kg_eff = {'k_g_range_rel_pct': (max(kg_vals) - min(kg_vals)) / ef_base * 100.0,
                  'rows': kg_rows}

    result = {
        'material': MAT, 'psi': PSI, 'S0': S0, 'k_g_default': K_G,
        'baseline_ef': float(ef_base),
        'factorial': {'params': ['beta3', 'beta4', 'beta5'], 'levels': levels, 'rows': rows},
        'main_effects': main_effects,
        'interactions': interactions,
        'k_g_sweep': kg_eff,
        'note': 'Finite 3-level full-factorial design (27 runs) + baseline + k_g 1-D sweep; '
                'approximation to a global sensitivity analysis under explicit-Taylor cost constraints.',
    }
    with open(os.path.join(OUT, 'sensitivity_r5_beta.json'), 'w') as f:
        json.dump(result, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in result.items() if k in ('baseline_ef', 'main_effects', 'interactions', 'k_g_sweep')},
                     indent=1, default=float))
    print(f"DONE {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
