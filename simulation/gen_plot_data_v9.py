"""
v9 图件数据生成 — gen_plot_data_v9.py
=====================================
用 v9 纯投影律 S0（MODEL：Ti64=0.32, 316L=3.2, AlSi=0.2，文献 gs，无 a45/a90）
重跑 9 个标定 case，输出 plot_data_v9.json：
每 case: strain / stress / damage / ef_pred / ef_exp / uts_pred / uts_exp
供 Fig7（应力应变）、Fig8（损伤）、Fig9（误差对比）重绘。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

N_GRAINS = 20
STRAIN_RATE = 1e-3
EPS_STEP_DEFAULT = 5e-4
EPS_STEP_316L = 1e-3
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]
S0_MODEL = {'Ti64': 0.32, '316L': 3.2, 'AlSi10Mg': 0.2}
OUT = os.path.join(os.path.dirname(__file__), 'output')


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def _gs_literature(mat):
    if mat == 'Ti64':
        return {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}


def run_case(mat, psi, S0):
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    model = TaylorCPCDM(
        mat, psi, S0_override=S0, n_grains=N_GRAINS,
        eps_step=_eps_step(mat), n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=False,
        param_overrides=_gs_literature(mat), seed=42,
    )
    return model.run_uniaxial(max_strain)


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {}
    for mat in MATERIALS:
        data[mat] = {}
        for psi in ORIENTS:
            r = run_case(mat, psi, S0_MODEL[mat])
            data[mat][f'{psi}deg'] = {
                'strain': r['strain'].tolist(),
                'stress': (r['stress'] / 1e6).tolist(),      # MPa
                'damage': r['damage'].tolist(),
                'ef_pred': float(r['fracture_strain']),
                'ef_exp': EXPERIMENTAL[mat]['ef'][psi],
                'uts_pred_MPa': float(r['uts'] / 1e6),
                'uts_exp_MPa': EXPERIMENTAL[mat]['uts'][psi],
            }
            print(f"{mat} {psi}deg: ef_pred={data[mat][f'{psi}deg']['ef_pred']:.4f} "
                  f"exp={EXPERIMENTAL[mat]['ef'][psi]:.4f} | "
                  f"UTS pred={data[mat][f'{psi}deg']['uts_pred_MPa']:.1f} "
                  f"exp={EXPERIMENTAL[mat]['uts'][psi]:.1f}", flush=True)
    with open(os.path.join(OUT, 'plot_data_v9.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1)
    print("saved plot_data_v9.json")


if __name__ == '__main__':
    main()
