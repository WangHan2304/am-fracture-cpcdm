# -*- coding: utf-8 -*-
"""Dc 敏感性扫描（R6③）：临界损伤 Dc ±40% 3 水平，Ti64 0°，S0=0.32"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), 'output')
MAT = 'Ti64'
PSI = 0
S0 = 0.32
BASE_DC = 0.42


def run(dc):
    model = TaylorCPCDM(
        MAT, PSI, S0_override=S0, n_grains=20,
        eps_step=5e-4, n_sub=30, strain_rate=1e-3,
        model_version='v4', force_homogeneous=False,
        param_overrides={'Dc': dc}, seed=42,
    )
    return float(model.run_uniaxial(0.17)['fracture_strain'])


def main():
    rows = []
    for dc in (BASE_DC * 0.6, BASE_DC, BASE_DC * 1.4):
        ef = run(dc)
        rows.append({'Dc': dc, 'ef': ef})
        print(f"Dc={dc:.3f}  ef={ef:.5f}", flush=True)
    base = rows[1]['ef']
    res = {
        'material': MAT, 'psi': PSI, 'S0': S0, 'Dc_base': BASE_DC,
        'rows': rows,
        'span_rel_pct': (max(r['ef'] for r in rows) - min(r['ef'] for r in rows)) / base * 100.0,
        'note': 'Critical damage Dc sensitivity, +/-40% levels (0.25/0.42/0.59).',
    }
    with open(os.path.join(OUT, 'sensitivity_Dc.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f"span_rel={res['span_rel_pct']:.2f}%  DONE saved sensitivity_Dc.json")


if __name__ == '__main__':
    main()
