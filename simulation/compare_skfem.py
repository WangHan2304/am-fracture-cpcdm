# -*- coding: utf-8 -*-
"""compare_skfem.py — 把 scikit-fem 外部全场、自研 in-house 全场、Taylor 三者并列，
并对 Taylor 采用足够大的 max_strain（0.5）避免截断，产出可写进论文的一致对比表。"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, 'src'))
OUT = os.path.join(HERE, 'output')
from materials import EXPERIMENTAL            # noqa
from taylor_cpcdm import TaylorCPCDM          # noqa
from am_correction_v4 import xi_eff            # noqa

KG, XI0 = 0.15, {'Ti64': 3.5, '316L': 2.5, 'AlSi10Mg': 1.5}


def _gs_directional(mat, psi):
    xi0 = XI0[mat]
    s = np.sin(np.radians(psi)) ** 2
    c = np.cos(np.radians(psi)) ** 2
    ratio = xi_eff(xi0, psi) / xi0   # xi_eff(psi)/xi0 (am_correction_v4)
    h = 1.0 - KG * (1.0 - ratio)
    if mat == 'Ti64':
        base = {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}
    else:
        base = {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}
    return {k: v * h for k, v in base.items()}

NGRAIN, EPS_STEP, N_SUB, STRAIN_RATE, SEED = 16, 2e-3, 30, 1e-3, 42
S0_CAL = {'Ti64': 0.32, 'AlSi10Mg': 0.20}

sk = json.load(open(os.path.join(OUT, 'fullfield_skfem_results.json'), encoding='utf-8'))
ih = json.load(open(os.path.join(OUT, 'fullfield_2d_cpfem_results.json'), encoding='utf-8'))
ihmap = {(c['material'], c['orientation']): c for c in ih['cases']}

rows = []
for c in sk['cases']:
    mat, psi = c['material'], c['orientation']
    # 非截断 Taylor（max_strain=0.5 足够大，Taylor 自己按 Dc 判停）
    t = TaylorCPCDM(mat, psi, S0_override=S0_CAL[mat], n_grains=NGRAIN,
                    eps_step=EPS_STEP, n_sub=N_SUB, strain_rate=STRAIN_RATE,
                    model_version='v4', param_overrides=_gs_directional(mat, psi),
                    seed=SEED)
    tr = t.run_uniaxial(0.5)
    ef_tay = float(tr['fracture_strain'])
    ef_sk = c['ef_skfem']
    ef_ih = ihmap[(mat, psi)]['ef_fullfield']
    bias_sk = (ef_sk - ef_tay) / ef_tay * 100.0
    bias_ih = (ef_ih - ef_tay) / ef_tay * 100.0
    rows.append({'material': mat, 'orientation': psi, 'criterion_sk': c['fracture_type'],
                 'ef_taylor': round(ef_tay, 5), 'ef_inhouse': ef_ih, 'ef_skfem': round(ef_sk, 5),
                 'bias_inhouse_pct': round(bias_ih, 2), 'bias_skfem_pct': round(bias_sk, 2),
                 'skfem_vs_inhouse_pct': round((ef_sk - ef_ih) / ef_ih * 100.0, 2)})
    print(f"{mat:9s} {psi:2d}  tayl={ef_tay:.4f}  inhouse={ef_ih:.4f}({bias_ih:+.1f}%)  "
          f"skfem={ef_sk:.4f}({bias_sk:+.1f}%)  skfem_vs_inhouse={(ef_sk-ef_ih)/ef_ih*100:+.1f}%")

ti = [r['bias_skfem_pct'] for r in rows if r['material'] == 'Ti64']
print("Ti64 mean bias skfem=%.1f%%  inhouse mean=%.1f%%" % (
    float(np.mean(ti)), float(np.mean([r['bias_inhouse_pct'] for r in rows if r['material'] == 'Ti64']))))
with open(os.path.join(OUT, 'skfem_vs_inhouse_comparison.json'), 'w', encoding='utf-8') as f:
    json.dump({'note': 'Taylor ef computed with generous max_strain=0.5 (un-truncated); '
                       'in-house = fullfield_2d_cpfem; skfem = fullfield_skfem (scikit-fem 12)',
               'rows': rows}, f, indent=1)
print("WROTE skfem_vs_inhouse_comparison.json")
