"""
正式模型（v6）正典校准 — experiment_v6_canonical.py
====================================================
响应审稿 P0-5（消除 λ,ξ 重复计数）与 P0-2（明确校准/验证）：
- 模型：S ≡ S0（常数，损伤能量强度不再被 λ,ξ 缩放）
        f_AM(ψ) = f_φ · f_θ(θ̄(ψ)) · f_D0 · f_λ(λ̄(ψ)) · f_ξ(ξ̄(ψ)) · f_AM^aniso(ψ)
  - 前五项为无量纲物理投影（am_correction_v4，无拟合参数）
  - f_AM^aniso(ψ) 为取向校准乘数：f_AM^aniso(0°)=1，45°/90° 每材料各 1 个
    自由参数（v4 诚实协议已证明纯物理投影不足以覆盖取向各向异性，
    故保留 2 个取向乘数；每工况一个校准自由度，9 工况全部为校准案例）。
- 标定：0° → 二分 S0；45° → 二分 a45；90° → 二分 a90（各自精确匹配实验 ef）
- 输出：9 工况 ef/UTS 预测与误差、S0/a45/a90、f_AM 各因子 → experiment_v6_results.json
- 316L 用 eps_step=1e-3（与 LOOCV 协议一致，材料内标定/预测一致）
"""

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402

N_GRAINS = 20
STRAIN_RATE = 1e-3
EPS_STEP_DEFAULT = 5e-4
EPS_STEP_316L = 1e-3
MAX_BISECT = 45
A_MULT_LO, A_MULT_HI = 1.0, 8.0
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]


def _eps_step(mat):
    return EPS_STEP_316L if mat == '316L' else EPS_STEP_DEFAULT


def run_case(mat, psi, S0, a_multi):
    # 316L 单次模拟最贵：max_strain 截断到 1.6x+0.02（UTS 峰值应变已验证
    # 0.21-0.22，断裂 <0.41，截断值 0.43-0.61 均有 >7% 余量）；其余材料原样
    max_strain = (EXPERIMENTAL[mat]['ef'][psi] * 1.6 + 0.02
                  if mat == '316L' else EXPERIMENTAL[mat]['ef'][psi] * 2.0 + 0.01)
    model = TaylorCPCDM(
        mat, psi, S0_override=S0, n_grains=N_GRAINS,
        eps_step=_eps_step(mat), n_sub=30, strain_rate=STRAIN_RATE,
        model_version='v4', force_homogeneous=False,
        param_overrides={'aniso_multiplier': {0: 1.0, 45: a_multi, 90: a_multi}}
        if a_multi != 1.0 else None,
        seed=42,
    )
    return model.run_uniaxial(max_strain)


def bisect(mat, psi, target, lo, hi, key):
    """二分匹配某参数 key ∈ (lo,hi) 使 ef@psi = target。key: 'S0' 或 'a'"""
    for _ in range(MAX_BISECT):
        mid = 0.5 * (lo + hi)
        a_m = mid if key == 'a' else 1.0
        s0 = mid if key == 'S0' else None
        r = run_case(mat, psi, s0 if s0 is not None else S0_CUR[mat], a_m)
        ef = r['fracture_strain']
        # 单调性：S0↑→ef↑；a↑→f_AM↑→ef↓（方向相反）
        if key == 'S0':
            if ef < target:
                lo = mid
            else:
                hi = mid
        else:
            if ef < target:
                hi = mid      # 预测偏低 → 减小 a
            else:
                lo = mid      # 预测偏高 → 增大 a
        # 终止仅按相对区间宽度：绝对 ef 容差在区间尚宽时会提前误判，
        # 返回未收敛的中点（实测造成 3% 标定误差）
        if hi - lo < 5e-4 * max(1.0, mid):
            break
    return 0.5 * (lo + hi)


# 每材料 0° 标定的 S0（先 S0_CUR 占位，依次标定）
S0_CUR = {}


def calibrate_material(mat):
    """0° 二分 S0；45°/90° 二分 a。返回 (S0, a45, a90)。"""
    t0 = time.time()
    S0 = bisect(mat, 0, EXPERIMENTAL[mat]['ef'][0], 0.02, 400.0, 'S0')
    S0_CUR[mat] = S0
    a45 = bisect(mat, 45, EXPERIMENTAL[mat]['ef'][45], A_MULT_LO, A_MULT_HI, 'a')
    a90 = bisect(mat, 90, EXPERIMENTAL[mat]['ef'][90], A_MULT_LO, A_MULT_HI, 'a')
    print(f"[calibrate] {mat}: S0={S0:.4f}  a45={a45:.4f}  a90={a90:.4f}  "
          f"({time.time()-t0:.0f}s)")
    return S0, a45, a90


def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 88)
    print("CANONICAL v6 CALIBRATION (9 cases; S0 + 2 orientation multipliers)")
    print("=" * 88)

    cases = []
    for m in MATERIALS:
        S0, a45, a90 = calibrate_material(m)
        am = {0: 1.0, 45: a45, 90: a90}
        for psi in ORIENTS:
            r = run_case(m, psi, S0, am[psi])
            ef_exp = EXPERIMENTAL[m]['ef'][psi]
            uts_exp = EXPERIMENTAL[m]['uts'][psi]
            ef_err = abs(r['fracture_strain'] - ef_exp) / ef_exp * 100.0
            uts_err = abs(r['uts'] / 1e6 - uts_exp) / uts_exp * 100.0
            case = {
                'material': m, 'orientation': psi,
                'S0': float(S0), 'a_aniso': float(am[psi]),
                'f_AM': float(r['f_AM']),
                'ef_exp': float(ef_exp), 'ef_pred': float(r['fracture_strain']),
                'error_pct': float(ef_err),
                'uts_exp_MPa': float(uts_exp), 'uts_pred_MPa': float(r['uts'] / 1e6),
                'uts_error_pct': float(uts_err),
            }
            cases.append(case)
            print(f"  {m:9s} {psi:>3d}° | S0={S0:7.4f} a={am[psi]:6.3f} "
                  f"f_AM={r['f_AM']:6.3f} | ef {ef_exp:.4f}->{r['fracture_strain']:.4f} "
                  f"err={ef_err:5.2f}% | UTS {uts_exp:.0f}->{r['uts']/1e6:.0f} "
                  f"err={uts_err:5.2f}%")

    errs = [c['error_pct'] for c in cases]
    uts = [c['uts_error_pct'] for c in cases]
    avg_ef = float(np.mean(errs))
    avg_uts = float(np.mean(uts))
    print("\n" + "=" * 88)
    print(f"SUMMARY: avg ef error = {avg_ef:.2f}%  |  avg UTS error = {avg_uts:.2f}%"
          f"  |  pass <10%: {int(np.sum(np.array(errs) < 10.0))}/9")
    print("ef errors per case:", [round(e, 2) for e in errs])
    print("=" * 88)

    payload = {
        'model': 'v6_canonical',
        'model_form': 'S=S0 constant; f_AM = f_phi*f_theta*f_D0*f_lambda(psi)'
                      '*f_xi(psi)*f_AM_aniso(psi); a(0)=1, a(45), a(90) calibrated',
        'eps_step': {'Ti64': EPS_STEP_DEFAULT, '316L': EPS_STEP_316L,
                     'AlSi10Mg': EPS_STEP_DEFAULT},
        'calibration': 'each case calibrated (S0 on 0deg, multipliers on 45/90deg); '
                       'all 9 are calibration cases; validation = LOOCV (v5)',
        'avg_error_pct': avg_ef, 'avg_uts_error_pct': avg_uts,
        'pass_count': int(np.sum(np.array(errs) < 10.0)),
        'cases': cases,
    }
    path = os.path.join(out_dir, 'experiment_v6_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
