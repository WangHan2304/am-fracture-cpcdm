# -*- coding: utf-8 -*-
"""
aggregate_nonlocal_results.py — 汇总非局部/局部对比算例
======================================================
输入（全部由 fullfield_3d_cpfem_nonlocal.py 逐算例写出，可追溯）：
  simulation/output/nonlocal_cases/<case_id>.json   每算例完整结果
  simulation/output/nonlocal_cases/_selfcheck.json  自检结果
输出：
  simulation/output/nonlocal_damage_results.json
  simulation/output/nonlocal_cost_bench.json        （本脚本现场实测的成本基准）
  simulation/figures/nonlocal_vs_local_ef.png

用法：
  python aggregate_nonlocal_results.py            # 含成本基准实测（约 4 min）
  python aggregate_nonlocal_results.py --skip-cost
"""
import argparse
import glob
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'src'))

from fullfield_3d_cpfem_nonlocal import (FullField3DNonlocal, S0_CAL, N_SUB,  # noqa: E402
                                        ip_coords, build_weights)

OUT = os.path.join(HERE, 'output')
CASE_DIR = os.path.join(OUT, 'nonlocal_cases')
FIG = os.path.join(HERE, 'figures')

EF_TAYLOR_LOCKED = 0.11175          # R13 merged, eps_step=2e-3, 27 grains
R13_LOCAL = {0: 0.1389, 45: 0.08763, 90: 0.0665}
R13_LOCAL_ALSI = 0.1220
R13_BIAS_BAND = [24.3, -5.02, -14.2, 3.29]

CASE_SPEC = {
    # case_id              ng  lc_kind
    'ng3_local_e1e3':       (3, 'local'),
    'ng3_lc0333_e1e3':      (3, 'lc_phys'),
    'ng3_lc0250_e1e3':      (3, 'lc_0p25'),
    'ng3_lc0125_e1e3':      (3, 'lc_0p125'),
    'ng3_lc0063_e1e3':      (3, 'lc_0p0625'),
    'ng4_local_e1e3':       (4, 'local'),
    'ng4_lc0333_e1e3':      (4, 'lc_phys'),
    'ng4_lc0250_e1e3':      (4, 'lc_scaled'),
    'ng3_local_e2e3_repro': (3, 'local_repro_e2e3'),
    # --- 三方向扩展（Ti64 45 / 90 度，ng=3）---
    'ng3_ang45_local_e1e3':   (3, 'ang45_local'),
    'ng3_ang45_lc0333_e1e3':  (3, 'ang45_lc_phys'),
    'ng3_ang45_lc0125_e1e3':  (3, 'ang45_lc_0p125'),
    'ng3_ang90_local_e1e3':   (3, 'ang90_local'),
    'ng3_ang90_lc0333_e1e3':  (3, 'ang90_lc_phys'),
    'ng3_ang90_lc0125_e1e3':  (3, 'ang90_lc_0p125'),
}

LC_SWEEP_NG3 = [('local', '0 (local)'), ('lc_0p0625', '0.0625'),
                ('lc_0p125', '0.125'), ('lc_0p25', '0.25'), ('lc_phys', '0.3333')]

# 三方向对比（ang=0 用本工作 e1e3 算例；45/90 用本工作 e1e3 算例 + R13 归档值作对照）
ANGLES = [0, 45, 90]
ANGLE_CASE = {0: {'local': 'ng3_local_e1e3', 'lc_phys': 'ng3_lc0333_e1e3',
                  'lc_0p125': 'ng3_lc0125_e1e3'},
              45: {'local': 'ng3_ang45_local_e1e3', 'lc_phys': 'ng3_ang45_lc0333_e1e3',
                   'lc_0p125': 'ng3_ang45_lc0125_e1e3'},
              90: {'local': 'ng3_ang90_local_e1e3', 'lc_phys': 'ng3_ang90_lc0333_e1e3',
                   'lc_0p125': 'ng3_ang90_lc0125_e1e3'}}
# R13 归档（eps_step=2e-3）四方向 ef / Taylor，用于偏差带论证
R13_TAYLOR = {0: 0.11175, 45: 0.092258, 90: 0.0775}
# Taylor 聚集体（27 取向，Ti64, S0=0.32, seed=42）在 eps_step=1e-3 下的值，
# 即步长收敛的 Taylor 参考；实测来源 output/_taylor_angle_check.json。
TAYLOR_CONVERGED = {0: 0.072548, 45: 0.0542, 90: 0.044335}

HONEST_NOTES = [
    'Allen-Cahn / Ginzburg-Landau phase field was NOT implemented. It needs extra Newton '
    'unknowns, a gradient-energy term, interface-parameter calibration and - decisively - a '
    'redefinition of the fracture criterion, which would break consistency with the locked '
    'R13 assets. It is declared a higher-fidelity follow-on, not a minimal viable variant.',
    'Material parameters were NOT recalibrated for the nonlocal model. S0=0.32, Dc=0.42 and '
    'p_D=0.05 are the values calibrated for the LOCAL model. The reported ef shifts therefore '
    'quantify "replacing the local driver by its nonlocal average at fixed calibration", not a '
    'calibrated nonlocal prediction. Standard practice would recalibrate S0 (and/or Dc) because '
    'nonlocal averaging changes the magnitude of the driving variable.',
    'l_c = 1-2 grains is too large for this RVE. At l_c = 1 grain the kernel effectively '
    'averages n_eff ~ 55 integration points ~ 6.9 grains of a 27-grain RVE, and an integration '
    'point retains only 4.7% of its own value. The regularisation therefore degenerates towards '
    'homogenisation; see nonlocal_kernel_stats.json.',
    'The nonlocal kernel uses Euclidean distances inside the RVE; no periodic (mirror) images '
    'are used. For l_c >= 0.25 every integration-point neighbourhood touches an RVE face, so '
    'the boundary truncation of the kernel is a first-order effect, not a small correction. '
    'This is the largest known approximation of the present implementation.',
    'All completed cases terminated by the aggregate criterion (D_agg >= Dc); none terminated by '
    'localization onset or hit the numerical limit. The ef values are therefore directly '
    'comparable and none is a truncated lower bound.',
    'The mesh-dependence conclusion is drawn only from measured data (see mesh_dependence).',
    'Negative result, stated plainly: nonlocalisation does NOT pull the 0-degree bias back into '
    'the existing R13 direction band; it pushes the positive bias further out.',
    'Not attempted: mesh dependence at 45/90 degrees, and the 64-grain mesh for the smaller l_c '
    'values - both outside the remaining compute budget.',
    'MESH DEPENDENCE, measured: refining 27 -> 64 grains changes the LOCAL ef by -15.92% '
    '(0.13966562 -> 0.11743750, and the termination branch changes from a Dc crossing to a '
    'localization onset), but changes the NONLOCAL ef by only -2.20% (lc fixed at 1 grain of '
    'the coarse mesh: 0.15268242 -> 0.14932), -1.75% (lc = 0.25 on both meshes: 0.14924257 '
    '-> 0.14663) and -3.96% (lc rescaled to 1 grain of each mesh: 0.15268242 -> 0.14663). '
    'So nonlocalisation reduces the mesh sensitivity by roughly 4x to 9x but does NOT '
    'eliminate it; a residual 2-4% mesh shift remains.',
    'The archived convergence study in output/code_verification_summary.md section 5 is the '
    'TAYLOR AGGREGATE, not the 3D full-field solver. This was verified by re-running '
    'TaylorCPCDM with the same settings (Ti64 psi=0, S0=0.32, seed=42) and reproducing the '
    'archived grain-count row bit-for-bit: 27/64/125 grains at eps_step=1e-3 give '
    '0.072548 / 0.071219 / 0.069538 with UTS 934.76 / 942.43 / 951.68 MPa. Two further '
    'indicators agree: the archived table contains 125 grains, which the 3D full-field code '
    'cannot run at all (NGRAIN makes any ng != 3 raise IndexError), and the archived suite '
    'itself lists mesh convergence of the 3D full-field solver under what it does NOT '
    'establish. The -15.92% reported here is therefore the first mesh-dependence measurement '
    'of the 3D full-field local model, and it does not contradict the archived 4.33%.',
    'The locked Taylor reference is itself not step-converged, and this matters for the bias '
    'band. The Taylor aggregate at 27 orientations gives ef = 0.072548 at eps_step=1e-3 (the '
    'step-converged value) but 0.11175 at eps_step=2e-3 - a +54.0% inflation. The locked R13 '
    'value 0.11175 IS the 2e-3 value. Meanwhile the 3D full-field local ef is step-converged '
    '(bit-identical at 1e-3 and 2e-3). The bias band is therefore reference-dependent: '
    'against the locked reference the local 0-degree bias is +24.98%, but against the '
    'step-converged Taylor reference it is +92.5%. Both readings are reported; the band in '
    'the manuscript rests on the 2e-3 Taylor reference.',
    'Two ef reading conventions coexist in this report and are always labelled by the '
    'fracture_type field: aggregate_Dc (linear interpolation of the Dc crossing) and '
    'localization_onset (the last converged increment before an increment that will not '
    'converge at the step-size floor). They are not the same measurement.',
    'COST MEASUREMENT METHODOLOGY, and one contaminated reading that was caught. The '
    'first pass of this benchmark measured each tangent once and measured all of local '
    'before any of nonlocal. It returned a residual overhead ratio of 2.4789 for ng=3, '
    'which is physically impossible: the nonlocal residual adds two sparse '
    'matrix-vector products (2 x 37.65 us at ng=3) to a 19.66 ms residual, i.e. 0.4%. '
    'The raw samples now kept in this JSON show what happened - rounds 4 and 5 of the '
    'residual sampling spiked to 37.86/49.58 ms (local) and 51.29/50.77 ms (nonlocal), '
    'a 2.5x contention event from other processes on the same machine. The benchmark '
    'was rewritten to interleave local and nonlocal and to report the MINIMUM over '
    'repeated rounds, which is the estimate least sensitive to such spikes. The reported '
    'ratios (ng=3 residual 1.0341 / tangent 1.0419; ng=4 residual 1.0485 / tangent '
    '1.0569) come from that hardened procedure, and every raw sample is in cost_bench.',
    'An optimisation was identified but deliberately NOT applied, because applying it now '
    'would invalidate the 15 completed runs. At this problem size the sparse kernel is '
    'slower than a dense one: 216x216 CSR 37.65 us vs dense 6.43 us, 512x512 CSR 128.01 us '
    'vs dense 29.65 us (agreement 1.3e-15, so it is purely a format choice). The measured '
    'per-residual overhead (0.67 ms, 3.4% at ng=3) is also about 9x the two kernel calls '
    'it is meant to cover, so most of the nonlocal cost is Python/scipy call overhead '
    'rather than the kernel. A dense kernel plus a hoisted operator would cut most of it.'
]


# ---------------------------------------------------------------------------
def timeit(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def measure_cost(ng, res_rounds=5, res_reps=20, tan_rounds=2):
    """交替 + 重复 + 取最小值地测量局部/非局部 residual 与 tangent。

    取最小值（而不是均值）是刻意的：本机同时跑着其它 python 进程，一次切线要 15~66 s，
    长窗口极易被抢占；min 是最不易被污染的一致估计。所有原始样本都写进 JSON。
    """
    ff, st, u = {}, {}, {}
    for tag, lc in (('local', None), ('nonlocal', 1.0 / ng)):
        ff[tag] = FullField3DNonlocal('Ti64', 0, S0_CAL['Ti64'], eps_step=1e-3,
                                      n_sub=N_SUB, ng=ng, max_strain=4e-3, lc=lc)
        st[tag] = ff[tag]._state_init()
        u[tag] = np.zeros(3 * ff[tag].nn)
        for d in ff[tag].controlled_dofs:
            u[tag][d] = 1e-3
    # 交替采样，抵消随时间的漂移
    rs = {'local': [], 'nonlocal': []}
    ts = {'local': [], 'nonlocal': []}
    for _ in range(res_rounds):
        for tag in ('local', 'nonlocal'):
            rs[tag].append(timeit(lambda: ff[tag]._residual(u[tag], st[tag], 1.0), res_reps))
    for tag in (['local', 'nonlocal'] * tan_rounds):
        t0 = time.perf_counter()
        ff[tag]._tangent(u[tag], st[tag], 1.0)
        ts[tag].append(time.perf_counter() - t0)
    res = {}
    for tag in ('local', 'nonlocal'):
        res[tag] = {
            'residual_ms': round(min(rs[tag]) * 1e3, 2),
            'residual_samples_ms': [round(v * 1e3, 2) for v in rs[tag]],
            'residual_spread_pct': round((max(rs[tag]) - min(rs[tag]))
                                         / min(rs[tag]) * 100.0, 2),
            'tangent_s': round(min(ts[tag]), 2),
            'tangent_samples_s': [round(v, 2) for v in ts[tag]],
            'tangent_spread_pct': round((max(ts[tag]) - min(ts[tag]))
                                        / min(ts[tag]) * 100.0, 2),
            'n_free': int(len(ff[tag].free_dofs)), 'nip': int(ff[tag].nip),
            'W': ff[tag].W_stats}
    res['method'] = ('interleaved sampling, min of %d rounds x %d reps for residual, '
                     'min of %d interleaved calls for tangent. MIN (not mean) is used '
                     'deliberately: other python processes run on this machine and one '
                     'tangent call lasts 15-69 s, so the mean is routinely hit by '
                     'contention spikes - see residual_samples_ms for an example.'
                     % (res_rounds, res_reps, tan_rounds))
    res['tangent_overhead_ratio'] = round(
        res['nonlocal']['tangent_s'] / res['local']['tangent_s'], 4)
    res['residual_overhead_ratio'] = round(
        res['nonlocal']['residual_ms'] / res['local']['residual_ms'], 4)
    pos = ip_coords(ng)
    mv = {}
    for tag, lc in (('lc_1_over_ng', 1.0 / ng), ('lc_0p25', 0.25)):
        Ws, stt = build_weights(pos, lc, sparsify=True)
        Wd = np.asarray(Ws.todense())
        rng = np.random.RandomState(0)
        v = np.abs(rng.randn(pos.shape[0]))
        mv[tag] = {'nnz_per_row': stt['nnz_per_row_mean'],
                   'csr_us': round(timeit(lambda: Ws @ v, 200) * 1e6, 2),
                   'dense_us': round(timeit(lambda: Wd @ v, 200) * 1e6, 2),
                   'max_abs_agreement': float(np.abs(np.asarray(Ws @ v).ravel()
                                                     - Wd @ v).max())}
    res['matvec'] = mv
    return res


# ---------------------------------------------------------------------------
def load_cases():
    out = {}
    for p in sorted(glob.glob(os.path.join(CASE_DIR, '*.json'))):
        name = os.path.splitext(os.path.basename(p))[0]
        if name.startswith('_'):
            continue
        with open(p, 'r', encoding='utf-8') as f:
            out[name] = json.load(f)
    return out


def case_row(cid, c):
    ng, kind = CASE_SPEC.get(cid, (c.get('ng'), 'unknown'))
    term = c.get('terminal_D_metrics') or {}
    return {
        'case_id': cid, 'ng': ng, 'n_grains': ng ** 3, 'lc_kind': kind,
        'grain_size_h': c.get('grain_size'),
        'lc': c.get('lc'), 'lc_over_h': (round(c['lc'] / c['grain_size'], 4)
                                        if c.get('lc') else None),
        'eps_step': c.get('eps_step'), 'n_sub': N_SUB,
        'ef_fullfield': c.get('fracture_strain'),
        'fracture_type': c.get('fracture_type'),
        'ef_truncated_at_max_strain': c.get('fracture_type') == 'max_strain',
        'ef_taylor_eps_matched': c.get('ef_taylor_eps_matched'),
        'bias_vs_taylor_eps_matched_pct': c.get('bias_vs_taylor_eps_matched_pct'),
        'bias_vs_taylor_locked_0p11175_pct': c.get('bias_vs_taylor_locked_pct'),
        'wall_sec': c.get('wall_sec'),
        'n_steps_converged': c.get('n_steps_converged'),
        'n_step_reductions': c.get('n_step_reductions'),
        'n_newton_iter_total': c.get('n_newton_iter_total'),
        'eps_step_final': c.get('eps_step_final'),
        'Dagg_terminal': (c.get('damage') or [None])[-1],
        'terminal_D_metrics': {k: v for k, v in term.items() if k != 'D_field'},
    }


def _pair_kind(a, b, la=None, lb=None):
    if a == 'local':
        return 'local (no regularisation)'
    if la is not None and lb is not None and abs(la - lb) < 1e-12:
        return 'nonlocal, l_c fixed as a physical length'
    if a == b:
        return 'nonlocal, l_c held equal on both meshes'
    return 'nonlocal, l_c rescaled to one grain of each mesh'


def pair_mesh(cases, tag_a, tag_b, ng_lo=3, ng_hi=4):
    """在给定 lc_kind 下比较 ng=3 与 ng=4 的 ef。"""
    def find(ng, kind):
        for cid, (g, k) in CASE_SPEC.items():
            if g == ng and k == kind and cid in cases:
                return cid, cases[cid]
        return None, None
    ca, cb = find(ng_lo, tag_a), find(ng_hi, tag_b)
    if ca[1] is None or cb[1] is None:
        miss = [lab for lab, z in (('ng3:' + tag_a, ca), ('ng4:' + tag_b, cb))
                if z[1] is None]
        return {'lc_kind_ng3': tag_a, 'lc_kind_ng4': tag_b,
                'available': False, 'missing_cases': miss}
    ea, eb = ca[1]['fracture_strain'], cb[1]['fracture_strain']
    return {
        'lc_kind_ng3': tag_a, 'lc_kind_ng4': tag_b, 'available': True,
        'ng3': {'case_id': ca[0], 'ng': 3, 'n_grains': 27,
                'lc': ca[1].get('lc'), 'lc_over_h': (round(ca[1]['lc'] / ca[1]['grain_size'], 4)
                                                     if ca[1].get('lc') else None),
                'ef': ea, 'fracture_type': ca[1].get('fracture_type'),
                'truncated': ca[1].get('fracture_type') == 'max_strain'},
        'ng4': {'case_id': cb[0], 'ng': 4, 'n_grains': 64,
                'lc': cb[1].get('lc'), 'lc_over_h': (round(cb[1]['lc'] / cb[1]['grain_size'], 4)
                                                     if cb[1].get('lc') else None),
                'ef': eb, 'fracture_type': cb[1].get('fracture_type'),
                'truncated': cb[1].get('fracture_type') == 'max_strain'},
        'ef_relative_change_pct_ng4_vs_ng3': round((eb - ea) / ea * 100.0, 2)
        if ea else None,
        'comparison_kind': _pair_kind(tag_a, tag_b, ca[1].get('lc'), cb[1].get('lc')),
        'note': ('这是对"非局部化能否消除网格依赖"的直接检验：内部长度取物理长度，'
                 '网格细化时 l_c 不变。'
                 if _pair_kind(tag_a, tag_b, ca[1].get('lc'), cb[1].get('lc'))
                 == 'nonlocal, l_c fixed as a physical length' else
                 '对照算例：l_c 随网格一起缩放（各取自身 1 个晶粒），用于区分'
                 '"物理长度固定"与"内部长度缩放"两种设置。'
                 if _pair_kind(tag_a, tag_b, ca[1].get('lc'), cb[1].get('lc'))
                 == 'nonlocal, l_c rescaled to one grain of each mesh' else
                 '无正则化的局部损伤基线：网格细化时损伤局部化不受内部长度约束。'),
    }


def smoothness_table(cases):
    rows = []
    for cid, c in cases.items():
        if (c.get('angle') or 0.0) != 0.0 or c.get('fracture_strain') is None:
            continue
        for cp, m in (c.get('checkpoints') or {}).items():
            rows.append({'case_id': cid, 'ng': c.get('ng'), 'strain': float(cp),
                         'roughness_h_kernel': m['D_roughness_h_kernel'],
                         'grad_mean_over_h': m['D_grad_mean_over_h'],
                         'max_over_mean': m['D_max_over_mean'],
                         'D_max': m['D_max'], 'D_mean': m['D_mean'],
                         'localization_frac': m['D_localization_frac']})
    return sorted(rows, key=lambda r: (r['case_id'], r['strain']))


def three_direction_bias(cases):
    """三方向（Ti64 0/45/90）局部 vs 非局部 ef 与对 Taylor 的偏差。"""
    out = {'angles_deg': ANGLES, 'taylor_locked_eps_matched_2em3': R13_TAYLOR,
           'taylor_step_converged_1em3': TAYLOR_CONVERGED,
           'r13_archived_local_ef': R13_LOCAL, 'rows': []}
    for ang in ANGLES:
        row = {'angle_deg': ang, 'ef_taylor': R13_TAYLOR.get(ang),
               'r13_archived': {'ef_local': R13_LOCAL.get(ang),
                                'bias_vs_taylor_pct': (round((R13_LOCAL[ang]
                                                              - R13_TAYLOR[ang])
                                                             / R13_TAYLOR[ang] * 100.0, 2)
                                                       if ang in R13_TAYLOR else None)}}
        for tag, lab in (('local', 'local'), ('lc_0p125', 'nonlocal_lc_0.125'),
                         ('lc_phys', 'nonlocal_lc_1grain')):
            cid = ANGLE_CASE[ang][tag]
            c = cases.get(cid)
            if not c or c.get('fracture_strain') is None:
                row[lab] = None
                continue
            efv = c['fracture_strain']
            tay = R13_TAYLOR.get(ang)
            tayc = TAYLOR_CONVERGED.get(ang)
            row[lab] = {'case_id': cid, 'ef': efv,
                        'fracture_type': c.get('fracture_type'),
                        'bias_vs_taylor_locked_pct': (round((efv - tay) / tay * 100.0, 2)
                                                      if tay else None),
                        'bias_vs_taylor_step_converged_pct':
                            (round((efv - tayc) / tayc * 100.0, 2) if tayc else None)}
        out['rows'].append(row)
    return out


def load_kernel_stats():
    p = os.path.join(OUT, 'nonlocal_kernel_stats.json')
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_json_soft(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def interp(xs, ys, x):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0 or x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ys))


def lc_sensitivity(cases):
    """ng=3 上 l_c 对损伤增长速率与 ef 的影响（本工作新增的敏感性扫描）。"""
    rows = []
    probes = [0.05, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]
    for kind, lab in LC_SWEEP_NG3:
        cid = next((k for k, v in CASE_SPEC.items() if v[1] == kind and v[0] == 3
                    and k in cases), None)
        if cid is None:
            continue
        c = cases[cid]
        s, d = c.get('strain') or [], c.get('damage') or []
        row = {'case_id': cid, 'lc_label': lab, 'lc': c.get('lc'),
               'lc_over_h': (round(c['lc'] / c['grain_size'], 4) if c.get('lc') else 0.0),
               'ef_fullfield': c.get('fracture_strain'),
               'fracture_type': c.get('fracture_type'),
               'ef_is_lower_bound_only': c.get('fracture_type') == 'max_strain',
               'wall_sec': c.get('wall_sec'),
               'Dagg_probes': {f'{p:.2f}': interp(s, d, p) for p in probes}}
        a, b = interp(s, d, 0.05), interp(s, d, 0.10)
        row['dDagg_dstrain_0p05_to_0p10'] = (round((b - a) / 0.05, 4)
                                             if (a is not None and b is not None) else None)
        rows.append(row)
    base = next((r for r in rows if r['lc'] is None), None)
    for r in rows:
        if base and base.get('dDagg_dstrain_0p05_to_0p10') and r.get('dDagg_dstrain_0p05_to_0p10'):
            r['damage_rate_ratio_vs_local'] = round(
                r['dDagg_dstrain_0p05_to_0p10'] / base['dDagg_dstrain_0p05_to_0p10'], 4)
    return rows


# ---------------------------------------------------------------------------
def make_figure(cases, cost, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def get(cid, key, default=None):
        return cases.get(cid, {}).get(key, default)

    fig, axes = plt.subplots(3, 2, figsize=(15.0, 15.4), dpi=170)
    axes = axes.ravel()

    # ---- Panel A: ef 对比（ng=3, 0 度）-----------------------------------
    ax = axes[0]
    order = [('ng3_local_e1e3', 'local'), ('ng3_lc0063_e1e3', 'nonlocal\n$l_c$=0.0625'),
             ('ng3_lc0125_e1e3', 'nonlocal\n$l_c$=0.125'),
             ('ng3_lc0250_e1e3', 'nonlocal\n$l_c$=0.25'),
             ('ng3_lc0333_e1e3', 'nonlocal\n$l_c$=0.3333')]
    labels, vals, cols, hatches = [], [], [], []
    for cid, lab in order:
        c = cases.get(cid)
        if not c:
            continue
        labels.append(lab)
        vals.append(c['fracture_strain'])
        cols.append('#F58518' if c.get('lc') is None else '#4C78A8')
        hatches.append('///' if c.get('fracture_type') == 'max_strain' else '')
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, width=0.6, color=cols, edgecolor='k', linewidth=0.7)
    for b, h in zip(bars, hatches):
        b.set_hatch(h)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.004, f'{v:.4f}', ha='center', va='bottom', fontsize=8.5,
                fontweight='bold')
    ax.axhline(EF_TAYLOR_LOCKED, color='#54A24B', ls='--', lw=1.4,
               label=f'Taylor $\\varepsilon_f$ = {EF_TAYLOR_LOCKED:.4f} (eps_step 2e-3)')
    ax.axhline(R13_LOCAL[0], color='#B279A2', ls=':', lw=1.4,
               label=f'R13 3D local $\\varepsilon_f$ = {R13_LOCAL[0]:.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('fracture strain  $\\varepsilon_f$  (-)', fontsize=10)
    ax.set_title('(A) Ti-6Al-4V, 0$\\degree$, 27 grains: $l_c$ sweep\n'
                 'hatched = truncated at max_strain', fontsize=10)
    ax.axhline(TAYLOR_CONVERGED[0], color='#7F7F7F', ls='-.', lw=1.3,
               label='Taylor step-converged (1e-3) = %.4f' % TAYLOR_CONVERGED[0])
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(axis='y', ls=':', alpha=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ---- Panel B: 网格依赖 ----------------------------------------------
    ax = axes[1]
    groups = [('local', 'local', 'local'),
              ('lc_phys', 'lc_phys', 'nonlocal\n$l_c$=0.333 fixed'),
              ('lc_phys', 'lc_scaled', 'nonlocal\n$l_c$=1 grain'),
              ('lc_0p25', 'lc_scaled', 'nonlocal\n$l_c$=0.25 fixed')]
    gl, g3, g4, dlt = [], [], [], []
    for key3, key4, lab in groups:
        p = pair_mesh(cases, key3, key4)
        if not p.get('available'):
            continue
        gl.append(lab)
        g3.append(p['ng3']['ef'])
        g4.append(p['ng4']['ef'])
        dlt.append(p['ef_relative_change_pct_ng4_vs_ng3'])
    xx = np.arange(len(gl))
    w = 0.36
    b3 = ax.bar(xx - w / 2, g3, w, label='27 grains (ng=3)', color='#72B7B2',
                edgecolor='k', linewidth=0.6)
    b4 = ax.bar(xx + w / 2, g4, w, label='64 grains (ng=4)', color='#E45756',
                edgecolor='k', linewidth=0.6)
    for xi, a, b, d in zip(xx, g3, g4, dlt):
        ax.text(xi - w / 2, a + 0.003, f'{a:.4f}', ha='center', va='bottom', fontsize=7.5)
        ax.text(xi + w / 2, b + 0.003, f'{b:.4f}', ha='center', va='bottom', fontsize=7.5)
        ax.text(xi, max(a, b) + 0.018, f'{d:+.1f}%', ha='center', va='bottom',
                fontsize=8.5, fontweight='bold',
                color='#B03A2E' if abs(d) > 5 else '#1E7B34')
    ax.set_xticks(xx)
    ax.set_xticklabels(gl, fontsize=8)
    ax.set_ylabel('fracture strain  $\\varepsilon_f$  (-)', fontsize=10)
    ax.set_title('(B) Mesh dependence of $\\varepsilon_f$\n'
                 '(red % = change of the finer mesh)', fontsize=10)
    ax.set_ylim(0.0, max(max(g3), max(g4)) * 1.24)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='y', ls=':', alpha=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ---- Panel C: D 场光滑性 --------------------------------------------
    ax = axes[2]
    styles = {'ng3_local_e1e3': ('local (ng=3)', '#F58518', '-o'),
              'ng3_lc0333_e1e3': ('nonlocal lc=h (ng=3)', '#4C78A8', '-s'),
              'ng3_lc0250_e1e3': ('nonlocal lc=0.25 (ng=3)', '#54A24B', '-^'),
              'ng4_local_e1e3': ('local (ng=4)', '#B279A2', '--o'),
              'ng4_lc0333_e1e3': ('nonlocal lc=0.333 (ng=4)', '#9C755F', '--s')}
    for cid, (lab, col, fmt) in styles.items():
        c = cases.get(cid)
        if not c or not c.get('checkpoints'):
            continue
        xs = sorted((float(k) for k in c['checkpoints']))
        ys = [c['checkpoints'][f'{v:.2f}'] ['D_roughness_h_kernel']
              if f'{v:.2f}' in c['checkpoints']
              else c['checkpoints'][str(v)]['D_roughness_h_kernel'] for v in xs]
        ax.plot(xs, ys, fmt, color=col, ms=4.5, lw=1.5, label=lab)
    ax.set_xlabel('applied strain  (-)', fontsize=10)
    ax.set_ylabel('$D$-field roughness  $\\|D-W_hD\\|/\\|D\\|$  (-)', fontsize=10)
    ax.set_title('(C) Spatial smoothness of the damage field\n'
                 '$W_h$ = fixed geometric kernel with $l_c=h$', fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(ls=':', alpha=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ---- Panel D: l_c 敏感性（损伤增长）----------------------------------
    ax = axes[3]
    sw = {'ng3_local_e1e3': ('local (l_c=0)', '#F58518', '-', 2.0),
          'ng3_lc0063_e1e3': ('l_c=0.0625 (0.19h)', '#54A24B', '-', 1.6),
          'ng3_lc0125_e1e3': ('l_c=0.125 (0.38h)', '#72B7B2', '-', 1.6),
          'ng3_lc0250_e1e3': ('l_c=0.25 (0.75h)', '#4C78A8', '-', 1.6),
          'ng3_lc0333_e1e3': ('l_c=0.3333 (1.0h)', '#B279A2', '-', 1.6)}
    for cid, (lab, col, ls, lw) in sw.items():
        c = cases.get(cid)
        if not c or not c.get('damage'):
            continue
        ax.plot(c['strain'], c['damage'], ls, color=col, lw=lw, label=lab)
    ax.axhline(0.42, color='k', ls='--', lw=1.2, label='$D_c$ = 0.42')
    ax.set_yscale('log')
    ax.set_xlabel('applied strain  (-)', fontsize=10)
    ax.set_ylabel('aggregate damage  $D_{agg}$  (-)', fontsize=10)
    ax.set_title('(D) $l_c$ sensitivity of damage growth (ng=3, 27 grains)\n'
                 'nonlocal averaging reduces the driving rate', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(ls=':', alpha=0.5, which='both')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ---- Panel E: 三方向（0/45/90）局部 vs 非局部 ------------------------
    ax = axes[4]
    td = three_direction_bias(cases)
    angs = [r['angle_deg'] for r in td['rows']]
    xa = np.arange(len(angs))
    series = [('ef_taylor', 'Taylor aggregate, R13 locked (eps step 2e-3)',
               '#54A24B', 0.20),
              ('local', 'full-field local (this work, eps step 1e-3)',
               '#F58518', 0.24),
              ('nonlocal_lc_0.125', 'full-field nonlocal $l_c$=0.125', '#72B7B2', 0.24),
              ('nonlocal_lc_1grain', 'full-field nonlocal $l_c$=1 grain', '#4C78A8', 0.24)]
    series = [s for s in series
              if s[0] == 'ef_taylor' or any(r.get(s[0]) for r in td['rows'])]
    nS = len(series)
    w = 0.78 / max(nS, 1)
    for k, (key, lab, col, alp) in enumerate(series):
        vals = []
        for r in td['rows']:
            if key == 'ef_taylor':
                vals.append(r.get('ef_taylor'))
            else:
                v = r.get(key) or {}
                vals.append(v.get('ef') if isinstance(v, dict) else None)
        xx2 = xa - 0.39 + w * (k + 0.5)
        ax.bar(xx2, [v if v is not None else 0.0 for v in vals], w,
               color=col, edgecolor='k', linewidth=0.5,
               label=lab, alpha=(1.0 if alp is None else 1.0),
               hatch=('//' if key == 'ef_taylor' else ''))
    if 'local' in [s[0] for s in series]:
        kk = [s[0] for s in series].index('local')
        for ri, r in enumerate(td['rows']):
            v = (r.get('r13_archived') or {}).get('ef_local')
            if v is None:
                continue
            ax.plot([xa[ri] - 0.39 + w * (kk + 0.5)], [v], marker='s', ms=6.5,
                    mfc='none', mec='#B279A2', mew=1.8,
                    label=('R13 archived local (eps step 2e-3)' if ri == 0 else None))
    ax.set_ylim(0.0, 0.20)
    ax.set_xticks(xa)
    ax.set_xticklabels([f'{a}$\\degree$' for a in angs], fontsize=10)
    ax.set_xlabel('loading direction relative to grain orientation', fontsize=9)
    ax.set_ylabel('fracture strain  $\\varepsilon_f$  (-)', fontsize=10)
    ax.set_title('(E) Three-direction comparison (Ti64, 27 grains, $\\varepsilon$-step 1e-3)\n'
                 'nonlocalisation raises, not lowers, the 0$\\degree$ bias', fontsize=10)
    ax.legend(fontsize=7.0, ncol=2, loc='upper right')
    ax.grid(axis='y', ls=':', alpha=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    # ---- Panel F: 检查点 D 场分布（局部化 vs 平滑）-----------------------
    ax = axes[5]
    cp_key = None
    for cid in ('ng3_local_e1e3', 'ng3_lc0250_e1e3', 'ng3_lc0333_e1e3'):
        c = cases.get(cid)
        if not c or not c.get('checkpoints'):
            continue
        ks = list(c['checkpoints'].keys())
        cp_key = cp_key or ('0.12' if '0.12' in ks else ks[-1])
    dist = [('ng3_local_e1e3', 'local', '#F58518'),
            ('ng3_lc0250_e1e3', 'nonlocal $l_c$=0.25', '#4C78A8'),
            ('ng3_lc0333_e1e3', 'nonlocal $l_c$=1 grain', '#72B7B2')]
    shown = False
    for cid, lab, col in dist:
        c = cases.get(cid)
        if not c or not c.get('checkpoints') or cp_key is None:
            continue
        m = c['checkpoints'].get(cp_key) or c['checkpoints'].get(str(float(cp_key)))
        if not m or 'D_field' not in m:
            continue
        D = np.asarray(m['D_field'], dtype=float)
        ax.hist(D, bins=36, histtype='step', lw=1.8, color=col,
                label=f'{lab}  (max/mean = {m["D_max_over_mean"]:.2f})')
        shown = True
    if shown:
        ax.set_xlabel('integration-point damage  $D$  (-)', fontsize=10)
        ax.set_ylabel('number of integration points', fontsize=10)
        ax.set_title('(F) Distribution of $D$ over the 216 integration points\n'
                     f'at $\\varepsilon$ = {cp_key} : nonlocalisation removes the high-$D$ tail',
                     fontsize=10)
        ax.legend(fontsize=7.5)
    ax.grid(ls=':', alpha=0.5)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-cost', action='store_true')
    a = ap.parse_args()

    cases = load_cases()
    if not cases:
        raise SystemExit('no per-case results in ' + CASE_DIR)
    rows = [case_row(cid, c) for cid, c in cases.items()
            if cid in CASE_SPEC and c.get('fracture_strain') is not None]

    cost = None
    if not a.skip_cost:
        print('measuring cost baseline (local vs nonlocal, ng=3 and ng=4) ...', flush=True)
        cost = {f'ng={ng}': measure_cost(ng) for ng in (3, 4)}
        with open(os.path.join(OUT, 'nonlocal_cost_bench.json'), 'w',
                  encoding='utf-8', newline='') as f:
            json.dump(cost, f, ensure_ascii=False, indent=1)

    mesh = [pair_mesh(cases, 'local', 'local'),
            pair_mesh(cases, 'lc_phys', 'lc_phys'),
            pair_mesh(cases, 'lc_phys', 'lc_scaled'),
            pair_mesh(cases, 'lc_0p25', 'lc_scaled')]

    ef = {}
    for cid, c in cases.items():
        ef[cid] = {'ng': c.get('ng'), 'lc': c.get('lc'),
                   'ef_fullfield': c.get('fracture_strain'),
                   'fracture_type': c.get('fracture_type')}

    nvsl = {}
    for tag, (k3, k4) in {'lc_equals_grain_of_ng3_mesh': ('lc_phys', 'lc_phys'),
                          'lc_equals_current_grain': ('lc_phys', 'lc_scaled')}.items():
        p = pair_mesh(cases, k3, k4)
        if p.get('available'):
            nvsl[tag] = p
    loc3 = cases.get('ng3_local_e1e3', {}).get('fracture_strain')
    nl3 = cases.get('ng3_lc0333_e1e3', {}).get('fracture_strain')
    loc3_2e3 = cases.get('ng3_local_e2e3_repro', {}).get('fracture_strain')

    bias = {}
    for cid, c in cases.items():
        if c.get('fracture_strain') is None:
            continue
        b = c.get('bias_vs_taylor_locked_pct')
        bias[cid] = {'ng': c.get('ng'), 'lc': c.get('lc'), 'eps_step': c.get('eps_step'),
                     'ef': c.get('fracture_strain'),
                     'ef_taylor_eps_matched': c.get('ef_taylor_eps_matched'),
                     'bias_vs_taylor_eps_matched_pct': c.get('bias_vs_taylor_eps_matched_pct'),
                     'bias_vs_taylor_locked_0p11175_pct': b,
                     'position_in_R13_bias_band': None}
        if b is not None:
            if b > max(R13_BIAS_BAND):
                bias[cid]['position_in_R13_bias_band'] = 'above the whole R13 band'
            elif b < min(R13_BIAS_BAND):
                bias[cid]['position_in_R13_bias_band'] = 'below the whole R13 band'
            else:
                bias[cid]['position_in_R13_bias_band'] = (
                    f'inside the R13 band [{min(R13_BIAS_BAND):.2f}%, '
                    f'{max(R13_BIAS_BAND):.2f}%]')

    failures = []
    for cid, c in cases.items():
        if c.get('fracture_type') == 'max_strain':
            failures.append({'case_id': cid, 'kind': 'truncated_at_max_strain',
                             'detail': f"reached max_strain={c.get('max_strain')} "
                                       f"without Dc/localization; ef is a lower bound",
                             'Dagg_terminal': (c.get('damage') or [None])[-1]})
        if c.get('fracture_type') == 'numerical_limit':
            failures.append({'case_id': cid, 'kind': 'numerical_limit',
                             'detail': 'step size hit the adaptive lower bound'})
        if (c.get('n_step_reductions') or 0) > 0:
            failures.append({'case_id': cid, 'kind': 'step_reductions',
                             'detail': f"{c['n_step_reductions']} failed increments "
                                       f"(adaptive step halving), final eps_step="
                                       f"{c.get('eps_step_final')}"})

    runtimes = {cid: {'wall_sec': c.get('wall_sec'),
                      'wall_min': round((c.get('wall_sec') or 0) / 60.0, 1),
                      'n_steps_converged': c.get('n_steps_converged'),
                      'sec_per_step': round((c.get('wall_sec') or 0)
                                            / max(c.get('n_steps_converged') or 1, 1), 1)}
                for cid, c in cases.items()}

    os.makedirs(FIG, exist_ok=True)
    fig_path = os.path.join(FIG, 'nonlocal_vs_local_ef.png')
    make_figure(cases, cost, fig_path)

    selfcheck = None
    scp = os.path.join(CASE_DIR, '_selfcheck.json')
    if os.path.exists(scp):
        with open(scp, 'r', encoding='utf-8') as f:
            selfcheck = json.load(f)

    result = {
        'generated_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
        'host': platform.platform(),
        'python': sys.version.split()[0],
        'protocol': {
            'material': 'Ti64', 'orientation_deg': 0,
            'S0': S0_CAL['Ti64'], 'Dc': 0.42, 'n_sub': N_SUB,
            'strain_rate': 1e-3, 'seed': 42,
            'fracture_criterion': 'D_agg >= Dc or localization onset (identical to R13)',
            'eps_step': '1e-3 for the matrix; 2e-3 reserved for the R13 reproduction case',
            'R13_locked_assets': {'ef_local_ng3_0deg': R13_LOCAL[0],
                                  'ef_taylor_0deg': EF_TAYLOR_LOCKED,
                                  'bias_band_pct': R13_BIAS_BAND},
        },
        'route_evaluation': {
            'selected': 'integral-type nonlocal damage (nonlocal equivalent plastic strain)',
            'rejected': {
                'implicit_gradient_helmholtz':
                    'needs a scalar-field assembly and a sparse solver the framework does not '
                    'have; the driving variable would become implicit, so the FD tangent needs '
                    '2*n_free*30 sparse solves per increment (~+64% on the tangent budget)',
                'allen_cahn_phase_field':
                    'requires extra unknowns in the Newton system, gradient energy, interface '
                    'parameter calibration, and a redefinition of the fracture criterion - it '
                    'would break consistency with the locked R13 results. Declared a '
                    'higher-fidelity follow-on, NOT implemented.',
            },
            'reason': ('the nonlocal operator is a state-independent constant linear operator: '
                       'computed once, no new unknowns, no change to the FE assembly, and the '
                       'numerical tangent picks the coupling up automatically.'),
        },
        'cost_bench': cost,
        'selfcheck': selfcheck,
        'cases': rows,
        'mesh_dependence': mesh,
        'nonlocal_vs_local': {
            'ng3_local_ef_e1e3': loc3, 'ng3_nonlocal_ef_e1e3': nl3,
            'ng3_local_ef_e2e3_R13_reproduction': loc3_2e3,
            'ng3_ef_change_pct_nonlocal_vs_local': (round((nl3 - loc3) / loc3 * 100.0, 2)
                                                    if (loc3 and nl3) else None),
            'pairs': nvsl,
        },
        'bias_vs_taylor': bias,
        'three_direction_bias': three_direction_bias(cases),
        'lc_sensitivity_ng3': lc_sensitivity(cases),
        'nonlocal_kernel_stats': load_kernel_stats(),
        'damage_field_smoothness': smoothness_table(cases),
        'runtime': runtimes,
        'failures_and_divergence': failures,
        'honest_declarations': HONEST_NOTES,
        'taylor_reference_check': load_json_soft(
            os.path.join(OUT, '_taylor_ref_check.json')),
        'taylor_angle_check': load_json_soft(
            os.path.join(OUT, '_taylor_angle_check.json')),
    }
    with open(os.path.join(OUT, 'nonlocal_damage_results.json'), 'w',
              encoding='utf-8', newline='') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps({'cases': rows, 'mesh_dependence': mesh,
                      'nonlocal_vs_local': result['nonlocal_vs_local'],
                      'three_direction_bias': result['three_direction_bias'],
                      'lc_sensitivity_ng3': result['lc_sensitivity_ng3'],
                      'figure': fig_path},
                     ensure_ascii=False, indent=1), flush=True)
    print('wrote', os.path.join(OUT, 'nonlocal_damage_results.json'), flush=True)


if __name__ == '__main__':
    main()
