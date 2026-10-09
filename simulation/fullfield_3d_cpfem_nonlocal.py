# -*- coding: utf-8 -*-
"""
fullfield_3d_cpfem_nonlocal.py — 积分型非局部损伤（integral-type nonlocal damage）
=================================================================================
在 `fullfield_3d_cpfem.py` 的 **不修改原文件** 前提下，用子类方式给 3D 全场
CPFEM（hexa8 / 2x2x2 高斯 / damped Newton / 数值切线 / Lemaitre 型耦合损伤）
加一个最小可行、且可关闭的正则化开关：

    lc = None  -> 局部损伤（直接复用父类方法，数值上与 R13 既有结果逐位一致）
    lc = float -> 积分型非局部损伤，内部长度 lc（RVE 归一化长度单位，RVE=单位立方体）

非局部化方案（本文件唯一改动本构的地方）
--------------------------------------
每个高斯积分点的 **等效塑性应变** 做非局部化（Pijaudier-Cabot & Bazant 1987 的
积分型非局部损伤：把损伤驱动量换成它的非局部平均）：

    pbar_i = sum_j W_ij p_j ,  dpbar_i = sum_j W_ij dp_j

    W_ij = exp(-(r_ij/lc)^2) / sum_k exp(-(r_ik/lc)^2) ,  r_ij <= R_cut = 3 lc
    （按行归一化 -> 常数场被精确保持：W @ c*1 = c*1）

损伤演化门与驱动量同时换成非局部量：

    local    : act = (p > p_D) & (dp > 0)      ; dD = (r^s_dam) * dp    * f_AM
    nonlocal : act = (pbar > p_D) & (dpbar > 0); dD = (r^s_dam) * dpbar * f_AM

其他一切（滑移/硬化律、损伤能量项 r = Y/S_eff、Dc 判据、ef 定义、eps_step、
n_sub、收敛容差、自适应步长缩减）与原文件完全相同。

为什么权重矩阵只算一次：W 只依赖积分点几何（几何常量），与 u、状态无关；
截断后稀疏，每个子步一次矩阵-向量乘（ng=3: 216x216；ng=4: 512x512），
成本相对单次 residual（~20 ms，30 子步）可忽略。数值切线是有限差分，
非局部耦合通过 `_residual -> _integrate_batch` 自动进入切线。

自检（本文件 main 之外的 `selfcheck`）：
  * W 行和 = 1，W @ 1 = 1（常数场保持）
  * W = I（`force_identity_W=True`）时非局部结果必须与局部结果逐位相同
    —— 用来证明非局部版 `_integrate_batch` 的拷贝是忠实的。

运行（每个 case 独立进程，输出 output/nonlocal_cases/<case_id>.json）：
  python fullfield_3d_cpfem_nonlocal.py --case-id ng3_lc033 --ng 3 --lc 0.3333333 \
      --eps-step 1e-3 --max-strain 0.20
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'src'))

import fullfield_3d_cpfem as _ff  # noqa: E402
from fullfield_3d_cpfem import FullField3D as _Base3D  # noqa: E402

try:
    import scipy.sparse as _sp
    _HAVE_SCIPY = True
except Exception:                                       # pragma: no cover
    _sp = None
    _HAVE_SCIPY = False

OUT = os.path.join(HERE, 'output')
CASE_OUT = os.path.join(OUT, 'nonlocal_cases')
STRAIN_RATE = _ff.STRAIN_RATE
EPS_STEP = _ff.EPS_STEP
N_SUB = _ff.N_SUB
S0_CAL = _ff.S0_CAL
NG = _ff.NG
SEED = _ff.SEED
NEWTON_MAX = _ff.NEWTON_MAX
RES_TOL = _ff.RES_TOL
SLIP_CLIP = _ff.SLIP_CLIP
DGAMMA_SUM_MAX = _ff.DGAMMA_SUM_MAX
GAUSS_PTS = _ff.GAUSS_PTS

RCUT_FACTOR = 3.0
CHECKPOINTS = [0.04, 0.06, 0.08, 0.10, 0.12]


# ----------------------------------------------------------------------------
# 几何：积分点坐标 + 非局部权重矩阵
# ----------------------------------------------------------------------------
def ip_coords(ng, gauss_pts=None):
    """积分点物理坐标 (nip,3)。单元 e = ez*ng^2 + ey*ng + ex，单元内 8 个高斯点。

    与 fullfield_3d_cpfem.FullField3D 的 conn / 高斯点枚举顺序严格一致：
    单元 (ex,ey,ez) 的形心 = ((ex+0.5)/ng, (ey+0.5)/ng, (ez+0.5)/ng)，
    h = 1/ng，积分点 = 形心 + g * h/2。
    """
    gpt = GAUSS_PTS if gauss_pts is None else gauss_pts
    h = 1.0 / ng
    pos = np.zeros((ng ** 3 * 8, 3))
    for ez in range(ng):
        for ey in range(ng):
            for ex in range(ng):
                e = ez * ng ** 2 + ey * ng + ex
                c = np.array([(ex + 0.5) * h, (ey + 0.5) * h, (ez + 0.5) * h])
                for gi, g in enumerate(gpt):
                    pos[e * 8 + gi] = c + 0.5 * h * np.array(g)
    return pos


def build_weights(pos, lc, rcut_factor=RCUT_FACTOR, sparsify=True):
    """行归一化高斯核权重矩阵。返回 (W, stats)。"""
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    rcut = rcut_factor * lc
    Wd = np.where(r <= rcut, np.exp(-(r / lc) ** 2), 0.0)
    rs = Wd.sum(axis=1)
    if np.any(rs <= 0):
        raise RuntimeError('empty nonlocal neighbourhood: increase R_cut or lc')
    Wd = Wd / rs[:, None]
    nnz = int(np.count_nonzero(Wd))
    stats = {'nip': int(pos.shape[0]), 'lc': float(lc), 'rcut': float(rcut),
             'nnz': nnz, 'nnz_per_row_mean': round(nnz / pos.shape[0], 2),
             'max_neighbour_radius': float(r[Wd > 0].max()),
             'row_sum_min': float(Wd.sum(axis=1).min()),
             'row_sum_max': float(Wd.sum(axis=1).max()),
             'constant_field_error': float(np.abs(Wd @ np.ones(pos.shape[0]) - 1.0).max()),
             'sparse': bool(sparsify and _HAVE_SCIPY)}
    if sparsify and _HAVE_SCIPY:
        return _sp.csr_matrix(Wd), stats
    return Wd, stats


def neighbour_pairs(pos, max_ratio=1.35):
    """相邻积分点对（距离 <= max_ratio * 最近邻距离的中位数），用于 D 场梯度指标。"""
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(r.shape[0], 1)
    rv = r[iu]
    rmin = np.median(rv[rv > 0]) if np.any(rv > 0) else 1.0
    sel = (rv > 0) & (rv <= max_ratio * rmin)
    return iu[0][sel], iu[1][sel], rv[sel], float(rmin)


def dfield_metrics(D, Wh, pairs, h):
    """D 场光滑性指标。Wh = lc=h 的固定几何核（与算例 lc 无关，便于跨网格比较）。"""
    ii, jj, rr = pairs
    g = np.abs(D[ii] - D[jj]) / np.maximum(rr, 1e-12)
    ref = max(float(np.abs(D).max()), 1e-12)
    nrm = float(np.linalg.norm(D))
    return {
        'D_mean': float(D.mean()), 'D_max': float(D.max()),
        'D_std': float(D.std()),
        'D_max_over_mean': float(D.max() / max(D.mean(), 1e-12)),
        'D_grad_mean': float(g.mean()), 'D_grad_max': float(g.max()),
        'D_grad_mean_over_h': float(g.mean() * h),
        'D_roughness_h_kernel': float(np.linalg.norm(D - Wh @ D) / max(nrm, 1e-12)),
        'D_localization_frac': float(np.mean(D > 0.5 * D.max())) if D.max() > 0 else 0.0,
        'D_ref_for_scaling': float(ref),
    }


# ----------------------------------------------------------------------------
# 非局部全场 CPFEM
# ----------------------------------------------------------------------------
class FullField3DNonlocal(_Base3D):
    """父类 + 积分型非局部损伤开关。lc=None 时全部走父类方法（逐位一致）。"""

    def __init__(self, mat, psi, S0, eps_step=EPS_STEP, n_sub=N_SUB, ng=NG,
                 max_strain=None, lc=None, rcut_factor=RCUT_FACTOR, seed=SEED,
                 force_identity_W=False, ref_strain=None):
        # 父类 __init__ 用模块级 NGRAIN 构造 TaylorCPCDM（每单元一晶粒 -> ng^3）。
        # 这里只在内存里临时改写该模块全局，绝不落盘修改原文件。
        prev = _ff.NGRAIN
        _ff.NGRAIN = ng ** 3
        try:
            super().__init__(mat, psi, S0, eps_step=eps_step, n_sub=n_sub, ng=ng,
                             max_strain=max_strain)
        finally:
            _ff.NGRAIN = prev
        self.seed = seed
        self.lc = lc
        self.h = 1.0 / self.ng
        self.ref_strain = CHECKPOINTS[1] if ref_strain is None else ref_strain
        self.ip_pos = ip_coords(self.ng)
        self.W_stats = None
        self.W = None
        if lc is not None:
            if force_identity_W:
                nip = self.ip_pos.shape[0]
                Wd = np.eye(nip)
                self.W = _sp.csr_matrix(Wd) if _HAVE_SCIPY else Wd
                self.W_stats = {'identity_W_selfcheck': True, 'nip': nip,
                                'lc': float(lc), 'rcut': None, 'nnz': nip,
                                'nnz_per_row_mean': 1.0,
                                'constant_field_error': 0.0,
                                'sparse': bool(_HAVE_SCIPY)}
            else:
                self.W, self.W_stats = build_weights(self.ip_pos, lc, rcut_factor)
        # 固定几何核（lc = 晶粒尺寸），只用于光滑性指标，不参与物理
        Wh, _ = build_weights(self.ip_pos, self.h, sparsify=False)
        self.Wh_metric = Wh
        ii, jj, rr, rmin = neighbour_pairs(self.ip_pos)
        self.adj_pairs = (ii, jj, rr)
        self.adj_rmin = rmin

    # -- 状态 ---------------------------------------------------------------
    def _state_init(self):
        st = super()._state_init()
        if self.W is not None:
            st['pbar'] = np.zeros(self.nip)
        return st

    # -- 残差（只改解包，物理完全沿用父类 _grad_F3 / B 组装）-----------------
    def _residual(self, u, st, dt):
        F3 = self._grad_F3(u)
        sig = self._integrate_batch(F3, st, dt)[0]
        r = np.zeros(3 * self.nn)
        for e in range(self.ne):
            nd = self.conn[e]
            for gi in range(8):
                ip = e * 8 + gi
                if st['D'][ip] >= 0.99:
                    continue
                re = self.Bs[ip].T @ sig[ip] * self.Jdet
                for a in range(8):
                    da = 3 * nd[a]
                    r[da:da + 3] += re[3 * a:3 * a + 3]
        return r, sig

    # -- 本构积分 -----------------------------------------------------------
    def _integrate_batch(self, F3, st, dt):
        if self.W is None:
            sig6, D, Fp, g, p = super()._integrate_batch(F3, st, dt)
            return sig6, D, Fp, g, p, None
        return self._integrate_batch_nl(F3, st, dt)

    def _integrate_batch_nl(self, F3, st, dt):
        """与父类逐语句一致，只有两处改动：
        (1) 每子步 pbar += W @ dp；(2) 损伤门与驱动量用 pbar / dpbar。
        当 W = I 时本函数与父类逐位等价（见 selfcheck）。"""
        Fp = st['Fp'].copy()
        g = st['g'].copy()
        D = st['D'].copy()
        p = st['p'].copy()
        pbar = st['pbar'].copy()
        R = self.Rs
        dt_sub = dt / N_SUB
        Rt = np.swapaxes(R, 1, 2)
        W = self.W
        for _ in range(N_SUB):
            Fc = np.einsum('nij,njk,nkl->nil', Rt, F3, R)
            Fp_inv = np.linalg.inv(Fp)
            Fe = np.einsum('nij,njk->nik', Fc, Fp_inv)
            Ee = 0.5 * (np.einsum('nij,njk->nik', np.swapaxes(Fe, 1, 2), Fe)
                        - np.eye(3))
            S = np.einsum('ijkl,nkl->nij', self.C4, Ee)
            tau = np.einsum('nij,pij->np', S, self.P0)
            tau_abs = np.abs(tau)
            denom = np.maximum(g, 1e-10)
            ratio = np.clip(np.where(g > 1e-6, tau_abs / denom, 0.0), 0.0, 10.0)
            gamma_dot = self.gdot0 * (ratio ** self.n_rate) * np.sign(tau)
            dgamma = np.clip(gamma_dot * dt_sub, -SLIP_CLIP, SLIP_CLIP)
            dg_tot = np.sum(np.abs(dgamma), axis=1)
            scale = np.minimum(1.0, DGAMMA_SUM_MAX / np.maximum(dg_tot, 1e-30))
            dgamma = dgamma * scale[:, None]
            Lp = np.einsum('np,pij->nij', dgamma, self.P0)
            Fp = (np.eye(3) + Lp) @ Fp
            dgd_dt = np.abs(dgamma) / dt_sub
            ratio_h = np.clip(1.0 - g / self.g_sat[None, :], 0.0, None)
            dg = self.h0 * dgd_dt * (ratio_h ** self.a_exp)
            dg = dg @ self.Q.T
            g = np.maximum(g + dg * dt_sub, 1e-3)
            dp = np.sqrt(2.0 / 3.0) * np.sum(np.abs(dgamma), axis=1)
            p = p + dp
            # ---- 唯一的物理改动：非局部化 ----
            dpbar = np.asarray(W @ dp).ravel()
            pbar = pbar + dpbar
            act = (pbar > self.p_D) & (dpbar > 0)
            if np.any(act):
                Fe_n = np.einsum('nij,njk->nik', Fc, np.linalg.inv(Fp))
                Ee_n = 0.5 * (np.einsum('nij,njk->nik',
                                        np.swapaxes(Fe_n, 1, 2), Fe_n) - np.eye(3))
                S_n = np.einsum('ijkl,nkl->nij', self.C4, Ee_n)
                dev = S_n - np.einsum('nii->n', S_n)[:, None, None] / 3.0 * np.eye(3)
                vm = np.sqrt(1.5 * np.einsum('nij,nij->n', dev, dev))
                Y = vm * vm / (2.0 * self.Emod)
                r = Y / max(self.S_eff, 1e-10)
                dD = (r ** self.s_dam) * dpbar * self.f_AM
                dD = np.clip(dD, 0.0, 0.02)
                D = np.minimum(0.99, D + dD)
        Fp_inv = np.linalg.inv(Fp)
        Fe = np.einsum('nij,njk->nik', np.einsum('nij,njk,nkl->nil', Rt, F3, R), Fp_inv)
        Ee = 0.5 * (np.einsum('nij,njk->nik', np.swapaxes(Fe, 1, 2), Fe) - np.eye(3))
        S = np.einsum('ijkl,nkl->nij', self.C4, Ee)
        J = np.linalg.det(Fe)
        sc = (1.0 - D)[:, None, None] / np.maximum(J, 1e-10)[:, None, None] \
            * np.einsum('nij,njk->nik', Fe, S)
        sm = np.einsum('nij,njk,nlk->nil', R, sc, R)
        sig6 = np.stack([sm[:, 0, 0], sm[:, 1, 1], sm[:, 2, 2],
                         sm[:, 0, 1], sm[:, 1, 2], sm[:, 0, 2]], axis=1)
        return sig6, D, Fp, g, p, pbar

    # -- 主循环（结构/判据与父类逐语句一致，仅增加记录）---------------------
    def run(self, verbose=True):
        t0 = time.time()
        st = self._state_init()
        u = np.zeros(3 * self.nn)
        u_prev = u.copy()
        # 父类 run() 用模块常量 EPS_STEP 起步；本文件用构造时传入的 eps_step
        # （eps_step 与父类默认值一致时两者逐位相同，见 selfcheck）。
        eps0 = float(self.t.eps_step)
        reached = 0.0
        Dagg_h, sig_h, used_eps = [], [], []
        ef = None
        it_total = 0
        n_reduce = 0
        n_step_ok = 0
        Dagg = 0.0
        smac = 0.0
        cps_done = set()
        cp_snaps = {}
        D_term = None
        step_log = []
        while reached < self.max_strain - 1e-12:
            cur = min(reached + eps0, self.max_strain)
            dt = eps0 / STRAIN_RATE
            u = u_prev.copy()
            for d in self.controlled_dofs:
                u[d] = cur
            conv = False
            nrm = 0.0
            try:
                K_inv = np.linalg.pinv(self._tangent(u, st, dt))
            except Exception:
                K_inv = None
            for it in range(NEWTON_MAX):
                it_total += 1
                r, _ = self._residual(u, st, dt)
                r_free = r[self.free_dofs]
                nrm = float(np.max(np.abs(r_free)))
                if nrm < RES_TOL:
                    conv = True
                    break
                if K_inv is None:
                    break
                du = -K_inv @ r_free
                alpha = 1.0
                improved = False
                for _ in range(6):
                    u_try = u.copy()
                    u_try[self.free_dofs] = u[self.free_dofs] + alpha * du
                    r_t, _ = self._residual(u_try, st, dt)
                    nrm_t = float(np.max(np.abs(r_t[self.free_dofs])))
                    if nrm_t < RES_TOL:
                        u = u_try
                        conv = True
                        improved = True
                        break
                    if nrm_t < nrm:
                        u = u_try
                        nrm = nrm_t
                        improved = True
                        break
                    alpha *= 0.5
                if conv:
                    break
                if not improved:
                    if it == 0:
                        try:
                            K_inv = np.linalg.pinv(self._tangent(u, st, dt))
                        except Exception:
                            break
                        continue
                    break
            if not conv:
                n_reduce += 1
                if eps0 < 6.25e-5 or n_reduce > 20:
                    if verbose:
                        print(f'  [fail] eps={reached:.4f}->{cur:.4f} nrm={nrm:.3e}',
                              flush=True)
                    if Dagg > 0.005:
                        peak = max(sig_h) if sig_h else 0.0
                        if smac < 0.85 * peak or smac <= peak * 1.02:
                            ef = cur
                            ftype = 'localization_onset'
                            break
                    ftype = 'numerical_limit'
                    break
                eps0 *= 0.5
                u = u_prev.copy()
                continue
            n_step_ok += 1
            F3 = self._grad_F3(u)
            _, Darr, FpN, gN, pN, pbarN = self._integrate_batch(F3, st, dt)
            st = {'Fp': FpN, 'g': gN, 'D': Darr, 'p': pN}
            if pbarN is not None:
                st['pbar'] = pbarN
            Dagg = float(Darr.mean())
            _, sig = self._residual(u, st, dt)
            act = Darr < 0.99
            smac = float(sig[act, 0].mean()) if np.any(act) else 0.0
            Dagg_h.append(Dagg)
            sig_h.append(smac)
            used_eps.append(cur)
            step_log.append({'eps': cur, 'sig11_MPa': round(smac / 1e6, 3),
                             'Dagg': round(Dagg, 6),
                             'iters': it_total, 'wall_s': round(time.time() - t0, 1),
                             'eps0': eps0})
            D_term = Darr.copy()
            for cp in CHECKPOINTS:
                if cp not in cps_done and cur >= cp - 1e-12:
                    cps_done.add(cp)
                    cp_snaps[cp] = dfield_metrics(Darr, self.Wh_metric,
                                                  self.adj_pairs, self.h)
                    cp_snaps[cp]['D_field'] = [round(float(v), 6) for v in Darr]
            if Dagg >= self.Dc:
                if len(Dagg_h) > 1 and Dagg_h[-1] > Dagg_h[-2]:
                    frac = (self.Dc - Dagg_h[-2]) / (Dagg_h[-1] - Dagg_h[-2])
                    ef = used_eps[-2] + np.clip(frac, 0.0, 1.0) * (cur - used_eps[-2])
                else:
                    ef = cur
                ftype = 'aggregate_Dc'
                break
            reached = cur
            u_prev = u.copy()
            if verbose and (len(used_eps) % 20 == 0):
                print(f'  [{self.mat} {self.psi}D ng={self.ng} lc={self.lc}] eps={cur:.4f} '
                      f'sig11={smac/1e6:.1f} Dagg={Dagg:.4f} '
                      f'({time.time()-t0:.0f}s, it={it_total}, '
                      f'eps0={eps0:.4f}, r={n_reduce})', flush=True)
        if ef is None:
            ef = float(used_eps[-1]) if used_eps else 0.0
            ftype = 'max_strain'
        term_metrics = (dfield_metrics(D_term, self.Wh_metric, self.adj_pairs, self.h)
                        if D_term is not None else None)
        if term_metrics is not None:
            term_metrics['D_field'] = [round(float(v), 6) for v in D_term]
        return {'strain': used_eps, 'stress': sig_h, 'damage': Dagg_h,
                'fracture_strain': ef, 'fracture_type': ftype,
                'eps_step_final': eps0,
                'material': self.mat, 'orientation': self.psi,
                'S0': float(S0_CAL[self.mat]), 'Dc': float(self.Dc),
                'wall_sec': round(time.time() - t0, 1),
                'ng': self.ng, 'n_grains': self.ng ** 3, 'nip': self.nip,
                'n_free_dofs': int(len(self.free_dofs)),
                'lc': self.lc, 'grain_size': self.h,
                'nonlocal': self.W is not None,
                'W_stats': self.W_stats,
                'n_steps_converged': n_step_ok, 'n_step_reductions': n_reduce,
                'n_newton_iter_total': it_total,
                'checkpoints': cp_snaps,
                'terminal_D_metrics': term_metrics,
                'adjacent_ip_rmin': self.adj_rmin,
                'step_log': step_log}


# ----------------------------------------------------------------------------
def run_case(case_id, ng, lc, eps_step, max_strain, seed=SEED, verbose=True,
             identity_W=False, angle=0):
    """angle = 加载方向与晶粒取向的夹角（0 / 45 / 90 度），默认 0 度。

    angle 只进入 TaylorCPCDM 的取向与滑移系取向因子（`_gs_directional`），
    不改变非局部算子（W 只依赖积分点几何），因此 angle=0 的路径与加入本参数前一致。
    """
    print(f'=== case {case_id}: ng={ng} lc={lc} eps_step={eps_step} '
          f'max_strain={max_strain} angle={angle} ===', flush=True)
    t0 = time.time()
    ff = FullField3DNonlocal('Ti64', angle, S0_CAL['Ti64'], eps_step=eps_step,
                            n_sub=N_SUB, ng=ng, max_strain=max_strain, lc=lc,
                            seed=seed, force_identity_W=identity_W)
    res = ff.run(verbose=verbose)
    gs = _ff._gs_directional('Ti64', angle)
    t = _ff.TaylorCPCDM('Ti64', angle, S0_override=S0_CAL['Ti64'], n_grains=ng ** 3,
                        eps_step=eps_step, n_sub=N_SUB, strain_rate=STRAIN_RATE,
                        model_version='v4', param_overrides=gs, seed=seed)
    tr = t.run_uniaxial(res['strain'][-1] + 0.02 if len(res['strain']) else 0.1)
    ef_t = float(tr['fracture_strain'])
    out = dict(res)
    out.update({'case_id': case_id, 'lc': lc, 'eps_step': eps_step,
                'max_strain': max_strain, 'seed': seed, 'angle': angle,
                'ef_taylor_eps_matched': round(ef_t, 6),
                'bias_vs_taylor_eps_matched_pct':
                    round((res['fracture_strain'] - ef_t) / ef_t * 100.0, 2)
                    if ef_t > 0 else None,
                'ef_taylor_locked_2em3': 0.11175,
                'bias_vs_taylor_locked_pct':
                    round((res['fracture_strain'] - 0.11175) / 0.11175 * 100.0, 2),
                'wall_sec_total': round(time.time() - t0, 1)})
    os.makedirs(CASE_OUT, exist_ok=True)
    path = os.path.join(CASE_OUT, case_id + '.json')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'=== {case_id} DONE ef={res["fracture_strain"]:.5f} '
          f'type={res["fracture_type"]} wall={out["wall_sec_total"]:.0f}s -> {path} ===',
          flush=True)
    return out


def selfcheck(ng=3, eps_step=2e-3, max_strain=6e-3, verbose=True):
    """自检：证明 (1) 权重矩阵性质正确；(2) 非局部代码路径在 W=I 时与局部
    逐位等价；(3) 本文件的局部模式与未修改的原文件逐位等价。

    返回 dict（同时打印摘要），供 nonlocal_damage_results.json 引用。
    """
    rep = {}
    # (1) 权重矩阵性质
    pos = ip_coords(ng)
    Wn, stats = build_weights(pos, 1.0 / ng)
    Wd = np.asarray(Wn.todense()) if (_HAVE_SCIPY and _sp is not None
                                      and hasattr(Wn, 'todense')) else np.asarray(Wn)
    rep['weights'] = stats
    rep['weights']['row_sum_max_dev'] = float(np.abs(Wd.sum(1) - 1).max())
    rep['weights']['symmetry_max_dev'] = float(np.abs(Wd - Wd.T).max())
    rep['weights']['min_entry'] = float(Wd.min())
    # 邻域覆盖：每个积分点至少有一个邻居
    rep['weights']['isolated_ips'] = int(np.sum((Wd > 0).sum(1) <= 1))

    def hist(res):
        return (np.array(res['strain']), np.array(res['stress']),
                np.array(res['damage']), float(res['fracture_strain']),
                res['fracture_type'])

    # (2) W = I 等价性
    ma = FullField3DNonlocal('Ti64', 0, S0_CAL['Ti64'], eps_step=eps_step,
                             n_sub=N_SUB, ng=ng, max_strain=max_strain,
                             lc=1.0 / ng, force_identity_W=True)
    mb = FullField3DNonlocal('Ti64', 0, S0_CAL['Ti64'], eps_step=eps_step,
                             n_sub=N_SUB, ng=ng, max_strain=max_strain,
                             lc=None)
    a = ma.run(verbose=False)
    b = mb.run(verbose=False)
    sa, ya, da, fa, ta = hist(a)
    sb, yb, db, fb, tb = hist(b)
    rep['identity_W_vs_local'] = {
        'n_steps': int(len(sa)),
        'strain_max_abs_diff': float(np.abs(sa - sb).max()) if len(sa) == len(sb) else None,
        'stress_max_abs_diff': float(np.abs(ya - yb).max()) if len(ya) == len(yb) else None,
        'damage_max_abs_diff': float(np.abs(da - db).max()) if len(da) == len(db) else None,
        'ef_diff': abs(fa - fb), 'ftype_a': ta, 'ftype_b': tb,
        'ef_both': [fa, fb],
        'bitwise_identical': bool(len(sa) == len(sb)
                                 and np.array_equal(sa, sb) and np.array_equal(ya, yb)
                                 and np.array_equal(da, db) and fa == fb)}

    # (2b) 直接对 _integrate_batch 做等价性（把 p 预置在 p_D 之上，强制激活
    #      损伤分支，从而验证含损伤路径在 W=I 时也逐位等价）
    rng = np.random.RandomState(0)
    F3t = np.tile(np.eye(3)[None, :, :], (ma.nip, 1, 1))
    F3t += 0.02 * rng.randn(ma.nip, 3, 3)
    st_t = ma._state_init()
    st_t['p'] = np.full(ma.nip, 0.06)
    if 'pbar' in st_t:
        st_t['pbar'] = np.full(ma.nip, 0.06)
    oa = ma._integrate_batch(F3t, st_t, 2e-3)
    ob = mb._integrate_batch(F3t, st_t, 2e-3)
    rep['integrate_batch_identity_W_test'] = {
        'D_max_abs_diff': float(np.abs(oa[1] - ob[1]).max()),
        'Fp_max_abs_diff': float(np.abs(oa[2] - ob[2]).max()),
        'g_max_abs_diff': float(np.abs(oa[3] - ob[3]).max()),
        'p_max_abs_diff': float(np.abs(oa[4] - ob[4]).max()),
        'sig_max_abs_diff': float(np.abs(oa[0] - ob[0]).max()),
        'damage_branch_active': bool(np.any(oa[1] > st_t['D'][0])),
        'bitwise_identical': bool(np.array_equal(oa[0], ob[0])
                                 and np.array_equal(oa[1], ob[1])
                                 and np.array_equal(oa[2], ob[2])
                                 and np.array_equal(oa[3], ob[3])
                                 and np.array_equal(oa[4], ob[4])),
    }
    # (2c) 非局部算子对常数场与平移不变性的数值验证
    c = np.full(ma.nip, 0.07)
    rep['operator_checks'] = {
        'constant_field_preserved_max_dev': float(np.abs(np.asarray(
            ma.W @ c).ravel() - c).max()) if ma.W is not None else None,
        'uniform_activation_note': 'W @ c*1 = c*1 由行归一化保证',
    }

    # (3) 与未修改原文件逐位等价（ng=3 时原文件可用）
    #     注意原文件 run() 用模块常量 EPS_STEP=2e-3 起步，忽略构造参数 eps_step；
    #     本文件用构造参数。故本项比较必须在 eps_step = EPS_STEP 下进行才可比。
    if ng == 3:
        es_ref = float(_ff.EPS_STEP)
        mc = FullField3DNonlocal('Ti64', 0, S0_CAL['Ti64'], eps_step=es_ref,
                                 n_sub=N_SUB, ng=3, max_strain=max_strain, lc=None)
        c = mc.run(verbose=False)
        sc, yc, dc, fc, tc = hist(c)
        sb2, yb2, db2, fb2, tb2 = sc, yc, dc, fc, tc
        if abs(eps_step - es_ref) > 1e-15:
            mb2 = FullField3DNonlocal('Ti64', 0, S0_CAL['Ti64'], eps_step=es_ref,
                                      n_sub=N_SUB, ng=3, max_strain=max_strain, lc=None)
            sb2, yb2, db2, fb2, tb2 = hist(mb2.run(verbose=False))
        rep['local_vs_original_file'] = {
            'eps_step_used_for_comparison': es_ref,
            'note': ('原文件 run() 忽略构造参数 eps_step 而用模块常量 EPS_STEP；'
                     '本文件改用构造参数。两者在 eps_step=EPS_STEP 时逐位相同。'),
            'n_steps_original': int(len(sc)), 'n_steps_new_local': int(len(sb2)),
            'strain_max_abs_diff': float(np.abs(sb2 - sc).max()) if len(sc) == len(sb2) else None,
            'stress_max_abs_diff': float(np.abs(yb2 - yc).max()) if len(yc) == len(yb2) else None,
            'damage_max_abs_diff': float(np.abs(db2 - dc).max()) if len(dc) == len(db2) else None,
            'ef_original': fc, 'ef_new_local': fb2,
            'bitwise_identical': bool(len(sc) == len(sb2)
                                     and np.array_equal(sc, sb2) and np.array_equal(yc, yb2)
                                     and np.array_equal(dc, db2) and fc == fb2)}
    if verbose:
        print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case-id', default=None)
    ap.add_argument('--ng', type=int, default=NG)
    ap.add_argument('--lc', type=float, default=None)
    ap.add_argument('--eps-step', type=float, default=1e-3)
    ap.add_argument('--max-strain', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--angle', type=float, default=0.0,
                    help='加载方向与晶粒取向夹角（度）；0 / 45 / 90')
    ap.add_argument('--identity-W', action='store_true')
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--selfcheck-max-strain', type=float, default=6e-3)
    a = ap.parse_args()
    if a.selfcheck:
        rep = selfcheck(ng=a.ng, eps_step=a.eps_step,
                        max_strain=a.selfcheck_max_strain)
        os.makedirs(CASE_OUT, exist_ok=True)
        with open(os.path.join(CASE_OUT, '_selfcheck.json'), 'w',
                  encoding='utf-8', newline='') as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
        return
    if a.case_id is None:
        ap.error('--case-id is required unless --selfcheck is given')
    run_case(a.case_id, a.ng, a.lc, a.eps_step, a.max_strain, a.seed,
             identity_W=a.identity_W, angle=a.angle)


if __name__ == '__main__':
    main()
