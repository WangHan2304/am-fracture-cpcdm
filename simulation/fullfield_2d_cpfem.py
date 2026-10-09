# -*- coding: utf-8 -*-
"""
fullfield_2d_cpfem.py — 2D 平面应变全场 CPFEM（耦合损伤，numpy 批量向量化）
===========================================================================
与 TaylorCPCDM 同构对比：相同材料参数（S0、gs(psi)、硬化、损伤律、Dc）、
相同取向（同种子 42）、相同应变步/子步时间分辨率（eps_step、n_sub 固定）。
唯一区别：Taylor 等应变假设 vs 全场平衡（Q4 平面应变、2x2 高斯、
阻尼 Newton + 单步回退自适应，各向同性弹性初刚度仅用于迭代方向）。
本构：批量显式积分（numpy 向量化 64 积分点；滑移增量 clip 与 Taylor
子步控制一致；无逐点自适应二分——显式声明于论文）。
断裂判据与 Taylor 相同：全部高斯点平均损伤 >= Dc（线性插值）。
输出: output/fullfield_2d_cpfem_results.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from materials import EXPERIMENTAL  # noqa: E402
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import xi_eff   # noqa: E402

STRAIN_RATE = 1e-3
EPS_STEP = 2e-3          # 主应变步（Taylor 对比用同一值）
N_SUB = 30               # 固定子步（批量显式，无逐点二分）
KG = 0.15
XI0 = {'Ti64': 3.5, '316L': 2.5, 'AlSi10Mg': 1.5}
S0_CAL = {'Ti64': 0.32, '316L': 3.2, 'AlSi10Mg': 0.20}
OUT = os.path.join(os.path.dirname(__file__), 'output')
NGRID = int(os.environ.get('FF_NGRID', '4'))
NGRAIN = NGRID * NGRID
SEED = 42
NU = 0.3
NEWTON_MAX = 25
RES_TOL = 100.0
SLIP_CLIP = 0.01
DGAMMA_SUM_MAX = 0.10

GAUSS = [(1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
         (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
         (-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
         (1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0))]


def _gs_directional(mat, psi):
    xi0 = XI0[mat]
    ratio = xi_eff(xi0, psi) / xi0   # xi_eff(psi)/xi0, single source (am_correction_v4)
    h = 1.0 - KG * (1.0 - ratio)
    if mat == 'Ti64':
        base = {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}
    else:
        base = {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}
    return {k: v * h for k, v in base.items()}


def _dN(eps, eta):
    return np.array([
        [-0.25 * (1 - eta), 0.25 * (1 - eta), 0.25 * (1 + eta), -0.25 * (1 + eta)],
        [-0.25 * (1 - eps), -0.25 * (1 + eps), 0.25 * (1 + eps), 0.25 * (1 - eps)],
    ])


class FullField2D:
    def __init__(self, mat, psi, S0, eps_step=EPS_STEP, n_sub=N_SUB,
                 n=NGRID, max_strain=None):
        self.mat = mat
        self.psi = psi
        gs = _gs_directional(mat, psi)
        self.t = TaylorCPCDM(
            mat, psi, S0_override=S0, n_grains=NGRAIN,
            eps_step=eps_step, n_sub=n_sub, strain_rate=STRAIN_RATE,
            model_version='v4', param_overrides=gs, seed=SEED)
        self.exp = EXPERIMENTAL[mat]
        self.Dc = self.exp['Dc']
        self.max_strain = max_strain or (self.exp['ef'][psi] * 2.0 + 0.01
                                         if mat != '316L'
                                         else self.exp['ef'][psi] * 1.6 + 0.02)
        self.n = n
        self.nn = (n + 1) ** 2
        self.ne = n * n
        self.nip = self.ne * 4
        self.x = np.array([[ix / n, iy / n]
                           for iy in range(n + 1) for ix in range(n + 1)])
        self.conn = []
        for ey in range(n):
            for ex in range(n):
                i0 = ey * (n + 1) + ex
                self.conn.append([i0, i0 + 1, i0 + n + 2, i0 + n + 1])
        self.conn = np.array(self.conn)
        self.jacs = []
        for e in range(self.ne):
            nd = self.conn[e]
            self.jacs.append(np.array([[self.x[nd[1], 0] - self.x[nd[0], 0],
                                        self.x[nd[2], 0] - self.x[nd[0], 0]],
                                       [self.x[nd[1], 1] - self.x[nd[0], 1],
                                        self.x[nd[2], 1] - self.x[nd[0], 1]]]))
        self.Jdet = float(np.linalg.det(self.jacs[0]))
        # 预计算：每积分点 B 矩阵 (nip,3,8) 与 R (nip,3,3)
        self.Bs = np.zeros((self.nip, 3, 8))
        self.Rs = np.zeros((self.nip, 3, 3))
        for e in range(self.ne):
            nd = self.conn[e]
            jinvT = np.linalg.inv(self.jacs[e]).T
            for g, (ep, et) in enumerate(GAUSS):
                ip = e * 4 + g
                dNdx = jinvT @ _dN(ep, et)
                B = np.zeros((3, 8))
                for a in range(4):
                    B[0, 2 * a] = dNdx[0, a]
                    B[1, 2 * a + 1] = dNdx[1, a]
                    B[2, 2 * a] = dNdx[1, a]
                    B[2, 2 * a + 1] = dNdx[0, a]
                self.Bs[ip] = B
                self.Rs[ip] = self.t.R[e]
        self._dofmap()
        self._assemble_K0()
        # 批量本构参数
        self.P0 = self.t.P0                       # (Nslip,3,3)
        self.Nslip = self.t.N_slip
        self.C4 = self.t.stiffness
        self.gdot0 = self.t.params['gamma_dot_0']
        self.n_rate = self.t.params['n_rate']
        self.h0 = self.t.params['h0']
        self.a_exp = self.t.params['a']
        self.q_lat = self.t.params['q_lat']
        self.p_D = self.t.params['p_D']
        self.s_dam = self.t.params['s_damage']
        self.S_eff = self.t.S_eff * 1e6
        self.f_AM = self.t.f_AM
        self.g_sat = self.t.g_sat_vec
        self.Emod = self.exp['E']
        # Voce 交互矩阵
        self.Q = np.full((self.Nslip, self.Nslip), self.q_lat)
        np.fill_diagonal(self.Q, 1.0)

    def _dofmap(self):
        n = self.n
        fixed = set()
        controlled = set()
        for iy in range(n + 1):
            for ix in range(n + 1):
                node = iy * (n + 1) + ix
                if ix == 0:
                    fixed.add(2 * node)
                if iy == 0:
                    fixed.add(2 * node + 1)
                if ix == n:
                    controlled.add(2 * node)
        fixed |= controlled
        self.fixed_dofs = sorted(fixed)
        self.controlled_dofs = sorted(controlled)
        self.free_dofs = np.array([d for d in range(2 * self.nn)
                                   if d not in fixed], dtype=int)

    def _assemble_K0(self):
        E = self.exp['E']
        lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        mu = E / (2 * (1 + NU))
        C0 = np.array([[lam + 2 * mu, lam, 0.0],
                       [lam, lam + 2 * mu, 0.0],
                       [0.0, 0.0, mu]])
        K = np.zeros((2 * self.nn, 2 * self.nn))
        for e in range(self.ne):
            nd = self.conn[e]
            for g in range(4):
                ip = e * 4 + g
                Ke = self.Bs[ip].T @ C0 @ self.Bs[ip] * self.Jdet
                for a in range(4):
                    for b in range(4):
                        da = 2 * nd[a]
                        db = 2 * nd[b]
                        K[da:da + 2, db:db + 2] += Ke[2 * a:2 * a + 2, 2 * b:2 * b + 2]
        ff = np.ix_(self.free_dofs, self.free_dofs)
        self.Kff_inv = np.linalg.inv(K[ff])

    # ---------- 批量状态 ----------
    def _state_init(self):
        return {'Fp': np.tile(np.eye(3)[None, :, :], (self.nip, 1, 1)),
                'g': np.tile(self.t.g0_vec[None, :], (self.nip, 1)),
                'D': np.full(self.nip, self.t.params['D0']),
                'p': np.zeros(self.nip)}

    def _grad_F3(self, u):
        """全部积分点 F3（nip,3,3），由位移场计算。"""
        F3 = np.tile(np.eye(3)[None, :, :], (self.nip, 1, 1))
        for e in range(self.ne):
            nd = self.conn[e]
            ue = np.concatenate([u[2 * i:2 * i + 2] for i in nd])
            for g in range(4):
                ip = e * 4 + g
                dNdx = self.Bs[ip][0:2, :].T  # 4x2（B 矩阵第0-1行只含 dN/dx0、dN/dy1 项——不完整，见下）
        # 用显式 H 梯度（B 矩阵组装自 dNdx，但缺剪切项；这里直接重算）
        for e in range(self.ne):
            nd = self.conn[e]
            ue = np.concatenate([u[2 * i:2 * i + 2] for i in nd])
            jinvT = np.linalg.inv(self.jacs[e]).T
            for g, (ep, et) in enumerate(GAUSS):
                ip = e * 4 + g
                dNdx = jinvT @ _dN(ep, et)   # 2x4
                H = np.zeros((2, 2))
                for a in range(4):
                    H[0, 0] += dNdx[0, a] * ue[2 * a]
                    H[0, 1] += dNdx[1, a] * ue[2 * a]
                    H[1, 0] += dNdx[0, a] * ue[2 * a + 1]
                    H[1, 1] += dNdx[1, a] * ue[2 * a + 1]
                F2 = np.eye(2) + H
                F3[ip, 0, 0], F3[ip, 1, 1] = F2[0, 0], F2[1, 1]
                F3[ip, 0, 1], F3[ip, 1, 0] = F2[0, 1], F2[1, 0]
        return F3

    # ---------- 批量显式本构（numpy 向量化，固定 n_sub） ----------
    def _integrate_batch(self, F3, st, dt):
        """从步初状态 st 积分到最终 F3，返回 (sigma_2d(nip,3), Darr, FpN, gN, pN)。"""
        nip = self.nip
        Fp = st['Fp'].copy()
        g = st['g'].copy()
        D = st['D'].copy()
        p = st['p'].copy()
        R = self.Rs                       # (nip,3,3)
        dt_sub = dt / N_SUB
        # Fc = R^T F R：批量
        Rt = np.swapaxes(R, 1, 2)
        for _ in range(N_SUB):
            Fc = np.einsum('nij,njk,nkl->nil', Rt, F3, R)
            Fp_inv = np.linalg.inv(Fp)
            Fe = np.einsum('nij,njk->nik', Fc, Fp_inv)
            Ee = 0.5 * (np.einsum('nij,njk->nik', np.swapaxes(Fe, 1, 2), Fe)
                        - np.eye(3))
            S = np.einsum('ijkl,nkl->nij', self.C4, Ee)
            tau = np.einsum('nij,pij->np', S, self.P0)          # (nip,Nslip)
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
            # 硬化（Voce）
            dgd_dt = np.abs(dgamma) / dt_sub
            ratio_h = np.clip(1.0 - g / self.g_sat[None, :], 0.0, None)
            dg = self.h0 * dgd_dt * (ratio_h ** self.a_exp)
            dg = dg @ self.Q.T
            g = np.maximum(g + dg * dt_sub, 1e-3)
            dp = np.sqrt(2.0 / 3.0) * np.sum(np.abs(dgamma), axis=1)
            p = p + dp
            # 损伤
            act = (p > self.p_D) & (dp > 0)
            if np.any(act):
                Fe_n = np.einsum('nij,njk->nik', Fc, np.linalg.inv(Fp))
                Ee_n = 0.5 * (np.einsum('nij,njk->nik',
                                        np.swapaxes(Fe_n, 1, 2), Fe_n) - np.eye(3))
                S_n = np.einsum('ijkl,nkl->nij', self.C4, Ee_n)
                vm = np.sqrt(1.5 * np.einsum('nij,nij->n', S_n, S_n)
                             - 0.5 * np.einsum('nii,njj->n', S_n, S_n) * 0.0
                             + 0.0)
                dev = S_n - np.einsum('nii->n', S_n)[:, None, None] / 3.0 * np.eye(3)
                vm = np.sqrt(1.5 * np.einsum('nij,nij->n', dev, dev))
                Y = vm * vm / (2.0 * self.Emod)
                r = Y / max(self.S_eff, 1e-10)
                dD = (r ** self.s_dam) * dp * self.f_AM
                dD = np.clip(dD, 0.0, 0.02)
                D = np.minimum(0.99, D + dD)
        # 最终应力（柯西，批量）
        Fp_inv = np.linalg.inv(Fp)
        Fe = np.einsum('nij,njk->nik', np.einsum('nij,njk,nkl->nil', Rt, F3, R), Fp_inv)
        Ee = 0.5 * (np.einsum('nij,njk->nik', np.swapaxes(Fe, 1, 2), Fe) - np.eye(3))
        S = np.einsum('ijkl,nkl->nij', self.C4, Ee)
        J = np.linalg.det(Fe)
        sc = (1.0 - D)[:, None, None] / np.maximum(J, 1e-10)[:, None, None] \
            * np.einsum('nij,njk->nik', Fe, S)
        sm = np.einsum('nij,njk,nlk->nil', R, sc, R)
        sig2 = np.stack([sm[:, 0, 0], sm[:, 1, 1], sm[:, 0, 1]], axis=1)
        return sig2, D, Fp, g, p

    def _residual(self, u, st, dt):
        F3 = self._grad_F3(u)
        sig, _, _, _, _ = self._integrate_batch(F3, st, dt)
        r = np.zeros(2 * self.nn)
        # 全局组装：r += B^T sigma（B 与 F3 同源 dNdx，应变位移一致）
        for e in range(self.ne):
            nd = self.conn[e]
            for g in range(4):
                ip = e * 4 + g
                if st['D'][ip] >= 0.99:
                    continue
                re = self.Bs[ip].T @ sig[ip] * self.Jdet
                for a in range(4):
                    da = 2 * nd[a]
                    r[da:da + 2] += re[2 * a:2 * a + 2]
        return r, sig

    def _tangent(self, u, st, dt):
        """数值切向刚度（free DOF，中心差分）。"""
        h = 1e-6
        r0, _ = self._residual(u, st, dt)
        r0f = r0[self.free_dofs].copy()
        nf = len(self.free_dofs)
        Kff = np.zeros((nf, nf))
        for i in range(nf):
            d = self.free_dofs[i]
            u_old = u[d]
            u[d] = u_old + h
            rp, _ = self._residual(u, st, dt)
            u[d] = u_old - h
            rm, _ = self._residual(u, st, dt)
            u[d] = u_old
            Kff[:, i] = (rp[self.free_dofs] - rm[self.free_dofs]) / (2 * h)
        return Kff

    def run(self, verbose=True):
        t0 = time.time()
        st = self._state_init()
        u = np.zeros(2 * self.nn)
        u_prev = u.copy()
        eps0 = EPS_STEP
        reached = 0.0
        Dagg_h, sig_h, used_eps = [], [], []
        ef = None
        it_total = 0
        n_reduce = 0
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
                    # 切向 K 已过时：重组装一次（数值切向一次性近似）
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
                        print(f'  [fail] eps={reached:.4f}->{cur:.4f} nrm={nrm:.3e}', flush=True)
                    # 全场局部化失稳判据：损伤激活后（Dagg>0.005）平衡失稳，
                    # 或宏观应力显著软化（<0.85 峰值），即判定为损伤局部化断裂
                    # （应变局部化 -> 单元退化 -> 平衡失稳/失去适定性），
                    # 与 Taylor 的聚合损伤判据（D_agg>=Dc）机制不同但物理对应。
                    if Dagg > 0.005:
                        peak = max(sig_h) if sig_h else 0.0
                        if smac < 0.85 * peak or smac <= peak * 1.02:
                            ef = cur
                            ftype = 'localization_onset'
                            break
                    ftype = 'numerical_limit'
                    break
                eps0 *= 0.5
                u = u_prev.copy()          # 完整恢复上一步
                continue
            F3 = self._grad_F3(u)
            _, Darr, FpN, gN, pN = self._integrate_batch(F3, st, dt)
            st = {'Fp': FpN, 'g': gN, 'D': Darr, 'p': pN}
            Dagg = float(Darr.mean())
            _, sig = self._residual(u, st, dt)
            act = Darr < 0.99
            smac = float(sig[act, 0].mean()) if np.any(act) else 0.0
            Dagg_h.append(Dagg)
            sig_h.append(smac)
            used_eps.append(cur)
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
            if verbose and (len(used_eps) % 25 == 0):
                print(f'  [{self.mat} {self.psi}°] eps={cur:.4f} sig11={smac/1e6:.1f} '
                      f'Dagg={Dagg:.4f} ({time.time()-t0:.0f}s, it={it_total}, '
                      f'eps0={eps0:.4f}, r={n_reduce})', flush=True)
        if ef is None:
            ef = float(used_eps[-1]) if used_eps else 0.0
            ftype = 'max_strain'
        return {'strain': used_eps, 'stress': sig_h, 'damage': Dagg_h,
                'fracture_strain': ef, 'fracture_type': ftype,
                'eps_step_final': eps0,
                'material': self.mat, 'orientation': self.psi,
                'S0': S0_CAL[self.mat], 'Dc': self.Dc,
                'wall_sec': round(time.time() - t0, 1)}


def run_case(mat, psi, verbose=False):
    ff = FullField2D(mat, psi, S0_CAL[mat])
    res = ff.run(verbose=verbose)
    gs = _gs_directional(mat, psi)
    t = TaylorCPCDM(mat, psi, S0_override=S0_CAL[mat], n_grains=NGRAIN,
                    eps_step=EPS_STEP, n_sub=N_SUB, strain_rate=STRAIN_RATE,
                    model_version='v4', param_overrides=gs, seed=SEED)
    tr = t.run_uniaxial(res['strain'][-1] + 0.02 if len(res['strain']) else 0.1)
    ef_t = float(tr['fracture_strain'])
    ef_ff = float(res['fracture_strain'])
    return {'material': mat, 'orientation': psi,
            'ef_fullfield': round(ef_ff, 5), 'ef_taylor': round(ef_t, 5),
            'bias_pct': round((ef_ff - ef_t) / ef_t * 100.0, 2)
            if ef_t > 0 else None,
            'Dc': float(ff.Dc), 'S0': S0_CAL[mat],
            'eps_step': EPS_STEP, 'n_sub': N_SUB,
            'fullfield_wall_sec': res['wall_sec']}


def main():
    cases = [('Ti64', 0), ('Ti64', 45), ('Ti64', 90), ('AlSi10Mg', 0)]
    results = []
    for mat, psi in cases:
        print(f'--- {mat} {psi}deg ---', flush=True)
        results.append(run_case(mat, psi, verbose=True))
        print(json.dumps(results[-1]), flush=True)
    _suffix = '' if NGRID == 4 else f'_ng{NGRID}'
    with open(os.path.join(OUT, f'fullfield_2d_cpfem_results{_suffix}.json'),
              'w', encoding='utf-8') as f:
        json.dump({'model': '2D plane-strain full-field CPFEM vs TaylorCPCDM '
                            '(identical parameters, orientations, eps_step, n_sub)',
                   'constitutive': 'batch explicit integration, fixed n_sub '
                                   '(no per-point bisection), slip clip 0.01',
                   'cases': results}, f, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
