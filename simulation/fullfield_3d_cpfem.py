# -*- coding: utf-8 -*-
"""
fullfield_3d_cpfem.py — 3D 全场 CPFEM（耦合损伤，numpy 批量向量化，最小规模）
=============================================================================
与 TaylorCPCDM 同构对比（同 2D 协议）：相同材料参数（S0、gs(psi)、硬化、
损伤律、Dc）、相同取向（同种子 42 取前 N 个）、相同应变步/子步分辨率
（eps_step、n_sub 固定）。唯一区别：Taylor 等应变假设 vs 全场平衡。

网格：ng x ng x ng = 27 个 hexa8 单元（每晶粒 1 单元，3x3x3），
每单元 2x2x2=8 高斯点；单轴拉伸（x 方向位移控制，y/z 自由——单轴应力态）。
本构：与 2D 相同的批量显式积分（F 3x3 全量，Cauchy 应力 3x3 -> 6 分量）。
断裂判据：D_agg>=Dc（聚合）或 局部化失稳（损伤激活后平衡失稳/软化）。
输出: output/fullfield_3d_cpfem_results.json
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
EPS_STEP = 2e-3
N_SUB = 30
KG = 0.15
XI0 = {'Ti64': 3.5, '316L': 2.5, 'AlSi10Mg': 1.5}
S0_CAL = {'Ti64': 0.32, '316L': 3.2, 'AlSi10Mg': 0.20}
OUT = os.path.join(os.path.dirname(__file__), 'output')
NG = 3                      # 3x3x3 = 27 grains
NGRAIN = NG ** 3
SEED = 42
NU = 0.3
NEWTON_MAX = 25
RES_TOL = 100.0
SLIP_CLIP = 0.01
DGAMMA_SUM_MAX = 0.10

# 8-node trilinear hexahedron, 2x2x2 Gauss
GAUSS_PTS = [(gx, gy, gz)
             for gx in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
             for gy in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
             for gz in (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))]

N8 = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)


def _dN3(eps, eta, zeta):
    """dN/d(xi,eta,zeta) for 8-node hexa (8x3)."""
    d = np.zeros((8, 3))
    for a in range(8):
        xi = N8[a, 0] * (1 + eps * N8[a, 0]) * (1 + eta * N8[a, 1]) * (1 + zeta * N8[a, 2])
        # 用标准公式
        pass
    d = np.zeros((8, 3))
    for a in range(8):
        ex = 1 + eps * N8[a, 0]
        ey = 1 + eta * N8[a, 1]
        ez = 1 + zeta * N8[a, 2]
        d[a, 0] = 0.125 * N8[a, 0] * ey * ez
        d[a, 1] = 0.125 * N8[a, 1] * ex * ez
        d[a, 2] = 0.125 * N8[a, 2] * ex * ey
    return d


def _gs_directional(mat, psi):
    xi0 = XI0[mat]
    ratio = xi_eff(xi0, psi) / xi0   # xi_eff(psi)/xi0, single source (am_correction_v4)
    h = 1.0 - KG * (1.0 - ratio)
    if mat == 'Ti64':
        base = {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}
    else:
        base = {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}
    return {k: v * h for k, v in base.items()}


class FullField3D:
    def __init__(self, mat, psi, S0, eps_step=EPS_STEP, n_sub=N_SUB,
                 ng=NG, max_strain=None):
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
        self.ng = ng
        self.nn = (ng + 1) ** 3
        self.ne = ng ** 3
        self.nip = self.ne * 8
        # nodes (ng+1)^3
        self.x = np.array([[ix / ng, iy / ng, iz / ng]
                           for iz in range(ng + 1)
                           for iy in range(ng + 1)
                           for ix in range(ng + 1)], dtype=float)
        # connectivity: element (ex,ey,ez)
        self.conn = []
        for ez in range(ng):
            for ey in range(ng):
                for ex in range(ng):
                    i0 = ez * (ng + 1) ** 2 + ey * (ng + 1) + ex
                    self.conn.append([i0,
                                      i0 + 1,
                                      i0 + (ng + 2),
                                      i0 + (ng + 1),
                                      i0 + (ng + 1) ** 2,
                                      i0 + (ng + 1) ** 2 + 1,
                                      i0 + (ng + 1) ** 2 + (ng + 2),
                                      i0 + (ng + 1) ** 2 + (ng + 1)])
        self.conn = np.array(self.conn)
        # per-element Jacobian at reference node (linear mapping: affine hexa)
        self.Jdet = 1.0 / (ng ** 3)
        self.jinv = np.eye(3) * ng
        # B matrices per ip (6,24), grain orientation R per ip
        self.Bs = np.zeros((self.nip, 6, 24))
        self.Rs = np.zeros((self.nip, 3, 3))
        for e in range(self.ne):
            nd = self.conn[e]
            gid = e                      # 每单元一晶粒
            for gi, (ep, et, ez) in enumerate(GAUSS_PTS):
                ip = e * 8 + gi
                dNd = _dN3(ep, et, ez)   # 8x3
                dNdx = dNd @ self.jinv   # 8x3
                B = np.zeros((6, 24))
                for a in range(8):
                    for j in range(3):
                        B[j, 3 * a + j] = dNdx[a, j]
                    B[3, 3 * a] = dNdx[a, 1]
                    B[3, 3 * a + 1] = dNdx[a, 0]
                    B[4, 3 * a + 1] = dNdx[a, 2]
                    B[4, 3 * a + 2] = dNdx[a, 1]
                    B[5, 3 * a] = dNdx[a, 2]
                    B[5, 3 * a + 2] = dNdx[a, 0]
                self.Bs[ip] = B
                self.Rs[ip] = self.t.R[gid]
        self._dofmap()
        # 批量本构参数（同 2D）
        self.P0 = self.t.P0
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
        self.Q = np.full((self.Nslip, self.Nslip), self.q_lat)
        np.fill_diagonal(self.Q, 1.0)

    def _dofmap(self):
        ng = self.ng
        nn3 = (ng + 1) ** 3
        fixed = set()
        controlled = set()
        for iz in range(ng + 1):
            for iy in range(ng + 1):
                for ix in range(ng + 1):
                    node = iz * (ng + 1) ** 2 + iy * (ng + 1) + ix
                    if ix == 0:
                        fixed.add(3 * node)          # x=0 face: fix x
                    if ix == ng:
                        controlled.add(3 * node)     # x=L face: control x
                    if ix == 0 and iy == 0 and iz == 0:
                        fixed.add(3 * node + 1); fixed.add(3 * node + 2)
                    if ix == 0 and iy == ng and iz == 0:
                        fixed.add(3 * node + 2)
        fixed |= controlled
        self.fixed_dofs = sorted(fixed)
        self.controlled_dofs = sorted(controlled)
        self.free_dofs = np.array([d for d in range(3 * nn3)
                                   if d not in fixed], dtype=int)

    def _state_init(self):
        return {'Fp': np.tile(np.eye(3)[None, :, :], (self.nip, 1, 1)),
                'g': np.tile(self.t.g0_vec[None, :], (self.nip, 1)),
                'D': np.full(self.nip, self.t.params['D0']),
                'p': np.zeros(self.nip)}

    def _grad_F3(self, u):
        F3 = np.tile(np.eye(3)[None, :, :], (self.nip, 1, 1))
        for e in range(self.ne):
            nd = self.conn[e]
            ue = np.concatenate([u[3 * i:3 * i + 3] for i in nd])
            for gi, (ep, et, ez) in enumerate(GAUSS_PTS):
                ip = e * 8 + gi
                dNdx = _dN3(ep, et, ez) @ self.jinv   # 8x3
                H = np.zeros((3, 3))
                for a in range(8):
                    for j in range(3):
                        H[j, :] += dNdx[a, :] * ue[3 * a + j]
                F3[ip] = np.eye(3) + H
        return F3

    def _integrate_batch(self, F3, st, dt):
        nip = self.nip
        Fp = st['Fp'].copy()
        g = st['g'].copy()
        D = st['D'].copy()
        p = st['p'].copy()
        R = self.Rs
        dt_sub = dt / N_SUB
        Rt = np.swapaxes(R, 1, 2)
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
            act = (p > self.p_D) & (dp > 0)
            if np.any(act):
                Fe_n = np.einsum('nij,njk->nik', Fc, np.linalg.inv(Fp))
                Ee_n = 0.5 * (np.einsum('nij,njk->nik',
                                        np.swapaxes(Fe_n, 1, 2), Fe_n) - np.eye(3))
                S_n = np.einsum('ijkl,nkl->nij', self.C4, Ee_n)
                dev = S_n - np.einsum('nii->n', S_n)[:, None, None] / 3.0 * np.eye(3)
                vm = np.sqrt(1.5 * np.einsum('nij,nij->n', dev, dev))
                Y = vm * vm / (2.0 * self.Emod)
                r = Y / max(self.S_eff, 1e-10)
                dD = (r ** self.s_dam) * dp * self.f_AM
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
        return sig6, D, Fp, g, p

    def _residual(self, u, st, dt):
        F3 = self._grad_F3(u)
        sig, _, _, _, _ = self._integrate_batch(F3, st, dt)
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

    def _tangent(self, u, st, dt):
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
        u = np.zeros(3 * self.nn)
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
            if verbose and (len(used_eps) % 20 == 0):
                print(f'  [{self.mat} {self.psi}° 3D] eps={cur:.4f} '
                      f'sig11={smac/1e6:.1f} Dagg={Dagg:.4f} '
                      f'({time.time()-t0:.0f}s, it={it_total}, '
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
    ff = FullField3D(mat, psi, S0_CAL[mat])
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
    cases = [('Ti64', 0)]
    results = []
    for mat, psi in cases:
        print(f'--- 3D {mat} {psi}deg ---', flush=True)
        results.append(run_case(mat, psi, verbose=True))
    with io_open_utf8(os.path.join(OUT, 'fullfield_3d_cpfem_results.json')) as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, indent=1), flush=True)


def io_open_utf8(path):
    return open(path, 'w', encoding='utf-8', newline='')


if __name__ == '__main__':
    main()
