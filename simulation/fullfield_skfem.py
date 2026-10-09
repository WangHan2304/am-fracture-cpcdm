# -*- coding: utf-8 -*-
"""
fullfield_skfem.py — 用第三方有限元引擎 scikit-fem 驱动 CP-CDM 耦合损伤平面应变全场
=====================================================================================
目的（论文 Item 1）：把"全场耦合损伤参考"的**离散化/装配/线性求解**交给独立的第三
方有限元库 scikit-fem（MeshQuad + ElementVector(Q4) + Basis 高斯梯度 + BilinearForm
装配切向 + LinearForm 装配内力 + 稀疏求解 + interpolate 恢复应变），本构物理（晶体塑
性滑移幂律 + Voce 硬化 + AM 修正 Lemaitre 损伤 + 断裂判据）与自研 TaylorCPCDM 逐字一致
（复制 _integrate_batch）。与自研 Q4 手搓装配相比，唯一变化即"由外部 FE 引擎负责求解"。

与 fullfield_2d_cpfem.py 同构：相同材料、相同取向（种子 42、16 元=16 晶粒、每元 4 高
斯点）、相同 eps_step / n_sub / strain_rate、相同断裂判据（全体高斯点平均 D>=Dc，线性
插值）。输出 output/fullfield_skfem_results.json，含 ef_skfem / ef_taylor / bias_pct。
"""
import json
import os
import sys
import time

import numpy as np

# scikit-fem 第三方引擎
from skfem import (MeshQuad, ElementQuad1, ElementVector, Basis,
                   asm, LinearForm, BilinearForm)
from skfem.helpers import inner, grad
import scipy.sparse as sp
import scipy.sparse.linalg as spl

# 复用自研本构对象（仅取材料/取向/参数，不用其求解器）
SIM = os.environ.get("SIMHOME", os.path.expanduser("~/sim"))
sys.path.insert(0, os.path.join(SIM, "src"))
from materials import EXPERIMENTAL            # noqa: E402
from taylor_cpcdm import TaylorCPCDM          # noqa: E402
from am_correction_v4 import xi_eff            # noqa: E402

STRAIN_RATE = 1e-3
EPS_STEP = 2e-3
N_SUB = 30
KG = 0.15
XI0 = {'Ti64': 3.5, '316L': 2.5, 'AlSi10Mg': 1.5}
S0_CAL = {'Ti64': 0.32, '316L': 3.2, 'AlSi10Mg': 0.20}
OUT = os.path.join(SIM, 'output')
NGRID = int(os.environ.get('FF_NGRID', '4'))
NGRAIN = NGRID * NGRID
SEED = 42
NU = 0.3
NEWTON_MAX = 30
RES_TOL = 100.0
SLIP_CLIP = 0.01
DGAMMA_SUM_MAX = 0.10


def _gs_directional(mat, psi):
    xi0 = XI0[mat]
    ratio = xi_eff(xi0, psi) / xi0   # xi_eff(psi)/xi0, single source (am_correction_v4)
    h = 1.0 - KG * (1.0 - ratio)
    if mat == 'Ti64':
        base = {'gs_basal': 700e6, 'gs_prism': 720e6, 'gs_pyr': 800e6}
    else:
        base = {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat]}
    return {k: v * h for k, v in base.items()}


class SKFEMFullField:
    """scikit-fem 驱动的平面应变全场 CP-CDM。"""

    def __init__(self, mat, psi, S0):
        self.mat, self.psi = mat, psi
        gs = _gs_directional(mat, psi)
        t = TaylorCPCDM(mat, psi, S0_override=S0, n_grains=NGRAIN,
                        eps_step=EPS_STEP, n_sub=N_SUB, strain_rate=STRAIN_RATE,
                        model_version='v4', param_overrides=gs, seed=SEED)
        self.t = t
        self.exp = EXPERIMENTAL[mat]
        self.Dc = self.exp['Dc']
        self.max_strain = (self.exp['ef'][psi] * 2.0 + 0.01
                           if mat != '316L' else self.exp['ef'][psi] * 1.6 + 0.02)
        # --- scikit-fem 网格 / 基函数 / 装配 ---
        self.mesh = MeshQuad.init_tensor(np.linspace(0, 1, NGRID + 1),
                                         np.linspace(0, 1, NGRID + 1))
        self.ev = ElementVector(ElementQuad1())
        self.ib = Basis(self.mesh, self.ev, intorder=2)
        self.ne = self.ib.nelems
        self.nq = 4
        self.nip = self.ne * self.nq
        # 每积分点取向 = 所属单元晶粒取向（与自研一致：元=粒）
        self.Rs = np.zeros((self.nip, 3, 3))
        for e in range(self.ne):
            self.Rs[e * self.nq:(e + 1) * self.nq] = t.R[e]
        # --- 本构参数（来自 TaylorCPCDM，逐字与自研相同）---
        self.P = {'P0': t.P0, 'C4': t.stiffness, 'Nslip': t.N_slip,
                  'gdot0': t.params['gamma_dot_0'], 'n_rate': t.params['n_rate'],
                  'h0': t.params['h0'], 'a_exp': t.params['a'],
                  'q_lat': t.params['q_lat'], 'p_D': t.params['p_D'],
                  's_dam': t.params['s_damage'], 'S_eff': t.S_eff * 1e6,
                  'f_AM': t.f_AM, 'g_sat': t.g_sat_vec, 'E': self.exp['E'],
                  'N_SUB': N_SUB, 'R': self.Rs, 'g0': t.g0_vec,
                  'D0': t.params['D0']}
        Q = np.full((t.N_slip, t.N_slip), t.params['q_lat'])
        np.fill_diagonal(Q, 1.0)
        self.P['Q'] = Q
        # --- 边界条件（元→节点 dof，(component,node) 索引）---
        ND = self.ib.nodal_dofs
        xn, yn = self.mesh.p[0], self.mesh.p[1]
        self.left = xn <= 1e-12
        self.bottom = yn <= 1e-12
        self.right = xn >= 1.0 - 1e-12
        self.fixed = np.concatenate([ND[0, self.left], ND[1, self.bottom]])
        self.controlled = ND[0, self.right]
        self.free = self.ib.complement_dofs(
            np.concatenate([self.fixed, self.controlled]))
        self._elastic_precond()

    def _elastic_precond(self):
        """弹性切向刚度（仅用于 Newton 迭代方向，常量、一次装配）。"""
        E = self.exp['E']
        lam = E * NU / ((1 + NU) * (1 - 2 * NU))
        mu = E / (2 * (1 + NU))
        ib = self.ib

        @BilinearForm
        def a(u, v, w):
            gu = grad(u)
            gv = grad(v)
            eu = 0.5 * (gu + np.swapaxes(gu, 0, 1))
            evv = 0.5 * (gv + np.swapaxes(gv, 0, 1))
            s11 = (lam + 2 * mu) * eu[0, 0] + lam * eu[1, 1]
            s22 = lam * eu[0, 0] + (lam + 2 * mu) * eu[1, 1]
            s12 = mu * (eu[0, 1] + eu[1, 0])
            sig = np.zeros(eu.shape)
            sig[0, 0] = s11; sig[1, 1] = s22
            sig[0, 1] = s12; sig[1, 0] = s12
            return inner(sig, evv)
        K = sp.csr_matrix(asm(a, ib))
        Kff = K[self.free][:, self.free].tocsc()
        self.Kff_lu = spl.splu(Kff)
        self.Kfull = K

    # ---- F3 由 scikit-fem 位移梯度恢复 ----
    def _grad_F3(self, u):
        gf = self.ib.interpolate(u).grad          # (2,2,nelem,nq)
        F3 = np.zeros((self.nip, 3, 3))
        F3[:, 2, 2] = 1.0
        for e in range(self.ne):
            for q in range(self.nq):
                H = gf[:, :, e, q]
                p = e * self.nq + q
                F3[p, 0, 0] = 1.0 + H[0, 0]
                F3[p, 0, 1] = H[0, 1]
                F3[p, 1, 0] = H[1, 0]
                F3[p, 1, 1] = 1.0 + H[1, 1]
        return F3

    # ---- 本构（复制自研 _integrate_batch，逐字相同，仅数据源换 P）----
    def _integrate_batch(self, F3, st, dt):
        P = self.P
        N_SUB_ = P['N_SUB']
        Fp = st['Fp'].copy(); g = st['g'].copy()
        D = st['D'].copy(); p = st['p'].copy()
        R = P['R']; C4 = P['C4']; P0 = P['P0']
        dt_sub = dt / N_SUB_
        Rt = np.swapaxes(R, 1, 2)
        for _ in range(N_SUB_):
            Fc = np.einsum('nij,njk,nkl->nil', Rt, F3, R)
            Fp_inv = np.linalg.inv(Fp)
            Fe = np.einsum('nij,njk->nik', Fc, Fp_inv)
            Ee = 0.5 * (np.einsum('nij,njk->nik', np.swapaxes(Fe, 1, 2), Fe) - np.eye(3))
            S = np.einsum('ijkl,nkl->nij', C4, Ee)
            tau = np.einsum('nij,pij->np', S, P0)
            tau_abs = np.abs(tau)
            denom = np.maximum(g, 1e-10)
            ratio = np.clip(np.where(g > 1e-6, tau_abs / denom, 0.0), 0.0, 10.0)
            gamma_dot = P['gdot0'] * (ratio ** P['n_rate']) * np.sign(tau)
            dgamma = np.clip(gamma_dot * dt_sub, -SLIP_CLIP, SLIP_CLIP)
            dg_tot = np.sum(np.abs(dgamma), axis=1)
            scale = np.minimum(1.0, DGAMMA_SUM_MAX / np.maximum(dg_tot, 1e-30))
            dgamma = dgamma * scale[:, None]
            Lp = np.einsum('np,pij->nij', dgamma, P0)
            Fp = (np.eye(3) + Lp) @ Fp
            dgd_dt = np.abs(dgamma) / dt_sub
            ratio_h = np.clip(1.0 - g / P['g_sat'][None, :], 0.0, None)
            dg = P['h0'] * dgd_dt * (ratio_h ** P['a_exp'])
            dg = dg @ P['Q'].T
            g = np.maximum(g + dg * dt_sub, 1e-3)
            dp = np.sqrt(2.0 / 3.0) * np.sum(np.abs(dgamma), axis=1)
            p = p + dp
            act = (p > P['p_D']) & (dp > 0)
            if np.any(act):
                Fe_n = np.einsum('nij,njk->nik', Fc, np.linalg.inv(Fp))
                Ee_n = 0.5 * (np.einsum('nij,njk->nik', np.swapaxes(Fe_n, 1, 2), Fe_n) - np.eye(3))
                S_n = np.einsum('ijkl,nkl->nij', C4, Ee_n)
                dev = S_n - np.einsum('nii->n', S_n)[:, None, None] / 3.0 * np.eye(3)
                vm = np.sqrt(1.5 * np.einsum('nij,nij->n', dev, dev))
                Y = vm * vm / (2.0 * P['E'])
                r = Y / max(P['S_eff'], 1e-10)
                dD = (r ** P['s_dam']) * dp * P['f_AM']
                dD = np.clip(dD, 0.0, 0.02)
                D = np.minimum(0.99, D + dD)
        Fp_inv = np.linalg.inv(Fp)
        Fe = np.einsum('nij,njk->nik', np.einsum('nij,njk,nkl->nil', Rt, F3, R), Fp_inv)
        Ee = 0.5 * (np.einsum('nij,njk->nik', np.swapaxes(Fe, 1, 2), Fe) - np.eye(3))
        S = np.einsum('ijkl,nkl->nij', C4, Ee)
        J = np.linalg.det(Fe)
        sc = (1.0 - D)[:, None, None] / np.maximum(J, 1e-10)[:, None, None] \
            * np.einsum('nij,njk->nik', Fe, S)
        sm = np.einsum('nij,njk,nlk->nil', R, sc, R)
        sig2 = np.stack([sm[:, 0, 0], sm[:, 1, 1], sm[:, 0, 1]], axis=1)
        return sig2, D, Fp, g, p

    def _sigma_field(self, sig2):
        sig = np.zeros((2, 2, self.ne, self.nq))
        for e in range(self.ne):
            for q in range(self.nq):
                p = e * self.nq + q
                s11, s22, s12 = sig2[p]
                sig[0, 0, e, q] = s11
                sig[1, 1, e, q] = s22
                sig[0, 1, e, q] = s12
                sig[1, 0, e, q] = s12
        return sig

    def _residual(self, u, st, dt):
        ib = self.ib

        @LinearForm
        def internal(v, w):
            return inner(w.sigma, grad(v))
        F3 = self._grad_F3(u)
        sig2, Darr, _, _, _ = self._integrate_batch(F3, st, dt)
        sigma = self._sigma_field(sig2)
        r = asm(internal, ib, sigma=sigma)
        return r, sig2, Darr

    def run(self, verbose=True):
        t0 = time.time()
        P = self.P
        st = {'Fp': np.tile(np.eye(3)[None], (self.nip, 1, 1)),
              'g': np.tile(P['g0'][None, :], (self.nip, 1)),
              'D': np.full(self.nip, P['D0']),
              'p': np.zeros(self.nip)}
        u = np.zeros(self.ib.N)
        u_prev = u.copy()
        eps0 = EPS_STEP
        reached = 0.0
        used_eps, sig_h, Dagg_h = [], [], []
        ef, smac, Dagg = None, 0.0, 0.0
        n_reduce = 0
        while reached < self.max_strain - 1e-12:
            cur = min(reached + eps0, self.max_strain)
            dt = eps0 / STRAIN_RATE
            u = u_prev.copy()
            for d in self.controlled:
                u[d] = cur
            st0 = {k: v.copy() for k, v in st.items()}
            conv = False
            nrm = np.inf
            for it in range(NEWTON_MAX):
                r, _, _ = self._residual(u, st0, dt)
                r_free = r[self.free]
                nrm = float(np.max(np.abs(r_free)))
                if nrm < RES_TOL:
                    conv = True
                    break
                du = self.Kff_lu.solve(-r_free)
                alpha, improved = 1.0, False
                for _ in range(6):
                    ut = u.copy()
                    ut[self.free] = u[self.free] + alpha * du
                    rt, _, _ = self._residual(ut, st0, dt)
                    nt = float(np.max(np.abs(rt[self.free])))
                    if nt < RES_TOL:
                        u = ut; conv = True; improved = True; break
                    if nt < nrm:
                        u = ut; nrm = nt; improved = True; break
                    alpha *= 0.5
                if conv:
                    break
                if not improved:
                    break
            if not conv:
                n_reduce += 1
                if eps0 < 6.25e-5 or n_reduce > 20:
                    if Dagg > 0.005 and ef is None:
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
            _, sig, _ = self._residual(u, st, dt)
            act = Darr < 0.99
            smac = float(sig[act, 0].mean()) if np.any(act) else 0.0
            used_eps.append(cur); sig_h.append(smac); Dagg_h.append(Dagg)
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
            if verbose and len(used_eps) % 25 == 0:
                print(f'  [{self.mat} {self.psi}] eps={cur:.4f} '
                      f'sig11={smac/1e6:.1f} Dagg={Dagg:.4f} '
                      f'({time.time()-t0:.0f}s r={n_reduce})', flush=True)
        if ef is None:
            ef = float(used_eps[-1]) if used_eps else 0.0
            ftype = 'max_strain'
        return {'fracture_strain': float(ef), 'fracture_type': ftype,
                'material': self.mat, 'orientation': self.psi,
                'S0': S0_CAL[self.mat], 'Dc': float(self.Dc),
                'wall_sec': round(time.time() - t0, 1),
                'strain': used_eps, 'stress': sig_h, 'damage': Dagg_h}


def run_case(mat, psi, verbose=False):
    ff = SKFEMFullField(mat, psi, S0_CAL[mat])
    res = ff.run(verbose=verbose)
    gs = _gs_directional(mat, psi)
    t = TaylorCPCDM(mat, psi, S0_override=S0_CAL[mat], n_grains=NGRAIN,
                    eps_step=EPS_STEP, n_sub=N_SUB, strain_rate=STRAIN_RATE,
                    model_version='v4', param_overrides=gs, seed=SEED)
    tr = t.run_uniaxial(res['strain'][-1] + 0.02 if res['strain'] else 0.1)
    ef_t = float(tr['fracture_strain'])
    ef_ff = float(res['fracture_strain'])
    return {'material': mat, 'orientation': psi,
            'ef_skfem': round(ef_ff, 5), 'ef_taylor': round(ef_t, 5),
            'bias_pct': round((ef_ff - ef_t) / ef_t * 100.0, 2) if ef_t > 0 else None,
            'fracture_type': res['fracture_type'],
            'Dc': float(ff.Dc), 'S0': S0_CAL[mat],
            'eps_step': EPS_STEP, 'n_sub': N_SUB,
            'skfem_wall_sec': res['wall_sec']}


def main():
    import skfem
    cases = [('Ti64', 0), ('Ti64', 45), ('Ti64', 90), ('AlSi10Mg', 0)]
    results = []
    for mat, psi in cases:
        print(f'--- {mat} {psi}deg ---', flush=True)
        results.append(run_case(mat, psi, verbose=True))
        print(json.dumps(results[-1]), flush=True)
    out = {'engine': 'scikit-fem ' + skfem.__version__,
           'model': 'Third-party FE (scikit-fem) plane-strain full-field CP-CDM '
                    'vs TaylorCPCDM (identical constitutive/parameters/orientations)',
           'constitutive': 'batch explicit integration copied verbatim from '
                           'fullfield_2d_cpfem._integrate_batch (fixed n_sub, slip clip)',
           'cases': results}
    _suffix = '' if NGRID == 4 else f'_ng{NGRID}'
    with open(os.path.join(OUT, f'fullfield_skfem_results{_suffix}.json'),
              'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
