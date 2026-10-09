"""
Taylor 多晶 CP-CDM 模型
=========================
Taylor 等应变假设：所有晶粒承受相同的宏观变形梯度。
应力通过取向平均得到，修复单晶模型中的 UTS 方向依赖性。

核心公式：
- 宏观F_total → 各晶粒 F_total^i = R_i^T @ F_total_macro @ R_i
- 晶粒内积分（晶体坐标系一致）
- Cauchy应力旋转回宏观：σ_macro^i = R_i @ σ_crystal^i @ R_i^T
- 平均：σ̄ = (1/N) Σ σ_macro^i, D̄ = (1/N) Σ D^i
"""

import numpy as np


class TaylorCPCDM:
    """Taylor多晶CP-CDM（基于晶体塑性显式积分）"""

    def __init__(self, material_key, orientation_deg=0, n_grains=30, S0_override=None,
                 eps_step=5e-4, n_sub=30, strain_rate=1e-3, seed=42,
                 aniso_override=None, model_version='v3', beta4_override=None,
                 force_homogeneous=False, param_overrides=None,
                 damage_model='lem'):
        """
        Args:
            material_key: 'Ti64', '316L', 'AlSi10Mg'
            orientation_deg: 宏观加载相对于打印方向的角度（0/45/90）
            n_grains: 晶粒数量（25-50为收敛性良好平衡点）
            S0_override: 指定S0（None则从params获取）
            eps_step: 应变步长
            n_sub: 每步步数
            strain_rate: 应变率（s⁻¹）
            seed: 随机取向种子
            model_version: 'v3'（旧版，向后兼容）或 'v4'（重构版：
                S≡S0 常数、取向依赖仅经投影描述符进入 f_AM、Y=σ̃²/2E）
            beta4_override: v4 专用，界面耦合系数 β4 覆盖值
            force_homogeneous: v4 专用，f_AM≡1（Lemaitre 基线对照）
            param_overrides: 任意参数键值覆盖（UQ/敏感性扰动用）
        """
        self.material_key = material_key
        self.orientation_deg = orientation_deg
        self.n_grains = n_grains
        self.eps_step = eps_step
        self.n_sub = n_sub
        self.strain_rate = strain_rate
        self.model_version = model_version
        self.use_v4 = (model_version == 'v4')
        self.force_homogeneous = force_homogeneous

        # 延迟导入避免循环依赖
        from materials import (
            MATERIAL_DATABASE,
            get_orientation_adjusted_params, build_stiffness_tensor,
            get_slip_systems_FCC, get_slip_systems_HCP, EXPERIMENTAL
        )

        # 加载材料参数
        if self.use_v4:
            # v4：原始微结构描述符（不做旧版的取向临时缩放），
            # D0 统一取自实验数据库（单一来源）
            self.params = dict(MATERIAL_DATABASE[material_key])
            self.params['D0'] = EXPERIMENTAL[material_key]['D0']
            if beta4_override is not None:
                self.params['beta4'] = beta4_override
        else:
            self.params = get_orientation_adjusted_params(material_key, orientation_deg)
        if S0_override is not None:
            self.params['S0'] = S0_override
        if (not self.use_v4) and (aniso_override is not None):
            self.params['_f_AM_aniso'] = aniso_override
        if param_overrides:
            # 最末应用：用于敏感性/不确定性扰动（可含 S0、beta4、_w_iso、theta_tex 等）
            self.params.update(param_overrides)

        # GTN 基线参数：f_c=0.15 文献固定（临界孔隙率）；
        # A（应变控制常数形核率）为每材料唯一拟合参数
        self.damage_model = damage_model
        self.gtn_A  = self.params.get('A', 0.40)
        self.gtn_fc = self.params.get('f_c', 0.15)

        # 弹性常数（晶体坐标系 rank-4 → 使用 Voigt 运算）
        self.stiffness = build_stiffness_tensor(self.params)

        # 实验参考
        self.exp = EXPERIMENTAL[material_key]

        # 滑移系（晶体坐标系）
        cs = self.params['crystal_structure']
        if cs == 'FCC':
            slip_systems = get_slip_systems_FCC()
        elif cs == 'HCP':
            slip_systems = get_slip_systems_HCP()
        else:
            raise ValueError(f"Unknown crystal structure: {cs}")

        self.N_slip = len(slip_systems)

        # Schmid张量 P0^(α) = s^(α) ⊗ m^(α) （晶体坐标系）
        s_dir = np.array([s[1] for s in slip_systems])      # (N_slip, 3)
        m_normal = np.array([s[0] for s in slip_systems])    # (N_slip, 3)
        self.P0 = np.einsum('pi,pj->pij', s_dir, m_normal)   # (N_slip, 3, 3)

        # 生成晶粒取向（先于此，v4 的 θ(ψ) 由取向组直接计算）
        np.random.seed(seed)
        self.R = self._generate_orientations()  # (n_grains, 3, 3), macro→crystal

        # AM修正（宏观参数，所有晶粒共享）
        if self.use_v4:
            from am_correction_v4 import factors_v4
            self.theta_bar = self._mean_schmid(orientation_deg)
            self.f_AM, self.am_factors = factors_v4(
                self.params, orientation_deg, self.theta_bar)
            if self.force_homogeneous:
                self.f_AM = 1.0   # Lemaitre 基线：不做任何 AM 修正
            # 可选取向乘数修正（正式模型 v6 与 LOOCV 协议用，默认关闭，向后兼容）：
            # 1) aniso_multiplier: dict {ψ: 乘数}，每个取向一个校准乘数（v6 正典协议，
            #    0°≡1，45°/90° 校准）——与 f_AM 中的物理投影叠加；
            # 2) aniso_amp: 标量，单参数取向修正 f_AM *= 1 + aniso_amp·sin²ψ（LOOCV 协议，
            #    形式与 λ(ψ) 弱界面横观各向同性投影同源）。
            aniso_multi = self.params.get('aniso_multiplier', None)
            if aniso_multi is not None:
                self.f_AM *= float(aniso_multi.get(orientation_deg, 1.0))
                if self.am_factors is not None:
                    self.am_factors['aniso_multiplier'] = dict(aniso_multi)
            aniso_amp = self.params.get('aniso_amp', 0.0)
            if aniso_amp != 0.0:
                self.f_AM *= (1.0 + aniso_amp * np.sin(np.radians(orientation_deg)) ** 2)
                if self.am_factors is not None:
                    self.am_factors['aniso_amp'] = float(aniso_amp)
            self.S_eff = float(self.params['S0'])  # v4: S ≡ S0（常数）
        else:
            self.theta_bar = None
            self.am_factors = None
            self.f_AM, self.S_eff = self._compute_am_correction()

        # 初始CRSS
        self.g0_vec = self._init_crss()
        # 饱和CRSS（HCP 按 family 区分，FCC 标量广播）
        self.g_sat_vec = self._init_gsat()

    # ============================================================
    # 初始化助手
    # ============================================================
    def _compute_am_correction(self):
        """AM修正（等价于 experiment_v2.am_correction）"""
        p = self.params
        LAMBDA_REF = 50.0
        lam = p['lambda_mp'] / LAMBDA_REF

        g1 = 1.0 + p['alpha1'] * lam ** p['n1']
        g2 = 1.0 + p['alpha2'] * max(p['xi_grain'] - 1.0, 0.0) ** p['n2']
        S_eff = p['S0'] * g1 * g2

        f_phi   = 1.0 + p['beta1'] * (p['phi'] / max(p['phi_crit'], 1e-10)) ** p['m1']
        f_theta = 1.0 + p['beta2'] * (1.0 - p['theta_tex']) ** p['m2']
        f_D0    = np.exp(min(p['beta3'] * p['D0'], 50.0))
        f_lam   = 1.0 + p['beta4'] * lam ** p['m3']
        f_xi    = 1.0 + p['beta5'] * (1.0 / max(p['xi_grain'], 1e-10) - 1.0)

        f_AM = f_phi * f_theta * f_D0 * f_lam * f_xi
        f_AM *= p.get('_f_AM_aniso', 1.0)
        return f_AM, S_eff

    def _init_crss(self):
        if 'g0' in self.params:
            return np.full(self.N_slip, self.params['g0'], dtype=float)
        else:
            g = np.zeros(self.N_slip)
            g[0:3]  = self.params['g0_basal']
            g[3:6]  = self.params['g0_prism']
            g[6:12] = self.params['g0_pyr']
            return g

    def _init_gsat(self):
        """饱和 CRSS 向量（与滑移系一一对应）。
        FCC 用标量 gs 广播；HCP 按 basal/prism/pyr 各自饱和值，
        避免旧实现把 prisma/pyr 也强行按 700 MPa 收敛导致的失真。"""
        p = self.params
        if 'gs' in p:
            return np.full(self.N_slip, p['gs'], dtype=float)
        g = np.zeros(self.N_slip)
        g[0:3]  = p.get('gs_basal', 700e6)
        g[3:6]  = p.get('gs_prism', 720e6)
        g[6:12] = p.get('gs_pyr', 800e6)
        return g

    @staticmethod
    def _draw_rotation(rng, theta_tex):
        """按织构律采样一个旋转矩阵（macro→crystal）。

        rng 可为 np.random 模块或 RandomState 实例；theta_tex 为
        [001] 纤维织构概率，其余为 Haar 均匀随机取向。
        """
        if rng.random_sample() < theta_tex:
            # [001]纤维织构（z轴=打印方向）：绕z轴随机旋转
            phi = 2.0 * np.pi * rng.random_sample()
            c, s = np.cos(phi), np.sin(phi)
            return np.array([[c, -s, 0.0],
                             [s,  c, 0.0],
                             [0.0, 0.0, 1.0]])
        # 均匀随机取向（随机四元数法→旋转矩阵）
        u1, u2, u3 = rng.random_sample(3)
        qw = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
        qx = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
        qy = np.sqrt(u1)       * np.sin(2.0 * np.pi * u3)
        qz = np.sqrt(u1)       * np.cos(2.0 * np.pi * u3)
        return np.array([
            [1.0 - 2.0*(qy**2 + qz**2),      2.0*(qx*qy - qz*qw),      2.0*(qx*qz + qy*qw)],
            [    2.0*(qx*qy + qz*qw),  1.0 - 2.0*(qx**2 + qz**2),      2.0*(qy*qz - qx*qw)],
            [    2.0*(qx*qz - qy*qw),      2.0*(qy*qz + qx*qw),  1.0 - 2.0*(qx**2 + qy**2)]
        ])

    def _generate_orientations(self):
        """随机旋转矩阵（Haar测度 SO(3) 均匀采样 + 织构偏向）"""
        theta_tex = self.params.get('theta_tex', 0.0)
        return np.stack([self._draw_rotation(np.random, theta_tex)
                         for _ in range(self.n_grains)])

    def _mean_schmid(self, psi_deg, n_mc=2000):
        """取向分布下 ψ 加载方向的平均最大 Schmid 因子 θ(ψ)。

        与 _generate_orientations 共享同一织构生成律（_draw_rotation），
        但用独立大样本（n_mc=2000，固定种子）估计，消除有限晶粒组的
        蒙特卡洛噪声；θ(ψ) 与 n_grains 解耦、确定可复现、不含拟合参数。
        """
        theta_tex = self.params.get('theta_tex', 0.0)
        rng = np.random.RandomState(1234567)
        u = np.array([np.sin(np.radians(psi_deg)), 0.0,
                      np.cos(np.radians(psi_deg))])
        P0 = self.P0
        tot = 0.0
        for _ in range(n_mc):
            c = self._draw_rotation(rng, theta_tex) @ u
            m = np.abs(np.einsum('i,pij,j->p', c, P0, c))
            tot += m.max()
        return float(tot / n_mc)

    # ============================================================
    # 晶粒级体力
    # ============================================================
    @staticmethod
    def _resolved_shear(S, P0):
        """τ^α = S : P0^α"""
        return np.einsum('ij,pij->p', S, P0)  # (N_slip,)

    @staticmethod
    def _mises(S):
        dev = S - np.trace(S) / 3.0 * np.eye(3)
        return np.sqrt(1.5 * np.einsum('ij,ij->', dev, dev))

    @staticmethod
    def _stiffness_contract(C_rank4, E):
        """S_ij = C_ijkl E_kl (rank-4 double contraction)"""
        return np.einsum('ijkl,kl->ij', C_rank4, E)

    @staticmethod
    def _Y(sigma_mises, D, E_mod):
        D_clip = np.clip(D, 0.0, 0.95)
        denom = 2.0 * E_mod * max((1.0 - D_clip)**2, 1e-12)
        return sigma_mises**2 / denom

    @staticmethod
    def _hardening(g_current, g_sat, dgamma_dt, h0, a_exp, q_lat):
        """Voce硬化率 dg/dt"""
        N = len(dgamma_dt)
        Q = np.full((N, N), q_lat)
        np.fill_diagonal(Q, 1.0)
        ratio = np.clip(1.0 - g_current / g_sat, 0.0, None)
        base = h0 * np.abs(dgamma_dt) * (ratio ** a_exp)
        return Q @ base

    # ============================================================
    # 单步步积分器（显式 + 子步步）
    # ============================================================
    def _step_single_grain(self, F_crystal, Fp, g, D, p_acc,
                         dt_total, n_rate, gdot0, g_sat):
        """单晶粒子步步积分，返回 (Fp_new, g_new, D_new, p_acc_new)"""
        dt_sub = dt_total / self.n_sub
        C4 = self.stiffness
        h0   = self.params['h0']
        a_exp = self.params['a']
        q_lat = self.params['q_lat']
        p_D   = self.params['p_D']
        s_dam = self.params['s_damage']
        S_eff_pa = self.S_eff * 1e6  # MPa→Pa

        Fp_c = Fp.copy()
        g_c  = g.copy()
        D_c  = D
        p_c  = p_acc

        dt_cap = dt_total / self.n_sub   # 目标子步长
        dt_sub = dt_cap
        t_done = 0.0
        max_refine = 4                   # 最多二分 16 倍缩小
        SLIP_TOL = 0.02                  # 子步总滑移稳定上限

        while t_done < dt_total - 1e-14:
            if D_c >= 0.99:
                break

            dt_sub = min(dt_sub, dt_total - t_done)

            # 弹性试探
            Fe = F_crystal @ np.linalg.inv(Fp_c)
            Ee = 0.5 * (Fe.T @ Fe - np.eye(3))
            S  = self._stiffness_contract(C4, Ee)

            # 分解剪应力
            tau = self._resolved_shear(S, self.P0)  # (N_slip,)

            # 幂律剪切率（方案B：损伤不直接影响塑性流）
            # 弹性预测用未退化刚度 C:Ee（tau 与 D 无关），滑移阻力 g 亦不随损伤缩放；
            # 损伤的影响仅经 (i) 名义应力输出退化 (1-D) 与 (ii) 损伤演化/断裂判据。
            # （第四轮审稿：选择"承认损伤对塑性流动无直接影响"分支，删除 g_eff=g/(1-D)。）
            tau_abs = np.abs(tau)
            denom_eff = np.maximum(g_c, 1e-10)
            ratio = np.where(g_c > 1e-6, tau_abs / denom_eff, 0.0)
            ratio = np.clip(ratio, 0.0, 10.0)
            gamma_dot = gdot0 * (ratio ** n_rate) * np.sign(tau)

            # 剪切增量
            dgamma = gamma_dot * dt_sub
            dgamma = np.clip(dgamma, -0.01, 0.01)
            dg_tot = np.sum(np.abs(dgamma))

            # 稳定子步控制：显式格式在多系共激活时会出现"滑移-应力"
            # 极限环振荡（p 无界增长、损伤爆炸）。若本子步总滑移超限时
            # 二分 dt_sub 重算（已算的量全部丢弃），直到滑移增量受控。
            refine = 0
            while dg_tot > SLIP_TOL and refine < max_refine:
                dt_sub *= 0.5
                refine += 1
                dgamma = gamma_dot * dt_sub
                dgamma = np.clip(dgamma, -0.01, 0.01)
                dg_tot = np.sum(np.abs(dgamma))
            if dg_tot > SLIP_TOL:
                dgamma = dgamma * (SLIP_TOL / dg_tot)  # 兜底比例缩放

            # 更新Fp（显式前向）
            Lp = np.einsum('p,pij->ij', dgamma, self.P0)
            Fp_c = (np.eye(3) + Lp) @ Fp_c

            # 硬化（使用实际剪切率而非幂律初始爆炸值，避免过度硬化锁定）
            dg_dt = self._hardening(g_c, g_sat, dgamma / dt_sub, h0, a_exp, q_lat)
            g_c += dg_dt * dt_sub
            g_c = np.maximum(g_c, 1e-3 * np.abs(g_c).max())

            # 累积塑性应变
            dp = np.sqrt(2.0 / 3.0) * np.sum(np.abs(dgamma))
            p_c += dp

            # 损伤更新
            if p_c > p_D and dp > 0:
                # 重新计算当前应力（用于损伤判定）
                Fe_n = F_crystal @ np.linalg.inv(Fp_c)
                Ee_n = 0.5 * (Fe_n.T @ Fe_n - np.eye(3))
                S_n  = self._stiffness_contract(C4, Ee_n)
                vm   = self._mises(S_n)
                if self.use_v4:
                    # Y = sigma~^2 / (2E)：σ̃ 为有效（未退化）应力（应变等价假设）
                    Y = vm * vm / (2.0 * self.exp['E'])
                else:
                    Y = self._Y(vm, D_c, self.exp['E'])
                if self.damage_model == 'gtn':
                    # GTN-type 孔隙演化：应变控制常数形核率 ḟ=A·dp
                    # （AM 缺陷/未熔合连续形核；growth 项在 Taylor 滑移
                    # 不可压下为零）→ ef = f_c/A，单调平滑、无阈值奇点
                    dD   = self.gtn_A * dp
                    dD   = np.clip(dD, 0.0, 0.05)
                else:
                    r    = Y / max(self.S_eff * 1e6, 1e-10)
                    dD   = (r ** s_dam) * dp * self.f_AM
                    # 损伤增量上限收紧到 SLIP_TOL 量级：抑制多粒级联断裂
                    # （avalanche）带来的 ef 断阶噪声，保证 ef(S0) 足够平滑
                    dD   = np.clip(dD, 0.0, 0.02)
                D_c  = min(0.99, D_c + dD)

            t_done += dt_sub
            # 步长回升：连续顺利则逐步恢复到目标子步长
            dt_sub = min(dt_sub * 1.5, dt_cap)

            if D_c >= 0.99:
                break  # 断裂后尽早退出，减少无效子步

        return Fp_c, g_c, D_c, p_c

    # ============================================================
    # 主积分循环
    # ============================================================
    def integrate(self, strain_path):
        """沿宏观应变路径积分

        Args:
            strain_path: (n_steps, 6) Voigt应变 [ε11, ε22, ε33, γ12, γ13, γ23]

        Returns:
            stress_history: (n_steps, 6)
            D_history: (n_steps,)
            p_history: (n_steps,)
            D_grain_history: (n_steps, n_grains)
        """
        n_steps = len(strain_path)

        # 晶粒状态初始化
        Fp   = np.tile(np.eye(3)[None, :, :], (self.n_grains, 1, 1))  # (G,3,3)
        g    = np.tile(self.g0_vec[None, :], (self.n_grains, 1))      # (G,N_slip)
        D_g  = np.full(self.n_grains, self.params['D0'])
        p_g  = np.zeros(self.n_grains)

        # 宏观输出
        stress_out = np.zeros((n_steps, 6))
        D_out = np.zeros(n_steps)
        p_out = np.zeros(n_steps)
        DG_out = np.zeros((n_steps, self.n_grains))

        D_out[0] = self.params['D0']
        DG_out[0] = D_g.copy()

        # 材料参数
        n_rate  = self.params['n_rate']
        gdot0   = self.params['gamma_dot_0']
        g_sat   = self.g_sat_vec

        for step in range(1, n_steps):
            eps_cur = strain_path[step]
            deps = eps_cur - strain_path[step - 1]
            eps_norm = np.linalg.norm(deps)

            if eps_norm < 1e-14:
                stress_out[step] = stress_out[step - 1]
                D_out[step] = D_out[step - 1]
                p_out[step] = p_out[step - 1]
                DG_out[step] = D_g.copy()
                continue

            # 宏观总 F
            F_macro = np.diag([1.0 + eps_cur[0],
                               1.0 + eps_cur[1],
                               1.0 + eps_cur[2]])

            dt_total = eps_norm / self.strain_rate

            # 逐晶粒子步
            for i in range(self.n_grains):
                if D_g[i] >= 0.99:
                    continue
                R_i = self.R[i]
                F_crystal = R_i.T @ F_macro @ R_i

                Fp_i_new, g_i_new, D_i_new, p_i_new = \
                    self._step_single_grain(
                        F_crystal, Fp[i], g[i], D_g[i], p_g[i],
                        dt_total, n_rate, gdot0, g_sat
                    )

                Fp[i], g[i], D_g[i], p_g[i] = Fp_i_new, g_i_new, D_i_new, p_i_new

            # 收敛后平均应力
            sigma_avg = np.zeros((3, 3))
            active = 0
            for i in range(self.n_grains):
                if D_g[i] >= 0.99:
                    continue
                R_i = self.R[i]
                Fc = R_i.T @ F_macro @ R_i
                Fe_i = Fc @ np.linalg.inv(Fp[i])
                Ee_i = 0.5 * (Fe_i.T @ Fe_i - np.eye(3))
                S_i  = self._stiffness_contract(self.stiffness, Ee_i)
                J    = np.linalg.det(Fe_i)
                sig_crys = (1.0 - D_g[i]) / max(J, 1e-10) * (Fe_i @ S_i @ Fe_i.T)
                sig_mac  = R_i @ sig_crys @ R_i.T
                sigma_avg += sig_mac
                active += 1

            if active > 0:
                sigma_avg /= active

            stress_out[step, 0] = sigma_avg[0, 0]
            stress_out[step, 1] = sigma_avg[1, 1]
            stress_out[step, 2] = sigma_avg[2, 2]
            stress_out[step, 3] = sigma_avg[0, 1]
            stress_out[step, 4] = sigma_avg[0, 2]
            stress_out[step, 5] = sigma_avg[1, 2]

            active = D_g < 0.99
            D_out[step] = np.mean(D_g[active]) if np.any(active) else 0.99
            p_out[step] = np.mean(p_g[active]) if np.any(active) else p_out[step - 1]
            DG_out[step] = D_g.copy()

            if np.mean(D_g) >= 0.99:
                break

        return stress_out, D_out, p_out, DG_out

    def _macro_sigma(self, F_macro, Fp, g, D_g):
        """逐晶平均宏观柯西应力张量。"""
        C4 = self.stiffness
        s = np.zeros((3, 3)); cnt = 0
        for i in range(self.n_grains):
            if D_g[i] >= 0.99:
                continue
            Fc = self.R[i].T @ F_macro @ self.R[i]
            Fe = Fc @ np.linalg.inv(Fp[i])
            Ee = 0.5 * (Fe.T @ Fe - np.eye(3))
            S = self._stiffness_contract(C4, Ee)
            J = np.linalg.det(Fe)
            sc = (1.0 - D_g[i]) / max(J, 1e-10) * (Fe @ S @ Fe.T)
            s += self.R[i] @ sc @ self.R[i].T
            cnt += 1
        return s / max(cnt, 1)

    def run_uniaxial(self, max_strain, inner_tol_pa=3e6, n_inner=25):
        """混合边界条件单轴拉伸模拟。

        轴向位移受控（ε11 指定），侧向自由度通过逐步迭代使
        宏观平均侧向应力 ⟨σ22⟩=⟨σ33⟩=0 —— 即自由表面单轴拉伸的标准
        CP 边界条件。这样避免固定泊松比强加的三轴弹性过应力，
        单晶时 ν 从弹性 ~0.3 自然过渡到塑性 ~0.5。
        """
        C4 = self.stiffness
        n_rate = self.params['n_rate']
        gdot0 = self.params['gamma_dot_0']
        g_sat = self.g_sat_vec
        es = self.eps_step
        G = self.n_grains

        Fp = np.tile(np.eye(3)[None, :, :], (G, 1, 1))
        g = np.tile(self.g0_vec[None, :], (G, 1))
        D_g = np.full(G, self.params['D0'])
        p = np.zeros(G)
        if self.damage_model == 'gtn':
            Dc = self.gtn_fc                   # GTN 临界孔隙率（文献固定 0.15）
        else:
            Dc = self.exp['Dc']

        n_steps = max(10, int(max_strain / es)) + 1
        eps_arr = np.linspace(0.0, max_strain, n_steps)
        sig_h = np.zeros(n_steps)
        Dm_h = np.zeros(n_steps); Dm_h[0] = D_g.mean()
        frac_broken = np.zeros(n_steps)

        # 集合观测量：对已断裂晶粒按 D=0.99 计入的全晶粒平均。
        # 断裂判据 = D_agg >= Dc：物理上等价于"损失承载能力的晶粒质量分数 ≥ Dc"。
        # 与旧的"仅活动晶粒忽均"相比是连续量，消除了晶粒退出导致的跳变。
        lat = 0.30  # 初始猜测（弹性泊松比）
        for s in range(1, n_steps):
            cur = eps_arr[s]
            dt = es / self.strain_rate
            saved = (Fp.copy(), g.copy(), D_g.copy(), p.copy())

            def eval_lat(lv):
                Fp[:] = saved[0]; g[:] = saved[1]; D_g[:] = saved[2]; p[:] = saved[3]
                F_macro = np.diag([1.0 + cur, 1.0 - lv * cur, 1.0 - lv * cur])
                for i in range(G):
                    if D_g[i] >= 0.99:
                        continue
                    Fc = self.R[i].T @ F_macro @ self.R[i]
                    Fp[i], g[i], D_g[i], p[i] = self._step_single_grain(
                        Fc, Fp[i], g[i], D_g[i], p[i], dt, n_rate, gdot0, g_sat)
                sm = self._macro_sigma(F_macro, Fp, g, D_g)
                return sm, 0.5 * (sm[1, 1] + sm[2, 2])

            # 有界阻尼牛顿解横向收缩 lat（平均侧向应力→0）
            # slat(lat) 曲线刚性且含噪声，裸 secant 大步长会在多解间跳
            # 振荡并触发滑移-应力极限环。限制单步 |Δlat|、用上一步收敛
            # 值热启动、并始终提交 |slat| 最小的探测点，保证轨迹平滑。
            x = lat
            sm_x, fx = eval_lat(x)
            best_x, best_fx, best_sm = x, fx, sm_x
            x_prev, fx_prev = None, None
            for _ in range(n_inner):
                if abs(fx) < inner_tol_pa:
                    break
                if x_prev is not None and fx != fx_prev:
                    slope = (fx - fx_prev) / (x - x_prev)
                else:
                    slope = None
                if slope is None or abs(slope) < 1e-12 or slope > 0:
                    # 斜率缺失或方向异常：向历史中点估值斜率
                    xp = min(0.58, x + 0.01) if x + 0.01 <= 0.58 else max(0.05, x - 0.01)
                    fp = eval_lat(xp)[1]
                    dx = 0.01 if x + 0.01 <= 0.58 else -0.01
                    slope = (fp - fx) / dx if dx != 0 else None
                    if slope is None or abs(slope) < 1e-12:
                        break
                dx = -0.5 * fx / slope              # 阻尼=0.5 防过冲
                dx = min(0.02, max(-0.02, dx))      # 信赖域
                x_new = min(0.58, max(0.05, x + dx))
                if abs(x_new - x) < 1e-9:
                    break
                fx_new_sm, fx_new = eval_lat(x_new)
                x_prev, fx_prev = x, fx
                x, fx = x_new, fx_new
                if abs(fx) < abs(best_fx):
                    best_x, best_fx, best_sm = x, fx, fx_new_sm
            l_conv = best_x
            sm, _ = eval_lat(l_conv)
            lat = l_conv

            sig_h[s] = sm[0, 0]
            act = D_g < 0.99
            # 全体平均：已断晶粒取 0.99，未断为真值。连续、随应变单调不减。
            Dm_h[s] = float(np.clip(D_g, 0.0, 0.99).mean())
            frac_broken[s] = 1.0 - np.count_nonzero(act) / G
            if Dm_h[s] >= Dc:
                break

        eps_arr = eps_arr[:s + 1]
        sig_h = sig_h[:s + 1]
        Dm_h = Dm_h[:s + 1]
        frac_broken = frac_broken[:s + 1]

        # 插值断裂应变：落在 Dm 首次跨过 Dc 的两个相邻步之间的线性插值
        idx = np.where(Dm_h >= Dc)[0]
        if len(idx) > 0:
            i0 = idx[0]
            if i0 > 0 and Dm_h[i0] > Dm_h[i0 - 1]:
                frac = (Dc - Dm_h[i0 - 1]) / (Dm_h[i0] - Dm_h[i0 - 1])
                ef = eps_arr[i0 - 1] + np.clip(frac, 0.0, 1.0) * es
            else:
                ef = eps_arr[i0]
        else:
            ef = eps_arr[-1]

        sig_out = sig_h.copy()
        if len(idx) > 0 and idx[0] < len(sig_out):
            sig_out[idx[0]:] = 0.0
        uts = np.max(sig_out[:idx[0] + 1] if len(idx) > 0 else sig_out)

        return {
            'strain':          eps_arr,
            'stress':          sig_out,
            'damage':          Dm_h,
            'frac_broken':     frac_broken,
            'material':        self.material_key,
            'orientation':     self.orientation_deg,
            'fracture_strain': ef,
            'uts':             uts,
            'S0':              self.params['S0'],
            'f_AM':            self.f_AM,
            'S_eff':           self.S_eff,
        }