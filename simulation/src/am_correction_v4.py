"""
AM修正函数 v4 — 重构版（响应审稿意见）
========================================

相对 v3 的三项结构修改：

1. 消除 S 与 f_AM 之间的双重计数/方向矛盾：
   旧版 S_eff = S0·g1(λ)·g2(ξ) 与 f_λ、f_ξ 重复使用同一描述符；
   且 λ 在 g1 中增大 S_eff（减速损伤）而在 f_λ 中加速损伤，方向矛盾。
   v4: S ≡ S0（常数，仅在 0° 标定），全部微结构/取向效应只经 f_AM 进入。

2. 各向异性不再使用自由拟合乘数 f_AM_aniso；
   取向依赖完全由"取向分辨的微结构描述符"承载：
     λ(ψ): 熔池层界面的法向随加载轴投影（弱界面网络横观各向同性）
     ξ(ψ): 柱状晶沿加载方向的解析长宽比（椭球几何）
     θ(ψ): 取向集合在该加载方向下的平均 Schmid 因子（由模拟用取向组直接计算）
   这些投影律是几何假设（文献可核），不含拟合取向数据的参数。

3. Y 统一用有效应力定义（应变等价假设）：
   Y = σ̃_eq²/(2E)，σ̃ = σ/(1−D)，不再出现 (1−D)² 重复折算。

归一化约定：f_AM 的每个子因子为无量纲量，f=1 为标准各向同性基准；
f>1 表示加速损伤，f<1 表示减速损伤（每个因子的符号方向在代码与论文中明确）。
"""

import numpy as np

LAMBDA_REF = 50.0   # MPB 密度归一化参考值 [mm⁻¹]
W_ISO = 0.25        # 熔池界面网络的各向同性面积分数（立体测量学假设）


def lambda_eff(lam0, psi_deg, w_iso=W_ISO):
    """取向分辨的有效熔池边界密度 [mm⁻¹]

    物理图像：LPBF 熔池边界/柱状晶间界面构成横观各向同性的弱界面网络，
    其法向主要分布在垂直于建造方向的平面内。沿与 BD 成 ψ 角方向拉伸时，
    垂直于加载轴的弱界面面积密度 ∝ (w_iso + (1−w_iso)·sin²ψ)：
      ψ=0   → 仅各向同性部分被正应力直接拉开（界面法向 ⊥ 加载方向为主）
      ψ=90° → 全部界面网络面临正拉应力分量
    """
    s2 = np.sin(np.radians(psi_deg)) ** 2
    return lam0 * (w_iso + (1.0 - w_iso) * s2)


def xi_eff(xi0, psi_deg):
    """取向分辨的晶粒形态因子：旋转椭球晶粒沿加载方向的解析长宽比

    晶粒建模为绕 BD 的旋转椭球（半轴 a, a, c=ξ·a）。加载轴与 BD 成 ψ 角时，
    沿加载线到晶面的径向长度为 r(ψ)=a·ξ/√(ξ²sin²ψ+cos²ψ)，横向共轭半径为
    r(ψ+90°)=a·ξ/√(ξ²cos²ψ+sin²ψ)，故沿加载方向的表观长度 L(ψ)=2r(ψ)、垂直于
    加载方向的表观宽度 W(ψ)=2r(ψ+90°)，取向分辨长宽比为二共轭半径之比：
      ξ_eff(ψ) = L/W = √((ξ²cos²ψ+sin²ψ)/(ξ²sin²ψ+cos²ψ))
      ψ=0 → ξ，ψ=45° → 1（45° 投影各向同性中性），ψ=90° → 1/ξ（横向短轴）
    注：早前实现误将"参数角半径"2a·√(ξ²cos²ψ+sin²ψ) 当作沿加载方向的径向长度，
      给出 ξ_eff=(ξ²cos²ψ+sin²ψ)/ξ；0°/90° 端点相同，但 45° 严重偏大（ξ₀=3.5 时
      给 1.89 而正确值为 1.0），直接影响 f_xi 与 ĝ_s(ψ) 的方向响应，已在此修正。
    损伤方向的物理意义：f_xi 采用 1/ξ_eff（横向短韧带机制）——
      90° 加载时有效长宽比最小（短轴主导），损伤在晶粒宽度尺度的短韧带上
      沿柱间弱界面快速连通（加速）；0° 加载时沿长轴加载，韧带长（减速）。
    """
    s2 = np.sin(np.radians(psi_deg)) ** 2
    c2 = 1.0 - s2
    return np.sqrt((xi0 ** 2 * c2 + s2) / (xi0 ** 2 * s2 + c2))


def factors_v4(params, psi_deg, theta_bar):
    """计算 v4 版 f_AM 及各分量。

    Args:
        params: 材料参数（MATRIX 原值，不做旧版取向缩放）
        psi_deg: 加载轴与建造方向夹角
        theta_bar: 取向集合在该加载方向的平均 Schmid 因子（由 Taylor 取向组计算）

    Returns:
        (f_AM, dict_of_components)
    """
    p = params
    w_iso = float(p.get('_w_iso', W_ISO))                    # 界面网络各向同性分数
    s2_lam = np.sin(np.radians(psi_deg)) ** 2
    lam_mode = p.get('_lambda_norm', 'global')              # 'global'（生产）| 'rel'（方案A相对归一化）
    lambda_eff_mm = p['lambda_mp'] * (w_iso + (1.0 - w_iso) * s2_lam)  # 有量纲 λ_eff [mm⁻¹]
    if lam_mode == 'rel':
        # 方案A：相对本材料 0° 值归一化 λ̄_rel(ψ)=λ_eff(ψ)/λ_eff(0°)=(w_iso+(1−w_iso)sin²ψ)/w_iso
        # 与材料无关，横向增强恒为 1/w_iso；消除全局参考 λ_ref 对低密度合金（316L λ0=35）
        # 在 90° 读作弱网络（f_λ<1，减速）的方向反转。
        lam = (w_iso + (1.0 - w_iso) * s2_lam) / w_iso
    else:
        lam = lambda_eff_mm / LAMBDA_REF                    # 生产：全局参考归一化（无量纲）
    xi = xi_eff(p['xi_grain'], psi_deg)                      # 无量纲

    f_phi = 1.0 + p['beta1'] * (p['phi'] / max(p['phi_crit'], 1e-10)) ** p['m1']
    f_theta = 1.0 + p['beta2'] * max(0.0, 1.0 - 2.0 * theta_bar) ** p['m2']
    f_D0 = np.exp(min(p['beta3'] * p['D0'], 50.0))
    f_lam = 1.0 + p['beta4'] * (lam - 1.0)   # 参考态 λ̄=1 → f_λ=1（审稿 2a）
    # f_xi: 横向短韧带机制（审稿 R7：原 (xi-1) 形式在 0° 加速、90° 减速，与实验相反；
    #  重定义为 1/xi-1：细长柱状晶横向加载时损伤沿柱间弱界面在短横向韧带尺度快速连通）
    f_xi = 1.0 + p['beta5'] * (1.0 / xi - 1.0)
    # f_shield: 几何屏蔽项（审稿 R8）——层状缺陷面法向平行建造方向：加载方向垂直缺陷面
    #  （ψ=0°）时缺陷直接张开（无屏蔽），平行缺陷面（ψ=90°）时缺陷被几何屏蔽（不张开）
    #  → 90° 损伤减速。仅缺陷/界面主导断裂的材料需要 β6>0；柱状加速主导的标定集材料
    #  β6=0（f_shield≡1，现有路径零变化）。
    beta6 = float(p.get('beta6', 0.0))
    s2 = np.sin(np.radians(psi_deg)) ** 2
    f_shield = 1.0 - beta6 * s2

    f_AM = f_phi * f_theta * f_D0 * f_lam * f_xi * f_shield

    comp = {
        'f_phi': float(f_phi), 'f_theta': float(f_theta), 'f_D0': float(f_D0),
        'f_lambda': float(f_lam), 'f_xi': float(f_xi), 'f_shield': float(f_shield),
        'lambda_eff_mm': float(lambda_eff_mm), 'lambda_norm': str(lam_mode),
        'xi_eff': float(xi), 'theta_bar': float(theta_bar),
        'f_AM': float(f_AM),
    }
    return f_AM, comp
