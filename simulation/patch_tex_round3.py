# -*- coding: utf-8 -*-
"""tex 第三轮修补 — patch_tex_round3.py
修复三个残留问题：
1. 591 行重复句（experimental data are compiled 出现两次）
2. 结论第 4 条引用取向乘子 a45/a90 数值（与 v9 取消乘子矛盾）
3. 结论第 2 条 DAMASK 措辞（DAMASK 已用于无损伤全场对比，非"后续工作"）
"""
import io

TEX = r'D:\20260618断裂模型论文\manuscript\manuscript_main.tex'


def main():
    with io.open(TEX, 'r', encoding='utf-8') as f:
        s = f.read()

    # 1. 591 行重复句：删除第二个 "The experimental data are compiled..."
    dup = (r'The experimental data are compiled from literature values for '
           r'as-built LPBF material of comparable process windows, and '
           r'standardized to ensure consistent comparison (sources and test '
           r'conditions are detailed in Section~\ref{sec:experimental_sources}). '
           r'The experimental data are compiled from literature values for '
           r'as-built LPBF material of comparable process windows, and '
           r'standardized to ensure consistent comparison (sources and test '
           r'conditions are detailed in Section~\ref{sec:experimental_sources}).')
    fix = (r'The experimental data are compiled from literature values for '
           r'as-built LPBF material of comparable process windows, and '
           r'standardized to ensure consistent comparison (sources and test '
           r'conditions are detailed in Section~\ref{sec:experimental_sources}).')
    assert s.count(dup) == 1, f'dup count = {s.count(dup)}'
    s = s.replace(dup, fix)

    # 2. 结论第 4 条重写（v9 语境）
    old_aniso = r'''    \item \textbf{Quantified anisotropic damage susceptibility}: The calibrated orientation multipliers reveal systematic microstructure-dependent trends: Ti-6Al-4V exhibits the strongest transverse damage acceleration ($f_{\text{AM}}^{\text{aniso}} = 2.740$ at $\ang{90}$, with a substantial 1.246 at $\ang{45}$), followed by 316L SS (1.231 at $\ang{90}$), while AlSi10Mg retains the weakest residual multiplier (0.558 at $\ang{45}$, 0.563 at $\ang{90}$). The strong transverse acceleration of the HCP alloy is consistent with transverse cracking along colony boundaries and melt-pool-boundary paths, whereas the FCC alloys distribute the anisotropy between the physical projection terms and the calibrated multiplier.'''
    new_aniso = r'''    \item \textbf{Quantified anisotropic damage susceptibility}: The physically projected descriptors ($\lambda_{\mathrm{eff}}$, $\xi_{\mathrm{eff}}$, $\theta$) alone compress the predicted $0^\circ/90^\circ$ fracture-strain ratio to 1.12 (Ti-6Al-4V), 1.14 (316L) and 2.19 (AlSi10Mg) against the measured 1.67, 1.45 and 1.51 (Section~\ref{sec:calibration}). This exposes a genuine modeling gap: the empirical orientation multipliers of the earlier parametrization absorbed the remainder, and removing them (as required by the leakage-free protocol) degrades the transverse calibration accuracy. The strong measured transverse damage acceleration of the HCP alloy is consistent with transverse cracking along colony boundaries and melt-pool-boundary paths, but a purely projection-based representation is insufficient to reproduce it quantitatively; this residual direction dependence is identified as the primary target of the tensorial and microstructure-coupled extensions outlined below.'''
    assert s.count(old_aniso) == 1, f'aniso count = {s.count(old_aniso)}'
    s = s.replace(old_aniso, new_aniso)

    # 3. 结论第 2 条 DAMASK 措辞
    old_impl = r'''    \item \textbf{Fully-coupled CP-CDM implementation}: The modified damage model is implemented at the constitutive level in a DAMASK-compatible crystal plasticity solver, enabling concurrent evolution of crystallographic slip and damage. The fully-coupled formulation captures the two-way feedback essential for accurate fracture prediction.'''
    new_impl = r'''    \item \textbf{Fully-coupled CP-CDM implementation}: The modified damage model is implemented at the constitutive level in our in-house Taylor-type solver TaylorCPCDM, which follows DAMASK's constitutive interface conventions; the fully-coupled formulation enables concurrent evolution of crystallographic slip and damage, capturing the two-way feedback essential for accurate fracture prediction. DAMASK itself is used here exclusively as an external full-field spectral-FFT reference to quantify the Taylor homogenization bias (Section~\ref{sec:damask_fft}); porting the damage law into the DAMASK native solver remains follow-on work.'''
    assert s.count(old_impl) == 1, f'impl count = {s.count(old_impl)}'
    s = s.replace(old_impl, new_impl)

    with io.open(TEX, 'w', encoding='utf-8', newline='') as f:
        f.write(s)
    print('OK: patch applied')


if __name__ == '__main__':
    main()
