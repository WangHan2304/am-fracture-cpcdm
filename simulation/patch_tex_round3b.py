# -*- coding: utf-8 -*-
"""tex 第三轮修补2 — patch_tex_round3b.py
1. 删除重复的 DAMASK 文献条目（1018-1022 行，前轮已存在相同条目）
2. 修正摘要/引言中 "DAMASK-native port targeted as follow-on work" 与
   新 DAMASK 全场对比节的矛盾（DAMASK 已实际用于无损伤全场对比）
"""
import io

TEX = r'D:\20260618断裂模型论文\manuscript\manuscript_main.tex'


def main():
    with io.open(TEX, 'r', encoding='utf-8') as f:
        s = f.read()

    # 1. 删除重复 DAMASK 文献 —— 已通过 rm_dup_bib.py 完成，此处跳过
    dup_bib = None

    # 2a. 摘要：DAMASK 表述
    old_abs_d = ('implemented in our in-house Taylor-type polycrystal solver TaylorCPCDM, '
                 'which follows DAMASK\'s constitutive interface conventions; all results '
                 'reported here are produced by TaylorCPCDM itself, and a DAMASK-native '
                 'spectral port is targeted as follow-on work.')
    new_abs_d = ('implemented in our in-house Taylor-type polycrystal solver TaylorCPCDM, '
                 'which follows DAMASK\'s constitutive interface conventions; all calibration '
                 'and cross-validation results are produced by TaylorCPCDM itself, while the '
                 'DAMASK spectral-FFT solver is used as an external full-field reference to '
                 'quantify the Taylor homogenization bias (Section~\ref{sec:damask_fft}).')
    assert s.count(old_abs_d) == 1, f'abs count = {s.count(old_abs_d)}'
    s = s.replace(old_abs_d, new_abs_d)

    # 2b. 引言第 1 条
    old_i1 = ('The modified damage model is implemented at the constitutive level in the '
              'TaylorCPCDM crystal plasticity solver, which shares DAMASK\'s constitutive '
              'interface conventions and is designed for a future DAMASK-native port.')
    new_i1 = ('The modified damage model is implemented at the constitutive level in the '
              'TaylorCPCDM crystal plasticity solver, which shares DAMASK\'s constitutive '
              'interface conventions; DAMASK\'s spectral-FFT solver is used as an external '
              'full-field reference to quantify the Taylor homogenization bias '
              '(Section~\ref{sec:damask_fft}).')
    assert s.count(old_i1) == 1, f'i1 count = {s.count(old_i1)}'
    s = s.replace(old_i1, new_i1)

    # 2c. 引言第 3 条
    old_i3 = ('The workflow is solver-agnostic and demonstrated with the TaylorCPCDM '
              'framework, with an HPC-grade DAMASK-native execution targeted as follow-on work.')
    new_i3 = ('The workflow is solver-agnostic and demonstrated with the TaylorCPCDM '
              'framework, with the DAMASK full-field solver employed for the '
              'homogenization-bias assessment.')
    assert s.count(old_i3) == 1, f'i3 count = {s.count(old_i3)}'
    s = s.replace(old_i3, new_i3)

    # 2d. 454 行实现段
    old_impl = ('TaylorCPCDM shares DAMASK\'s constitutive interface conventions (material '
                'parameter keys, orientation-independent state layout, and per-integration-point '
                'output), so the model description here can be directly transcribed into a '
                'DAMASK-native phase module as follow-on work. \emph{All results reported in '
                'this paper (calibration, orientation cross-validation, and the Lemaitre '
                'baseline) are produced by the TaylorCPCDM implementation described above}; '
                'a DAMASK-native spectral FFT solver for explicit polycrystal RVE coupling is '
                'targeted as follow-on work and is not used for any of the calibration or '
                'cross-validation results.')
    new_impl = ('TaylorCPCDM shares DAMASK\'s constitutive interface conventions (material '
                'parameter keys, orientation-independent state layout, and per-integration-point '
                'output), so the model description here can be directly transcribed into a '
                'DAMASK-native phase module. \emph{All results reported in this paper '
                '(calibration, orientation cross-validation, and the Lemaitre baseline) are '
                'produced by the TaylorCPCDM implementation described above}; the DAMASK '
                'spectral-FFT solver is used exclusively as an external full-field reference '
                'for the homogenization-bias assessment of Section~\ref{sec:damask_fft} and '
                'does not contribute to any calibration or cross-validation result.')
    assert s.count(old_impl) == 1, f'impl count = {s.count(old_impl)}'
    s = s.replace(old_impl, new_impl)

    # 2e. 572 行 workflow 段落
    old_wf = ('the full nine-case calibration plus the leave-one-orientation-out '
              'cross-validation suite requires on the order of hours on a mid-range CPU; a '
              'DAMASK-native FFT implementation on HPC clusters, targeted as follow-on work, '
              'would scale to large $64^3$--$256^3$ voxel RVEs using MPI, with near-linear '
              'scaling up to $\\sim$512 cores as documented in the DAMASK literature '
              '\\cite{roters2019damask}).')
    new_wf = ('the full nine-case calibration plus the leave-one-orientation-out '
              'cross-validation suite requires on the order of hours on a mid-range CPU; the '
              'DAMASK spectral-FFT full-field reference (Section~\\ref{sec:damask_fft}) was '
              'run on a $32^3$ voxel RVE with MPI and would scale to large $64^3$--$256^3$ '
              'voxel RVEs with near-linear scaling up to $\\sim$512 cores as documented in '
              'the DAMASK literature \\cite{roters2019damask}).')
    assert s.count(old_wf) == 1, f'wf count = {s.count(old_wf)}'
    s = s.replace(old_wf, new_wf)

    with io.open(TEX, 'w', encoding='utf-8', newline='') as f:
        f.write(s)
    print('OK: patch2 applied')


if __name__ == '__main__':
    main()
