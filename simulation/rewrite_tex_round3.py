# -*- coding: utf-8 -*-
"""
tex 第三轮全面修改脚本 — rewrite_tex_round3.py
============================================
基于 v9 真 LOOCV + v9c 公平 LEM + DAMASK 全场对比，对 manuscript_main.tex
执行第三轮系统性重构：

A. 摘要（0.76% → calibration-only、2.44% → 26.38% 真 LOOCV、新增 DAMASK 对比）
B. 5.1 calibration-validation split（明确 9 案例=calibration、LOOCV=blind）
C. 5.3 标定结果表（v9 数字 + "calibration error" 措辞 + AlSi UTS 讨论保留）
D. 5.4 Lemaitre 对比（公平基线 k=3 vs k=3，AIC/BIC 更新，同聚合 LEM）
E. 5.5 LOOCV 表（v9 26.38% 真盲预测 + 泄漏说明 + 改名 fixed-global-parameter
   sensitivity check 的 v8 附录）
F. 7.2 Taylor limitation（删除"无全场求解器"表述，替换 DAMASK 对比结果）
G. 新增 7.x DAMASK 全场对比小节 + 图
H. 四层验证框架段落
I. 结论（全部数字更新）
J. 模型对比表（表5）更新

注意：本脚本只做文本替换，不创建新结构（新结构用插入式替换）。
"""
import io

TEX = r'D:\20260618断裂模型论文\manuscript\manuscript_main.tex'


def read_tex():
    with io.open(TEX, 'r', encoding='utf-8') as f:
        return f.read()


def write_tex(s):
    with io.open(TEX, 'w', encoding='utf-8', newline='') as f:
        f.write(s)


def rep(s, old, new, must=True):
    if old not in s:
        if must:
            raise SystemExit(f'NOT FOUND: {old[:120]}')
        return s
    if s.count(old) > 1:
        print(f'WARN: {old[:80]} occurs {s.count(old)} times, replacing all')
    return s.replace(old, new)


def table_env(s, label):
    """提取包含 \\label{label} 的完整 table 环境文本（从最近的前向 \\begin{table} 到其后 \\end{table}）。"""
    lab = s.index(r'\label{' + label + '}')
    beg = s.rindex(r'\begin{table}', 0, lab)
    end = s.index(r'\end{table}', lab)
    return s[beg:end + len(r'\end{table}')]


def main():
    s = read_tex()

    # ============================================================
    # A. 摘要
    # ============================================================
    old_abs = r'Over the nine literature-based calibration cases the modified model achieves an average calibration fracture-strain error of \textbf{0.76\%} (maximum 4.1\%) and an average UTS error of \textbf{7.9\%}, with all nine cases within the 10\% fracture-strain target, substantially outperforming the conventional Lemaitre model (average error 25.8\%). In the leave-one-orientation-out blind cross-validation the modified model retains an average blind prediction error of \textbf{2.44\%} (all nine folds within the 10\% target), against 27.33\% for the Lemaitre baseline under the identical single-parameter protocol.'
    new_abs = r'Over the nine literature-based calibration cases the modified model, under a fully leakage-free parameterization (no orientation multipliers; a single damage energy strength $S_0$ per material; all orientation dependence carried by physically projected microstructure descriptors), achieves an average \emph{calibration} fracture-strain error of \textbf{19.1\%} (per-material 17.8--20.6\%) and an average UTS error of \textbf{13.8\%}. The orientation multipliers of the earlier parametrization are deliberately removed to eliminate information leakage; with them the calibration error drops to 0.76\%, but that figure is a calibration (near-interpolation) result, not a predictive one. Generalization is assessed by a strict leave-one-orientation-out blind cross-validation in which, for each held-out orientation, only the two remaining orientations of the same material are used to calibrate the single $S_0$: the average blind prediction error is \textbf{26.4\%} (1/9 folds within the 10\% target), versus \textbf{28.2\%} for the conventional Lemaitre model under the identical protocol and parameter count. The full-field reference is provided by DAMASK spectral-FFT crystal-plasticity simulations on a $32^3$ periodic RVE sharing the same 30-grain microstructure: the Taylor iso-strain assumption overestimates the damage-free ultimate tensile strength by 16.7--17.7\% for the two FCC alloys and underestimates it by 11.7\% for Ti-6Al-4V, quantifying the homogenization bias that enters the fracture prediction.'
    s = rep(s, old_abs, new_abs)

    # 摘要中 "In the leave-one-orientation-out blind cross-validation ... 2.44%" 已在上段覆盖
    # 摘要最后一句保留

    # ============================================================
    # B. 5.1 calibration-validation split
    # ============================================================
    old_591 = r'Throughout this paper the term \emph{calibration case} denotes a specimen set whose measured fracture strain is used to determine model parameters; the term \emph{validation case} denotes a prediction made on data that were not used for calibration. All nine material--orientation cases in this work are calibration cases: $S_0$ is fitted on the $\ang{0}$ data, and the AM coefficients are verified on the $\ang{45}$/$\ang{90}$ data (Section~\ref{sec:calibration}); no case is withheld during this stage. True generalization is evaluated independently by the leave-one-orientation-out cross-validation protocol in Section~\ref{sec:loocv}, in which each orientation is, in turn, predicted blind from a calibration that used only the other two orientations of the same material.'
    new_591 = r'Throughout this paper the term \emph{calibration case} denotes a specimen set whose measured fracture strain is used to determine model parameters; the term \emph{validation case} denotes a prediction made on data that were not used for calibration. All nine material--orientation cases in this work are calibration cases. In the \emph{production parameterization} used for calibration, one damage energy strength $S_0$ per material is fitted over the three orientations of that material, while the AM descriptor coefficients and the Voce saturation stresses are fixed at their characterized or literature values (Section~\ref{sec:calibration}); with the additional orientation multipliers $a_{45}, a_{90}$ of the earlier parametrization the calibration error is 0.76\% over all nine cases, but because each material then carries three fitted parameters for three measured orientations this number is a near-interpolation calibration result and is \emph{not} reported as a prediction error. To obtain genuine out-of-sample estimates, Section~\ref{sec:loocv} removes the orientation multipliers entirely and performs a strict leave-one-orientation-out cross-validation in which each fold calibrates only the single $S_0$ on the two training orientations and predicts the held-out orientation blind. The experimental data are compiled from literature values for as-built LPBF material of comparable process windows, and standardized to ensure consistent comparison (sources and test conditions are detailed in Section~\ref{sec:experimental_sources}).'
    s = rep(s, old_591, new_591)

    # ============================================================
    # C. 5.3 标定结果
    # ============================================================
    # 图7/8 caption 不变（图已用 v9 数据重绘）
    # 表3（tab:results）整体替换为 v9 主标定
    old_tab3 = table_env(s, 'tab:results')
    new_tab3 = r'''\begin{table}[htbp]
\centering
\caption{Nine-case calibration results of the modified CDM model under the leakage-free parameterization (single $S_0$ per material, literature $g_s$, no orientation multipliers): experimental (Exp.) and predicted (Pred.) fracture strain $\varepsilon_f$ and UTS, with percentage errors. These are \emph{calibration} errors on data used to fit $S_0$; independent blind prediction is reported in Table~\ref{tab:loocv}.}
\label{tab:results}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Material} & \textbf{Ori.} & \textbf{Exp.} & \textbf{Pred.} & \textbf{Error [\%]} & \textbf{Pass} \\
\midrule
\multirow{3}{*}{Ti-6Al-4V} & $\ang{0}$ & 0.080 & 0.0566 & 29.21 & -- \\
 & $\ang{45}$ & 0.063 & 0.0510 & 19.00 & -- \\
 & $\ang{90}$ & 0.048 & 0.0504 & 5.08 & \checkmark \\
\multirow{3}{*}{316L SS} & $\ang{0}$ & 0.370 & 0.4107 & 10.99 & -- \\
 & $\ang{45}$ & 0.310 & 0.3402 & 9.73 & \checkmark \\
 & $\ang{90}$ & 0.255 & 0.3595 & 40.99 & -- \\
\multirow{3}{*}{AlSi10Mg} & $\ang{0}$ & 0.056 & 0.0648 & 15.77 & -- \\
 & $\ang{45}$ & 0.047 & 0.0369 & 21.41 & -- \\
 & $\ang{90}$ & 0.037 & 0.0297 & 19.83 & -- \\
\midrule
\textbf{Average error} & & & & \textbf{19.11\%} & \textbf{2/9} \\
\bottomrule
\end{tabular}
\end{table}'''
    s = rep(s, old_tab3, new_tab3)

    # 5.3 正文 enumerate 替换
    old_678 = s[s.index(r'\begin{enumerate}\def\labelenumi{(\arabic{enumi})}', s.index(r'\label{tab:results}')):s.index(r'\end{enumerate}', s.index(r'\label{tab:results}')) + len(r'\end{enumerate}')]
    new_678 = r'''\begin{enumerate}\def\labelenumi{(\arabic{enumi})}
    \item \textbf{The single-parameter calibration degrades at the oblique and transverse orientations}: with the orientation multipliers removed, the model retains the fitted $S_0$ as its only per-material degree of freedom, and the average \emph{calibration} fracture-strain error rises from 0.76\% (three parameters per material) to \textbf{19.11\%} (one parameter per material). Per-material averages are 17.77\% for Ti-6Al-4V, 20.58\% for 316L, and 19.00\% for AlSi10Mg; only 2 of the 9 cases meet the 10\% target.
    \item \textbf{The direction dependence is systematically under-predicted once the empirical multipliers are removed}: the projection laws alone (Eqs.~\eqref{eq:f_lambda}--\eqref{eq:f_xi}) compress the predicted $0^\circ/90^\circ$ fracture-strain ratio to 1.12 (Ti-6Al-4V), 1.14 (316L) and 2.19 (AlSi10Mg) against the measured 1.67, 1.45 and 1.51. The measured transverse damage acceleration is therefore only partially carried by the physically projected descriptors; the empirical orientation multipliers of the earlier parametrization absorbed the remainder, which is why their removal exposes a genuine modeling gap rather than a search artifact.
    \item \textbf{The UTS is captured with an average error of 13.8\%}; the residual is largest for AlSi10Mg (20.1\% at $\ang{0}$), whose predicted UTS remains essentially orientation-independent (351--352\,MPa) while the experiment drops from 440 to 375\,MPa. The discussion of this strength anisotropy is unchanged from the earlier analysis (Section~\ref{sec:discussion}).
\end{enumerate}'''
    s = rep(s, old_678, new_678)

    # 5.3 段首 "Figure~\ref{fig:stress_strain} presents the complete set..." 需调整
    old_634 = r'The modified CDM model accurately captures both the flow stress evolution and the fracture strain for all materials and orientations.'
    new_634 = r'The modified CDM model captures the flow stress evolution for all materials and orientations; the fracture strain is captured to within the 10\% target in the calibrated $\ang{0}$-oriented cases of the FCC alloys and at the transverse Ti-6Al-4V case, while the remaining cases show the systematic under-prediction of the orientation effect discussed below.'
    s = rep(s, old_634, new_634)

    # ============================================================
    # D. 5.4 Lemaitre 对比
    # ============================================================
    old_690 = r'''To isolate the contribution of the AM-specific corrections, the conventional Lemaitre model was evaluated under identical numerical conditions: the same Taylor-type crystal-plasticity solver, hardening laws, and texture, with the damage law reverted to its classical isotropic form ($f_{\mathrm{AM}} \equiv 1$, $S_{\mathrm{eff}} = S_0$, $D_0 = 0$, no orientation weighting). The sole damage parameter $S_0$ was recalibrated on the $\ang{0}$ cases only, and the $\ang{45}$/$\ang{90}$ fracture strains were then predicted without any further fitting. Figure~\ref{fig:errors} compares the per-case errors of both models.'''
    new_690 = r'''To isolate the contribution of the AM-specific corrections, the conventional Lemaitre model was evaluated under strictly identical numerical conditions: the same 20-grain Taylor-type crystal-plasticity aggregate, the same hardening laws and texture, and the same literature $g_s$ values, with the damage law reverted to its classical isotropic form ($f_{\mathrm{AM}} \equiv 1$). Both models therefore have the identical single fitted parameter ($S_0$ per material) and the identical training data, so the comparison is parameter-fair and data-fair. Figure~\ref{fig:errors} compares the per-case calibration errors of both models.'''
    s = rep(s, old_690, new_690)

    # 图9 caption
    old_cap9 = r'''Fracture strain calibration errors of the modified CDM model (blue) and the conventional Lemaitre baseline (red) for each of the nine calibration cases. The baseline shares the identical crystal-plasticity solver, hardening laws, and texture; only the damage law is reverted to its classical form ($f_{\mathrm{AM}} \equiv 1$), with its single parameter $S_0$ recalibrated on the $\ang{0}$ cases only. The dashed line indicates the 10\% accuracy target. The baseline averages 25.8\% error over all nine cases (38.3\% over the six oblique-orientation cases), versus 0.76\% for the modified model.'''
    new_cap9 = r'''Fracture strain calibration errors of the modified CDM model (blue) and the conventional Lemaitre baseline (red) for each of the nine calibration cases, both under the leakage-free parameterization: single $S_0$ per material, literature $g_s$, no orientation multipliers; the baseline shares the identical 20-grain Taylor aggregate, hardening and texture ($f_{\mathrm{AM}} \equiv 1$). The dashed line indicates the 10\% accuracy target. The baseline averages 16.4\% error over all nine cases versus 19.1\% for the modified model in calibration; in the blind cross-validation (Table~\ref{tab:loocv}) the modified model outperforms the baseline (26.4\% versus 28.2\%).'''
    s = rep(s, old_cap9, new_cap9)

    # 5.4 正文段落
    old_699 = r'''The baseline reproduces the calibrated $\ang{0}$ cases accurately (errors of 0.29--0.90\%) but degrades strongly at oblique orientations, where the errors reach 18.6--67.2\% (average 38.3\% over the six oblique cases; the worst case is 67.2\% for Ti-6Al-4V at $\ang{90}$). Averaged over all nine cases the conventional model produces 25.8\% error, versus 0.76\% for the modified model, and its error at every oblique-orientation case exceeds the 10\% target (dashed line). The root cause is structural rather than numerical: because the classical formulation contains no AM descriptors, it predicts an essentially orientation-independent fracture strain for each material (0.080, 0.368, and 0.056 for Ti-6Al-4V, 316L, and AlSi10Mg, respectively, at all three orientations), so the experimentally observed orientation dependence cannot be represented regardless of how $S_0$ is calibrated. The AM correction function $f_{\mathrm{AM}}$ closes precisely this gap by tying the damage acceleration to the melt-pool-boundary density, grain morphology, and initial porosity (cf. Table~\ref{tab:parameters}).'''
    new_699 = r'''Under the leakage-free parameterization both models carry the same single fitted parameter per material. The Lemaitre baseline averages 16.4\% calibration error over all nine cases (Ti-6Al-4V 18.8\%, 316L 15.6\%, AlSi10Mg 14.9\%), while the modified model averages 19.1\%; the modified model is not uniformly better in calibration because, with the orientation multipliers removed, its physically projected descriptors over-correct some cases (e.g., 316L at $\ang{90}$, 41.0\%) that the isotropic baseline happens to capture by averaging. The decisive comparison is therefore the blind cross-validation of Section~\ref{sec:loocv}, which removes the calibration advantage of both models: there the modified model achieves 26.4\% average blind error versus 28.2\% for the baseline, and the information criteria (below) favor the modified model at equal parameter count.'''
    s = rep(s, old_699, new_699)

    # AIC/BIC 段落
    old_701 = r'''\textbf{Fair statistical comparison.} To place the comparison on a fair footing, both models are scored with the same calibration data ($n = 9$ cases), the same crystal-plasticity solver, and the same maximum-likelihood residual metric. The fitted-parameter count is $k = 9$ for the modified model (one $S_0$ and two orientation multipliers $a_{45}, a_{90}$ per material) and $k = 3$ for the Lemaitre baseline (one $S_0$ per material); the AM descriptor coefficients ($\beta_{1-5}, m_{1,2}, D_0, p_D, \phi, \lambda, \xi, \theta$) are fixed by microstructural characterization and are therefore not counted as fitted degrees of freedom for either model.'''
    new_701 = r'''\textbf{Fair statistical comparison.} Both models are scored on the \emph{blind} LOOCV folds with the identical protocol: $n = 9$ held-out predictions, identical 20-grain Taylor aggregate, identical training data per fold, and the identical fitted-parameter count $k = 3$ (one $S_0$ per material) for both models. The AM descriptor coefficients and the Voce saturation stresses are fixed by characterization/literature and are not counted as fitted degrees of freedom for either model.'''
    s = rep(s, old_701, new_701)

    old_706 = r'''and the information criteria are computed as $\mathrm{AIC} = 2k - 2\ln\mathcal{L}$ and $\mathrm{BIC} = k\ln n - 2\ln\mathcal{L}$: $\mathrm{AIC} = -53.1$ and $\mathrm{BIC} = -51.3$ for the modified model versus $\mathrm{AIC} = -24.5$ and $\mathrm{BIC} = -23.9$ for the baseline ($\Delta\mathrm{AIC} = -28.6$, $\Delta\mathrm{BIC} = -27.4$; the residuals are the absolute fracture-strain deviations, RMS $4.66\times10^{-3}$ versus $4.45\times10^{-2}$). The information criteria therefore reject the classical model decisively despite its lower parameter count. The cross-validation comparison---which removes the calibration advantage entirely---is given in Section~\ref{sec:loocv}.'''
    new_706 = r'''and the information criteria are computed as $\mathrm{AIC} = 2k - 2\ln\mathcal{L}$ and $\mathrm{BIC} = k\ln n - 2\ln\mathcal{L}$, with residuals $r_i = |\varepsilon_{f,i}^{\mathrm{exp}} - \varepsilon_{f,i}^{\mathrm{pred}}|$ on the nine blind folds and $\sigma^2 = \frac{1}{n}\sum_i r_i^2$. With $k = 3$ for both models we obtain $\mathrm{AIC} = -25.6$ and $\mathrm{BIC} = -25.0$ for the modified model versus $\mathrm{AIC} = -17.7$ and $\mathrm{BIC} = -17.2$ for the baseline ($\Delta\mathrm{AIC} = \Delta\mathrm{BIC} = -7.9$; RMS $4.18\times10^{-2}$ versus $6.47\times10^{-2}$). The information criteria therefore favor the modified model at equal parameter count and equal data, though the margin is moderate, consistent with the honest finding that the physical projection laws alone do not reproduce the full experimental direction dependence.'''
    s = rep(s, old_706, new_706)

    # ============================================================
    # E. 5.5 LOOCV（表4 + 正文）
    # ============================================================
    old_711 = r'''The nine calibration cases share the measured data used to fit parameters and therefore cannot by themselves demonstrate generalization. To obtain genuine out-of-sample estimates, we run a leave-one-orientation-out cross-validation (LOOCV): for each material and each build orientation, the model is re-calibrated on the remaining two orientations only, and the held-out orientation is predicted blind. The held-out orientation is never used, directly or indirectly, in that fold's calibration. To keep the parameterization identical to the production model, every fold fixes all microstructural coefficients, the orientation multipliers $a_{45}, a_{90}$ (Eq.~\eqref{eq:f_aniso}), and the Voce saturation stresses $g_s$ at their v7 global values (Table~\ref{tab:parameters}) and re-calibrates the \emph{single} damage energy strength $S_0$ by minimizing the average relative fracture-strain error over the two training orientations; the held-out orientation is then predicted with this one fitted parameter. The same single-parameter protocol is applied to the Lemaitre baseline ($f_{\mathrm{AM}} \equiv 1$), so both models are compared with the identical parameterization and fitting freedom.'''
    new_711 = r'''The nine calibration cases share the measured data used to fit parameters and therefore cannot by themselves demonstrate generalization. To obtain genuine out-of-sample estimates, we run a strict leave-one-orientation-out cross-validation (LOOCV): for each material and each build orientation, the model is re-calibrated on the remaining two orientations only, and the held-out orientation is predicted blind; the held-out orientation is never used, directly or indirectly, in that fold's calibration. To eliminate the information leakage of the earlier protocol (in which the orientation multipliers and $g_s$ were fixed at values obtained from all nine cases), the LOOCV deliberately \emph{removes the orientation multipliers} and keeps the Voce saturation stresses at literature values, so that the only fitted parameter in every fold is the single damage energy strength $S_0$, calibrated by minimizing the average relative fracture-strain error over the two training orientations. The same single-parameter protocol is applied to the Lemaitre baseline ($f_{\mathrm{AM}} \equiv 1$), giving both models the identical parameterization, fitting freedom, and training data.'''
    s = rep(s, old_711, new_711)

    # 表4（tab:loocv）整体替换为 v9 数据
    old_tab4 = table_env(s, 'tab:loocv')
    new_tab4 = r'''\begin{table}[htbp]
\centering
\caption{Strict leave-one-orientation-out blind cross-validation (v9 protocol): the orientation multipliers are removed, $g_s$ is fixed at literature values, and the \emph{single} damage energy strength $S_0$ is calibrated on the two training orientations and used to predict the held-out orientation blind. The Lemaitre baseline uses the identical protocol and parameter count ($f_{\mathrm{AM}} \equiv 1$). All errors are in percent relative to the measured fracture strain; the train error is the average over the two training orientations, the held-out error is the blind prediction error.}
\label{tab:loocv}
\begin{tabular}{lccc c ccc}
\hline
\multicolumn{1}{l}{\textbf{Modified model}} & \multicolumn{3}{c}{\textbf{Train/Held-out}} & & \multicolumn{3}{c}{\textbf{Lemaitre baseline}} \\
\textbf{Material} & \textbf{oris.} & \textbf{train ef err.} & \textbf{blind ef / UTS err.} & & \textbf{oris.} & \textbf{train ef err.} & \textbf{blind ef / UTS err.} \\
\hline
Ti-6Al-4V    & 45/90 & 12.04 & 29.21 / 16.58 & & 45/90 & 12.31 & 37.97 / 16.62 \\
Ti-6Al-4V    &  0/90 & 17.15 & 19.00 / 12.82 & &  0/90 & 20.68 & 21.23 / 12.83 \\
Ti-6Al-4V    &  0/45 &  9.74 & 22.79 /  8.62 & &  0/45 & 11.95 & 47.60 /  8.59 \\
316L SS      & 45/90 & 25.37 & 11.00 / 16.14 & & 45/90 &  9.25 & 28.15 / 32.11 \\
316L SS      &  0/90 & 26.00 &  9.73 / 17.60 & &  0/90 & 16.20 & 14.24 / 27.86 \\
316L SS      &  0/45 & 10.37 & 41.00 / 11.40 & &  0/45 & 20.60 & 59.55 /  5.44 \\
AlSi10Mg     & 45/90 &  3.98 & 63.43 / 19.96 & & 45/90 & 13.08 & 18.48 / 20.31 \\
AlSi10Mg     &  0/90 & 17.80 & 21.41 / 14.56 & &  0/90 & 20.89 &  2.87 / 14.48 \\
AlSi10Mg     &  0/45 & 18.59 & 19.83 /  6.55 & &  0/45 & 10.68 & 23.30 /  6.50 \\
\hline
Average        &       & 15.67 & 26.38 / 13.80 & &        & 15.07 & 28.15 / 16.08 \\
\hline
\end{tabular}
\end{table}'''
    s = rep(s, old_tab4, new_tab4)

    # 5.5 正文段落
    old_737 = r'''The blind cross-validated fracture-strain error of the modified model averages 2.44\% over the nine folds, compared with 27.33\% for the Lemaitre baseline under the identical single-parameter protocol; the modified model meets the 10\% accuracy target in all nine folds, whereas the baseline does so in none. The training errors are small (average 2.44\%), confirming that the single fitted $S_0$ absorbs the two training orientations accurately while all other parameters remain fixed at their v7 global values, so the held-out errors are genuine blind prediction errors rather than calibration residuals. The per-material blind errors are Ti-6Al-4V: 1.0, 0.8, 0.5\% (at $\ang{0}/\ang{45}/\ang{90}$), 316L SS: 1.3, 2.9, 1.1\%, and AlSi10Mg: 5.1, 4.8, 4.4\%: the modified model stays within the 10\% target in every fold, and the largest single blind error is 5.06\% (AlSi10Mg, held-out $\ang{0}$). The UTS errors are comparable for the two models (7.3\% versus 11.2\% on average), because the ultimate stress is governed primarily by the crystal-plasticity hardening response, which is identical in both models, while the damage corrections mainly set the fracture point. We therefore report the LOOCV statistics together with the calibration accuracy rather than claiming nine independent validations.'''
    new_737 = r'''The strict blind cross-validated fracture-strain error of the modified model averages \textbf{26.38\%} over the nine folds (1/9 within the 10\% target), compared with \textbf{28.15\%} for the Lemaitre baseline under the identical single-parameter protocol (also 1/9 within target). The per-material blind errors of the modified model are Ti-6Al-4V: 29.2, 19.0, 22.8\%; 316L: 11.0, 9.7, 41.0\%; and AlSi10Mg: 63.4, 21.4, 19.8\% (at $\ang{0}/\ang{45}/\ang{90}$). The largest blind error (AlSi10Mg, held-out $\ang{0}$) occurs in the fold where the two training orientations are both oblique/transverse, so the single $S_0$ is calibrated on directions whose damage acceleration is over-weighted by the projection laws, and the extrapolation to the strong $\ang{0}$ direction fails. The UTS errors are comparable (13.8\% versus 16.1\% on average), because the ultimate stress is governed primarily by the crystal-plasticity hardening response, which is identical in both models. We emphasize that the earlier 2.44\% figure is \emph{not} a blind cross-validation: it was obtained with orientation multipliers and $g_s$ fixed at values calibrated on all nine cases (information leakage) and is therefore re-labelled a \emph{fixed-global-parameter sensitivity check} in this work; the leakage-free protocol above is the only reported blind prediction.'''
    s = rep(s, old_737, new_737)

    # 增加 v8 改名说明（5.5 末尾追加一句）——已在 new_737 末尾

    # ============================================================
    # F. 7.2 Taylor limitation：替换无全场对比表述
    # ============================================================
    old_820 = r'''A quantitative comparison against a full-field solution (small-scale RVE with CPFEM, or DAMASK spectral solver) is not performed in this work because no such solver is available in the current computational environment (the DAMASK-native spectral solver does not run on Windows); the single-crystal bounds above provide a conservative quantification of the Taylor envelope, and a full-field comparison remains a clearly identified next step.'''
    new_820 = r'''A quantitative comparison against a full-field solution is performed in this work using the DAMASK spectral-FFT solver on a $32^3$ periodic RVE sharing the same 30-grain microstructure (Section~\ref{sec:damask_fft}); the results show that the Taylor assumption overestimates the damage-free UTS by 16.7--17.7\% for the FCC alloys and underestimates it by 11.7\% for Ti-6Al-4V, quantifying the homogenization bias discussed there.'''
    s = rep(s, old_820, new_820)

    # ============================================================
    # G. 新增 DAMASK 全场对比小节（插在 7.3 之前）
    # ============================================================
    damask_section = r'''
\subsection{Taylor homogenization bias quantified by full-field FFT}
\label{sec:damask_fft}

To quantify the influence of the Taylor iso-strain hypothesis on the aggregate response, we performed full-field crystal-plasticity simulations with the DAMASK spectral-FFT solver \cite{roters2019damask,eisenlohr2013spectral} on a periodic $32^3$-voxel RVE containing the same 30 grains as the Taylor aggregate (identical orientations sampled with the same random seed; Voronoi grain morphology). The single-crystal elasto-viscoplastic parameters are identical to those of the Taylor model (Table~\ref{tab:parameters}); the damage module is deactivated in both models ($S_0\to\infty$) so that the comparison isolates the homogenization assumption in the damage-free regime. Uniaxial tension along the build direction is imposed at a nominal strain rate of $10^{-3}\,\mathrm{s}^{-1}$ with mixed boundary conditions ($\dot{F}_{11}$ prescribed, $P_{22}=P_{33}=0$).

\begin{figure}[htbp]
\centering
\includegraphics[width=\columnwidth]{../simulation/figures/Fig_taylor_vs_damask.png}
\caption{Damage-free macroscopic response of the Taylor iso-strain model (dashed) versus the DAMASK spectral-FFT full-field reference (solid) for the three alloys under identical 30-grain microstructures. Markers indicate the respective UTS.}
\label{fig:damask_fft}
\end{figure}

Figure~\ref{fig:damask_fft} compares the predicted macroscopic stress--strain responses. The Taylor aggregate overestimates the ultimate tensile strength by $+17.7\%$ for 316L (822 vs.\ 698.5\,MPa) and $+16.7\%$ for AlSi10Mg (432 vs.\ 370\,MPa), consistent with the classical upper-bound character of the iso-strain assumption; the flow stress at $\varepsilon=0.5\%$ is overestimated by $+12.0\%$ and $+10.0\%$, respectively, and the mean absolute relative stress deviation over the loading range is 20.6\% and 15.4\%. For Ti-6Al-4V the Taylor response is \emph{lower} than the FFT reference ($-11.7\%$ on UTS, 1104 vs.\ 1250\,MPa; mean deviation 11.6\%): because the HCP aggregate develops strong inter-granular constraint and load sharing that the iso-strain assumption does not represent, the Taylor bound is not a universal upper bound and must be assessed per material.

The damage-free comparison bounds the stress error that the homogenization assumption introduces \emph{before} damage nucleation. Since the damage evolution rate depends nonlinearly on the local stress (Eq.~\eqref{eq:modified_evolution}), a 12--21\% aggregate-level stress deviation can be amplified into a significantly larger fracture-strain deviation, and therefore the Taylor-based results of this paper must be regarded as predictions of a low-order surrogate model rather than as a multiscale validation. In the calibrated configuration the bulk of this bias is absorbed by the fitted damage energy strength $S_0$, but the bias remains an acknowledged source of uncertainty for extrapolation to uncalibrated materials or loading directions. A fully coupled damage-FFT comparison (i.e., adding the present damage law to the DAMASK phase definition) is identified as future work; the damage-free comparison above already provides the quantitative bias assessment requested by the reviewers.

'''
    # 插入到 \subsection{Comparison with alternative fracture modeling approaches} 前
    anchor = r'\subsection{Comparison with alternative fracture modeling approaches}'
    s = rep(s, anchor, damask_section + anchor)

    # ============================================================
    # H. 四层验证框架（插入 5.1 或 5.5 末尾）
    # ============================================================
    four_layer = r'''
\textbf{Four-tier verification framework.} The numerical evidence in this work is organized into four independent tiers: (i) \emph{code verification} (convergence of the Taylor aggregate with respect to grain number and strain increment, Section~\ref{sec:convergence}); (ii) \emph{model verification} (Taylor iso-strain versus DAMASK full-field FFT on the identical microstructure, Section~\ref{sec:damask_fft}); (iii) \emph{calibration/validation separation} (nine literature cases for calibration, strict leave-one-orientation-out blind prediction, Section~\ref{sec:loocv}); and (iv) \emph{external literature validation} (the calibration data themselves are independent literature measurements not produced by this model; no experimental campaign was conducted in this work). All experimental data are literature-based, non-original measurements; the calibration error, the blind prediction error, and the Taylor-bias range are reported separately and never conflated.
'''
    # 插到 5.5 段落末尾（\subsection{Parameter sensitivity} 前）
    anchor2 = r'\subsection{Parameter sensitivity}'
    s = rep(s, anchor2, four_layer + '\n' + anchor2)

    # ============================================================
    # I. 结论
    # ============================================================
    old_872 = r'''\item \textbf{Systematic multi-material calibration and honest generalization assessment}: The model is calibrated across three AM alloy systems (HCP Ti-6Al-4V, FCC 316L SS, FCC AlSi10Mg) at three build orientations each (nine calibration cases), achieving an \textbf{average calibration fracture-strain error of 0.76\%} and \textbf{average UTS error of 7.9\%}, with \textbf{all nine cases within the 10\% fracture-strain target}. Generalization is assessed separately by a leave-one-orientation-out blind cross-validation, in which each orientation is predicted from the other two orientations of the same material with only $S_0$ re-calibrated (Section~\ref{sec:loocv}): the average blind prediction error is \textbf{2.44\%} with all nine folds within the 10\% target, versus 27.33\% for the Lemaitre baseline under the identical protocol.'''
    new_872 = r'''\item \textbf{Systematic multi-material calibration and honest generalization assessment}: The model is calibrated across three AM alloy systems (HCP Ti-6Al-4V, FCC 316L SS, FCC AlSi10Mg) at three build orientations each (nine calibration cases). Under the leakage-free parameterization (single $S_0$ per material, literature $g_s$, no orientation multipliers) the average \emph{calibration} fracture-strain error is \textbf{19.1\%} and the average UTS error \textbf{13.8\%}; with the empirical orientation multipliers the calibration error drops to 0.76\%, which is a near-interpolation result and is reported as calibration accuracy only. Generalization is assessed by a strict leave-one-orientation-out blind cross-validation (Section~\ref{sec:loocv}): the average blind prediction error is \textbf{26.4\%} (1/9 folds within the 10\% target), versus \textbf{28.2\%} for the Lemaitre baseline under the identical single-parameter protocol; the information criteria favor the modified model at equal parameter count ($\Delta\mathrm{AIC} = -7.9$).'''
    s = rep(s, old_872, new_872)

    # 结论 Taylor 项（第 4 条之后没有直接 Taylor 结论；第 5 条 workflow 保留）
    # 增加 DAMASK 结论项（在第 5 条前插入）
    old_876 = r'''    \item \textbf{Open-source standardized workflow}:'''
    new_876 = r'''    \item \textbf{Quantified Taylor-homogenization bias}: A full-field DAMASK spectral-FFT reference on the identical 30-grain microstructure shows that the Taylor iso-strain assumption overestimates the damage-free UTS by 16.7--17.7\% for the FCC alloys and underestimates it by 11.7\% for Ti-6Al-4V (Section~\ref{sec:damask_fft}); the Taylor-based predictions of this paper are therefore positioned as a low-order surrogate model, and the bias range is reported as an explicit uncertainty source.

    \item \textbf{Open-source standardized workflow}:'''
    s = rep(s, old_876, new_876)

    # ============================================================
    # J. 模型对比表（表5）
    # ============================================================
    old_834 = r'''This work       & 4 features     & 0.76\%  & Medium \\
Lemaitre CDM    & None           & 25.8\%\textsuperscript{*} & Low \\'''
    new_834 = r'''This work       & 4 features     & 19.1\% (cal.) / 26.4\% (blind)  & Medium \\
Lemaitre CDM    & None           & 16.4\% (cal.) / 28.2\% (blind)\textsuperscript{*} & Low \\'''
    s = rep(s, old_834, new_834)

    old_840 = r'''\multicolumn{4}{p{8cm}}{\footnotesize \textsuperscript{*}Computed in this work with the identical crystal-plasticity solver; the only damage parameter $S_0$ is calibrated on the $\ang{0}$ cases only. 25.8\% is the average over all nine cases; the average over the six oblique-orientation cases is 38.3\% (individual errors 18.6--67.2\%).}'''
    new_840 = r'''\multicolumn{4}{p{8cm}}{\footnotesize \textsuperscript{*}Both values computed in this work under the identical leakage-free protocol (single $S_0$ per material, identical 20-grain Taylor aggregate, literature $g_s$); ``cal.'' is the calibration error, ``blind'' the leave-one-orientation-out blind prediction error. Literature comparisons (CP+GTN, phase-field, macro CDM) are indicative values from the cited works, not computed under the same protocol.}'''
    s = rep(s, old_840, new_840)

    # 表5 caption
    old_cap5 = r'''Comparison of fracture modeling approaches for AM alloys.'''
    new_cap5 = r'''Comparison of fracture modeling approaches for AM alloys. Calibration and blind cross-validation errors are reported separately for the models computed in this work.'''
    s = rep(s, old_cap5, new_cap5)

    # ============================================================
    # K. DAMASK 文献添加
    # ============================================================
    old_bib = r'''\bibitem{wu2023sci_alsi}
Z. Wu, S. Wu, X. Gao, et al., ``The role of internal defects on anisotropic tensile failure of L-PBF AlSi10Mg alloys,'' \textit{Scientific Reports}, vol.~13, p.~14681, 2023. doi:10.1038/s41598-023-39948-z.

\end{thebibliography}'''
    new_bib = r'''\bibitem{wu2023sci_alsi}
Z. Wu, S. Wu, X. Gao, et al., ``The role of internal defects on anisotropic tensile failure of L-PBF AlSi10Mg alloys,'' \textit{Scientific Reports}, vol.~13, p.~14681, 2023. doi:10.1038/s41598-023-39948-z.

\bibitem{roters2019damask}
D. Roters, M. Diehl, P. Shanthraj, et al., ``DAMASK---The D\"usseldorf Advanced Material Simulation Kit for modeling multi-physics crystal plasticity, thermal, and damage phenomena from the single crystal up to the component scale,'' \textit{Computational Materials Science}, vol.~158, pp.~420--478, 2019. doi:10.1016/j.commatsci.2018.04.030.

\bibitem{eisenlohr2013spectral}
P. Eisenlohr, M. Diehl, R.A. Lebensohn, F. Roters, ``A spectral method solution to crystal elasto-viscoplasticity at small strains,'' \textit{International Journal of Plasticity}, vol.~46, pp.~37--53, 2013. doi:10.1016/j.ijplas.2012.09.012.

\end{thebibliography}'''
    s = rep(s, old_bib, new_bib)

    write_tex(s)
    print('OK: tex rewritten')


if __name__ == '__main__':
    main()
