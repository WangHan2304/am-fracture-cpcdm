# DAMASK 对比节完整 LaTeX 文本（待 v9 完成后与 LOOCV 数字一并写入）

## 建议插入位置
Section 7 (Discussion) → 7.2 的 Taylor limitation 条目之后，作为新的
\subsubsection{Taylor homogenization bias quantified by full-field FFT}
或在 7.2 Limitations 内替换原有 "no full-field comparison" 语句。

## LaTeX 文本

\subsubsection{Quantified Taylor-homogenization bias via full-field FFT reference}
\label{sec:damask_fft}

To quantify the influence of the Taylor iso-strain hypothesis on the aggregate
response, we performed full-field crystal-plasticity simulations with the
DAMASK spectral FFT solver \cite{roters2019damask,eisenlohr2013spectral}
(spectral\_basic solver, periodic boundary conditions) on small periodic RVEs of
$32^3$ voxels containing the same 30 grains as the Taylor aggregate (identical
orientations sampled with the same random seed; Voronoi grain morphology).
The single-crystal elasto-viscoplastic parameters are identical to those of the
Taylor model in Table~\ref{tab:parameters}; the damage module is deactivated in
both models ($S_0 \to \infty$) so that the comparison isolates the
homogenization assumption in the damage-free regime. Uniaxial tension along the
build direction ($\ang{0}$) at a nominal strain rate of $10^{-3}\,\mathrm{s}^{-1}$
is imposed with mixed boundary conditions ($\dot F_{11}$ prescribed, $P_{22}=P_{33}=0$).

Figure~\ref{fig:damask_fft} compares the predicted macroscopic stress--strain
responses. The Taylor aggregate overestimates the ultimate tensile strength by
$+17.7\%$ for 316L (822 vs.\ 698.5\,MPa) and $+16.7\%$ for AlSi10Mg
(432 vs.\ 370\,MPa), consistent with the classical upper-bound character of the
iso-strain assumption; the flow stress at $\varepsilon=0.5\%$ is overestimated by
$+12.0\%$ and $+10.0\%$, respectively, and the mean absolute relative stress
deviation over the loading range is 20.6\% and 15.4\%. For Ti-6Al-4V the
Taylor response is \emph{lower} than the FFT reference ($-11.7\%$ on UTS,
1104 vs.\ 1250\,MPa; mean deviation 11.6\%): because the HCP aggregate
develops strong inter-granular constraint and load sharing that the
iso-strain assumption does not represent, the Taylor bound is not a universal
upper bound and must be assessed per material.

The damage-free comparison bounds the stress error that the homogenization
assumption introduces \emph{before} damage nucleation. Since the damage
evolution rate depends nonlinearly on the local stress (Eq.~\eqref{eq:modified_evolution}),
a $12$--$21\%$ aggregate-level stress deviation can be amplified into a
significantly larger fracture-strain deviation, and therefore the Taylor-based
results of this paper must be regarded as predictions of a low-order surrogate
model rather than as a multiscale validation. In the calibrated configuration the
bulk of this bias is absorbed by the fitted damage energy strength $S_0$ (the
calibration absorbs the aggregate-level mismatch), but the bias remains an
acknowledged source of uncertainty for extrapolation to uncalibrated materials
or loading directions. A fully coupled damage-FFT comparison (i.e., adding the
present damage law to the DAMASK phase definition) is identified as future work;
the damage-free comparison above already provides the quantitative bias
assessment requested by the reviewers.
