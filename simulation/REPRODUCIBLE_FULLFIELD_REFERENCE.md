# Reproducible in-house full-field CPFEM reference

This file documents the full-field crystal-plasticity / continuum-damage-mechanics
(CPFEM/CP-CDM) references used to quantify the Taylor iso-strain homogenization bias
against a spatially resolved solver, rather than against the 0-D Taylor aggregate alone.

## Scope and honest status

- These solvers are **in-house research code, not a third-party framework**. The paper
  makes no claim that they were run inside DAMASK / FEniCSx / MOOSE: in the present
  environment those solvers are not installed (see `supplementary.tex`, Section S9), and
  DAMASK 3.1.0 ships no FEM solver and its spectral-FFT damage branch (isobrittle) diverges
  at the damage-activation increment. The spectral-FFT solver is used **only** as a
  damage-free full-field reference.
- The coupled-damage full-field references below are the authors' own implementations,
  released so that every reported bias number can be reproduced from a fixed seed, mesh
  and strain increment.
- **Third-party engine cross-check (Supplementary Section S13).** The identical coupled-damage
  law was re-driven on the externally maintained finite-element engine `scikit-fem` 12.0.2
  (a `pip`-installable library supplying mesh, shape functions, quadrature, assembly and linear
  solvers, but no constitutive content) on the plane-strain 4x4 Q4 RVE, in a WSL2 Ubuntu-24.04
  Python 3.12 venv. This is the only external FE engine installable here (MOOSE/FEniCSx/PRISMS-PF
  are not; DAMASK ships no FEM solver; the conda/`fenics-dolfinx` routes are blocked). The Ti-6Al-4V
  localisation-onset fracture strains agree with the in-house reference to within 11-17% at all three
  orientations and reproduce the same trend; the more uniform AlSi10Mg case diverges (-54%) and is
  reported honestly as a mesh-/solver-sensitivity of the localisation criterion, not as agreement. This
  cross-check validates the discretisation and linear solve against an independent framework; it is
  **not** a phase-field or fully coupled platform-solver result, which remain future work.

## Solvers

| Solver | File | Discretisation | Role |
|--------|------|----------------|------|
| 2-D plane-strain CPFEM (Q4) | `simulation/fullfield_2d_cpfem.py` | explicit Q4 grains | coupled-damage full-field reference; localisation-onset fracture strain |
| 3-D CPFEM (3x3x3 hex) | `simulation/fullfield_3d_cpfem.py` | 27-grain 3x3x3 hex, mesh-converged to 64 grains | coupled-damage full-field reference; aggregate-D_c and localisation-onset criteria |
| 3-D nonlocal CPFEM | `simulation/fullfield_3d_cpfem_nonlocal.py` | integral-type nonlocal damage operator | mesh-dependence compression (Supplementary Section S1) |
| 2-D external-engine cross-check | `simulation/fullfield_skfem.py` (+ `simulation/compare_skfem.py`) | plane-strain 4x4 Q4, `scikit-fem` 12 assembly/solve | third-party FE cross-check of the 2-D reference (Section S13) |

The damage law, orientation set, hardening and `f_AM` coefficients are taken unchanged
from the production parameter table (main-text Table 1), so the comparison isolates the
homogenization scheme and nothing else.

## Fixed reproducibility inputs

- Random seed: `42` for every aggregate/orientation draw.
- Converged strain increment: `Delta_epsilon = 1e-3` (3-D reference reported at this
  increment; the increment-resolved sensitivity is tabulated in Supplementary Section S1).
- Mesh convergence: fracture-strain statistics are stable to **below one percentage point**
  from 27 to 64 grains within a fixed criterion.
- Grain-count / increment convergence check: `simulation/convergence_test.py`.

## One-command reproduction

```
python simulation/convergence_test.py            # mesh- and increment-convergence evidence
python simulation/fullfield_2d_cpfem.py           # 2-D plane-strain coupled-damage bias
python simulation/fullfield_3d_cpfem.py           # 3-D 3x3x3 coupled-damage bias
python simulation/fullfield_3d_cpfem_nonlocal.py  # nonlocal regularization (Section S1)
python simulation/merge_ff3d_results.py           # assemble the 3-D orientation table
```

The external-engine cross-check (Section S13) runs under the WSL2 venv described above, not under the
Windows Python 3.13 production stack:

```
# inside the WSL2 scikit-fem venv (see Section S13 provenance)
python simulation/fullfield_skfem.py               # third-party FE coupled-damage RVE (4 cases)
python simulation/compare_skfem.py                 # skfem vs in-house vs un-truncated Taylor table
```

Mesh-refinement cross-check (external engine 16->64 grains, Section S13): set `FF_NGRID=8` before running
`fullfield_skfem.py` (emits `fullfield_skfem_results_ng8.json`). The in-house 2D CPFEM honours the same
`FF_NGRID` env but its damped-Newton collapses the step near localisation at 8x8 and was not carried to
convergence here, so the mesh-stability statement is reported for the external engine only.

## Number-to-source binding

Every full-field bias number quoted in the manuscript is produced by one of the scripts
above under the fixed seed/increment/mesh, and is version-fingerprinted in
`supplementary.tex`, Section S4 (reproducibility: version fingerprinting and
number-to-source binding). Where a value in an earlier draft was superseded, the
superseded figure is retained only in the Supplementary Material and never in the main
text.

## Post-hoc statistical analysis (directional-gain hierarchical Bayes)

- `simulation/directional_bayes.py` -> `simulation/output/directional_bayes_results.json`
  (Supplementary Section S7, main-text Sec. uncertainty). An `emcee` affine-invariant MCMC
  (seed `42`, 64 walkers, 12000 steps, 3000 burn-in, thin 15) fitting a per-group
  directional gain `beta_g ~ N(mu, tau^2)` to the within-group-centred log-fracture-strain
  contrasts of the complete 0/45/90 triples in `output/blind_eval_expanded_results.json`
  (six groups, 18 observations). This is a **purely post-hoc analysis**: it re-uses stored
  blind-set predictions and changes no production parameter, no `S_0`, and no `f_AM`
  number. Headline outputs: pooled `mu_beta = 0.121` HPD89 `[-0.302, 0.598]` (brackets zero);
  only the columnar `Ti64.v12` group has `beta_g = 0.36` with directional capture `R^2 = 0.94`
  (0/90 ratio exp 1.667 / mod 1.885 / lem 0.988); every FCC / reversed-polarity group has
  `R^2 < 0`. The Lemaitre per-group gain is not identifiable (predicted contrast ~ 0) and is
  reported through the vanishing contrast and `R^2 ~ 0`, not an unstable `beta`.
- Environment: Windows Python 3.13 production stack; requires `emcee` (available), NumPy,
  SciPy. Reproduce with `python simulation/directional_bayes.py`.

## Post-hoc statistical upgrade (Bayes factor, TOST, bootstrap Kendall)

- `simulation/statistical_upgrade_bayes.py` -> `simulation/output/statistical_upgrade_bayes_results.json`
  (Supplementary Section S11, main-text expanded-blind paragraph Sec. loocv). This is a **purely
  post-hoc** re-analysis of the stored paired blind errors in `output/blind_eval_expanded_results.json`
  (`abs_err_mod_pct` vs `abs_err_lem_pct`, `n = 23`); it changes no production parameter, no `S_0`, and
  no `f_AM` number. Three routes: (i) a Rouder Jeffreys--Zellner--Sioukas Bayes factor `BF01` on the
  paired difference (mean `-1.627`, SD `41.12`, SE `8.574` points) with a Cauchy prior on the
  standardized effect of width `0.2/0.35/0.5/1.0` giving `BF01 = 1.853/2.588/3.371/6.145` (H0 and H1
  are integrated on the *same* log-sigma grid under the *same* improper-prior constants so the
  `sigma^-n` factors cancel); (ii) a TOST equivalence test rejecting non-equivalence at `+/-20` pp
  (`p = 0.0217`) and `+/-25` pp (`p = 0.0062`) but not `+/-10` pp (`p = 0.1697`) or `+/-15` pp
  (`p = 0.0666`); (iii) a group-bootstrap (`20{,}000` resamples, seed `42`, sign-flip permutation) of
  the mean Kendall `tau` over the eight source--material groups: descriptor `+0.208` CI `[-0.333, 0.708]`
  `p = 0.53`, Lemaitre `-0.208` CI `[-0.750, 0.333]` `p = 0.51`, difference `+0.417` CI `[-0.417, 1.250]`
  `p = 0.43`. Honest reading: moderate-to-strong evidence for mean-accuracy equivalence and a positive
  but not individually significant directional advantage.
- Environment: Windows Python 3.13 production stack; NumPy, SciPy only. Reproduce with
  `python simulation/statistical_upgrade_bayes.py`.
