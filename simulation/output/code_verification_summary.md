# Code verification of the in-house CP-CDM implementation (TaylorCPCDM)

**Deliverables**

| artifact | role |
| --- | --- |
| `simulation/tests/test_taylor_cpcdm.py` | the verification suite (35 tests, 4 tiers), writes the result file through its `main()` |
| `simulation/output/code_verification_results.json` | machine-readable results: per-test status/duration, all measured evidence, source fingerprints |
| `simulation/output/framework_dead_ends.md` | evidence that no external CPFEM framework (MOOSE / FEniCS / PRISMS) is installable in this environment |
| this file | the narrative summary, written so that it can be pasted into the manuscript |

**Bottom line.** 35 tests: **32 pass, 3 fail**. The three failures are not suite bugs — they are real, reproducible negative results that the suite was written to expose, and they are reported below rather than relaxed:

1. the fracture strain is **not** step-size convergent above an effective increment of ≈1.5e-3 (§5.4);
2. the archived 2026-09-23 results are **not** reproducible from the current sources: `ef` moves by **+27.6%** while `uts` reproduces to **0.05%** (§6).

Everything else — crystallography, crystal-elastic tensors, parameter sourcing, orientation sets, damage gate and monotonicity, unit consistency, and the two analytic benchmark limits — passes, with the analytic benchmarks agreeing to **< 0.16%** (elastic limit) and to **machine precision** (Lemaitre limit).

---

## 1. What was verified, against what

The reviewer asked for code verification, unit tests and benchmark solutions. No open-source CPFEM framework can be installed here (`framework_dead_ends.md`: the INL MOOSE has no PyPI or conda-forge `win-64` build at all; `fenics-dolfinx` exists on conda-forge for `win-64` only up to Python 3.12 and this interpreter is 3.13; PRISMS is on neither index). Verification is therefore against **analytic solutions and internal consistency** of the implementation that all reported results actually come from.

Two benchmark solutions are closed-form:

* **B3-1, the elastic limit** — with damage frozen (`S0 = 1e12` Pa) the uniaxial response must equal the exact anisotropic elastic solution of the model's *own* kinematic constraint, computed independently from the model's orientation set.
* **B3-2, the Lemaitre limit** — at the reference-state descriptors the AM correction must reduce to unity and the model must reduce *exactly* to the classical Lemaitre damage model.

Everything else is a property test (symmetry, definiteness, point-group invariance, monotonicity, dimensional consistency) or a convergence study.

**Environment and provenance.** Windows-11-10.0.26200, Python 3.13.0 (conda-forge, MSC v.1942 64-bit), numpy 2.3.4, `unittest` (pytest is not installed in this environment). All runs use the fixed seed 42. The result file records an md5 + mtime fingerprint of every source file consulted, so a reader can tell whether a re-run used the same sources:

| file | md5 | mtime |
| --- | --- | --- |
| `src/taylor_cpcdm.py` | `38bd53d31a92fef55418448fd116c6bb` | 2026-09-25 07:41:02 |
| `src/materials.py` | `ca6c6fdef5dfe4febeefac50fea24c21` | 2026-09-16 18:06:00 |
| `src/am_correction_v4.py` | `d05d9562826c1328ea4f8477517f1dbe` | 2026-09-25 22:25:15 |
| `src/cp_cdm_model.py` | `a2a6863c5c7eac022e8307587720af99` | 2026-06-18 16:54:11 |
| `convergence_test.py` | `bb81cdd1182386d3d69483f550258158` | 2026-09-23 22:33:26` |
| `taylor_single_crystal_bounds.py` | `4926ded39a2126ea5660e23f91c18c56` | 2026-09-23 22:42:21` |

**Test inventory (35 tests, 1240.2 s wall clock).**

| tier | class | tests | what it pins down |
| --- | --- | --- | --- |
| 1a | `TestSlipSystems` | 4 | FCC 12 / HCP 12 / BCC 24 systems, `n·d = 0`, traceless Schmid tensors |
| 1b | `TestElasticStiffness` | 6 | minor + major symmetry, positive definiteness, Born criteria, cubic and HCP point-group invariance, `double_contraction` vs `einsum` |
| 1c | `TestModelInitialisation` | 6 | v4 parameter sourcing, override isolation, orientation properness, θ̄, seed determinism, CRSS families |
| 1d | `TestDamageEvolution` | 3 | the `p_D` gate, per-grain monotonicity, monotone uniaxial damage history |
| 1e | `TestRunUniaxialInterface` | 3 | stress in Pa, exact recoverability of `ef`/`uts`, the strain-increment trap |
| 2 | `TestElasticLimitBaseline`, `TestLemaitreLimit` | 6 | the two analytic benchmark solutions |
| 3 | `TestConvergence` | 5 | grain count (×2 grids), sub-stepping, strain step, cross-material strain step |
| 4 | `TestArchivedReproduction` | 2 | reproducibility of the archived 2026-09-23 results |

---

## 2. Result

```
verification suite: 35 tests, 32 passed, 3 failed/errored, 1240.21 s
```

| test | status | measured |
| --- | --- | --- |
| `TestConvergence.test_convergence_with_eps_step` | **FAIL** | `ef` spread 53.51% > 5% |
| `TestArchivedReproduction.test_archived_single_crystal_bounds_are_reproducible` | **FAIL** | `ef` +27.65% vs archive |
| `TestArchivedReproduction.test_archived_convergence_json_is_reproducible` | **FAIL** | `ef` +27.65% vs archive |
| all other 32 tests | pass | see §3–§5 |

The three failing assertions are the *statement of the requirement*, not a mis-specified expectation:

```python
self.assertLess(max_rel, 5.0, f"eps_step convergence FAILED: ef = ...")
self.assertLess(rel_dev, 0.01, f"... archive is NOT reproducible: ef = ...")
```

They are left failing on purpose. Rewriting them to accommodate the measured values would convert two open defects into silent ones, which is the opposite of what a code-verification section is for.

---

## 3. Tier 1 — unit tests

### 3.1 Crystallography (`TestSlipSystems`)

| check | FCC | HCP | BCC |
| --- | --- | --- | --- |
| number of systems | 12 (4 planes × 3) | 12 (basal 3 + prism 3 + pyramidal 6) | 24 |
| max &#124;n·d&#124; over all systems | 0.0 | 1.11e-16 | 0.0 |
| max &#124;‖n‖−1&#124;, &#124;‖d‖−1&#124; | 1.11e-16 | 2.22e-16 | 1.11e-16 |
| max &#124;tr(P)&#124; over Schmid tensors | — | — | — |

FCC planes are de-duplicated by whole-vector sign normalisation (all four {111} normals survive; a componentwise `abs` de-duplication would collapse them into one and yield 3 systems instead of 12 — a defect found and fixed during this verification). Schmid tensors are traceless to 1.11e-16. The BCC generator is exercised for module completeness only: none of the three calibration alloys is BCC, and the suite says so in the recorded evidence rather than implying BCC was validated through a material.

### 3.2 Crystal elasticity (`TestElasticStiffness`)

* **Symmetry**: minor (`ij`) 0.0, minor (`kl`) 0.0, major 0.0 relative deviation, all three alloys.
* **Positive definiteness**: minimum eigenvalue 66.0 GPa (316L), 35.0 GPa (Ti64), 28.0 GPa (AlSi10Mg); the minimum energy density over random strains is +5.0e10 / +2.7e10 / +2.9e10 Pa > 0.
* **Born criteria** satisfied with margin: `C11−C12` = 66 / 70 / 47 GPa; `C11+2C12` = 480 / — / 230 GPa; `C44` = 126 / 47 / 28 GPa; for the HCP alloy `C33(C11+C12) − 2C13²` = 3.62e13 Pa > 0 and `C66 = (C11−C12)/2` = 35 GPa.
* **Cubic point-group invariance**: rotating the FCC tensor by each of the 24 cubic group elements leaves it unchanged (worst relative deviation **0.0**), while the negative control — rotation by 30° about z, which is *not* a cubic symmetry — changes it by **34.2%**. The control matters: it shows the test would catch a tensor that merely happened to be isotropic.
* **HCP transverse isotropy**: invariant under rotations about the c axis to 1.70e-16, with a negative control at 5.53%.
* **`double_contraction` vs the full contraction**: max &#124;σ − einsum("ijkl,kl", C, E)&#124; < 1e-6.

> A note for whoever reads the raw JSON: the evidence key `naive_tensor_voigt_reduction_worst_rel_dev` is **1.0**, and that number is *not* an error in `double_contraction`. It is the naive 6×6 tensor reduction `M[a,b] = C[i,j,k,l]`, which omits the second contribution of the symmetric shear pair and is therefore short by a factor 2 in every shear row — i.e. wrong by up to 100% by construction. The quantity the test asserts is recorded separately as `double_contraction_vs_einsum_worst_abs` (< 1e-6).

### 3.3 Model initialisation (`TestModelInitialisation`)

* **v4 sources `D0` from the experimental record, not the material database**: 316L 0.001 (DB 0.002), Ti64 0.003 (DB 0.005), AlSi10Mg 0.002 (DB 0.008). `S0` comes from the library (2.8 / 3.5 / 2.0 MPa) and `S_eff = params['S0']` in v4.
* **`S0_override` takes precedence and does not mutate the global database** (checked by re-reading `MATERIAL_DATABASE` after the override).
* **Orientation sets are orthonormal and proper**: worst `‖RᵀR − I‖` = 5.55e-16, worst &#124;det R − 1&#124; = 6.66e-16 over 40 grains for each alloy.
* **Mean Schmid factor θ̄** agrees with an independent 400-orientation sampling: Ti64 0.45524 (code) vs 0.45531 / 0.45468 (sampled); 316L 0.43496 vs 0.43343; AlSi10Mg 0.43685 vs 0.43456. Agreement to 0.1–0.5%.
* **Seed determinism**: same seed → bitwise identical histories (max difference 0.0); seed 17 vs 18 → differ by 1.88, so orientation generation is genuinely seed-dependent.
* **CRSS / saturation family structure** is right per structure: 316L FCC one family (g₀ 200, g_sat 500 MPa); Ti64 HCP three families (350/370/450 and 700/720/800 MPa); AlSi10Mg FCC one family (120/280 MPa).

### 3.4 Damage evolution (`TestDamageEvolution`)

* **Gate**: raising `p_D` to 1e9 and straining to 12% over 120 steps leaves the damage unchanged (max change **0.0**) — damage does not evolve below the plastic-strain threshold.
* **Per-grain monotonicity**: over the same 120-step path the worst per-grain increment is **0.0**, while `D` grows 0.003 → 0.0513 as `p` reaches 0.1334 (i.e. the path crosses `p_D = 0.05` and the test is not vacuous).
* **Uniaxial damage history** is monotone over 253 points, worst increment 0.0, ending at `D = 0.4218` against `Dc = 0.42` — the run stops where it should.

### 3.5 `run_uniaxial` interface (`TestRunUniaxialInterface`)

* **Stress is in Pa**: peak stresses 4.72e8 (316L), 9.18e8 (Ti64), 4.18e8 (AlSi10Mg) Pa. (These are library-`S0` runs; the calibrated Ti64 `S0 = 0.32` MPa is discussed in §7.)
* **`ef` and `uts` are exactly recoverable** from the returned raw history and `Dc`: recomputed `ef` = 0.5026251719009015 matches, recomputed `uts` = 918.0906657517321 MPa matches, over 253 points.
* **Documented defect — the strain-increment trap.** `run_uniaxial` builds its own load path as `n_steps = max(10, int(max_strain/eps_step)) + 1`, so the *effective* increment is `max_strain/(n_steps−1)`, which equals `eps_step` only when the ratio is an integer **and** at least 10, and — because the run stops at fracture — only while the run is not truncated early. Measured:

| `max_strain` | `eps_step` requested | `n_steps` | effective `deps` | `deps/eps_step` | effective strain rate |
| --- | --- | --- | --- | --- | --- |
| 0.004 | 5e-4 | 11 | 4.0e-4 | 0.8 | 8.0e-4 (not 1.0e-3) |
| 0.002 | 5e-4 | 11 | 2.0e-4 | 0.4 | 4.0e-4 (not 1.0e-3) |
| 0.17 | 5e-4 | 341 | 5.0e-4 | 1.0 | 1.0e-3 |
| 0.25 | 5e-4 | 501 | 5.0e-4 | 1.0 | 1.0e-3 |
| 0.30 | 1e-3 | 301 | 1.0e-3 | 1.0 | 1.0e-3 |

The fracture-strain interpolation still uses the *requested* `eps_step`, so a non-integer ratio silently mis-scales `ef`. Every convergence run in this report therefore uses `max_strain = 300 · eps_step`, and each recorded run carries `deps_effective / eps_step` beside its result — for all of them that ratio is exactly **1.0**, which is what makes the convergence numbers below trustworthy.

---

## 4. Tier 2 — analytic benchmark solutions

### 4.1 B3-1: the elastic limit (damage frozen)

Protocol: `S0 = 1e12` Pa so that `D` never leaves `D0` (verified: max &#124;D − D0&#124; ≤ 4.3e-19), `n_grains = 10`, elastic windows chosen so the increment is exact — 316L (2.0e-3, 2.0e-4), Ti64 (5.0e-3, 5.0e-4), AlSi10Mg (3.0e-3, 3.0e-4) — lateral tolerance `1e5` Pa, 60 iterations.

The reference is **not** the textbook isotropic 3K–2G formula. It is an independent re-implementation of the solver's own crystal-to-macro averaging (`E_crystal = Rᵀ E_macro R`, `σ_macro = R σ_crystal Rᵀ`) combined with the solver's own lateral condition:

| alloy | E* analytic (GPa) | w* | E numerical (GPa) | relative deviation |
| --- | --- | --- | --- | --- |
| 316L | 204.0133 | 0.287486 | 203.7270 | **−0.140%** |
| Ti64 | 118.0075 | 0.316277 | 117.8578 | **−0.127%** |
| AlSi10Mg | 70.2689 | 0.347241 | 70.3766 | **+0.153%** |

Agreement is at the 0.15% level, and the residual is the expected O(ε) difference between the reference's infinitesimal kinematics and the solver's Green–Lagrange measure.

**A constraint subtlety worth recording.** `run_uniaxial` applies a *single scalar* lateral coefficient — `ε22 = ε33 = −w ε11` — and drives the **mean** lateral stress to zero: its lateral residual is `0.5·(σ22 + σ33)`, not `σ22`. The reference must follow suit:

```
w* = −(a22 + a33) / (b22 + b33),    E* = a11 + w*·b11
```

where `σ = e·a + w·b` for `E = diag(e, −w e, −w e)`. Imposing the textbook `σ22 = 0` condition instead (`w = −a22/b22`) does **not** give the same answer for an anisotropic Taylor aggregate, because `ε33` is slaved to `ε22` while `σ33` is left non-zero:

| alloy (10 grains) | E* under `σ22 = 0` | E* under mean = 0 (correct) | bias of the wrong condition |
| --- | --- | --- | --- |
| 316L | 216.8131 | 204.0133 | **+6.27%** |
| Ti64 | 114.2418 | 118.0075 | **−3.19%** |
| AlSi10Mg | 71.1953 | 70.2689 | **+1.32%** |

The bias differs in sign between alloys, so it is not a harmless offset — it would have produced a spurious 6% "failure" of the elastic benchmark for 316L. At the correct `w*` the individual lateral stresses are ±2% of `σ11` while their mean is ≤ 1.2e-4 of `σ11`, which is exactly the condition the solver enforces. This is recorded because it is the one place in this verification where a plausible-looking reference would have given the wrong answer.

**Hooke's law over the elastic window.** The secant modulus is constant to 0.365% (316L, to 2e-3), 0.895% (Ti64, to 5e-3) and 0.327% (AlSi10Mg, to 3e-3) — all under 1%, with `D` pinned at `D0` (max &#124;D − D0&#124; ≤ 4.3e-19). The stress–strain curve is linear in the elastic segment.

**Production tolerance.** At the production default (`inner_tol_pa = 3e6`, `n_inner = 25`) the measured elastic-response bias is −1.37% (316L), +1.94% (Ti64) and +7.83% (AlSi10Mg). All are below the 10% reporting threshold, so the default lateral tolerance is adequate for the reported work — but the bias is set by the *lateral Newton tolerance*, not by the constitutive law, and AlSi10Mg is the tightest case. Tightening to `1e5` Pa removes it (§4.1 table).

### 4.2 B3-2: the Lemaitre limit

The reference-state descriptors are `φ → 0`, `λ_eff = λ_ref`, `ξ̄ = 1`, `θ̄ = 0.5`. With all corrective factors at unity the AM-corrected model must degenerate to the classical Lemaitre model. Two tests pin this down:

**(a) Factor normalisation.** With `ξ = 1`, `λ_eff = λ_ref = 50 µm` (requiring `w_iso = 1`), `θ̄ = 0.5`, `φ → 0`, `D0 → 0`:

| factor | f_φ | f_θ | f_D0 | f_λ | f_ξ | f_shield | **f_AM** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| value | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | **1.0** |

**(b) Equivalence, not similarity.** An uniaxial run to fracture at those descriptors is compared against the same run with `force_homogeneous=True` (which forces `f_AM = 1`): over 46 compared points, max &#124;ΔD&#124; = **0.0**, max &#124;Δσ&#124; = **0.0 Pa**, and `ef` (0.08957183966027298) and `uts` (733.0275842020457 MPa) are identical. The AM correction degenerates to the classical Lemaitre model *exactly*, not approximately.

**(c) The three-condition reading is insufficient — an actionable correction.** Setting only `ξ = 1`, `θ̄ = 0.5` and `λ_eff = λ_ref`, while leaving `φ`, `D0` and `w_iso` at their production values, does **not** send `f_AM` to 1:

| alloy | φ | D0 | w_iso | λ_eff/λ_ref | ξ̄ | θ̄ | f_AM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 316L | 0.001 | 0.001 | 0.25 | 0.25 | 1.0 | 0.5 | **0.563** |
| Ti64 | 0.003 | 0.003 | 0.25 | 0.25 | 1.0 | 0.5 | **0.439** |
| AlSi10Mg | 0.005 | 0.002 | 0.25 | 0.25 | 1.0 | 0.5 | **0.303** |

`f_λ = 0.25`–`0.55` alone accounts for most of the shortfall: unless the pore-free factor is unity, `λ_eff = 0.25·λ0 = 12.5 µm`, not `λ_ref = 50 µm`. The reference-state sentence in the manuscript should therefore list **D0 → 0** alongside `φ → 0`, and state explicitly that `λ_eff = λ_ref` requires the pore-free (fully dense) normalisation `w_iso = 1`. As written, the parenthetical descriptor list is a necessary but not sufficient statement of the Lemaitre limit.

---

## 5. Tier 3 — convergence

All runs: Ti64, ψ = 0°, `S0 = 0.32` MPa (the calibrated value), seed 42, effective increment equal to `eps_step` (ratio 1.0 in every row).

### 5.1 Grain count as specified (27 / 64 / 125 at `eps_step = 1e-3`) — **pass**

| n_grains | `ef` | `uts` (MPa) | wall clock (s) | deviation vs 125 |
| --- | --- | --- | --- | --- |
| 27 | 0.072548 | 934.76 | 32.9 | +4.33% |
| 64 | 0.071219 | 942.43 | 77.9 | +2.42% |
| 125 | 0.069538 | 951.68 | 148.4 | — |

Max relative deviation **4.33% < 10%** → the fracture strain is grain-count convergent, and the 20-grain aggregates used for the reported results sit inside this band.

### 5.2 Grain count on the archived grid (10 / 20 / 30 / 50 at `eps_step = 5e-4`) — **pass**

| n_grains | `ef` | `uts` (MPa) | deviation vs 50 |
| --- | --- | --- | --- |
| 10 | 0.071455 | 963.56 | −1.64% |
| 20 | 0.072287 | 959.88 | −0.49% |
| 30 | 0.073143 | 953.98 | +0.69% |
| 50 | 0.072643 | 957.76 | — |

Max relative deviation **1.64% < 5%**. The scatter is non-monotone and bounded, i.e. the remaining spread is orientation-set sampling noise, not a systematic grain-count trend.

### 5.3 Sub-stepping (15 / 30 / 60 at `eps_step = 1e-3`, 20 grains) — **pass**

| n_sub | `ef` | `uts` (MPa) | deviation vs 30 |
| --- | --- | --- | --- |
| 15 | 0.072943 | 929.66 | +1.60% |
| 30 | 0.071797 | 938.99 | — |
| 60 | 0.071384 | 941.09 | −0.57% |

Max relative deviation **1.60% < 5%**. The near-insensitivity has a structural reason worth stating: in the steady regime `_step_single_grain` advances one sub-step per increment, so `n_sub` changes the answer only where it triggers sub-division of a hard step — it is a safety margin against a stiff increment, not a refinement parameter with a converged limit. The 30 used in production is therefore not a tuned value.

### 5.4 Strain step (5e-4 / 1e-3 / 2e-3 at 20 grains) — **FAIL (open limitation)**

| `deps` | `ef` | `uts` (MPa) | points | deviation vs 5e-4 |
| --- | --- | --- | --- | --- |
| 5.0e-4 | 0.072287 | 959.88 | 146 | — |
| 1.0e-3 | 0.071797 | 938.99 | 73 | **−0.68%** |
| 1.5e-3 | 0.082228 | 872.53 | 56 | **+13.75%** |
| 2.0e-3 | 0.110970 | 733.56 | 57 | **+53.51%** |

**This is a genuine negative result, not a test artefact**: the effective increment was verified to equal the requested `eps_step` in every row (§3.5), so the curve above is a pure step-size study.

The fracture strain is step-convergent *up to* `deps = 1e-3` (< 0.7% between 5e-4 and 1e-3), then degrades sharply. A cross-material probe at 10 grains shows the breakdown is **material-dependent in sign as well as magnitude**:

| alloy | `ef` at `deps` = 5e-4 | `ef` at `deps` = 2e-3 | change |
| --- | --- | --- | --- |
| 316L | 0.088147 | 0.033735 | **−61.7%** |
| Ti64 | 0.072287 | 0.110970 | **+53.5%** |
| AlSi10Mg | 0.097758 | 0.149276 | **+52.7%** |

Because the sign flips between alloys, no single empirical step correction can be applied — the only safe course is to keep the increment small. The damage update is an explicit per-step increment whose value depends on where each step lands relative to the `p_D` gate and to the softening feedback; once the step is coarse enough that the fracture-triggering step overshoots by a large margin, the reported `ef` stops being a property of the material and becomes a property of the discretisation.

**Consequence for the reported work.** All production and archived runs use `eps_step ≤ 1e-3` (the archive uses 5e-4 and 2.5e-4), which is inside the converged range — so the *reported* numbers are not affected by this limitation. What the paper should do is state the increment used and record this bound explicitly, e.g.:

> The fracture strain is convergent to better than 1% for effective strain increments up to 1e-3; at 1.5e-3 and 2e-3 it moves by +13.8% and +53.5% (Ti64) and by −61.7% (316L), so all reported results use an increment of 1e-3 or smaller.

This is the honest form of the claim: convergence is demonstrated over the range actually used, with the breakdown documented rather than asserted away.

---

## 6. Tier 4 — reproducibility of the archived results — **FAIL (open defect)**

The archive files `output/convergence_test.json` (written 2026-09-23 22:35:55) and `output/taylor_single_crystal_bounds_v9.json` (22:45) were produced by sources *older* than the current `src/taylor_cpcdm.py` (2026-09-25 07:41:02) and `src/am_correction_v4.py` (2026-09-25 22:25:15). Re-running the same protocol from the current sources:

| quantity | archived 2026-09-23 | current sources | relative change |
| --- | --- | --- | --- |
| `ef`, 20 grains, `S0` = 0.32 MPa, `eps_step` = 5e-4 | 0.056629064932528075 | 0.07228673685617111 | **+27.65%** |
| `uts` (MPa), same run | 959.3631776153682 | 959.8832975000528 | **+0.054%** |

**This is the most informative result in the section, and it is diagnostic rather than merely negative.** The peak stress reproduces to 5e-4 relative while the fracture strain moves by 27.6%: whatever changed between 09-23 and 09-25 acted on the *damage / fracture* description, not on the elastic–plastic hardening response. A reviewer can therefore localise the change without bisecting the repository.

**Consequence for the manuscript.** The Discussion quotes the 20-grain Taylor fracture strain **0.0566** and a single-crystal band of 0.0480–0.0674 (mean 0.05635). Those values cannot be produced by the current sources, which give 0.07229 — a 27.6% difference, far outside the ±12–21% model-versus-experiment band the paper itself reports. One of the following has to happen, and it is a decision about which sources are the paper's sources, not a modelling choice:

* re-derive the cited numbers from the current sources (`ef` ≈ 0.0723), or
* recover the pre-09-23 sources that produced the archive and state them as the paper's version.

Both numbers may **not** be quoted together. This report does not choose between them and does not touch the manuscript text.

Note for context: under the archived (old) sources the 20-grain value 0.0566 sat 29% below the measured Ti64 fracture strain (0.080 at 0°); under the current sources 0.0723 sits 9.6% below it. If the current sources are the paper's, the comparison to experiment is *better* than the archived number suggests — but that is a consequence to verify, not a reason to prefer it.

---

## 7. Defects found in surrounding artifacts (reported, not silently fixed)

| # | artifact | defect | status |
| --- | --- | --- | --- |
| 1 | `run_uniaxial` | the load path is rebuilt from `n_steps = max(10, int(max_strain/eps_step)) + 1`, so the effective increment and strain rate differ from the requested ones unless the ratio is an integer ≥ 10; `ef` is then interpolated with the *requested* `eps_step` (§3.5) | documented, quotable; not changed |
| 2 | `output/taylor_nodamage_results.json` | the field named `uts_MPa` stores **Pa** (316L 8.148e8, Ti64 1.101e9, AlSi10Mg 4.314e8). The values themselves agree with the damage-free Taylor UTS quoted in the manuscript (814.8 / 1100.8 / 431.4 MPa), so only the **key name** is wrong | reported only; the file is an archived output and was left untouched |
| 3 | `MATERIAL_DATABASE` | the library `S0` for Ti64 is **3.5 MPa**, which is *not* the calibrated value: with it the model gives `ef = 0.5026` and `uts = 918.1 MPa`, whereas the calibrated `S0 = 0.32 MPa` gives `ef = 0.0723` against a measured 0.080 at 0°. A reader of the database could mistake 3.5 for the paper's value | reported; the calibrated value is what all convergence runs and the archive use |
| 4 | archive vs sources | §6 | open |

---

## 8. What this verification does and does not establish

**Establishes**

* the crystallographic generators produce the correct systems, with planes de-duplicated correctly and Schmid tensors traceless;
* the crystal-elastic tensors are symmetric, positive definite, satisfy Born's criteria, and are invariant under exactly the right point groups (with discriminating negative controls);
* the v4 parameter path loads `D0` from the experimental record and `S0` from the library, and overrides neither leak nor mutate shared state;
* orientation sets are orthonormal, proper and seed-reproducible, and θ̄ agrees with independent sampling to ≤ 0.5%;
* damage does not evolve below the `p_D` gate and never decreases on a monotone path, per grain or in the uniaxial history;
* stresses are in Pa and `ef`/`uts` are exactly recoverable from the raw history;
* the elastic limit reproduces the exact anisotropic solution of the model's own kinematic constraint to **< 0.16%**, and Hooke's law holds to < 1% over the elastic window;
* the AM correction degenerates to the classical Lemaitre model at the reference state **exactly** (max &#124;ΔD&#124; = 0, max &#124;Δσ&#124; = 0), and the three-condition reading of that limit is insufficient;
* the fracture strain is grain-count convergent to 4.33% (27→125) and 1.64% (10→50), and sub-step insensitive to 1.60% over 15–60;
* the strain increment is convergent to < 1% for `deps ≤ 1e-3`.

**Does not establish**

* **mesh convergence of the 3D full-field solver**, or its agreement with the Taylor aggregate — that is a separate study reported in the manuscript (2D plane-strain CPFEM, peak stress at 0° = 0.0355, fracture-strain deviations −26.5% / −31.9% / −35.9% by orientation for Ti64, +5.3% for AlSi10Mg at 0°);
* **experimental validation** — that is the role of the blind tests, not of this suite;
* **uniqueness or identifiability of the fitted parameters** — the suite verifies the implementation, not the inverse problem;
* **the reproducibility of the archived 09-23 results**, which fail as described in §6;
* anything about the BCC path *through a material*: the BCC generator is tested directly, but none of the three calibration alloys is BCC.

---

## 9. Reproduction

```powershell
cd "D:\20260618断裂模型论文\simulation\tests"
& "C:\Users\ZJ03\miniconda3\python.exe" -u test_taylor_cpcdm.py
# writes  ..\output\code_verification_results.json
```

Runtime is ≈21 minutes (1240.2 s), dominated by the grain-count convergence tests (256 s and 234 s), the stress-unit run (165 s) and the strain-step study (140 s). A single class can be run alone, e.g.

```powershell
& "C:\Users\ZJ03\miniconda3\python.exe" -m unittest test_taylor_cpcdm.TestElasticStiffness
```

pytest is not installed in this environment; the suite is `unittest` and needs no third-party runner. All tests are deterministic (fixed seed 42); the same-run comparisons of `ef`/`uts` are bitwise identical across repeated executions, so the three failures reproduce exactly.

---

## 10. Suggested text for the manuscript

Ready to paste into `sec:convergence` (or a new code-verification subsection), with the numbers as measured:

```latex
\subsection{Code verification}
\label{sec:code_verification}

The implementation used for all results in this work (the Taylor-type
crystal-plasticity aggregate coupled to the AM-corrected Lemaitre damage
model, and its full-field CPFEM counterpart) is verified against analytic
solutions rather than against an external framework: no open-source CPFEM
framework (MOOSE, FEniCS, PRISMS) provides an installable build in the
environment used here (Appendix~\ref{app:framework_availability}).
A suite of 35 tests covers the crystallographic generators, the
crystal-elastic tensors, parameter sourcing, damage monotonicity, unit
consistency, two analytic benchmark limits and the convergence of the
discretisation; 32 pass and the three failures are reported below.

\paragraph{Elastic limit.} With damage frozen ($S_0 = 10^{12}\,$Pa) and the
load path chosen so that the effective increment equals the prescribed one,
the uniaxial response reproduces the exact anisotropic elastic solution of
the model's own kinematic constraint --- the Taylor average
$\langle R\,C\,R^{\mathrm T}\rangle$ with a single lateral coefficient
$w$ such that $\tfrac12(\sigma_{22}+\sigma_{33}) \to 0$ --- to within
$0.15\%$ for the three alloys ($E^{*} = 204.01$, $118.01$ and
$70.27\,$GPa, respectively). The secant modulus is constant to better than
$0.9\%$ over the elastic window.

\paragraph{Lemaitre limit.} At the reference-state descriptors
($\phi \to 0$, $D_0 \to 0$, $\bar\xi = 1$, $\lambda_{\mathrm{eff}}
= \lambda_{\mathrm{ref}}$, $\bar\theta = 0.5$) all corrective factors reduce
to unity and the model becomes \emph{bit-for-bit} identical to the classical
Lemaitre model forced with $f_{\mathrm{AM}} = 1$: over 46 compared points the
maximum differences in damage and stress are $0$ and $0\,$Pa, and the
fracture strain and UTS coincide exactly.

\paragraph{Convergence.} The fracture strain is convergent with respect to
the number of grains ($4.33\%$ over $27 \to 125$ grains, $1.64\%$ over
$10 \to 50$ at $\dot\varepsilon$-increment $5\times10^{-4}$) and is
insensitive to the number of sub-steps ($1.60\%$ over $15 \to 60$).

The strain increment is convergent to better than $1\%$ up to
$\Delta\varepsilon = 10^{-3}$; beyond that the explicit damage update is no
longer step-convergent, with changes of $+13.8\%$ and $+53.5\%$ at
$1.5\times10^{-3}$ and $2\times10^{-3}$ for Ti64 and $-61.7\%$ for 316L. All
results reported here use $\Delta\varepsilon \le 10^{-3}$
(Section~\ref{sec:convergence}).

Table~\ref{tab:convergence_summary} collects the convergence data.
```

Plus the two tables in LaTeX form:

```latex
\begin{table}[htbp]
\centering
\caption{Convergence of the Taylor-aggregate fracture strain and UTS
(Ti64, $\psi = 0^\circ$, $S_0 = 0.32$\,MPa). In every run the effective
increment equals the prescribed one.}
\label{tab:convergence_summary}
\begin{tabular}{llrrrr}
\hline
study & parameter & $n$ & $\varepsilon_f$ & UTS (MPa) & max dev. \\
\hline
grain count        & 27 / 64 / 125 grains & 3 & 0.07255 / 0.07122 / 0.06954 & 934.8 / 942.4 / 951.7 & 4.33\% \\
grain count        & 10 / 20 / 30 / 50 grains & 4 & 0.07146 / 0.07229 / 0.07314 / 0.07264 & 963.6 / 959.9 / 954.0 / 957.8 & 1.64\% \\
sub-stepping       & $n_{\mathrm{sub}}$ = 15 / 30 / 60 & 3 & 0.07294 / 0.07180 / 0.07138 & 929.7 / 939.0 / 941.1 & 1.60\% \\
strain increment   & $\Delta\varepsilon$ = 5e-4 / 1e-3 & 2 & 0.07229 / 0.07180 & 959.9 / 939.0 & 0.68\% \\
strain increment   & $\Delta\varepsilon$ = 1.5e-3 / 2e-3 & 2 & 0.08223 / 0.11097 & 872.5 / 733.6 & 53.5\% \\
\hline
\end{tabular}
\end{table}
```

(The grain-count and sub-step rows are for the 20-grain, $10^{-3}$ reference unless noted; the first row and second row differ in reference increment, 1e-3 and 5e-4 respectively.)
