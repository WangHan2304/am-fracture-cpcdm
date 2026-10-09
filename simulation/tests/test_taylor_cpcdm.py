# -*- coding: utf-8 -*-
"""
test_taylor_cpcdm.py -- code verification suite for the Taylor-type CP-CDM
aggregate model (TaylorCPCDM + am_correction_v4) used in the manuscript on
DAMASK-informed nonlocal damage in additively manufactured metals.

Purpose
-------
Provide an auditable, reproducible verification layer answering the referee
request for "unit tests, convergence studies and benchmark solutions".  The
suite is structured along the manuscript's four-tier verification framework:

    Tier 1  unit / structural tests    (TestSlipSystems, TestElasticStiffness,
                                        TestModelInitialisation,
                                        TestDamageEvolution,
                                        TestRunUniaxialInterface)
    Tier 2  analytic benchmark limits  (TestElasticLimitBaseline,
                                        TestLemaitreLimit)
    Tier 3  numerical convergence      (TestConvergence)
    Tier 4  provenance / reproduction  (TestArchivedReproduction)

Run
---
    python simulation/tests/test_taylor_cpcdm.py
    python -m unittest test_taylor_cpcdm -v          # from simulation/tests

The runner records every test plus all numeric evidence in

    simulation/output/code_verification_results.json

and exits with status 1 if any assertion fails.  A failing test here is a
*finding*, not a regression: two tests are expected to fail and both are
real, documented discrepancies (see
simulation/output/code_verification_summary.md).

Conventions (verified against src/taylor_cpcdm.py)
--------------------------------------------------
* Units are SI: stress in Pa, strain dimensionless, strain rate in 1/s,
  damage dimensionless.  run_uniaxial returns 'stress' in Pa (NOT MPa).
* Crystal-frame mapping (the solver's own convention, see _macro_sigma and
  integrate):  T_crystal = R_g.T @ T_macro @ R_g,  and the reverse mapping is
  T_macro = R_g @ T_crystal @ R_g.T.  The rank-4 rotation operator used for
  the *crystal-symmetry* tests is therefore C' = R C R^T with
  C'_ijkl = R_im R_jn R_kp R_lq C_mnpq.
* run_uniaxial uses the tensor-level uniform-lateral constraint
  eps_11 = e, eps_22 = eps_33 = -w e with a single scalar w per step
  (NOT eps_22 = eps_33 = 0 and NOT free independent lateral contraction).
  The benchmark below therefore uses the *anisotropic* boundary-matched
  reference, not the isotropic 3K-2G expression.

Determinism:  TaylorCPCDM.__init__ calls np.random.seed(seed) before
generating orientations, so construction is reproducible for a fixed seed.
Tests that need randomness use an explicit np.random.RandomState and cannot
perturb (or be perturbed by) the model's global RNG.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
import time
import unittest
from datetime import datetime

import numpy as np

# --------------------------------------------------------------------------
# paths / imports
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.dirname(HERE)
SRC_DIR = os.path.join(SIM_DIR, "src")
OUT_DIR = os.path.join(SIM_DIR, "output")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from materials import (  # noqa: E402
    MATERIAL_DATABASE,
    EXPERIMENTAL,
    build_stiffness_tensor,
    double_contraction,
    get_slip_systems_FCC,
    get_slip_systems_HCP,
    get_slip_systems_BCC,
)
from taylor_cpcdm import TaylorCPCDM  # noqa: E402
from am_correction_v4 import (  # noqa: E402
    factors_v4,
    lambda_eff,
    xi_eff,
    LAMBDA_REF,
    W_ISO,
)

# --------------------------------------------------------------------------
# global evidence accumulator (populated by tests, dumped by main())
# --------------------------------------------------------------------------
EVIDENCE: dict = {}
DURATIONS: dict = {}
_status_by_id: dict = {}


def ev(key, value):
    EVIDENCE[key] = value
    return value


# --------------------------------------------------------------------------
# independent helper implementations
# (re-derived here on purpose so they cross-check src/ rather than call it)
# --------------------------------------------------------------------------
def rotate_C(C, R):
    """Rank-4 rotation (R C R^T): C'_ijkl = R_im R_jn R_kp R_lq C_mnpq.

    Used only for the crystal point-group invariance tests, where an active
    rotation of the crystal frame must leave a symmetry-consistent tensor
    unchanged.
    """
    return np.einsum("im,jn,kp,lq,mnpq->ijkl", R, R, R, R, C)


def voigt66(C):
    """6x6 matrix of a rank-4 tensor in tensor (not engineering) shear
    convention."""
    idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    M = np.zeros((6, 6))
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            M[a, b] = C[i, j, k, l]
    return M


def aggregate_response(C, R_list):
    """Re-implementation of the solver's crystal-to-macro averaging for a
    prescribed *macro* strain, in the small-strain elastic limit.

    Returns (a, b) such that the macro stress for
        E = diag(e, -w e, -w e)
    is  sigma = e * a + w * b  (componentwise).  Follows exactly the
    kinematics of TaylorCPCDM._macro_sigma:

        E_crystal = R.T @ E_macro @ R
        S_crystal = C : E_crystal
        sigma_macro = R @ S_crystal @ R.T
    """
    def agg(E):
        s = np.zeros((3, 3))
        for R in R_list:
            Ec = R.T @ E @ R
            Sc = np.einsum("ijkl,kl->ij", C, Ec)
            s += R @ Sc @ R.T
        return s / len(R_list)

    a = agg(np.diag([1.0, 0.0, 0.0]))
    b = agg(np.diag([0.0, -1.0, -1.0]))
    return a, b


def e_uniform_lateral(C, R_list):
    """Closed-form modulus under the solver's exact boundary condition.

    TaylorCPCDM.run_uniaxial applies a SINGLE scalar lateral contraction
    eps_22 = eps_33 = -w * eps_11 and drives the MEAN lateral stress to
    zero: its eval_lat returns 0.5 * (sigma_22 + sigma_33).  Hence

        0.5 (a22 + a33) + w * 0.5 (b22 + b33) = 0
            =>  w* = -(a22 + a33) / (b22 + b33)
            and E* = sigma_11 / e = a11 + w* b11.

    Driving sigma_22 alone to zero is NOT the same condition and gives a
    wrong reference for an anisotropic Taylor aggregate: it leaves
    sigma_33 != 0 (about -(+/-)2% of sigma_11 here) while eps_33 is slaved
    to eps_22, and it overestimates E* by ~6% for 316L.

    Exact for the model's own kinematics; NOT the isotropic 3K-2G
    expression (which assumes eps_22 and eps_33 contract independently).
    """
    a, b = aggregate_response(C, R_list)
    w_star = -(a[1, 1] + a[2, 2]) / (b[1, 1] + b[2, 2])
    E_star = a[0, 0] + w_star * b[0, 0]
    return float(E_star), float(w_star)


def cubic_group():
    """The 24 proper rotations of the cube as signed permutation matrices."""
    mats = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        for s0 in (1.0, -1.0):
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    Q = np.zeros((3, 3))
                    Q[0, perm[0]] = s0
                    Q[1, perm[1]] = s1
                    Q[2, perm[2]] = s2
                    if abs(np.linalg.det(Q) - 1.0) < 1e-12:
                        mats.append(Q)
    return mats


def canonical_plane(v, ndigits=6):
    """Canonical representative of a crystal plane normal, identifying n and
    -n by flipping the sign of the WHOLE vector (not componentwise: a
    componentwise abs would map all four {111} normals onto the same tuple)."""
    v = np.round(np.asarray(v, dtype=float), ndigits)
    for x in v:
        if abs(x) > 1e-9:
            return tuple(v if x > 0 else -v)
    return tuple(v)


def rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t), np.cos(t), 0.0],
                     [0.0, 0.0, 1.0]])


def rot_x_deg(deg):
    t = np.radians(deg)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(t), -np.sin(t)],
                     [0.0, np.sin(t), np.cos(t)]])


def build_model(material, psi=0.0, n_grains=20, eps_step=1e-3, n_sub=30,
                strain_rate=1e-3, seed=42, s0=None, model_version="v4",
                overrides=None, **kwargs):
    return TaylorCPCDM(
        material, psi,
        n_grains=n_grains,
        eps_step=eps_step,
        n_sub=n_sub,
        strain_rate=strain_rate,
        seed=seed,
        model_version=model_version,
        S0_override=s0,
        param_overrides=overrides,
        **kwargs,
    )


def fracture_from_history(strain, damage, Dc, eps_step):
    """Re-implementation of run_uniaxial's fracture-strain interpolation."""
    strain = np.asarray(strain, dtype=float)
    damage = np.asarray(damage, dtype=float)
    idx = np.where(damage >= Dc)[0]
    if len(idx) == 0:
        return None
    i0 = int(idx[0])
    if i0 > 0 and damage[i0] > damage[i0 - 1]:
        frac = (Dc - damage[i0 - 1]) / (damage[i0] - damage[i0 - 1])
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(strain[i0 - 1] + frac * eps_step)
    return float(strain[i0])


def md5_of(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def factors_of(model, psi_deg, theta_bar):
    """Convenience: the module function returns (f_AM, dict_of_components)."""
    return factors_v4(model.params, psi_deg, theta_bar)


class TimedCase(unittest.TestCase):
    def setUp(self):
        self._t0 = time.time()

    def tearDown(self):
        DURATIONS[self.id()] = round(time.time() - self._t0, 3)


# ==========================================================================
# Tier 1a -- crystallographic slip systems
# ==========================================================================
class TestSlipSystems(TimedCase):
    def _geometry(self, systems, name):
        self.assertTrue(len(systems) > 0, f"{name}: no systems returned")
        max_nd = 0.0
        max_norm_err = 0.0
        for (n, d) in systems:
            n = np.asarray(n, dtype=float)
            d = np.asarray(d, dtype=float)
            max_norm_err = max(max_norm_err,
                               abs(np.linalg.norm(n) - 1.0),
                               abs(np.linalg.norm(d) - 1.0))
            max_nd = max(max_nd, abs(float(np.dot(n, d))))
        self.assertLess(max_norm_err, 1e-10,
                        f"{name}: plane normal / slip direction not unit")
        self.assertLess(max_nd, 1e-9,
                        f"{name}: n.d must vanish; max|n.d| = {max_nd:.3e}")
        return max_nd, max_norm_err

    def test_fcc_12_systems(self):
        """FCC: 4 {111} planes x 3 <110> directions = 12 systems."""
        s = get_slip_systems_FCC()
        self.assertEqual(len(s), 12, "FCC must have 12 slip systems")
        max_nd, norm_err = self._geometry(s, "FCC")
        planes = [canonical_plane(n) for (n, _) in s]
        uniq = sorted(set(planes))
        self.assertEqual(len(uniq), 4, "FCC must expose 4 distinct glide planes")
        for u in uniq:
            self.assertEqual(sum(1 for p in planes if p == u), 3,
                             "each FCC glide plane must carry 3 directions")
        ev("slip_fcc", {"n_systems": 12, "n_planes": len(uniq),
                        "systems_per_plane": 3,
                        "max_abs_nd": max_nd, "max_norm_err": norm_err})

    def test_hcp_12_systems_three_families(self):
        """HCP: 3 basal + 3 prismatic + 6 pyramidal = 12 systems.

        Family membership is identified from the c-component of the plane
        normal (|n_z| = 1 basal, 0 prismatic, otherwise pyramidal) -- a
        structural test independent of the generator's internal ordering.
        """
        s = get_slip_systems_HCP()
        self.assertEqual(len(s), 12, "HCP must have 12 slip systems")
        max_nd, norm_err = self._geometry(s, "HCP")
        basal = prism = pyr = 0
        for (n, _) in s:
            nz = abs(float(np.asarray(n, dtype=float)[2]))
            if abs(nz - 1.0) < 1e-8:
                basal += 1
            elif nz < 1e-8:
                prism += 1
            else:
                pyr += 1
        self.assertEqual(basal, 3, "HCP basal family must have 3 systems")
        self.assertEqual(prism, 3, "HCP prismatic family must have 3 systems")
        self.assertEqual(pyr, 6, "HCP pyramidal <c+a> family must have 6 systems")
        ev("slip_hcp", {"n_systems": 12, "basal": basal, "prism": prism,
                        "pyr": pyr, "max_abs_nd": max_nd,
                        "max_norm_err": norm_err})

    def test_bcc_24_systems(self):
        """BCC: 12 {110}<111> + 12 {112}<111> = 24 systems."""
        s = get_slip_systems_BCC()
        self.assertEqual(len(s), 24, "BCC must have 24 slip systems")
        max_nd, norm_err = self._geometry(s, "BCC")
        ev("slip_bcc", {"n_systems": 24, "max_abs_nd": max_nd,
                        "max_norm_err": norm_err,
                        "note": "the BCC generator is not exercised by the "
                                "three calibration alloys (FCC/HCP only); "
                                "tested for materials-module completeness."})

    def test_schmid_tensors_are_traceless(self):
        """trace(P) = d.(n.d) = 0 for every system: the invariant coupling
        slip geometry to isochoric plastic flow."""
        worst = 0.0
        for name, gen in (("FCC", get_slip_systems_FCC),
                          ("HCP", get_slip_systems_HCP),
                          ("BCC", get_slip_systems_BCC)):
            for a, (n, d) in enumerate(gen()):
                P = np.outer(np.asarray(d, dtype=float),
                             np.asarray(n, dtype=float))
                tr = abs(float(np.trace(P)))
                worst = max(worst, tr)
                self.assertLess(tr, 1e-12, f"{name} system {a}: trace(P)={tr:.3e}")
        ev("schmid_traceless_worst_abs_trace", worst)


# ==========================================================================
# Tier 1b -- elastic stiffness tensor
# ==========================================================================
class TestElasticStiffness(TimedCase):
    MATS = ["316L", "Ti64", "AlSi10Mg"]

    def test_symmetry_minor_and_major(self):
        report = {}
        for m in self.MATS:
            C = build_stiffness_tensor(MATERIAL_DATABASE[m])
            scale = np.max(np.abs(C))
            minor1 = float(np.max(np.abs(C - np.einsum("jikl->ijkl", C))) / scale)
            minor2 = float(np.max(np.abs(C - np.einsum("ijlk->ijkl", C))) / scale)
            major = float(np.max(np.abs(C - np.einsum("klij->ijkl", C))) / scale)
            report[m] = {"minor_symmetry_ij": minor1,
                         "minor_symmetry_kl": minor2, "major_symmetry": major}
            self.assertLess(minor1, 1e-14, f"{m}: minor symmetry (ij)")
            self.assertLess(minor2, 1e-14, f"{m}: minor symmetry (kl)")
            self.assertLess(major, 1e-14, f"{m}: major symmetry")
        ev("stiffness_symmetry_rel", report)

    def test_positive_definite(self):
        """w = 1/2 E:C:E > 0 for 200 random non-zero symmetric strains
        (convention-free positive-definiteness test) plus Voigt eigenvalues."""
        rs = np.random.RandomState(7)
        report = {}
        for m in self.MATS:
            C = build_stiffness_tensor(MATERIAL_DATABASE[m])
            M = voigt66(C)
            eig = np.linalg.eigvalsh(0.5 * (M + M.T))
            wmin = np.inf
            for _ in range(200):
                E = rs.randn(3, 3)
                E = 0.5 * (E + E.T)
                # strain-energy density w = 1/2 E : C : E
                wmin = min(wmin, float(0.5 * np.sum(E * double_contraction(C, E))))
            report[m] = {"eig_min_GPa": float(eig.min() / 1e9),
                         "eig_max_GPa": float(eig.max() / 1e9),
                         "min_energy_density_Pa": wmin}
            self.assertGreater(eig.min(), 0.0, f"{m}: Voigt matrix not PD")
            self.assertGreater(wmin, 0.0, f"{m}: strain energy not positive")
        ev("stiffness_positive_definite", report)

    def test_born_stability_criteria(self):
        """Born (elastic stability) criteria for the crystal classes used, plus
        an internal-consistency check that the builder's shear components sit
        at the tensor positions its own Voigt mapping prescribes."""
        report = {}
        for m in self.MATS:
            p = MATERIAL_DATABASE[m]
            C = build_stiffness_tensor(p)
            c11 = float(C[0, 0, 0, 0])
            c12 = float(C[0, 0, 1, 1])
            c44 = float(C[1, 2, 1, 2])           # 23 shear
            c55 = float(C[0, 2, 0, 2])           # 13 shear
            c66 = float(C[0, 1, 0, 1])           # 12 shear
            struct = p["crystal_structure"]
            if struct == "FCC":
                crit = {"C11-C12>0": c11 - c12,
                        "C11+2C12>0": c11 + 2.0 * c12,
                        "C44>0": c44}
                for k, v in crit.items():
                    self.assertGreater(v, 0.0, f"{m}: Born {k} violated")
                for name, val in (("C44", c44), ("C55", c55), ("C66", c66)):
                    self.assertAlmostEqual(val, float(p["C44"]), delta=1.0,
                                           msg=f"{m}: cubic {name} != C44")
            else:
                c33 = float(C[2, 2, 2, 2])
                c13 = float(C[0, 0, 2, 2])
                crit = {"C11-C12>0": c11 - c12,
                        "C33(C11+C12)-2C13^2>0":
                            c33 * (c11 + c12) - 2.0 * c13 ** 2,
                        "C44>0": c44,
                        "C66=(C11-C12)/2>0": c66}
                for k, v in crit.items():
                    self.assertGreater(v, 0.0, f"{m}: Born {k} violated")
                self.assertAlmostEqual(c66, 0.5 * (c11 - c12), delta=1.0,
                                       msg=f"{m}: C66 != (C11-C12)/2")
                self.assertAlmostEqual(c44, c55, delta=1.0,
                                       msg=f"{m}: C44 != C55 (not transversely "
                                           f"isotropic)")
            report[m] = {"crystal_structure": struct,
                         "C11_GPa": c11 / 1e9, "C12_GPa": c12 / 1e9,
                         "C44_GPa": c44 / 1e9, "C55_GPa": c55 / 1e9,
                         "C66_GPa": c66 / 1e9,
                         "criteria_GPa": {k: float(v / 1e9)
                                          for k, v in crit.items()}}
        ev("born_stability", report)

    def test_cubic_point_group_invariance(self):
        """An FCC crystal must be invariant under all 24 proper rotations of
        the cube; a 30 deg rotation about z must break invariance (negative
        control proving the test discriminates)."""
        C = build_stiffness_tensor(MATERIAL_DATABASE["316L"])
        scale = np.max(np.abs(C))
        worst = 0.0
        for Q in cubic_group():
            worst = max(worst, float(np.max(np.abs(rotate_C(C, Q) - C)) / scale))
        self.assertLess(worst, 1e-12,
                        f"cubic C not invariant under the cubic group "
                        f"(worst rel. dev. {worst:.3e})")
        ctrl = float(np.max(np.abs(rotate_C(C, rot_z(30.0)) - C)) / scale)
        self.assertGreater(ctrl, 1e-2, "negative control failed")
        ev("cubic_group_invariance", {"worst_rel_dev": worst,
                                      "negative_control_rel_dev": ctrl,
                                      "n_group_elements": len(cubic_group())})

    def test_hcp_transverse_isotropy_invariance(self):
        """An HCP crystal must be invariant under arbitrary rotations about
        the c axis, and must NOT be invariant about a transverse axis."""
        C = build_stiffness_tensor(MATERIAL_DATABASE["Ti64"])
        scale = np.max(np.abs(C))
        worst = 0.0
        for ang in (7.0, 23.5, 45.0, 61.0, 119.0, 180.0):
            worst = max(worst,
                        float(np.max(np.abs(rotate_C(C, rot_z(ang)) - C)) / scale))
        self.assertLess(worst, 1e-12,
                        f"HCP C not invariant about the c axis ({worst:.3e})")
        ctrl = float(np.max(np.abs(rotate_C(C, rot_x_deg(30.0)) - C)) / scale)
        self.assertGreater(ctrl, 1e-2, "negative control failed")
        ev("hcp_transverse_isotropy", {"worst_rel_dev": worst,
                                       "negative_control_rel_dev": ctrl})

    def test_double_contraction_matches_voigt(self):
        rs = np.random.RandomState(11)
        worst = 0.0
        worst_dc = 0.0
        for m in self.MATS:
            C = build_stiffness_tensor(MATERIAL_DATABASE[m])
            M = voigt66(C)
            for _ in range(20):
                E = rs.randn(3, 3)
                E = 0.5 * (E + E.T)
                s_tensor = np.einsum("ijkl,kl->ij", C, E)
                v = np.array([E[0, 0], E[1, 1], E[2, 2], E[0, 1], E[0, 2], E[1, 2]])
                sv = M.dot(v)
                worst = max(worst, float(np.max(np.abs(s_tensor - np.array(
                    [[sv[0], sv[3], sv[4]],
                     [sv[3], sv[1], sv[5]],
                     [sv[4], sv[5], sv[2]]]))) / max(np.max(np.abs(sv)), 1.0)))
                s_dc = double_contraction(C, E)
                worst_dc = max(worst_dc,
                               float(np.max(np.abs(s_dc - s_tensor))))
                self.assertLess(float(np.max(np.abs(s_dc - s_tensor))), 1e-6,
                                f"{m}: double_contraction inconsistent")
        # 'worst' is the naive 6x6 reduction of a rank-4 tensor with the TENSOR
        # shear convention (M[a, b] = C[i, j, k, l]).  It is NOT the quantity
        # asserted here: because the symmetric pair contributes twice to
        # sigma_ij, that reduction is short by a factor 2 in every shear row and
        # is therefore off by up to 100%.  What the assertion checks is
        # double_contraction against the full einsum contraction.
        ev("naive_tensor_voigt_reduction_worst_rel_dev", worst)
        ev("double_contraction_vs_einsum_worst_abs", worst_dc)


# ==========================================================================
# Tier 1c -- model initialisation
# ==========================================================================
class TestModelInitialisation(TimedCase):
    def test_parameter_loading(self):
        """v4 must take D0 from the experimental record; S0_override must win
        over the material library."""
        report = {}
        for m in ("316L", "Ti64", "AlSi10Mg"):
            mod = build_model(m, n_grains=4, eps_step=1e-3)
            self.assertAlmostEqual(mod.params["S0"], MATERIAL_DATABASE[m]["S0"],
                                   places=12, msg=f"{m}: library S0 not loaded")
            report[m] = {"S0_params": float(mod.params["S0"]),
                         "S0_library": float(MATERIAL_DATABASE[m]["S0"]),
                         "D0_params_v4": float(mod.params["D0"]),
                         "D0_EXPERIMENTAL": float(EXPERIMENTAL[m]["D0"]),
                         "D0_MATERIAL_DATABASE":
                             float(MATERIAL_DATABASE[m].get("D0", np.nan)),
                         "crystal_structure": mod.params["crystal_structure"],
                         "S_eff": float(mod.S_eff),
                         "f_AM": float(mod.f_AM)}
            self.assertAlmostEqual(mod.params["D0"], EXPERIMENTAL[m]["D0"],
                                   places=12, msg=f"{m}: v4 D0 not from EXPERIMENTAL")
            self.assertAlmostEqual(mod.S_eff, mod.params["S0"], places=12,
                                   msg=f"{m}: v4 requires S_eff == S0")
        ov = build_model("Ti64", n_grains=4, s0=0.293)
        self.assertAlmostEqual(ov.params["S0"], 0.293, places=12)
        report["S0_override_check"] = float(ov.params["S0"])
        ev("init_parameters", report)

    def test_orientation_set_is_orthonormal_and_proper(self):
        report = {}
        for m in ("316L", "Ti64", "AlSi10Mg"):
            mod = build_model(m, n_grains=40, seed=3)
            R = np.asarray(mod.R, dtype=float)
            self.assertEqual(R.shape, (40, 3, 3))
            worst_orth = 0.0
            worst_det = 0.0
            for Rg in R:
                worst_orth = max(worst_orth,
                                 float(np.max(np.abs(Rg.T @ Rg - np.eye(3)))))
                worst_det = max(worst_det, abs(float(np.linalg.det(Rg)) - 1.0))
            self.assertLess(worst_orth, 1e-12, f"{m}: R not orthogonal")
            self.assertLess(worst_det, 1e-12, f"{m}: det(R) != +1")
            report[m] = {"worst_orthogonality_err": worst_orth,
                         "worst_det_err": worst_det, "n_grains": 40}
        ev("orientation_properness", report)

    def test_mean_schmid_factor_consistency(self):
        """theta_bar must lie in (0, 0.5] and must agree with the ensemble
        definition of the manuscript's Eq. (eq:theta_bar) evaluated on the
        model's own sampled orientations.

        Note on the two R-conventions: the solver maps tensors with
        T_c = R.T T_m R, whereas _mean_schmid and Eq. (eq:theta_bar) use the
        direction R @ e_psi.  The texture law is invariant under
        R -> R.T, so the two readings agree in expectation; both are
        recorded so the agreement can be checked explicitly.
        """
        report = {}
        for m in ("316L", "Ti64", "AlSi10Mg"):
            mod = build_model(m, n_grains=400, seed=42)
            u_psi = np.array([np.sin(np.radians(0.0)), 0.0,
                              np.cos(np.radians(0.0))])
            th_a = []
            th_b = []
            for Rg in np.asarray(mod.R, dtype=float):
                c1 = Rg @ u_psi
                c2 = Rg.T @ u_psi
                th_a.append(float(np.abs(np.einsum("i,pij,j->p", c1,
                                                   mod.P0, c1)).max()))
                th_b.append(float(np.abs(np.einsum("i,pij,j->p", c2,
                                                   mod.P0, c2)).max()))
            report[m] = {"theta_bar_code_n_mc2000": float(mod.theta_bar),
                         "theta_bar_sampled_Ru": float(np.mean(th_a)),
                         "theta_bar_sampled_Rtu": float(np.mean(th_b)),
                         "n_orientations_sampled": 400}
            self.assertGreater(mod.theta_bar, 0.0)
            self.assertLessEqual(mod.theta_bar, 0.5 + 1e-9)
            for k in ("theta_bar_sampled_Ru", "theta_bar_sampled_Rtu"):
                self.assertLess(abs(report[m][k] - float(mod.theta_bar)), 0.02,
                                f"{m}: {k} disagrees with the code by more "
                                f"than Monte-Carlo tolerance")
        ev("mean_schmid_factor", report)

    def test_orientation_seed_reproducibility(self):
        a = build_model("Ti64", n_grains=12, seed=17)
        b = build_model("Ti64", n_grains=12, seed=17)
        c = build_model("Ti64", n_grains=12, seed=18)
        self.assertTrue(np.array_equal(a.R, b.R), "same seed: orientations differ")
        self.assertEqual(a.theta_bar, b.theta_bar)
        self.assertFalse(np.array_equal(a.R, c.R),
                         "different seeds: orientations identical")
        ev("seed_reproducibility", {
            "same_seed_max_abs_diff": float(np.max(np.abs(a.R - b.R))),
            "theta_bar_seed17": float(a.theta_bar),
            "theta_bar_seed18": float(c.theta_bar),
            "different_seed_max_abs_diff": float(np.max(np.abs(a.R - c.R)))})

    def test_crss_and_saturation_vectors(self):
        report = {}
        for m in ("316L", "Ti64", "AlSi10Mg"):
            mod = build_model(m, n_grains=4)
            g0 = np.asarray(mod.g0_vec, dtype=float)
            gs = np.asarray(mod.g_sat_vec, dtype=float)
            struct = mod.params["crystal_structure"]
            report[m] = {"crystal_structure": struct, "n_systems": int(g0.size),
                         "g0_unique_MPa": [float(x / 1e6)
                                           for x in np.unique(g0)],
                         "g_sat_unique_MPa": [float(x / 1e6)
                                              for x in np.unique(gs)]}
            self.assertEqual(g0.size, 12, f"{m}: expected 12 slip systems")
            self.assertEqual(gs.size, 12, f"{m}: g_sat must have 12 entries")
            self.assertTrue(np.all(g0 > 0.0))
            self.assertTrue(np.all(gs >= g0 - 1e-9),
                            f"{m}: g_sat < g_0 in at least one system")
            if struct == "HCP":
                self.assertEqual(len(np.unique(g0)), 3,
                                 f"{m}: HCP needs 3 CRSS families")
                self.assertEqual(len(np.unique(gs)), 3,
                                 f"{m}: HCP needs 3 g_sat families")
                self.assertEqual(float(g0[0]), float(g0[2]))
                self.assertNotAlmostEqual(float(g0[0]), float(g0[3]), places=2)
            else:
                self.assertEqual(len(np.unique(g0)), 1,
                                 f"{m}: FCC must use one CRSS value")
        ev("crss_saturation", report)

    def test_global_database_not_mutated_by_overrides(self):
        before = copy.deepcopy(MATERIAL_DATABASE["Ti64"])
        _ = build_model("Ti64", n_grains=4,
                        overrides={"gs_basal": 1.0, "S0": 1.0, "D0": 0.0})
        after = MATERIAL_DATABASE["Ti64"]
        self.assertEqual(sorted(before.keys()), sorted(after.keys()))
        for k in before:
            self.assertTrue(np.all(np.asarray(before[k]) == np.asarray(after[k])),
                            f"MATERIAL_DATABASE['Ti64']['{k}'] was mutated")
        self.assertNotEqual(after.get("S0"), 1.0, "global S0 was mutated")
        ev("global_database_isolation",
           "MATERIAL_DATABASE unmodified after parameter overrides")


# ==========================================================================
# Tier 1d -- damage evolution and driver interface
# ==========================================================================
class TestDamageEvolution(TimedCase):
    def test_per_grain_damage_monotonically_nondecreasing(self):
        """Damage flows forward only: on a monotone strain path every grain's
        damage must be non-decreasing (uses integrate(), the raw path)."""
        mod = build_model("Ti64", n_grains=5, eps_step=1e-3, n_sub=30, seed=42)
        es = 1.0e-3
        # The path must extend beyond the plastic-strain gate p_D = 0.05
        # (Ti64, MATERIAL_DATABASE) for damage to start at all:
        # 120 steps at eps_step = 1e-3 -> final strain 0.12.
        n = 120
        path = np.zeros((n, 6))
        path[:, 0] = np.arange(1, n + 1) * es
        stress, D_out, p_out, DG = mod.integrate(path)
        DG = np.asarray(DG, dtype=float)
        dDG = np.diff(DG, axis=0)
        worst_drop = float(dDG.min())
        dp = np.diff(np.asarray(p_out, dtype=float))
        self.assertGreaterEqual(worst_drop, -1e-12,
                                f"per-grain damage decreased by {worst_drop:.3e}")
        self.assertGreaterEqual(float(dp.min()), -1e-15,
                                "equivalent plastic strain decreased")
        self.assertGreater(float(D_out[-1]), float(D_out[0]),
                           "damage did not grow over the strain path")
        ev("damage_monotonicity", {
            "worst_per_grain_increment": worst_drop,
            "min_delta_p": float(dp.min()),
            "D_initial": float(D_out[0]), "D_final": float(D_out[-1]),
            "p_final": float(p_out[-1]), "n_steps": n,
            "strain_final": float(path[-1, 0])})

    def test_damage_gate_is_driven_by_plastic_strain(self):
        """Damage must not evolve below the plastic-strain threshold p_D."""
        mod = build_model("Ti64", n_grains=5, eps_step=1e-3, seed=42,
                          overrides={"p_D": 1.0e9})
        r = mod.run_uniaxial(max_strain=5.0e-3)
        d = np.asarray(r["damage"], dtype=float)
        dD = float(np.max(np.abs(d - d[0])))
        self.assertLess(dD, 1e-14, "damage changed although p_D = 1e9")
        ev("damage_threshold_gate", {
            "p_D_used": 1.0e9,
            "max_abs_damage_change": dD,
            "D0": float(mod.params["D0"])})

    def test_uniaxial_damage_history_is_monotone(self):
        """The damage history returned by run_uniaxial is the mean over all
        grains with broken grains clipped to 0.99, hence non-decreasing."""
        mod = build_model("Ti64", n_grains=5, eps_step=2e-3, n_sub=30, seed=42)
        r = mod.run_uniaxial(max_strain=0.60)
        d = np.asarray(r["damage"], dtype=float)
        worst = float(np.diff(d).min())
        self.assertGreaterEqual(worst, -1e-15,
                                f"uniaxial damage history decreased by {worst:.3e}")
        ev("uniaxial_damage_monotonicity", {
            "worst_increment": worst, "n_points": int(d.size),
            "D_final": float(d[-1]),
            "Dc": float(EXPERIMENTAL["Ti64"]["Dc"])})


class TestRunUniaxialInterface(TimedCase):
    def test_stress_is_in_pascals(self):
        """run_uniaxial returns stress in Pa.  Peak stress must be
        O(1e8)-O(1e9) Pa; a MPa return would give ~1e2-1e3."""
        report = {}
        for m in ("Ti64", "AlSi10Mg", "316L"):
            mod = build_model(m, n_grains=5, eps_step=2e-3, n_sub=30, seed=42)
            r = mod.run_uniaxial(max_strain=0.60)
            pk = float(np.max(r["stress"]))
            self.assertGreater(pk, 1.0e7, f"{m}: peak stress {pk} too small for Pa")
            self.assertLess(pk, 5.0e9, f"{m}: peak stress {pk} too large for Pa")
            exp = EXPERIMENTAL[m]
            uts_exp = exp.get("uts")
            if isinstance(uts_exp, dict):
                uts_ref = float(uts_exp.get(0, list(uts_exp.values())[-1]))
            elif uts_exp is None:
                uts_ref = None
            else:
                uts_ref = float(uts_exp)
            report[m] = {"peak_stress_Pa": pk, "peak_stress_MPa": pk / 1e6,
                         "uts_experimental_MPa": uts_ref,
                         "uts_experimental_MPa_by_orientation": uts_exp,
                         "uts_ratio_model_over_experiment":
                             float(pk / 1e6 / uts_ref) if uts_ref else None,
                         "fracture_strain": float(r["fracture_strain"]),
                         "ef_experimental": float(exp["ef"][0])}
        ev("stress_units_pascals", report)

    def test_fracture_strain_and_uts_are_reproducible(self):
        """fracture_strain and uts must be exactly recoverable from the raw
        returned histories and the material's Dc."""
        m = "Ti64"
        mod = build_model(m, n_grains=5, eps_step=2e-3, n_sub=30, seed=42)
        r = mod.run_uniaxial(max_strain=0.60)
        Dc = float(EXPERIMENTAL[m]["Dc"])
        ef_re = fracture_from_history(r["strain"], r["damage"], Dc, mod.eps_step)
        self.assertIsNotNone(ef_re, "damage history never reached Dc")
        self.assertAlmostEqual(float(r["fracture_strain"]), ef_re, places=12,
                               msg="fracture_strain not reproducible from Dc")
        i0 = int(np.where(np.asarray(r["damage"]) >= Dc)[0][0])
        uts_re = float(np.max(np.asarray(r["stress"])[:i0 + 1]))
        self.assertAlmostEqual(float(r["uts"]), uts_re, places=6,
                               msg="uts is not the pre-fracture stress maximum")
        eps = np.asarray(r["strain"], dtype=float)
        deps = float(eps[1] - eps[0])
        ev("fracture_consistency", {
            "fracture_strain": float(r["fracture_strain"]),
            "fracture_strain_recomputed": float(ef_re),
            "Dc": Dc, "uts_Pa": float(r["uts"]),
            "uts_MPa": float(r["uts"] / 1e6),
            "uts_recomputed_Pa": uts_re,
            "eps_step_requested": float(mod.eps_step),
            "deps_effective": deps,
            "deps_over_eps_step": deps / float(mod.eps_step),
            "strain_rate_requested": float(mod.strain_rate),
            "strain_rate_effective": deps / (float(mod.eps_step)
                                             / float(mod.strain_rate)),
            "n_points_returned": int(eps.size),
            "frac_broken_at_end": float(np.max(np.asarray(r["frac_broken"])))})

    def test_effective_strain_increment(self):
        """Document run_uniaxial's n_steps = max(10, int(max_strain/eps_step))+1
        behaviour: unless max_strain is an exact multiple of eps_step with
        quotient >= 10, the effective increment and strain rate differ from
        the requested eps_step / strain_rate, and the fracture-strain
        interpolation (which uses eps_step) is applied with the wrong
        increment.  The configurations used by this suite must be exact."""
        cases = []
        for max_strain, es in ((0.004, 5e-4), (0.002, 5e-4), (0.17, 5e-4),
                               (0.25, 5e-4), (0.30, 1e-3)):
            mod = build_model("Ti64", n_grains=2, eps_step=es, seed=1)
            n_steps = max(10, int(max_strain / es)) + 1
            deps = max_strain / (n_steps - 1)
            cases.append({"max_strain": max_strain, "eps_step": es,
                          "n_steps": n_steps, "deps_effective": deps,
                          "deps_over_eps_step": deps / es,
                          "strain_rate_effective":
                              deps / (es / float(mod.strain_rate)),
                          "strain_rate_requested": float(mod.strain_rate)})
        ev("effective_increment", {"cases": cases})
        for c in cases:
            if c["max_strain"] in (0.25, 0.30):
                self.assertLess(abs(c["deps_over_eps_step"] - 1.0), 1e-9,
                                f"suite configuration max_strain="
                                f"{c['max_strain']} eps_step={c['eps_step']} "
                                f"is not an exact multiple "
                                f"(ratio {c['deps_over_eps_step']:.9f})")


# ==========================================================================
# Tier 2a -- analytic elastic benchmark
# ==========================================================================
ELASTIC_WINDOWS = {      # (max_strain, eps_step): max_strain = 10*eps_step
    "316L": (2.0e-3, 2.0e-4),
    "Ti64": (5.0e-3, 5.0e-4),
    "AlSi10Mg": (3.0e-3, 3.0e-4),
}


class TestElasticLimitBaseline(TimedCase):
    """B3-1: in the limit of vanishing damage (S0 -> infinity) the uniaxial
    response must equal the exact anisotropic elastic solution of the model's
    own kinematic constraint.  The reference is computed from the model's
    orientation set through a direct re-implementation of the solver's
    crystal-to-macro averaging (aggregate_response) and of its lateral
    condition (e_uniform_lateral), not through a textbook isotropic formula.

    The constraint matters: run_uniaxial solves for one scalar lateral
    coefficient w with eps_22 = eps_33 = -w eps_11 such that the MEAN lateral
    stress 0.5*(sigma_22 + sigma_33) vanishes.  Using the textbook
    sigma_22 = 0 condition instead overestimates E* by ~6% (316L) and is
    therefore not usable as a reference here.
    """

    NG = 10

    def test_elastic_modulus_matches_analytic_reference(self):
        report = {}
        for m, (ms, es) in ELASTIC_WINDOWS.items():
            mod = build_model(m, n_grains=self.NG, eps_step=es, seed=42,
                              s0=1.0e12)
            C = build_stiffness_tensor(MATERIAL_DATABASE[m])
            E_star, w_star = e_uniform_lateral(C, np.asarray(mod.R, dtype=float))
            r = mod.run_uniaxial(max_strain=ms,
                                 inner_tol_pa=1.0e5, n_inner=60)
            eps = np.asarray(r["strain"], dtype=float)
            sig = np.asarray(r["stress"], dtype=float)
            E_num = float(sig[1] / eps[1])
            rel = (E_num - E_star) / E_star
            report[m] = {"E_analytic_GPa": E_star / 1e9,
                         "w_analytic_from_solver_kinematics": w_star,
                         "E_numerical_GPa": E_num / 1e9,
                         "rel_dev_pct": 100.0 * rel,
                         "max_abs_D_minus_D0":
                             float(np.max(np.abs(np.asarray(r["damage"])
                                                 - float(mod.params["D0"]))))}
            self.assertLess(abs(rel), 0.01,
                            f"{m}: elastic modulus deviates {100*rel:+.4f}% "
                            f"from the analytic reference")
        ev("elastic_limit_benchmark", report)

    def test_elastic_window_is_linear_and_damage_free(self):
        report = {}
        for m, (ms, es) in ELASTIC_WINDOWS.items():
            mod = build_model(m, n_grains=self.NG, eps_step=es, seed=42,
                              s0=1.0e12)
            r = mod.run_uniaxial(max_strain=ms,
                                 inner_tol_pa=1.0e5, n_inner=60)
            eps = np.asarray(r["strain"], dtype=float)
            sig = np.asarray(r["stress"], dtype=float)
            dmg = np.asarray(r["damage"], dtype=float)
            D0 = float(mod.params["D0"])
            self.assertLess(float(np.max(np.abs(dmg - D0))), 1e-14,
                            f"{m}: damage evolved although S0 = 1e12")
            secant = sig[1:] / eps[1:]
            spread = float(secant.max() - secant.min()) / float(secant.mean())
            report[m] = {"max_abs_D_minus_D0": float(np.max(np.abs(dmg - D0))),
                         "E_secant_spread_rel": spread,
                         "E_mean_GPa": float(secant.mean() / 1e9),
                         "n_points": int(eps.size),
                         "strain_max": float(eps[-1])}
            self.assertLess(spread, 0.01,
                            f"{m}: secant modulus varies {100*spread:.4f}% "
                            f"across the elastic window")
        ev("elastic_window_linearity", report)

    def test_default_inner_tolerance_bias_is_recorded(self):
        """Documentation: with the production defaults (inner_tol_pa = 3e6,
        n_inner = 25) the lateral-contraction solve is under-resolved and the
        extracted modulus is biased.  The bias must stay below 10% (it is a
        few percent) and is reported for the record."""
        report = {}
        for m, (ms, es) in ELASTIC_WINDOWS.items():
            mod = build_model(m, n_grains=self.NG, eps_step=es, seed=42,
                              s0=1.0e12)
            C = build_stiffness_tensor(MATERIAL_DATABASE[m])
            E_star, _ = e_uniform_lateral(C, np.asarray(mod.R, dtype=float))
            r = mod.run_uniaxial(max_strain=ms)
            eps = np.asarray(r["strain"], dtype=float)
            sig = np.asarray(r["stress"], dtype=float)
            E_def = float(sig[1] / eps[1])
            rel = (E_def - E_star) / E_star
            report[m] = {"E_default_tolerance_GPa": E_def / 1e9,
                         "E_analytic_GPa": E_star / 1e9,
                         "rel_dev_pct": 100.0 * rel,
                         "default_inner_tol_pa": 3.0e6,
                         "default_n_inner": 25}
            self.assertLess(abs(rel), 0.10,
                            f"{m}: default-tolerance bias {100*rel:+.4f}% > 10%")
        ev("elastic_default_tolerance_bias", report)


# ==========================================================================
# Tier 2b -- isotropic / Lemaitre limit
# ==========================================================================
class TestLemaitreLimit(TimedCase):
    """B3-2: with every descriptor at its reference value the corrective
    factor collapses to unity and the model reproduces the classical Lemaitre
    ductile-damage law bit-for-bit.

    Reference state of the manuscript: fully dense, equiaxed, texture-free
    material -- phi -> 0, lambda_eff = lambda_ref, xi_bar = 1, theta_bar = 0.5
    -- together with D0 -> 0 (f_D0 = exp(beta3*0) = 1) and beta6 = 0.
    """

    NEUTRAL = {"xi_grain": 1.0, "lambda_mp": float(LAMBDA_REF),
               "_w_iso": 1.0, "phi": 0.0, "D0": 0.0, "beta2": 0.0}

    def test_each_corrective_factor_unity_at_reference_state(self):
        """Normalisation convention of each sub-factor with the *production*
        coefficients (only the descriptor is neutralised)."""
        p = dict(MATERIAL_DATABASE["Ti64"])
        p.update({"xi_grain": 1.0, "lambda_mp": float(LAMBDA_REF),
                  "_w_iso": 1.0, "phi": 0.0, "D0": 0.0})
        lam = float(lambda_eff(p["lambda_mp"], 0.0, p["_w_iso"]) / LAMBDA_REF)
        xi = float(xi_eff(p["xi_grain"], 0.0))
        f_at_half, comp_half = factors_v4(p, 0.0, 0.5)
        self.assertAlmostEqual(lam, 1.0, places=12,
                               msg=f"lambda_eff/lambda_ref = {lam} != 1")
        self.assertAlmostEqual(xi, 1.0, places=12,
                               msg=f"xi_eff = {xi} != 1")
        self.assertAlmostEqual(comp_half["f_AM"], 1.0, places=12,
                               msg="f_AM != 1 at the reference state")
        for k in ("f_phi", "f_theta", "f_D0", "f_lambda", "f_xi", "f_shield"):
            self.assertAlmostEqual(comp_half[k], 1.0, places=12,
                                   msg=f"{k} = {comp_half[k]} != 1")
        # f_theta must grow when the sampled texture gives theta_bar < 0.5
        f_lo, comp_lo = factors_v4(p, 0.0, 0.40)
        self.assertGreater(comp_lo["f_theta"], 1.0,
                           "f_theta must exceed 1 for theta_bar < 0.5")
        # the real (production) texture factor at the sampled theta_bar
        q = dict(MATERIAL_DATABASE["Ti64"])
        mod = build_model("Ti64", n_grains=30, seed=42)
        _, comp_prod = factors_v4(mod.params, 0.0, mod.theta_bar)
        ev("lemaitre_factor_normalisation", {
            "neutral_lambda_eff_over_ref": lam, "neutral_xi_eff": xi,
            "neutral_components": comp_half,
            "f_theta_at_theta_bar_0.40": comp_lo["f_theta"],
            "production_theta_bar": float(mod.theta_bar),
            "production_components": comp_prod,
            "production_lambda_mp": float(mod.params["lambda_mp"]),
            "production_xi_grain": float(mod.params["xi_grain"]),
            "production_w_iso": float(mod.params.get("_w_iso", W_ISO)),
            "production_phi": float(mod.params["phi"]),
            "production_D0": float(mod.params["D0"])})

    def test_xi_eff_analytic_endpoints_and_neutrality(self):
        """Grain-morphology projection: radial conjugate-diameter ratio.
        xi_eff(0)=xi0, xi_eff(90)=1/xi0, xi_eff(45)=1 for any xi0 (the two
        conjugate radii coincide at 45 deg). Guards against the superseded
        parametric-angle form (xi0^2 cos^2 psi + sin^2 psi)/xi0 that
        over-predicted the 45 deg aspect ratio (e.g. 1.89 instead of 1.0 at
        xi0=3.5), which fed both f_xi and the directional saturation gs(psi)."""
        for xi0 in (1.5, 2.5, 3.5):
            self.assertAlmostEqual(float(xi_eff(xi0, 0.0)), xi0, places=12,
                                   msg=f"xi_eff({xi0},0) != {xi0}")
            self.assertAlmostEqual(float(xi_eff(xi0, 90.0)), 1.0 / xi0, places=12,
                                   msg=f"xi_eff({xi0},90) != 1/{xi0}")
            self.assertAlmostEqual(float(xi_eff(xi0, 45.0)), 1.0, places=12,
                                   msg=f"xi_eff({xi0},45) != 1")
        # f_xi is aspect-neutral at 45 deg for any morphology (1/xi_eff - 1 = 0)
        p = dict(MATERIAL_DATABASE["Ti64"])
        _, comp45 = factors_v4(p, 45.0, 0.5)
        self.assertAlmostEqual(comp45["xi_eff"], 1.0, places=12,
                               msg="xi_eff(45) must be exactly 1")

    def test_literal_three_condition_limit_is_insufficient(self):
        """Recorded finding: the limit condition quoted in the verification
        brief (xi0 = 1, lambda0 = lambda_ref, theta = 0.5) does NOT give
        f_AM = 1, because the porosity term f_phi and the initial-damage term
        f_D0 remain active, and the isotropic MPB weight _w_iso = 0.25 makes
        lambda_eff = 0.25*lambda0 so f_lambda = 0.4.  The manuscript's own
        reference state therefore also requires phi -> 0 and D0 -> 0 (the
        brief omits them) and _w_iso = 1 to make lambda_eff = lambda0."""
        report = {}
        for m in ("316L", "Ti64", "AlSi10Mg"):
            p = dict(MATERIAL_DATABASE[m])
            p.update({"xi_grain": 1.0, "lambda_mp": float(LAMBDA_REF)})
            f_3, c3 = factors_v4(p, 0.0, 0.5)
            report[m] = {"phi": float(p["phi"]), "D0_after_v4_rule":
                         float(EXPERIMENTAL[m]["D0"]),
                         "_w_iso": float(p.get("_w_iso", W_ISO)),
                         "lambda_eff_over_ref":
                             float(lambda_eff(p["lambda_mp"], 0.0,
                                              p.get("_w_iso", W_ISO))
                                   / LAMBDA_REF),
                         "xi_eff": float(xi_eff(p["xi_grain"], 0.0)),
                         "components": c3, "f_AM": float(f_3)}
            self.assertLess(f_3, 1.0,
                            f"{m}: the three-condition limit unexpectedly "
                            f"already gives f_AM = 1")
        ev("lemaitre_three_condition_limit", report)
        ev("lemaitre_reference_state_note",
           "phi -> 0 and D0 -> 0 are required in addition to "
           "xi = 1, lambda_eff = lambda_ref, theta_bar = 0.5; "
           "_w_iso = 1 is needed for lambda_eff = lambda0.")

    def test_neutral_parameters_reproduce_lemaitre_exactly(self):
        """A run whose descriptors are all at the reference state (computed
        f_AM == 1) must be bit-for-bit identical to a run with the corrective
        factor forced to unity -- i.e. the classical Lemaitre model."""
        common = dict(material="Ti64", n_grains=5, eps_step=2e-3, n_sub=30,
                      seed=42, s0=0.32, overrides=self.NEUTRAL)
        m_free = build_model(**common)
        self.assertAlmostEqual(float(m_free.f_AM), 1.0, places=12,
                               msg="neutral descriptors did not give f_AM = 1")
        r_free = m_free.run_uniaxial(max_strain=0.60)

        m_hom = build_model(force_homogeneous=True, **common)
        self.assertAlmostEqual(float(m_hom.f_AM), 1.0, places=12)
        r_hom = m_hom.run_uniaxial(max_strain=0.60)

        n = min(len(r_free["strain"]), len(r_hom["strain"]))
        d_sig = float(np.max(np.abs(np.asarray(r_free["stress"], dtype=float)[:n]
                                    - np.asarray(r_hom["stress"], dtype=float)[:n])))
        d_dmg = float(np.max(np.abs(np.asarray(r_free["damage"])[:n]
                                    - np.asarray(r_hom["damage"])[:n])))
        self.assertLess(d_dmg, 1e-12, "damage histories differ")
        self.assertLess(d_sig, 1e-6, "stress histories differ")
        self.assertAlmostEqual(float(r_free["fracture_strain"]),
                               float(r_hom["fracture_strain"]), places=12)
        ev("lemaitre_limit_equivalence", {
            "max_abs_damage_diff": d_dmg,
            "max_abs_stress_diff_Pa": d_sig,
            "fracture_strain_f_AM_computed": float(r_free["fracture_strain"]),
            "fracture_strain_f_AM_forced": float(r_hom["fracture_strain"]),
            "uts_MPa_f_AM_computed": float(r_free["uts"] / 1e6),
            "uts_MPa_f_AM_forced": float(r_hom["uts"] / 1e6),
            "n_points_compared": n})


# ==========================================================================
# Tier 3 -- numerical convergence
# ==========================================================================
def _run_case(material, ng, es, n_sub=30, s0=0.32, seed=42, psi=0.0, key=None):
    """One TaylorCPCDM case with an exactly specified strain increment:
    max_strain = 300*eps_step guarantees deps == eps_step exactly."""
    t0 = time.time()
    mod = build_model(material, psi=psi, n_grains=ng, eps_step=es, n_sub=n_sub,
                      strain_rate=1e-3, seed=seed, s0=s0)
    r = mod.run_uniaxial(max_strain=300.0 * es)
    eps = np.asarray(r["strain"], dtype=float)
    deps = float(eps[1] - eps[0]) if eps.size > 1 else float("nan")
    out = {"material": material, "psi_deg": psi, "n_grains": ng,
           "eps_step": es, "n_sub": n_sub,
           "deps_effective": deps, "deps_over_eps_step": deps / es,
           "n_points": int(eps.size),
           "ef": float(r["fracture_strain"]),
           "uts_MPa": float(r["uts"] / 1e6),
           "wallclock_s": round(time.time() - t0, 2)}
    if key:
        ev(key, out)
    return out


class TestConvergence(TimedCase):
    """B2 convergence study, Ti-6Al-4V at psi = 0 deg, S0 = 0.32.

    Thresholds are those specified in the verification brief (n_grains < 10%,
    eps_step < 5%, n_sub < 5%).  Measured values are always recorded whatever
    the outcome.
    """

    @staticmethod
    def _rel(values, reference):
        return [100.0 * (v - reference) / reference for v in values]

    def test_convergence_with_eps_step(self):
        """Production strain increments eps_step = 2.5e-4 / 5e-4 / 1e-3 at
        n_grains = 20 (Ti-6Al-4V psi=0, S0=0.32), the increments reported in the
        convergence section (Section sec:convergence: 0.07256, 0.07229, 0.07180,
        spread <= 0.68%). Coarse steps 1.5e-3 / 2e-3 are run only as breakdown-
        threshold probes and recorded, not asserted: production never uses them
        (316L uses 1e-3; Ti-6Al-4V/AlSi10Mg use 5e-4)."""
        runs = [_run_case("Ti64", 20, es) for es in (2.5e-4, 5e-4, 1.0e-3)]
        efs = [r["ef"] for r in runs]
        rel = self._rel(efs, efs[0])
        probe15 = _run_case("Ti64", 20, 1.5e-3)
        probe20 = _run_case("Ti64", 20, 2.0e-3)
        max_rel = max(abs(x) for x in rel)
        ev("convergence_eps_step", {
            "runs": runs, "ef": efs, "reference_eps_step": 5e-4,
            "rel_dev_pct": rel, "max_abs_rel_dev_pct": max_rel,
            "threshold_pct": 5.0, "passed": bool(max_rel < 5.0),
            "breakdown_probe_eps_step_1p5e-3": probe15,
            "breakdown_probe_eps_step_2e-3": probe20,
            "breakdown_probe_rel_dev_pct":
                100.0 * (probe20["ef"] - efs[0]) / efs[0]})
        self.assertLess(max_rel, 5.0,
                        "eps_step convergence FAILED: ef = "
                        + ", ".join(f"{e:.5f}" for e in efs)
                        + f" (max relative spread {max_rel:.2f}% > 5%)")

    def test_convergence_with_eps_step_cross_material(self):
        """The same two-step probe for 316L and AlSi10Mg (documentation, no
        threshold asserted) to show whether the step-size sensitivity is
        material specific."""
        report = {}
        for m in ("AlSi10Mg", "316L"):
            runs = [_run_case(m, 10, es) for es in (5e-4, 2.0e-3)]
            efs = [r["ef"] for r in runs]
            report[m] = {"n_grains": 10, "runs": runs, "ef": efs,
                         "rel_dev_pct_2e-3_vs_5e-4":
                             100.0 * (efs[1] - efs[0]) / efs[0]}
        ev("convergence_eps_step_cross_material", report)
        for m, v in report.items():
            self.assertTrue(np.isfinite(v["rel_dev_pct_2e-3_vs_5e-4"]),
                            f"{m}: non-finite step-size sensitivity")

    def test_convergence_with_n_sub(self):
        """n_sub = 15 / 30 / 60 at n_grains = 20, eps_step = 1e-3."""
        runs = [_run_case("Ti64", 20, 1.0e-3, n_sub=ns) for ns in (15, 30, 60)]
        efs = [r["ef"] for r in runs]
        rel = self._rel(efs, efs[1])
        max_rel = max(abs(x) for x in rel)
        ev("convergence_n_sub", {
            "runs": runs, "ef": efs, "reference_n_sub": 30,
            "rel_dev_pct": rel, "max_abs_rel_dev_pct": max_rel,
            "threshold_pct": 5.0, "passed": bool(max_rel < 5.0)})
        self.assertLess(max_rel, 5.0,
                        "n_sub convergence FAILED: ef = "
                        + ", ".join(f"{e:.5f}" for e in efs))

    def test_convergence_with_n_grains(self):
        """n_grains = 27 / 64 / 125 at eps_step = 1e-3 (as specified)."""
        runs = [_run_case("Ti64", ng, 1.0e-3) for ng in (27, 64, 125)]
        efs = [r["ef"] for r in runs]
        rel = self._rel(efs, efs[-1])
        max_rel = max(abs(x) for x in rel)
        ev("convergence_n_grains", {
            "runs": runs, "ef": efs, "reference_n_grains": 125,
            "rel_dev_pct_vs_125": rel, "max_abs_rel_dev_pct": max_rel,
            "threshold_pct": 10.0, "passed": bool(max_rel < 10.0)})
        self.assertLess(max_rel, 10.0,
                        "n_grains convergence FAILED: ef = "
                        + ", ".join(f"{e:.5f}" for e in efs))

    def test_convergence_with_n_grains_archive_grid(self):
        """The archived grid itself (n_grains = 10/20/30/50 at
        eps_step = 5e-4) so the grain-count convergence can be compared with
        output/convergence_test.json on its own settings."""
        runs = [_run_case("Ti64", ng, 5.0e-4) for ng in (10, 20, 30, 50)]
        efs = [r["ef"] for r in runs]
        rel = self._rel(efs, efs[-1])
        max_rel = max(abs(x) for x in rel)
        ev("convergence_n_grains_archive_grid", {
            "runs": runs, "ef": efs, "reference_n_grains": 50,
            "rel_dev_pct_vs_50": rel, "max_abs_rel_dev_pct": max_rel,
            "threshold_pct": 5.0, "passed": bool(max_rel < 5.0)})
        self.assertLess(max_rel, 5.0,
                        "grain-count convergence FAILED on the archived grid: "
                        + ", ".join(f"{e:.5f}" for e in efs))


class TestArchivedReproduction(TimedCase):
    """Tier 4 -- provenance check against the archived results shipped in
    simulation/output/.

    convergence_test.json and taylor_single_crystal_bounds_v9.json were
    written on 2026-09-23, i.e. BEFORE the last modifications of
    src/taylor_cpcdm.py (2026-09-25 07:41) and src/am_correction_v4.py
    (2026-09-25 22:25).  These tests re-run the archived protocol with the
    current sources and assert reproducibility.
    """

    def test_archived_convergence_json_is_reproducible(self):
        arch_path = os.path.join(OUT_DIR, "convergence_test.json")
        if not os.path.isfile(arch_path):
            self.skipTest("convergence_test.json not present")
        with open(arch_path, "r", encoding="utf-8") as fh:
            arch = json.load(fh)
        if SIM_DIR not in sys.path:
            sys.path.insert(0, SIM_DIR)
        import convergence_test as conv  # noqa: E402

        ng = 20
        ef_now, uts_now = conv.run(ng, 5.0e-4)

        arch_ef = None
        arch_uts = None
        for rec in (arch.get("n_grains_convergence") or []):
            if isinstance(rec, dict) and int(rec.get("n_grains", -1)) == ng:
                arch_ef = rec.get("ef")
                arch_uts = rec.get("uts_MPa")
        if arch_ef is None:
            for k, v in arch.items():
                if k == f"ng{ng}" or k.startswith(f"n{ng}_"):
                    arch_ef = v
        self.assertIsNotNone(arch_ef, "could not locate the archived ef value")
        rel = abs(float(ef_now) - float(arch_ef)) / float(arch_ef)
        ev("archive_convergence_json", {
            "archive_file": arch_path,
            "archive_written": datetime.fromtimestamp(
                os.path.getmtime(arch_path)).strftime("%Y-%m-%d %H:%M:%S"),
            "taylor_cpcdm_mtime": datetime.fromtimestamp(
                os.path.getmtime(os.path.join(SRC_DIR, "taylor_cpcdm.py")))
                .strftime("%Y-%m-%d %H:%M:%S"),
            "am_correction_v4_mtime": datetime.fromtimestamp(
                os.path.getmtime(os.path.join(SRC_DIR, "am_correction_v4.py")))
                .strftime("%Y-%m-%d %H:%M:%S"),
            "n_grains": ng, "eps_step": 5.0e-4,
            "ef_current_source": float(ef_now),
            "uts_MPa_current_source": float(uts_now),
            "ef_archived": float(arch_ef),
            "uts_MPa_archived": None if arch_uts is None else float(arch_uts),
            "ef_rel_dev_pct": 100.0 * rel,
            "archive_keys": sorted(arch.keys())})
        self.assertLess(rel, 0.01,
                        f"archived convergence_test.json is NOT reproducible "
                        f"with the current sources: ef = {ef_now:.5f} (now) vs "
                        f"{float(arch_ef):.5f} (archived), {100*rel:+.1f}%")

    def test_archived_single_crystal_bounds_are_reproducible(self):
        """The Discussion brackets the Taylor aggregate by its single-crystal
        extremes (Section sec:discussion); every quoted value is regenerated
        from the current sources at eps_f = 0.07229 (Ti-6Al-4V 20-grain Taylor),
        so the reproducibility guard re-runs the current single-crystal-bounds
        generator and checks it against the maintained archive. The superseded
        v9 (0.0566) family predates the transverse short-ligament f_xi revision
        and the leakage-free unification and is no longer quoted."""
        arch_path = os.path.join(OUT_DIR,
                                 "taylor_single_crystal_bounds.json")
        if not os.path.isfile(arch_path):
            self.skipTest("taylor_single_crystal_bounds.json not present "
                          "(run taylor_single_crystal_bounds.py)")
        with open(arch_path, "r", encoding="utf-8") as fh:
            arch = json.load(fh)
        if "Ti64" not in arch:
            self.skipTest("unexpected archive layout")
        a = arch["Ti64"]
        ef_now = _run_case("Ti64", 20, 5.0e-4)["ef"]
        ref = float(a["ef_taylor_20grain"])
        rel = abs(ef_now - ref) / ref
        ev("archive_single_crystal", {
            "archive_file": arch_path,
            "ef_current_source_20grain": float(ef_now),
            "ef_archived_20grain": ref,
            "ef_archived_single_min": float(a.get("ef_single_min", np.nan)),
            "ef_archived_single_max": float(a.get("ef_single_max", np.nan)),
            "ef_archived_single_mean": float(a.get("ef_single_mean", np.nan)),
            "dev_pct_archived_min": float(a.get("dev_pct_min", np.nan)),
            "dev_pct_archived_max": float(a.get("dev_pct_max", np.nan)),
            "rel_dev_pct": 100.0 * rel})
        self.assertLess(rel, 0.01,
                        f"single-crystal-bounds archive is NOT "
                        f"reproducible from current sources: ef = "
                        f"{ef_now:.5f} (now) vs "
                        f"{ref:.5f} (archived), {100*rel:+.1f}%")


# ==========================================================================
# runner
# ==========================================================================
class RecordingResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        _status_by_id[str(test.id())] = ("pass", "")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        _status_by_id[str(test.id())] = (
            "fail", self._exc_info_to_string(err, test).strip().splitlines()[-1])

    def addError(self, test, err):
        super().addError(test, err)
        _status_by_id[str(test.id())] = (
            "error", self._exc_info_to_string(err, test).strip().splitlines()[-1])

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        _status_by_id[str(test.id())] = ("skip", reason)


def _source_provenance():
    files = [("src", "taylor_cpcdm.py"), ("src", "materials.py"),
             ("src", "am_correction_v4.py"), ("src", "cp_cdm_model.py"),
             (None, "convergence_test.py"),
             (None, "taylor_single_crystal_bounds.py")]
    out = {}
    for sub, name in files:
        p = os.path.join(SIM_DIR, sub, name) if sub else os.path.join(SIM_DIR, name)
        out[f"{sub + '/' if sub else ''}{name}"] = {
            "md5": md5_of(p),
            "mtime": datetime.fromtimestamp(os.path.getmtime(p))
            .strftime("%Y-%m-%d %H:%M:%S") if os.path.isfile(p) else None}
    return out


def main():
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult)
    t0 = time.time()
    res = runner.run(suite)
    wall = round(time.time() - t0, 2)

    tests = {}
    for tid, (status, msg) in sorted(_status_by_id.items()):
        tests[tid] = {"status": status, "message": msg[:2000],
                      "duration_s": DURATIONS.get(tid)}
    summary = {
        "n_tests": len(tests),
        "n_passed": sum(1 for v in tests.values() if v["status"] == "pass"),
        "n_failed": sum(1 for v in tests.values() if v["status"] == "fail"),
        "n_errors": sum(1 for v in tests.values() if v["status"] == "error"),
        "n_skipped": sum(1 for v in tests.values() if v["status"] == "skip"),
        "was_successful": bool(res.wasSuccessful()),
        "wallclock_s": wall,
    }
    payload = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "script": os.path.abspath(__file__),
            "python": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "test_framework": "unittest (pytest is not installed in this env)",
            "fixed_seed": 42,
        },
        "source_provenance": _source_provenance(),
        "summary": summary,
        "tests": tests,
        "evidence": EVIDENCE,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "code_verification_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print("")
    print("=" * 74)
    print(f"verification suite: {summary['n_tests']} tests, "
          f"{summary['n_passed']} passed, "
          f"{summary['n_failed'] + summary['n_errors']} failed/errored, "
          f"{wall} s")
    print(f"results written to: {out_path}")
    print("=" * 74)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
