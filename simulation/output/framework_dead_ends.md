# Open-source CPFEM / phase-field frameworks: availability evidence on this platform

**Deliverable for referee comment on framework-based implementation and code verification**
**Machine-generated evidence document — all probes are read-only**

---

## 1. Purpose

The referee asks that the 3D crystal-plasticity + damage work be carried out inside an
established open-source framework (MOOSE, FEniCS, PRISMS, …) and accompanied by code
verification, unit tests and benchmark solutions.

This document records what was **measured** on the machine on which the present results were
produced, so that the statement in the manuscript

> "no open-source CPFEM framework (MOOSE, FEniCS, PRISMS) is installable in the present
> environment" (Sec. *Full-field coupled-damage reference: in-house 2D plane-strain CPFEM*)

is backed by reproducible evidence rather than by assertion. Section 6 lists what this
evidence does **not** show.

No framework was installed, upgraded, downgraded or configured. No conda environment was
created or modified. Every command below is either a version query (`pip index versions`), a
metadata-only resolution (`pip install --dry-run`, which downloads and inspects distributions
but writes nothing into site-packages), or a read-only repository query
(`conda search --platform win-64`, which loads repodata only).

## 2. Environment measured

| Item | Value |
|---|---|
| Date of probe | 2026-09-28 |
| OS | Windows 11 (Microsoft Windows NT 10.0.26200.0), 64-bit |
| Interpreter | `C:\Users\ZJ03\miniconda3\python.exe` — **CPython 3.13.0**, AMD64 |
| pip | 26.1.2 |
| conda | 26.1.1 |
| Package index used by pip | `https://mirrors.huaweicloud.com/repository/pypi/simple/` (PyPI mirror) |
| conda channels queried | `conda-forge`, plus the default `pkgs/main`, `pkgs/r`, `pkgs/msys2` channels; platform `win-64` |
| Installed scientific stack | numpy 2.3.4, scipy 1.17.1, matplotlib 3.11.0, pandas 2.3.3, PIL 12.2.0; **pytest is not installed** |
| Raw probe log (full, verbatim) | `simulation/output/framework_deadends_raw.txt` |

Every probe was issued through the interpreter above; the shell was PowerShell 5.1.

## 3. Method

For each candidate distribution name, three pip probes and one conda query were issued:

```powershell
python -m pip index versions               <name>
python -m pip install --dry-run            <name>          # full dependency resolution
python -m pip install --dry-run --only-binary=:all: --no-deps <name>   # wheel-only check
conda search -c conda-forge --platform win-64 <name>
```

The three pip probes answer three different questions and must be read together:

* `index versions` — does the *name* exist on the index at all?
* `--dry-run` — can the requirement set be resolved/built for this interpreter?
* `--only-binary=:all: --no-deps` — is there a **prebuilt wheel** for this platform and
  Python version (i.e. no compilation required)?

12 distribution names were probed:
`pymoose`, `moose`, `fenics-dolfinx`, `fenics`, `dolfin`, `fenics-basix`, `prisms`,
`prisms-pf`, `pyprisms`, `damask`, `damask-core`, `damask-parse`.

---

## 4. Results by framework

### 4.1 MOOSE (Idaho National Laboratory) — hard dead end on Windows

**The name "MOOSE" is ambiguous on PyPI, and neither PyPI package is the INL finite-element
framework.** This distinction is the whole result, and it is stated explicitly here because a
superficial probe would give the misleading impression that "MOOSE installs on Windows".

#### 4.1.1 `pymoose` — resolves, but it is the *neuroscience* MOOSE

`pip index versions pymoose` → `pymoose (4.3.1)`; available: 4.3.1, 4.3.0, 4.2.0, 4.1.4, 4.1.3,
4.1.2, 4.1.1, 3.1.5, 3.1.4.

`pip install --dry-run pymoose` → `exit=0`; the resolver downloaded a **cp313 / win_amd64
wheel** and pulled in the complete dependency closure:

```
Downloading pymoose-4.3.1-cp313-cp313-win_amd64.whl (4.3 MB)
Collecting vpython (from pymoose)                 -> vpython-7.6.5
Collecting python-libsbml (from pymoose)          -> python_libsbml-5.21.2-cp313-...whl
Collecting pyneuroml (from pymoose)               -> pyneuroml-1.3.22
Collecting pint (from pymoose)
Collecting pyqt5 (from pymoose)                   -> PyQt5-5.15.11-cp38-abi3-win_amd64.whl
Collecting pylems>=0.6.8 (from pyneuroml)
Collecting airspeed>=0.5.5 (from pyneuroml)
Collecting neuromllite>=0.6.1 (from pyneuroml)
Collecting libNeuroML>=0.6.7 (from pyneuroml)
Collecting modelspec>=0.2.6 (from neuromllite)
Collecting pymongo (from modelspec)
Collecting tables>3.10 (from libNeuroML)
...
```

The closure is decisive: **SBML, NeuroML, LEMS, neuronal-network description and a 3-D
animation package**. This is `MOOSE` = *Multiscale Object-Oriented Simulation Environment*, a
computational-neuroscience simulator for multiscale neuronal signalling. It has no
relationship to the Idaho National Laboratory `MOOSE` application framework for
finite-element multiphysics. It contains no finite-element mesh, assembly, solver, crystal
plasticity or phase-field capability.

#### 4.1.2 `moose` — a broken, unrelated third-party sdist

`pip index versions moose` → `moose (0.9.8)`; available 0.9.8, 0.9.4, 0.9.3, 0.9.2, 0.9.1, 0.9.0.
Only a source distribution exists (`Moose-0.9.8.tar.gz`, 968 kB). Resolution **fails**:

```
pip install --dry-run moose                                          -> exit=1
  × Getting requirements to build wheel did not run successfully.
  ╰─> error in Moose setup command: 'install_requires' must be a string or iterable of
      strings containing valid project/version requirement specifiers; ...
      openpyxl>=2.5.0Pillow>=3.4.1
ERROR: Failed to build 'moose' when getting requirements to build wheel

pip install --dry-run --only-binary=:all: --no-deps moose            -> exit=1
ERROR: Could not find a version that satisfies the requirement moose (from versions: none)
ERROR: No matching distribution found for moose
```

The malformed requirement string (`openpyxl>=2.5.0` concatenated with `Pillow>=3.4.1`) shows
this is an unrelated spreadsheet/data utility whose packaging metadata is invalid.

#### 4.1.3 conda-forge, win-64

```
conda search -c conda-forge --platform win-64 moose                  -> exit=1
No match found for: moose. Search: *moose*
PackagesNotFoundError: The following packages are not available from current channels: - moose
  Current channels: conda-forge/win-64, conda-forge/noarch, pkgs/main/win-64, ...
```

#### 4.1.4 Conclusion — MOOSE

The INL MOOSE framework **is not distributed on PyPI under any probed name**, and
**conda-forge publishes no `win-64` build**. The two PyPI packages that carry the name are
(a) a computational-neuroscience simulator and (b) an unrelated package with broken metadata.
Obtaining MOOSE on this machine would require a Linux environment (source build against PETSc
and libMesh, or the INL-hosted conda channel for `linux-64`/`osx-arm64`). Because the blocker
is the **platform** (no Windows distribution exists at any Python version), this is a hard
dead end, not a version-matching inconvenience.

---

### 4.2 FEniCS — soft dead end: available for win-64, but not for this interpreter

This is the one case where the honest answer is **more nuanced than "not installable"**, and it
is reported that way.

#### 4.2.1 `dolfin` (legacy FEniCS solver) — not on PyPI

```
pip index versions dolfin                                 -> exit=1
  ERROR: No matching distribution found for dolfin
pip install --dry-run dolfin                              -> exit=1
  ERROR: Could not find a version that satisfies the requirement dolfin (from versions: none)
pip install --dry-run --only-binary=:all: --no-deps dolfin -> exit=1
  ERROR: No matching distribution found for dolfin
```

DOLFIN — the mesh, assembly, linear-algebra and solver layer — is a compiled C++ library and
has **no PyPI distribution at all**.

#### 4.2.2 `fenics-dolfinx` (FEniCSx solver) — not on PyPI

```
pip index versions fenics-dolfinx                          -> exit=1
  ERROR: No matching distribution found for fenics-dolfinx
pip install --dry-run fenics-dolfinx                       -> exit=1
  ERROR: No matching distribution found for fenics-dolfinx
pip install --dry-run --only-binary=:all: --no-deps fenics-dolfinx -> exit=1
  ERROR: No matching distribution found for fenics-dolfinx
```

#### 4.2.3 `fenics` (legacy meta-package) — resolves, but delivers only the form compilers

```
pip install --dry-run fenics                                       -> exit=0
Would install fenics-2019.1.0 fenics-dijitso-2019.1.0 fenics-ffc-2019.1.0.post0
              fenics-fiat-2019.1.0 fenics-ufl-2019.1.0
```

These five distributions are `FFC` (form compiler), `FIAT` (element tabulation), `UFL` (form
language), `Dijitso` (just-in-time compilation) — all pure-Python code-generation layers.
**`dolfin` is not among them, and cannot be, because it has no distribution on PyPI** (§4.2.1).
A user who runs `pip install fenics` therefore obtains a form compiler that cannot assemble or
solve anything: the part that does the finite-element work is precisely the missing part.

#### 4.2.4 `fenics-basix` — a building block, not a framework

```
pip install --dry-run fenics-basix                                 -> exit=0
  Downloading fenics_basix-0.11.0-cp312-abi3-win_amd64.whl (6.4 MB)
Would install fenics-basix-0.11.0
```

BASIX is the finite-element *kernel* of FEniCSx: it tabulates basis functions and their
derivatives. It provides no mesh, no assembly, no solver, and no time integration, and is not
usable on its own for a CPFEM + damage problem.

#### 4.2.5 conda-forge, win-64 — available, but capped below this interpreter

```
conda search -c conda-forge --platform win-64 fenics-dolfinx        -> exit=0
# Name                       Version           Build  Channel
fenics-dolfinx                 0.9.0 py310h796b1d8_0  conda-forge
fenics-dolfinx                 0.9.0 py310h796b1d8_1  conda-forge
fenics-dolfinx                 0.9.0 py310h796b1d8_2  conda-forge
fenics-dolfinx                 0.9.0 py310h796b1d8_3  conda-forge
fenics-dolfinx                 0.9.0 py310h796b1d8_4  conda-forge
fenics-dolfinx                 0.9.0 py310he6fac46_5  conda-forge
fenics-dolfinx                 0.9.0 py310he6fac46_6  conda-forge
fenics-dolfinx                 0.9.0 py311h551409f_5  conda-forge
fenics-dolfinx                 0.9.0 py311h551409f_6  conda-forge
fenics-dolfinx                 0.9.0 py311h8b55968_0  conda-forge
fenics-dolfinx                 0.9.0 py311h8b55968_1  conda-forge
fenics-dolfinx                 0.9.0 py311h8b55968_2  conda-forge
fenics-dolfinx                 0.9.0 py311h8b55968_3  conda-forge
fenics-dolfinx                 0.9.0 py311h8b55968_4  conda-forge
fenics-dolfinx                 0.9.0 py312h1ac4452_0  conda-forge
fenics-dolfinx                 0.9.0 py312h1ac4452_1  conda-forge
fenics-dolfinx                 0.9.0 py312h1ac4452_2  conda-forge
fenics-dolfinx                 0.9.0 py312h1ac4452_3  conda-forge
fenics-dolfinx                 0.9.0 py312h1ac4452_4  conda-forge
fenics-dolfinx                 0.9.0 py312haf97b08_5  conda-forge
fenics-dolfinx                 0.9.0 py312haf97b08_6  conda-forge
fenics-dolfinx                 0.9.0  py39h4f5abcc_0  conda-forge
```

**A `win-64` build of FEniCSx 0.9.0 does exist on conda-forge — for Python 3.9, 3.10, 3.11 and
3.12 only. There is no `py313` build.** 22 builds in total; the `py39`/`py310`/`py311`/`py312`
tags are the complete set of interpreter versions offered. The `dolfinx` name alone returns no
package of its own — only the `adios4dolfinx` (0.8.1–0.10.0.post0) and `dolfinx-adjoint`
(0.2.1, 0.3.0) plugins plus a replay of the `fenics-dolfinx` rows:

```
conda search -c conda-forge --platform win-64 dolfinx               -> exit=0
No match found for: dolfinx. Search: *dolfinx*
# Name                       Version           Build  Channel
adios4dolfinx                  0.8.1    pyhd8ed1ab_0  conda-forge
...
dolfinx-adjoint                0.2.1    pyhc364b38_0  conda-forge
dolfinx-adjoint                0.3.0    pyhc364b38_0  conda-forge
```

```
conda search -c conda-forge --platform win-64 dolfinx               -> exit=0
No match found for: dolfinx. Search: *dolfinx*    (only adios4dolfinx, dolfinx-adjoint listed)
```

```
conda search -c conda-forge --platform win-64 dolfinx               -> exit=0
```

For comparison, `fenics-basix` *is* published for `win-64` including `py313` builds:

```
conda search -c conda-forge --platform win-64 fenics                -> exit=0
No match found for: fenics. Search: *fenics*
# Name                       Version           Build  Channel
fenics-basix                   0.9.0 py310h06fdc5a_4  conda-forge
...
fenics-basix                   0.9.0 py312hfdb765b_3  conda-forge
fenics-basix                   0.9.0 py313h1097ce1_4  conda-forge
fenics-basix                   0.9.0 py313h2f5a85f_3  conda-forge
fenics-basix                   0.9.0 py313h5c0e27c_3  conda-forge
```

i.e. the ecosystem does track Python 3.13 for the pure element kernel but not for the solver.

#### 4.2.6 Conclusion — FEniCS

Within the environment in which the present results were produced — **Python 3.13 with pip** —
FEniCS cannot be installed: neither the legacy solver (`dolfin`), nor the modern solver
(`fenics-dolfinx`), nor the legacy meta-package (which resolves only to the form compilers) is
available. The `win-64` conda-forge builds of `fenics-dolfinx` stop at Python 3.12, so using
FEniCS would require building a **separate conda environment pinned to Python ≤ 3.12**, i.e.
maintaining a second interpreter alongside the 3.13 stack on which every other result in this
work depends. The blocker is therefore the **interpreter version**, not the operating system —
a soft dead end. It is stated as such deliberately: it would be inaccurate to claim that
"FEniCS does not run on Windows".

---

### 4.3 PRISMS (PRISMS-PF) — hard dead end on Windows

```
pip index versions prisms                                   -> exit=1
  ERROR: No matching distribution found for prisms
pip install --dry-run prisms                                -> exit=1
  ERROR: Could not find a version that satisfies the requirement prisms (from versions: none)
pip install --dry-run --only-binary=:all: --no-deps prisms   -> exit=1
  ERROR: No matching distribution found for prisms

pip index versions prisms-pf                                -> exit=1
  ERROR: No matching distribution found for prisms-pf
pip install --dry-run prisms-pf                             -> exit=1
  ERROR: No matching distribution found for prisms-pf
pip install --dry-run --only-binary=:all: --no-deps prisms-pf -> exit=1
  ERROR: No matching distribution found for prisms-pf

pip index versions pyprisms                                 -> exit=1
  ERROR: No matching distribution found for pyprisms
pip install --dry-run pyprisms                              -> exit=1
  ERROR: Could not find a version that satisfies the requirement pyprisms (from versions: none)
pip install --dry-run --only-binary=:all: --no-deps pyprisms -> exit=1
  ERROR: No matching distribution found for pyprisms

conda search -c conda-forge --platform win-64 prisms        -> exit=0
No match found for: prisms. Search: *prisms*
# Name                       Version           Build  Channel
prisms-jobs                    4.0.2    pyhd8ed1ab_0  conda-forge
prisms-jobs                    4.0.3    pyh5ded981_3  conda-forge
prisms-jobs                    4.0.3    pyhd8ed1ab_0  conda-forge
prisms-jobs                    4.0.3    pyhd8ed1ab_1  conda-forge
prisms-jobs                    4.0.3    pyhd8ed1ab_2  conda-forge
```

No `prisms`, `prisms-pf` or `pyprisms` distribution exists on PyPI for any platform, and
conda-forge publishes no `prisms` package for `win-64`. The single conda-forge hit
(`prisms-jobs`) is an unrelated package matching the query only as a substring.

#### Conclusion — PRISMS

PRISMS-PF is distributed as a **source build / container image targeting Linux**. Nothing is
available for Windows — no wheel, no sdist, no conda-forge `win-64` package. Because the
blocker is the **platform**, this is a hard dead end.

---

### 4.4 DAMASK — installed, but it is an FFT reference only, with no FEM solver

DAMASK is not requested by the referee but is needed to close the argument, because it is the
framework that *is* present. It is treated as a reference, not as a candidate host.

```
pip index versions damask        -> exit=0
damask (3.1.0);  INSTALLED: 3.1.0;  LATEST: 3.1.0
pip install --dry-run damask     -> exit=0
  Requirement already satisfied: damask in ...site-packages (3.1.0)
  Requirement already satisfied: pandas>=1.3, numpy>=1.21, scipy>=1.8, h5py>=3.6,
              vtk>=9.1, matplotlib>=3.5, pyyaml>=5.4
```

`damask` 3.1.0 is installed and is the Python pre/post-processing and grid-solver interface.
`damask-core` does not exist on PyPI (`No matching distribution found`); `damask-parse` 0.6.2
resolves. On conda-forge only the *Python* package is present, and only for older interpreters:

```
conda search -c conda-forge --platform win-64 damask     -> exit=0
No match found for: damask. Search: *damask*
# Name                       Version           Build  Channel
python-damask                3.0.0a4 py37h03978a9_16  conda-forge
python-damask                3.0.0a4 py38haa244fe_16  conda-forge
python-damask                3.0.0a4 py39hcbf5309_16  conda-forge
python-damask                3.0.0a5 py310h5588dad_2  conda-forge
python-damask                3.0.0a5 py310h5588dad_3  conda-forge
python-damask                3.0.0a5 py310h5588dad_4  conda-forge
python-damask                3.0.0a5  py37h03978a9_0  conda-forge
  ... (py37/py38/py39 builds _0 through _4)
```

The interpreter tags present are exactly `py37`, `py38`, `py39` (build `3.0.0a4`) and `py37`,
`py38`, `py39`, `py310` (build `3.0.0a5`) — i.e. this conda package stops at Python 3.10 and is
five minor versions behind the interpreter used here.

Two established facts, already recorded in the manuscript (Sec. *Taylor homogenization bias
quantified by full-field FFT*), limit what DAMASK can contribute:

1. **No FEM solver.** The DAMASK 3.1.0 distribution available here provides only the spectral
   (FFT) solver `DAMASK_grid` and mesh utilities — not a finite-element solver. An
   explicit-FEM coupled validation therefore cannot be produced with it.
2. **The coupled-damage FFT computation diverges.** Augmenting the DAMASK phase with the
   built-in isobrittle phase-field damage on a $16^3$ periodic RVE with the identical 30-grain
   microstructure was attempted across $G_{\rm crit}=1.2\times10^6$–$3\times10^8$ Pa,
   $l_c=1$–$2$, $\mu=0.001$–$0.05$, cutback tolerance up to 20, and both `spectral_basic` and
   `spectral_polarization`. In every configuration the FFT iteration diverges at the
   damage-activation point ($\varepsilon\approx0.10$ for 316L; **DAMASK error 950**, cutback
   exhaustion).

**Scope statement (required by the paper's own protocol):** no damage model is implemented or
activated inside DAMASK in any simulation of this work. In the full-field FFT comparison the
damage module is *deactivated* there ($S_0\to\infty$), so that the comparison isolates the
homogenization assumption in the damage-free regime. DAMASK enters this work exclusively as an
external damage-free full-field reference.

#### Conclusion — DAMASK

Installed and usable as a spectral-FFT reference; **not** a host for the damage model, and not
an FEM solver. It contributes no coupled-damage validation.

---

## 5. Summary

| Framework | Probed names | PyPI | conda-forge `win-64` | Verdict in this environment | Kind of blocker |
|---|---|---|---|---|---|
| INL MOOSE (FE multiphysics) | `pymoose`, `moose` | `pymoose` resolves but is the *neuroscience* MOOSE; `moose` is an unrelated package with broken metadata | not found | **not installable** | platform (no Windows distribution exists) |
| FEniCSx solver | `fenics-dolfinx`, `dolfinx` | no matching distribution | **0.9.0 exists, Python 3.9–3.12 only** | **not installable on Python 3.13** | interpreter version (needs a py≤3.12 env) |
| FEniCS legacy solver | `dolfin` | no matching distribution | — | **not installable** | platform + interpreter |
| FEniCS legacy meta-package | `fenics` | resolves; form compilers **only** (FFC/FIAT/UFL/Dijitso) — no `dolfin` | — | **not usable** (no assembly/solver layer) | packaging (solver absent) |
| FEniCSx element kernel | `fenics-basix` | resolves (cp312-abi3 wheel) | 0.9.0 incl. `py313` | installable but **not a framework** | — (building block only) |
| PRISMS-PF | `prisms`, `prisms-pf`, `pyprisms` | no matching distribution | not found (only unrelated `prisms-jobs`) | **not installable** | platform (Linux source/container build) |
| DAMASK | `damask`, `damask-core`, `damask-parse` | `damask` 3.1.0 **already installed** | `python-damask` ≤ py310 | installable; **FFT solver only, no FEM; coupled damage diverges (error 950)** | capability, not availability |

Reading the table: **two** of the three frameworks named by the referee (MOOSE, PRISMS) are
blocked by the platform at every Python version; **one** (FEniCS) is blocked only by the
interpreter version and would be reachable through a second, Python ≤ 3.12 conda environment.
The distinction is stated because it is the difference between "this cannot be done here" and
"this was not done here", and only the second one would be a fair criticism of the paper.

## 6. Scope and limitations of this evidence

Stated explicitly, so the evidence is neither over- nor under-claimed.

1. **Read-only by construction.** All pip commands are `--dry-run` / `index versions`; the
   single conda command is `search` with an explicit `--platform`. `pip install --dry-run`
   downloads and unpacks distributions to a temporary directory to read their metadata but
   writes nothing into `site-packages`; no environment was created, modified or removed. No
   framework was installed at any point.
2. **A "no matching distribution" result is definitive for the configured index**, and the
   `--only-binary=:all:` probe isolates *wheel availability* from *buildability*. A
   `--dry-run` success proves that the metadata resolves and that a wheel exists; it does not
   prove that the resulting installation would import and run correctly. No such claim is made
   here.
3. **Index provenance.** pip is configured to a mirror
   (`mirrors.huaweicloud.com/repository/pypi/simple/`). A mirror serves the upstream PyPI
   content, so a distribution absent from it is absent from PyPI; the mirror URL is recorded so
   that the commands can be re-run verbatim and produce the same results.
4. **conda coverage.** The conda queries cover `conda-forge` and the default Anaconda channels
   for `win-64` only. They do **not** cover the INL-hosted conda channel
   (`conda.software.inl.gov`) or Spack. They do establish that conda-forge offers no `win-64`
   build of MOOSE or PRISMS.
5. **What this does not show.** It does not show that these frameworks are unsuitable in
   principle — they are not; they are the standard tools for this class of problem. It does not
   show that the present model could not be ported to them. It does not show that a Windows
   build of MOOSE or PRISMS is impossible, only that none exists for this platform.
6. **Why the port is a separate project rather than a substitution.** MOOSE, FEniCS and
   PRISMS-PF supply mesh handling, assembly, linear algebra, parallelism and time integration.
   None of them supplies the constitutive model of this work: the Taylor-type aggregate
   homogenization with per-grain crystal plasticity, the AM-corrected Lemaitre damage driving
   force with its five corrective descriptors and their projection laws, the plastic-strain
   activation gate, and the aggregate fracture criterion. Adopting a framework would change the
   *discretisation*, not the *physics*; every constitutive line, and with it the entire
   verification burden (unit tests, convergence studies, analytic limits), would still have to
   be written and validated. That is the sense in which the framework question and the code
   verification question are independent — and it is why the verification work reported in
   `code_verification_summary.md` is carried out on the in-house solver rather than deferred
   behind a framework migration.

## 7. Consequence for this work

Within the environment in which all reported results were produced, no framework capable of
hosting a 3D CPFEM + phase-field/non-local-damage formulation is installable (MOOSE, PRISMS:
blocked by platform; FEniCS: blocked by interpreter version). The damage-coupled full-field
reference therefore cannot be produced with an off-the-shelf framework here. Accordingly:

* the 2D plane-strain in-house full-field CPFEM (`fullfield_3d_cpfem.py`) supplies the
  full-field coupled-damage reference, and the 3D explicit-FEM coupled validation is identified
  in the manuscript as required follow-on work rather than silently assumed;
* DAMASK supplies the external **damage-free** spectral-FFT full-field comparison only;
* code verification, unit testing, convergence and analytic benchmark limits are carried out on
  the in-house solver, and are reported in
  `simulation/output/code_verification_summary.md` with the machine-readable record in
  `simulation/output/code_verification_results.json`.

## 8. Reproduction

```powershell
# from the project root
& "C:\Users\ZJ03\miniconda3\python.exe" -m pip index versions pymoose
& "C:\Users\ZJ03\miniconda3\python.exe" -m pip install --dry-run --only-binary=:all: --no-deps fenics-dolfinx
& "C:\Users\ZJ03\miniconda3\python.exe" -m pip install --dry-run fenics
& "C:\Users\ZJ03\miniconda3\python.exe" -m pip install --dry-run --only-binary=:all: --no-deps prisms-pf
conda search -c conda-forge --platform win-64 fenics-dolfinx
conda search -c conda-forge --platform win-64 moose
```

Full verbatim output of all 12 distributions × 3 pip probes plus 6 conda queries:
`simulation/output/framework_deadends_raw.txt` (1051 lines, 81 339 bytes, generated
2026-09-28, terminator `DONE`).
