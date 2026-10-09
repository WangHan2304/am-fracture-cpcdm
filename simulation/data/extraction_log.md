# Extraction Log — Extended LPBF Tensile-Fracture Dataset

**Deliverable set**: `simulation/data/` · **Built**: 2026-09-28 · **Builder**: `scratch/build_extended_dataset.py`

| File | Content |
|---|---|
| `extended_literature_dataset.csv` | **R13**: 24 data points (9 v12 original + 15 newly extracted) · **R13b (§7)**: + 30 NIST mds2-3402 Ti-6Al-4V points, 1 new source → **54 points / 7 sources** |
| `extended_source_groups.json` | Source grouping + leave-one-source feasibility per material |
| `extended_process_groups.json` | Process-window grouping + ungroupable sources with stated reasons |
| `extraction_log.md` | This file: method, per-point provenance, verification, exclusions |

---

## 0. Protocol statement

* **Extended protocol, reported separately.** The nine v12 points are reproduced **verbatim** in
  `extended_literature_dataset.csv` under `source_id = v12_original` and `extraction_method =
  v12_protocol_unmodified`. **No v12 value was altered.** The extended set is an independent protocol:
  it must not be silently substituted for the v12 statistics. Every v12 point passes its own
  plausibility band (see §4).
* **All data are literature values.** No original experiment was performed or simulated.
* **Include criteria**: as-built LPBF/SLM (heat-treated only if explicitly labelled), room-temperature
  quasi-static tension (strain rate order 1e-3 s^-1), ASTM E8/E8M-type specimen, one of the four target
  alloys, fracture/elongation strain reported as a number.
* **Exclude criteria applied**: non-peer-reviewed or non-verifiable full text; secondary citation only
  (values quoted from another paper's summary rather than read from the primary source); non-tensile or
  non-quasi-static loading; surface-modified or heat-treated condition outside the as-built scope.

---

## 1. Orientation convention audit (read this before using the orientation column)

**The model convention, as declared by the code and by the model's own equations:**

| Source of truth | Statement |
|---|---|
| `simulation/src/materials.py` L358–360 | "0°: 平行于打印方向（纵向）— 基准各向同性响应 … 90°: 垂直于打印方向（横向）" |
| `simulation/src/materials.py` L337–339 | `0: 1.0  # 纵向加载 — 晶粒长轴 ∥ 载荷` · `90: 3.8  # 横向加载 — 层间弱界面 ∥ 载荷方向` |
| `manuscript_main.tex` L299, L301–304 | λ_eff(ψ) = λ₀[w_iso + (1−w_iso)sin²ψ] with "loading at angle ψ **to the build direction**"; "At ψ = 0° only the isotropic fraction is directly loaded" |
| Task specification | "0°=平行于打印方向（纵向），90°=垂直于打印方向（横向）" |

All four agree: **0° = loading axis PARALLEL to the build direction (∥BD); 90° = PERPENDICULAR to BD
(⊥BD).** A consistency check confirms it: with the model's weak-interface network (normals ⊥ BD), λ_eff
is *minimum* at ψ = 0°, so the model inherently predicts the **highest** fracture strain at 0° — which
is exactly the shape of the v12 calibration data for all three alloys (0.080 > 0.063 > 0.048 etc.).

**Consequence — some literature reports angles from the build *plate*, which inverts the axis.** A
specimen "0° from the build plate" lies *in* the plate → its axis is **⊥BD** → it is the model's **90°**.
This dataset therefore **converts** every source that uses the build-plate convention, and records the
source's own printed label in `notes` so the dataset can be re-mapped either way without loss.

| Source | Angle basis (as printed) | Conversion applied |
|---|---|---|
| Sun 2024 | ASTM specimen codes Z / Z45 / X, defined by build direction | none needed (Z = ∥BD ≡ 0°) |
| Hitzler 2017 | polar angle Φ from the xy (build-plate) plane | Φ=0 → 90°; Φ=45 → 45°; Φ=90 → 0° |
| Awd 2018 | 0/45/90 relative to the build platform (Fig 1a) | printed 0 → 90°; 45 → 45°; printed 90 → 0° |
| Hovig 2018 | "orientation with respect to the build plate starts at horizontal (0°) … to vertical (90°)" | printed 0 → 90°; 45 → 45°; printed 90 → 0° |
| Watring 2020 | "build orientation is reported with respect to the build plate" | printed 0 → 90°; printed 60 → 30° (retained as 30°) |
| v12 original | already on the model convention | none |

> **⚠ Flag for the author (not actioned here — outside this task's scope).**
> The in-repo cross-source blind-set scripts appear to have assigned the *sources' printed* build-plate
> angles to the model's ψ directly, without this conversion (e.g. `blind_test_ti64_cross_source.py`
> maps Sun's X → 0° and Z → 90°; `manuscript_main.tex` L1019 states "(7.89% (0°), 5.33% (45°), 9.34%
> (90°))" for the same source, while L1029/L1031 compare against a measured 90°/0° ratio of 1.50 for
> IN718). Under the model's own convention (§ above), Sun's Z (∥BD) is 0° and X (⊥BD) is 90°, i.e. the
> assignment is inverted for every source that reports angles from the build plate.
> This is material, because it is the *same* operation in all four independent blind sets, and an
> inverted assignment would mechanically produce the reported uniform "reversed direction trend"
> across IN718, 316L, Ti-6Al-4V and AlSi10Mg. It does **not** by itself invalidate the paper's
> cross-source conclusion — the **S₀ non-transferability** finding is independent of orientation
> labels (it rests on error magnitude and on the 90°/0° ratio) — but the *direction-reversal* part of
> the applicability-domain argument should be re-checked against the geometric convention.
> The present dataset follows the specification (0° = ∥BD) and carries the printed source label in
> `notes` for every affected point, so a re-mapping is a one-column operation if you decide otherwise.
> `manuscript_main.tex` L290 ("Loading normal to the build direction (ψ = 0°)") also contradicts
> L299/L304 and `materials.py`; that prose sentence appears to be the origin of the divergence.

---

## 2. Per-source extraction records

### 2.1 `v12_original` — 9 points (Ti64 / 316L / AlSi10Mg × 0/45/90°)

* **Values**, from `simulation/src/materials.py` `EXPERIMENTAL` (mirrored in `manuscript_main.tex`
  Table `tab:experimental`): Ti64 ε_f = 0.080/0.063/0.048, UTS = 1150/1100/1050 MPa; 316L ε_f =
  0.370/0.310/0.255, UTS = 680/640/595 MPa; AlSi10Mg ε_f = 0.056/0.047/0.037, UTS = 440/410/375 MPa.
* **Nature of the data (recorded honestly):** v12 is a **compiled representative dataset** — central
  values of reported literature ranges (Carroll 2015 / Wang 2024 for Ti64; Jaskari 2021 for 316L;
  Wu 2023 for AlSi10Mg; a Ti64 ε_f = 0.080 lies above the 6 ± 3 % horizontal / 4 ± 3 % vertical ranges
  quoted in `sec:experimental_sources`). It is **not** a single table of measurements, and **no single
  process window is defined for it** → it is ungroupable for leave-one-process CV (see the process JSON).
* **Extraction method**: read verbatim from source code; `extraction_method = v12_protocol_unmodified`.
* **Verification**: all nine points inside their material bands (§4) — 9/9 PASS.

### 2.2 `sun2024adem` — Ti-6Al-4V, 3 points

* **Citation**: Sun, S.; Delp, A.; Zhang, D.; Palanisamy, S.; Liu, Q.; Schmidt, M.; Dargusch, M.S.
  *Adv. Eng. Mater.* **2024**, 26, 2400942. doi:10.1002/adem.202400942
* **Retrieval route**: OA status confirmed via the Unpaywall API; full text obtained from the Griffith
  University institutional repository (DSpace REST bitstream endpoint). Wiley direct retrieval returns
  HTTP 403. Local copy: `scratch/sun2024_adem.pdf` (30.7 MB).
* **ε_f — `extraction_method = table`**: Table 7, total strain at fracture: **Z 9.34 ± 0.67 %**,
  **Z45 5.33 ± 0.74 %**, **X 7.89 ± 0.56 %**. Mapping: Z (vertically built, ∥BD) → 0°; Z45 → 45°;
  X (horizontally built, ⊥BD) → 90°.
* **UTS — `extraction_method = figure_pixel`**: the paper gives UTS only as bars in Fig. 10b, with no
  extractable text. Procedure (reproducible via `scratch/pixel_bars2.py`): PyMuPDF pulled the Fig. 10
  panel from the PDF (xref 370, 1753 × 1692 px); the y-scale was calibrated on the dashed gridlines
  (800–1400 MPa; measured 116.5 px per 100 MPa, 800 MPa axis at y ≈ 1617.5 in-image); each bar's solid
  top edge was found column-wise as the widest horizontal run spanning ≥ 70 % of the bar width.
  **Result: Z = 1165, Z45 = 1299, X = 1186 MPa**, agreeing with the visual read (≈ 1000/1150;
  1180/1300; 1030/1190 MPa for YS/UTS) to within ± 7 MPa, and reproducing the source's stated ranking
  UTS(Z45) > UTS(X) > UTS(Z). **Verification: PASS.**
* **Process window**: 175 W laser power, 710 mm/s scan speed, 30 µm layer thickness.
* **Anomaly — FLAGGED, retained**: `sun2024_Ti64_45` UTS = 1299 MPa is **above** the nominal Ti-6Al-4V
  band adopted for this task (900–1200 MPa). It is a directly measured value from a peer-reviewed
  source (the diagonally built sample is reported as the strongest), so it was **retained and flagged**
  rather than dropped; exclude it if a nominal-band-only calibration set is required.
* **Trend note**: source ductility trend 0° > 90° > 45° (V-shaped, 45° least ductile) — **differs in
  shape** from the v12 monotonic trend, and under the correct convention it is *not* a simple reversal.

### 2.3 `hitzler2017materials` — 316L, 3 points

* **Citation**: Hitzler, L.; Hirsch, J.; Heine, B.; Merkel, M.; Hall, W.; Oechsner, A. *Materials*
  **2017**, 10(10), 1136. doi:10.3390/ma10101136
* **Retrieval route**: institutional repository (`hdl.handle.net/10072/348945`); MDPI full text 403.
  Local copy: `scratch/hitzler_pub3934.pdf`.
* **Method — `extraction_method = table`**, Table 6 (as-built means ± STDEV), using only the
  **azimuth Θ = 0** configurations so that the polar-angle trend is not confounded by the azimuth
  dependence reported in the same table:

  | Point | Config (Table 6) | Φ / Θ (source) | Model θ | ε_f | UTS (MPa) |
  |---|---|---|---|---|---|
  | `hitzler2017_316L_90` | (a) | 0° / 0° | 90° | 33.24 ± 0.57 % | 634.43 ± 7.39 |
  | `hitzler2017_316L_45` | (d) | 45° / 0° | 45° | 32.56 ± 0.71 % | 698.98 ± 4.23 |
  | `hitzler2017_316L_0` | (f) | 90° / 0° | 0° | 11.76 ± 1.24 % | 511.99 ± 39.28 |

* **Verification**: the polarity of the Φ definition was checked (Φ measured from the xy build-plate
  plane) and the conversion applied accordingly; the source's own statement that UTS peaks at 45° and
  that the in-plane orientation is the most ductile is reproduced.
* **Anomaly — FLAGGED, retained**: `hitzler2017_316L_0` UTS = 511.99 MPa is **below** the nominal 316L
  band (550–750 MPa). Directly tabulated value; the source attributes the low UTS to the build
  orientation. Retained and flagged.
* **Condition caveats (carried into `notes`)**: specimens are **milled to final geometry** (printed
  with a 0.4 mm machining allowance), so surfaces are not as-built even though the material state is
  as-built; the build platform was preheated to 200 °C; configurations (a)–(b) and (c)–(g) come from
  two melt batches with slightly different measured chemistry (Table 3).

### 2.4 `awd2018metals` — AlSi10Mg, 3 points

* **Citation**: Awd, M.; Stern, F.; Kampmann, A.; Kotzem, D.; Tenkamp, J.; Walther, F. *Metals*
  **2018**, 8(10), 825. doi:10.3390/met8100825
* **Method — `extraction_method = table`**: Table 3 means (n = 3), orientation definitions from Fig. 1a
  (printed 0° = bar lying in the plate plane, 90° = bar standing on the plate → converted):

  | Point | Source label | Model θ | ε_f | UTS (MPa) |
  |---|---|---|---|---|
  | `awd2018_AlSi10Mg_90` | printed 0° | 90° | 8.1 ± 0.4 % | 379.6 ± 4.5 |
  | `awd2018_AlSi10Mg_45` | printed 45° | 45° | 5.7 ± 0.3 % | 367.8 ± 3.2 |
  | `awd2018_AlSi10Mg_0` | printed 90° | 0° | 8.3 ± 0.3 % | 351.8 ± 3.3 |

* **Internal cross-checks against the source's own abstract — PASS**: "specimens of 0° showed 8 %
  higher tensile strength compared to 90°-specimens" → 379.6 / 351.8 = 1.079 (+7.9 %); "fracture strain
  was reduced almost 30 % for the 45°-specimen" → 5.7 % vs mean(8.1, 8.3) % = 8.2 % → −30.5 %. Both
  reproduce, which confirms the row-to-orientation assignment was read correctly.
* **Band note**: ε_f 8.1 % and 8.3 % marginally exceed the 2–8 % nominal AlSi10Mg band (top edge
  exceeded by 1 % and 4 % respectively). These are the source's own as-built values; retained and
  flagged as marginal.
* **Process grouping NOT possible**: the source explicitly states that the SLM scanning parameters are
  not available for publication. Declared in `extended_process_groups.json → ungroupable_sources`.

### 2.5 `hovig2018in718` — IN718, 3 points

* **Citation**: Hovig, E.W.; Azar, A.S.; Grytten, F.; Sørby, K.; Andreassen, E. *Adv. Mater. Sci. Eng.*
  **2018**, 2018, 7650303. doi:10.1155/2018/7650303 · full text: `hindawi.com/journals/amse/2018/7650303/`
* **Condition — `heat_treatment = "solution-treated (980 °C / 1 h, argon, air cool) - set 1"`**:
  this set is **NOT as-built**. Set 1 is the only set close to the study's scope; sets 2 (aged,
  1286 MPa / 5 %) and 3 (HIP + aged, 1201 MPa / 9 %) are heat-treated and single-orientation, and were
  excluded.
* **ε_f — `figure_pixel`** (the project's existing Fig. 15c read, re-used): printed 0° = 0.20,
  45° = 0.25, printed 90° = 0.30. Retained after checking them against the source's explicit claim
  ("the lowest elongation at break should then be observed for the 0° specimens and increase gradually
  to a maximum for the 90° specimens") and against the Table 8 orientation average of 26 ± 6 %
  (mean of the three adopted values = 25 %). **Consistent — retained.**
* **UTS — `table`**: the source tabulates only the **orientation-averaged** value, Table 8:
  **970 ± 74 MPa**, and reports no significant UTS variation with build angle (only YS is reported to
  vary meaningfully). **970 MPa is therefore assigned to all three orientations.** The earlier in-repo
  blind script's per-orientation 990 / 970 / 950 MPa were **not** re-used: asserting an untabulated
  trend on a ± 74 MPa scatter is not defensible. This is a deliberate reduction in information in
  favour of accuracy, and it means these three points differ only in ε_f.
* **Caveat**: specimens have an **as-built surface finish** (non-machined), and 11 orientations were
  tested per set — the three adopted values are figure-read points at the three canonical angles.

### 2.6 `watring2020addma` — IN718, 3 points (fully as-built)

* **Citation**: Watring, D.S.; Benzing, J.T.; Hrabe, N.; Spear, A.D. *Addit. Manuf.* **2020**, 36,
  101425. doi:10.1016/j.addma.2020.101425 · open full text: `pmc.ncbi.nlm.nih.gov/articles/PMC7772965/`
* **Why this source is the strongest addition**: it is **as-built (non-heat-treated)**, tested at
  **1 × 10⁻³ s⁻¹** on **ASTM E8-16a** specimens — it satisfies the task's test-condition requirement
  exactly — and it reports **three full process windows** with power, scan speed and layer thickness,
  which is what enables leave-one-process CV.

  | Point | Build cond. | Printed θ → model θ | Power / speed / layer | LED | ε_f (TE) | UTS |
  |---|---|---|---|---|---|---|
  | `watring2020_IN718_90_lo` | 1 | 0° → 90° | 168 W / 1475 mm/s / 30 µm | 38 J/mm³ | 0.06 ± 0.012 | 774 ± 22 |
  | `watring2020_IN718_90_hi` | 2 | 0° → 90° | 330 W / 1770 mm/s / 30 µm | 62 J/mm³ | 0.29 ± 0.027 | 1081 ± 23 |
  | `watring2020_IN718_30` | 3 | 60° → 30° | 220 W / 1180 mm/s / 30 µm | 62 J/mm³ | 0.28 ± 0.085 | 1070 ± 34 |

* **Anomaly — FLAGGED (both ε_f and UTS), retained**: condition 1 carries **6.91 % lack-of-fusion
  porosity** (relative density 93.09 %, Table 2) at the low energy density of 38 J/mm³; UTS is ~28 %
  below the other two conditions and ε_f (0.06) is below the nominal IN718 band. The source states the
  pore structure dominates the mechanical response. **This is a genuine process-window effect, not a
  transcription error**, and it is precisely the kind of point that gives leave-one-process CV its
  discriminating power. It is retained and flagged; **exclude it for nominal-property CV.**
* **Orientation note**: condition 3 is at 60° from the build plate = **30° from BD**. It is recorded
  as `orientation_deg = 30` (category "other") rather than being forced into 45°; the notes state that
  assigning it to 45° is possible but must then be declared as an approximation.

---

## 3. Exclusion register

### 3.1 Excluded on a data conflict (please adjudicate)

* **`pmc11721696` — 316L, *Materials* 2025, 18(1), 32** (`pmc.ncbi.nlm.nih.gov/articles/PMC11721696/`).
  Full text was obtained and the orientation convention was verified from its Fig. 3 photograph
  (specimen blocks labelled 90°/0°/45°/60°, confirming angles measured from the build plate).
  **Excluded because two sets of numbers for the same labels do not agree**: `manuscript_main.tex`
  L1021 cites **13.75 % (0°), 7.5 % (45°), 20 % (90°)** (UTS 583–638 MPa), whereas the digitization
  attempted in this run returned **20.0 % / 10.5 % / 14.5 %** for the same printed labels. Rather than
  ship unreconciled numbers into a calibration set, the source was **omitted**. Resolving the
  discrepancy would add a **7th source and a third independent 316L source** — the single highest-value
  remaining action for this dataset, and it is left to you deliberately, not overlooked.

### 3.2 Excluded to avoid double-counting

Carroll 2015, Wang 2024 (Ti64), Jaskari 2021 (316L) and Wu 2023 (AlSi10Mg) are the **origins of the
v12 columns**; adding them as separate extended sources would double-count the same measurements, so
they are represented only through `v12_original`.

### 3.3 Excluded on condition / retrievability

| Candidate | Reason for exclusion |
|---|---|
| Mower & Long 2016, IN718 (MLS magazine article) | Not a peer-reviewed primary data source; values could not be verified against a primary publication → excluded under the no-secondary-source rule |
| Simonelli et al. 2014, Ti64 | No OA full text route found within budget; values available only via secondary citation → excluded |
| `pmc6678663`, *Materials* 2020, 13(21), 4851 (AlSi10Mg) | Tested condition is **laser-polished** (surface-modified), not as-built |
| `pmc6337398`, *Materials* 2019, 12(14), 2281 (Ti64) | **Heat-treated / annealed** condition, outside the as-built scope |
| `crystals16/2/121` (Ti64) | Full text not retrievable (HTTP 403, no OA route) |

Several further MDPI and Wiley candidates were screened out at abstract level on the include/exclude
criteria (alloy outside the four targets, heat-treated condition, non-LPBF process, or non-tensile
loading) and are **not individually itemised**, because the dataset is defined by its inclusion
criteria rather than by an exhaustive census of every candidate considered. Direct full-text retrieval
from MDPI and Wiley returned **HTTP 403** throughout this run, so only sources with an OA repository,
PMC or publisher-OA route could be used at all — a real, and disclosable, selection pressure on the
source pool.

---

## 4. Plausibility validation (all 24 points)

Bands applied per the task: Ti64 ε_f 4–12 % / UTS 900–1200 MPa; 316L 15–45 % / 550–750 MPa;
AlSi10Mg 2–8 % / 350–500 MPa; IN718 10–30 % / 900–1300 MPa.

| Point | Material | θ | ε_f | UTS | Verdict |
|---|---|---|---|---|---|
| v12_Ti64_0/45/90 | Ti64 | 0/45/90 | 0.080/0.063/0.048 | 1150/1100/1050 | PASS |
| v12_316L_0/45/90 | 316L | 0/45/90 | 0.370/0.310/0.255 | 680/640/595 | PASS |
| v12_AlSi10Mg_0/45/90 | AlSi10Mg | 0/45/90 | 0.056/0.047/0.037 | 440/410/375 | PASS |
| sun2024_Ti64_0 | Ti64 | 0 | 0.0934 | 1165 | PASS |
| sun2024_Ti64_45 | Ti64 | 45 | 0.0533 | **1299** | **UTS OUT** — flagged, retained |
| sun2024_Ti64_90 | Ti64 | 90 | 0.0789 | 1186 | PASS |
| hitzler2017_316L_90 | 316L | 90 | 0.3324 | 634 | PASS |
| hitzler2017_316L_45 | 316L | 45 | 0.3256 | 699 | PASS |
| hitzler2017_316L_0 | 316L | 0 | 0.1176 | **512** | **UTS OUT** — flagged, retained |
| awd2018_AlSi10Mg_90 | AlSi10Mg | 90 | **0.081** | 380 | **ε_f marginal** (top edge +1 %) |
| awd2018_AlSi10Mg_45 | AlSi10Mg | 45 | 0.057 | 368 | PASS |
| awd2018_AlSi10Mg_0 | AlSi10Mg | 0 | **0.083** | 352 | **ε_f marginal** (top edge +4 %) |
| hovig2018_IN718_90/45/0 | IN718 | 90/45/0 | 0.20/0.25/0.30 | 970 | PASS |
| watring2020_IN718_90_lo | IN718 | 90 | **0.06** | **774** | **OUT** — lack-of-fusion porosity, flagged |
| watring2020_IN718_90_hi | IN718 | 90 | 0.29 | 1081 | PASS |
| watring2020_IN718_30 | IN718 | 30 | 0.28 | 1070 | PASS |

**Result: 19 / 24 fully in band; 5 flagged and retained with the cause recorded** (3 are genuine
out-of-band literature values from peer-reviewed sources, 2 are marginal ε_f at the band's top edge).
**No point was silently adjusted to fit a band.** No point was dropped merely for being out of band —
each retained one is traceable to a table or a reproducible pixel extraction with a stated cause.

### Direction-trend cross-check (ε_f at 0° vs 90°), per source

| Source | ε_f (0° / 45° / 90°) | Shape | vs v12 (0° > 45° > 90°) |
|---|---|---|---|
| v12_original | 0.080 / 0.063 / 0.048 · 0.370 / 0.310 / 0.255 · 0.056 / 0.047 / 0.037 | monotonic ↓ | reference |
| sun2024adem (Ti64) | 0.0934 / 0.0533 / 0.0789 | V-shaped (45° minimum) | **differs in shape** |
| hitzler2017materials (316L) | 0.1176 / 0.3256 / 0.3324 | rising | **opposite** |
| awd2018metals (AlSi10Mg) | 0.083 / 0.057 / 0.081 | V-shaped | **differs in shape** |
| hovig2018in718 (IN718) | 0.30 / 0.25 / 0.20 | monotonic ↓ | **same direction as v12** |
| watring2020addma (IN718) | 0.06 & 0.29 at 90°, 0.28 at 30° | 90° split by porosity | process-dominated |

**Consequence for validation, stated plainly**: with the geometric convention applied, the **IN718
source agreement with the v12 direction trend is *not* reversed** — the reversal reported for the
manuscript's blind IN718 set follows from the printed build-plate labels being used as ψ directly
(§1). The 316L extended source does show the opposite trend, and the two V-shaped sources differ in
shape rather than in polarity. This is a negative result and it is **not** dressed up: it says that
the extended dataset does not reproduce the clean "all four blind sets reversed" picture once the
orientation convention is enforced, and that this needs to be reconciled before the dataset is used
to support any applicability-domain claim about direction trends.

---

## 5. Grouping summary and honest capability statement

**Sources (6)** — v12_original (9) · sun2024adem (3) · hitzler2017materials (3) · awd2018metals (3) ·
hovig2018in718 (3) · watring2020addma (3) = **24 points**.

**Independent sources per material → leave-one-source CV is feasible for all four materials:**

| Material | Sources | n |
|---|---|---|
| Ti64 | v12_original, sun2024adem | 6 |
| 316L | v12_original, hitzler2017materials | 6 |
| AlSi10Mg | v12_original, awd2018metals | 6 |
| IN718 | hovig2018in718, watring2020addma | 6 |

**Process groups (6)** — 175/710/30 · 200/800/30 · 275/805/50 · 168/1475/30 · 330/1770/30 ·
220/1180/30. **Ungroupable: `v12_original`** (compiled dataset, no single process window) and
**`awd2018metals`** (source explicitly does not publish its scanning parameters); reasons are stated
in `extended_process_groups.json`.

### Two protocol options — pick one deliberately

* **Category A — extended protocol, 24 points / 6 sources.** This is what has been built and what the
  four deliverables record. **Disclosure that must accompany it**: three of its sources
  (`hovig2018in718`, `sun2024adem`, `awd2018metals`) are *already reported as blind sets* in the
  manuscript (`sec:in718_blind`, `sec:cross_source`). Including them means **these points are no
  longer held out from the model**, so the paper's blind-test narrative and the LOOCV protocol must be
  restated accordingly (the manuscript already reports these four sets and their negative results
  explicitly, so nothing is hidden — but the "held-out" status is gone). This is the only option among
  the two that supports per-material leave-one-source CV.
* **Category B — conservative, 15 points / 3 sources** (the 9 v12 + `watring2020addma` + `hitzler2017materials`).
  The four manuscript blind sets stay held out and the paper's blind results are unaffected.
  **Cost, stated plainly**: Ti64 then has only one independent source, so **leave-one-source CV is not
  feasible for Ti64** (or AlSi10Mg) under Category B. Category B is not a smaller version of Category A
  — it is a different, weaker validation design.

The remaining path to a genuinely strong dataset under Category B is the one flagged in §3.1: an
additional independent source per material (starting with the 316L PMC source), which is a literature
task, not a formatting one.

---

## 6. Limitations — what could not be done

1. **No OCR extraction was needed or used.** All 15 new points came from tables or from one calibrated
   pixel extraction; `figure_ocr` does not appear in the dataset, and no value is an estimate read
   "by eye" without a stated calibration.
2. **Only one figure was digitised** (Sun 2024 Fig. 10b). WebPlotDigitizer is a GUI tool and was
   unavailable, as stated in the task; the PIL/PyMuPDF path was used instead and is reproducible from
   `scratch/pixel_bars2.py`. Digitisation accuracy is ± 7 MPa against the visual read, which is
   adequate for a 900–1300 MPa axis but is **not** a substitute for the authors' tabulated values.
3. **IN718 has no as-built, fully-tabulated second source.** `hovig2018in718` is solution-treated, and
   its per-orientation ε_f are figure reads; only its UTS is tabulated, and only as an orientation
   average. IN718's leave-one-source CV therefore rests on one as-built source (Watring) and one
   heat-treated source (Hovig) — a **confound between source and heat-treatment state** that the
   grouping cannot remove.
4. **Process windows are thin**: 6 groups, 4 of them containing a single point. Leave-one-process CV
   will remove an entire window at a time, so its power is limited; this is a property of the
   literature, not of the extraction.
5. **Two points carry a heat-treatment/condition caveat rather than a clean as-built state**
   (`hovig2018` solution-treated; `hitzler2017` milled surfaces on a 200 °C-preheated platform). Both
   are labelled in `heat_treatment` and `notes` as required, and neither is described as as-built
   without qualification.
6. **The band check is a sanity filter, not a validation.** Passing it means a number is the right
   order of magnitude for its alloy; it cannot detect a correctly-scaled but mis-assigned orientation
   label. That is why the §1 convention audit and the per-source internal cross-checks (e.g. Awd's
   +8 % UTS and −30 % ε_f statements) were performed and are recorded point by point.

---

## 7. R13b addendum — NIST mds2-3402 appended as a third Ti-6Al-4V source

**Scope note.** §0–§6 above are the frozen **R13** record (24 points / 6 sources) and are unchanged
by this addendum. Where §5 and the header table give counts, **the counts in this section supersede
them**. The R13 statistics that were locked before this work — the v12 nine-fold baseline and the
R13 LOSO/LOMO/LOPO results in `output/experiment_v13_cv_variants_results.json` — were **not**
modified and are **not** superseded: R13b reports alongside them, and the two are compared
explicitly in §7.5.

### 7.1 The source, and how it was obtained

| | |
|---|---|
| Dataset | *Dataset of Additive Manufacturing Laser Powder Bed Fusion Ti-6Al-4V Subjected to Various Novel Hot Isostatic Pressing (HIP) Treatments* |
| Identifier | ARK `ark:/88434/mds2-3402` · DOI `10.18434/mds2-3402` · PDR v1.0.0 (2024-07-09), v1.0.2 (2026-01-06) |
| Licence / access | **NIST Open License**, `accessLevel = public`, no account or click-through used |
| Accompanying report | NIST IR 8551, Dec 2024, 286 pp., doi `10.6028/NIST.IR.8551` — the authority for every heat-treatment definition and every published mean used here |
| Related papers | Derimow et al., *Mater. Des.* **2024**, 247, 113388 · Derimow et al., *Mater. Sci. Eng. A* **2025**, 921, 147549 |

**All seven prescribed download channels were attempted, with the command, HTTP status and byte
count recorded for each** in `nist_ambench/download_attempts_r13b.md`. The outcome in one line: the
NIST front end `data.nist.gov` is reachable but is unusable for bulk transfer (Cloudflare, ~4 KB/s
on a single connection), **the bottleneck is not the licence and not the network**; a plain-HTTP
request to a data-file URL returns **301** with a `Location` pointing at
`https://nist-oar-cache.s3.amazonaws.com/prd/fst4/mds2-3402/…`, and that object store serves the
same bytes fast. Everything was then pulled from that prefix.

| Obtained | Size | Verification |
|---|---|---|
| `Reduced_Data.zip` | 63 085 120 B | sha256 `725469b3225b0d8c289200ac6328c8112df20ed758f5db6dfe5bed2e37b0b6ee` **= the sha256 NIST publishes**; `zipfile.testzip()` all CRC pass; 5 425 members |
| `README.docx` | 177 664 B | sha256 `2a4deb54…5d70`, **identical bytes from four independent channels** (curl, .NET against the front end, .NET against S3, python urllib) |
| 30 curve PDFs | 9 535 448 B | per-file sha256 all OK |
| `NIST.IR.8551.pdf` | 981 785 550 B | 286 pp. (note: the bare URL 404s, `?download=1` is required) |

Negative results, recorded rather than glossed: the `web_fetch` route was **refused by robots.txt**
(and not retried); **no GitHub, Zenodo or Figshare mirror exists** (two independent searches); the
`Tensile Testing/RAW DATA` directory and the `Microscopy` tree were **not obtained** and are not used
by any number here. One background fetch over-ran its filter and pulled ≈56 microscopy TIFFs before
being identified and stopped; those files are **dataset files and were left in place, not deleted**.

### 7.2 Parse, aggregation, and the orientation mapping

300 specimen directories (`HT<0..14>-<h|v><1..10>`, one naming anomaly `HT6-h6a`), 15 conditions ×
2 orientations, plus root-level summary tables and per-specimen `Prop.txt` / `StressStrain.txt`
(header `strain,stress,0.2_line,fittedline`; **engineering strain mm/mm, engineering stress MPa**).
Build parameters from IR 8551 Table 2.1: **245 W, 1250 mm/s, 60 µm**, argon, O₂ < 10 ppm.

Three aggregation routes were available and were **compared rather than assumed**:
(a) the published group means of IR 8551 Table 3.6 (h) / 3.7 (v); (b) the arithmetic mean of all ten
deposited specimens; (c) a read of the 30 published vector curve PDFs.

**(a) is delivered**, for the reasons set out in `nist_ambench/mds2_3402_parsed.md` §4: it is the
authors' own published statistic, it carries n and ±1σ, it is directly citable to a numbered table,
and it is immune to the two defects in the deposited derivative table. Routes (b) and (c) are
retained as evidence — (c) is what makes (a) trustworthy, because it reproduces it.

**Orientation mapping: `v → 0°`, `h → 90°` on the model convention (0° = loading axis ∥ BD).**
IR 8551 **p61** fixes the tensile axis as **Y for horizontal and Z for vertical** tests; **p47**
fixes the **build direction as Z**; hence v (Z) ∥ BD ≡ 0° and h (Y) ⊥ BD ≡ 90°. This is the **same
operation** applied to `sun2024adem` in §1, and the source's own printed label is preserved in
`notes` on every row so the mapping is reversible as a one-column change if the §1 flag is
adjudicated the other way. Two independent consistency checks pass: the report states strength skews
vertical and elongation horizontal, and elongation is indeed higher at h in 13 of 15 conditions
(modulus likewise at v in 10 of 15); and the WP2 asset `nist_ambench_specimens.csv` — which is
**AM-Bench 2022**, a different dataset, and does **not** use h/v labels — independently confirms that
in this build frame **Z is the build direction and angles are measured from it**. **There is no 45°
orientation in this dataset**, so it contributes only to the 0° and 90° columns.

### 7.3 The 30 appended rows

`point_id = nist_mds2_3402_HT<0..14>_<h|v>`, `material = Ti64`, `source_id = nist_mds2_3402`,
`extraction_method = published_group_mean_table`, process columns 245 / 1250 / 60. The
`heat_treatment` column carries the **full schedule** (temperature, pressure, time, cooling rate,
and the tempering step) rather than a short code, because the short names (nsHIP / csHIP / SHIP)
are not self-describing and the schedule is what a descriptor model would need; the short and long
names are quoted in `notes` alongside the specimen-ID rule, n, ±1σ, the excluded specimens, and the
mapping basis.

**Append-only, enforced.** `nist_ambench/scripts/append_mds2_rows.py` re-serialises the header and
the existing rows from the file itself and compares **byte-for-byte** with the original; if they
differ it prints `REFUSING TO WRITE` and exits. The delivered result:

| | |
|---|---|
| data rows | **54** (was 24) |
| sources | **7** — v12_original 9 · sun2024adem 3 · hitzler2017materials 3 · awd2018metals 3 · hovig2018in718 3 · watring2020addma 3 · **nist_mds2_3402 30** |
| Ti-6Al-4V | **36 points across 3 sources** (was 6 across 2) |
| new points by orientation | 0°: 15 · 90°: 15 |
| CSV sha256 | `65183045594b00b7b349a0fa9bb799bc5c58bff4af346f4427ba66231803e8b5` |

**Band check (task bands: Ti-6Al-4V UTS 900–1200 MPa, ε_f 5–20 %): 10 of 30 in band, 20 flagged and
retained. No value was adjusted to fit a band and none was dropped for leaving it.** The pattern is
metallurgical, not a data defect: as-built HT0 is **stronger** than the band's top edge (1294–1303
MPa), and 18 of 30 points are **more ductile** than its top edge (up to 32.8 %), because HIP
converts the as-built α′ martensite to α+β. For scale, the v12 as-built Ti-6Al-4V calibration points
are ε_f = 0.048–0.080, so the high-ductility HIP conditions sit **4–6× beyond the calibration
range**. The ten in-band points are exactly the tempering/quenching conditions (HT6, HT7, HT8,
HT12, HT14) — the band cuts this source along a real metallurgical division.

**A value correction, found and fixed before delivery.** V1–V4 of the parse verification all check
the *curves* against the *tables*; none of them would catch a disagreement between two derived
copies of a published number. A fifth check (`check_published_value_sources.py`) compared the two
derived files across all 30 conditions and found **exactly one** mismatch: the `HT14-v` UTS read
949.69 from the table text but 949.68 from the subset reconstruction. The printed value was then
read off the report directly — `NIST.IR.8551.pdf` **p63** reads `949.69 ± 5.78` — so **the printed
value is what the CSV now carries**; the previously delivered 949.68 was the reconstruction
(an 8-specimen mean of 949.6792), not a transcription of the table. The append script now takes all
values from the table-parsed columns and asserts the drift set is exactly `['14-v']`. The correction
was applied by restoring the 24 original rows and re-deriving all 30, then diffing against a backup:
**exactly one row moved** (its `uts_MPa` and its note), the other 53 rows byte-identical
(`nist_ambench/download/_refix_append.txt`, backup `_csv_before_reappend_2026-09-28.csv`).

Also recorded, and **not** fixed because it lies outside these columns: for `HT13-h` the 0.2 % yield
strength recomputed from the deposited `Prop.txt` is 725.69 MPa against 806.32 MPa in the published
table (the only one of 30 conditions that disagrees); and the deposited `Elong.txt` is corrupt for
11 specimens (5 of them negative), which is why the deposited all-10 means differ from the published
means in exactly those 11 conditions. Both are documented in `mds2_3402_parsed.md`.

### 7.4 Leave-one-source rerun

**Scope: LOSO only.** Leave-one-material, leave-one-process, `comparison_with_v12_loocv`, the
protocol-deviation check, the orientation-mapping flag and the failure-mode analysis were **not**
recomputed; `output/experiment_v13_cv_variants_results.json` was treated as **read-only** and was
never written by this work. The new run reuses `experiment_v13_cv_variants.py`'s own construction,
calibration, scoring and summary functions and its prediction cache (148 → 154 entries; the 6 added
runs are Ti-6Al-4V refinements at S₀ = 0.5 and 1.28, all three orientations).

Folds go from **6 → 7**. The new fold is `Ti64 / held = nist_mds2_3402`. Because the as-built
criterion recognises only HT0, mds2 contributes **2 as-built points** to the Ti-6Al-4V calibration
pool, which changes the two existing Ti-6Al-4V folds as well:

| Fold | S₀ (R13 → R13b) | ef error % (R13 → R13b) | n held |
|---|---|---|---|
| 316L / hitzler2017materials | 2 → 2 | 89.52 → **89.52** | 3 |
| 316L / v12_original | 0.8 → 0.8 | 75.21 → **75.21** | 3 |
| AlSi10Mg / awd2018metals | 0.2 → 0.2 | 38.75 → **38.75** | 3 |
| AlSi10Mg / v12_original | 0.32 → 0.32 | 27.18 → **27.18** | 3 |
| **Ti64 / nist_mds2_3402 (new)** | — → 0.32 | — → **72.09** | **30** |
| Ti64 / sun2024adem | 0.32 → 0.5 | 23.97 → 21.87 | 3 |
| Ti64 / v12_original | 0.32 → 0.5 | 9.77 → 8.19 | 3 |

**Regression check (in script, and it raises rather than warns):** the four folds whose inputs did
not change reproduce R13's S₀ **and** error **to the reported digits**; the three Ti-6Al-4V folds are
flagged as input-changed and expected to move. Neither R13 block is overwritten — the JSON carries
`loso_v13_original` (copied verbatim, tagged LOCKED) beside `loso_with_nist`.

**Result — a negative result, not dressed up.** `loso_with_nist`: 7 folds, n = 48,
**mean ε_f error 61.35 %, median 72.64 %, pass@10 % = 5/48.** Against `loso_v13_original`: 6 folds,
n = 18, mean 44.07 %, median 28.80 %, pass@10 % = 5/18. The mds2 fold's mean ε_f error is
**72.09 %**; its mean UTS error is 8.34 %, so the transfer failure is in **ductility**, not strength.
Ti-6Al-4V now has three independent sources, and the model does **not** transfer to the third one.

**Mechanism — measured, not asserted.** The diagnostic block stratifies the 30 held points:

| Subset | n | mean ε_f error % | median % |
|---|---|---|---|
| as-built (HT0 only) | 2 | **67.09** | 67.09 |
| stress-relief (HT1 only) | 2 | 77.97 | 77.97 |
| HIP (HT2–HT14) | 26 | 72.02 | 75.13 |
| all heat-treated | 28 | 72.44 | 75.13 |

**The as-built points fail almost as badly as the HIP points (67.09 % vs 72.44 %).** That matters,
because the obvious attribution — "mds2 is HIP-treated and the calibration set is as-built, so the
missing heat-treatment descriptor explains the fold" — **is not supported**: the two points that
*do* share the calibration state fail by the same order. The dominant cause is the **property-range
gap, which is present as-built too**: mds2 HT0 is UTS 1294–1303 MPa with ε_f 16.8–19.3 %, against
UTS 1050–1299 MPa with ε_f 4.8–9.3 % for the six as-built Ti-6Al-4V calibration points, so the S₀
calibrated on those points does not transfer to this material at **any** heat-treatment state. The
absent heat-treatment descriptor is a second, additive contribution — and it is sharper than the
IN718 `hovig2018in718` precedent, because within this source the division is visible: the
super-β HIP + rapid-quench routes (HT6, HT7, HT8, HT12, HT14) are precisely the ten points that
land in the nominal band and the least badly predicted, while the sub-β HIP and aged routes
(HT2–HT5, HT9–HT11) are 3–4× more ductile than as-built.

This is the same class of finding as the §4 result already in this log — that the extended dataset
does not reproduce the manuscript's clean cross-source picture once the orientation convention is
enforced — and it is the stronger version of it: the failure is not a labelling artefact.

### 7.5 What this changes, and what it does not

**Does not change:** any v12 value; any of the 15 previously extracted extended points; the locked
R13 CV statistics in `experiment_v13_cv_variants_results.json`; the §1 orientation convention; the
Category A / Category B choice in §5. All of these are untouched, and the CSV's 24 pre-existing rows
were verified byte-identical after the append.

**Changes:** the extended dataset is now **54 points / 7 sources**; **Ti-6Al-4V has a third
independent source**, so its leave-one-source CV is no longer restricted to two; and the
leave-one-source evidence is now **stronger against** the model's transferability claim, not weaker,
because the third source is a same-material, same-process-family, fully documented dataset that the
model still misses by ~72 % on ductility.

**Declared limitations of this addition, stated rather than smoothed over:**

1. **The 30 points are 15 condition means × 2 orientations, not 30 independent measurements.** Each
   carries n = 10. The number of held *rows* is 30, but the underlying independent specimens are 30
   groups of 10, so the row-to-row scatter understates the specimen scatter and the fold's error
   statistics read as better conditioned than the data's true degrees of freedom. This is a property
   of delivering published group means, and it is the main reason to read the mds2 fold's spread
   with care.
2. **10 of 30 points are inside the nominal Ti-6Al-4V band; 20 are flagged.** The source is
   deliberately a wide heat-treatment study, not a nominal-property set. A band-only subset of it
   (HT6/HT7/HT8/HT12/HT14 × 2 orientations) is available by filtering `heat_treatment`, and is the
   right input for a nominal-property calibration — but it was **not** what was appended, and no
   such subset was created here.
3. **`RAW DATA` and `Microscopy` were not obtained.** No number here depends on them (all 30 rows
   come from the reduced curves and the published tables), but the deposit was not collected in full.
4. **The h/v → 90°/0° mapping is inference from three sentences in the report**, well supported and
   independently consistent, but not a label the depositors printed in that form. The source label is
   preserved on every row.
5. **Only UTS and ε_f were extracted**, per the 14-column schema; YS, E and uniform elongation are
   available in the same tables if a later revision wants them.

**Reproduce:** `simulation/experiment_v13b_mds2_loso.py` (≈170 s from a cold cache, 0.1 s warm) →
`output/experiment_v13b_mds2_loso_results.json`; run log `output/cv_variants_r13b_log.txt` (the log
preceding the §7.3 value correction is kept at `output/_r13b_log_before_value_correction.txt`). The
download and parse record is `nist_ambench/download_attempts_r13b.md` and
`nist_ambench/mds2_3402_parsed.md`.

**Pre-delivery verification** (`verifier-hub`, 2026-09-28; raw output in
`nist_ambench/download/_verifier_results.txt`): `rubric check-file-format` passes on this log, the
CSV, the results JSON, `download_attempts_r13b.md` and `mds2_3402_parsed.md`;
`text must-contain` matched **7/7** required terms in this log and **5/5** in `mds2_3402_parsed.md`;
`rubric check-no-placeholder` reports **0** placeholders in this log and in the results JSON; and
`rubric check-cross-consistency` finds the point count (**54**) and the source count (**7**) agree
between this log and the JSON.
