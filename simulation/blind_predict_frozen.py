# -*- coding: utf-8 -*-
"""blind_predict_frozen.py — strict frozen-parameter blind prediction on external literature.

Protocol asked for by the task
-----------------------------
The model is calibrated ONLY on the nine v12 main-calibration cases (three orientations
x three materials Ti-6Al-4V / 316L / AlSi10Mg, source_id = v12_original). That calibration
freezes exactly one adjustable parameter per material, the damage threshold S0, to the
published v12 values (experiment_v13_cv_variants.S0_KNOWN_V12 = Ti64 0.32, 316L 3.2,
AlSi10Mg 0.20 MPa). Every other material/model constant is fixed a priori in
src/materials.py and the projection/size laws in cv (am_correction_v4, gs(psi), XI0).

These parameters are then FROZEN and applied DIRECTLY to tensile data drawn from OTHER
papers. No external point is ever used to re-fit anything: there is no grid search, no
refinement, no per-fold recalibration — the single per-material S0 is taken verbatim from
the nine-case calibration. This is a strict out-of-sample blind test, distinct from the
LOSO/LOMO folds in experiment_v13_cv_variants.py (which re-fit S0 on each fold's training
set). Here the SAME frozen S0 is reused for every blind point of a material.

Scope of the blind set
----------------------
In-domain = as-built material points whose base material is one of the three calibrated
metals. External sources supplying such points (source_id != v12_original):
  * sun2024adem     Ti-6Al-4V  as-built, full 0/45/90 trio
  * hitzler2017materials  316L as-built, full 0/45/90 trio
  * awd2018metals   AlSi10Mg   as-built, full 0/45/90 trio
  * nist_mds2_3402  Ti-6Al-4V  as-built (HT0, no post-build treatment) h/v pair
Excluded (cannot be reached by the frozen 9-case calibration without a refit or a
material/heat-treatment transfer that is NOT part of "calibrate then freeze"):
  * v12_original        -> these ARE the calibration set, never blind
  * hovig2018in718      -> solution-treated, and IN718 is not among the 9 cases
  * watring2020addma    -> IN718 (not among the 9 cases)
  * nist HT1..HT14      -> heat-treated (HIP/anneal/temper), out of the as-built domain
The 9-point trio set (sun/hitzler/awd) is the headline in-domain blind test; NIST HT0 is
reported separately as a supplementary out-of-band-strength Ti-6Al-4V check.

Outputs  simulation/output/blind_predict_frozen_results.json
         simulation/output/blind_predict_frozen_log.txt
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import experiment_v13_cv_variants as cv                       # noqa: E402

OUT = os.path.join(HERE, 'output')
RESULT_PATH = os.path.join(OUT, 'blind_predict_frozen_results.json')
LOG_PATH = os.path.join(OUT, 'blind_predict_frozen_log.txt')

# Frozen from the nine-case v12 calibration — reused verbatim, NEVER re-fitted here.
FROZEN_S0 = dict(cv.S0_KNOWN_V12)
CALIB_MATERIALS = set(FROZEN_S0)

# Headline external sources: one complete as-built 0/45/90 trio per calibrated material.
HEADLINE_SOURCES = {'sun2024adem', 'hitzler2017materials', 'awd2018metals'}
# Supplementary in-domain as-built external (Ti-6Al-4V, out-of-band strength).
SUPPLEMENT_SOURCES = {'nist_mds2_3402'}


def select_external(points):
    """External as-built points on the three calibrated materials, split headline/ suppl."""
    headline, suppl = [], []
    for p in points:
        if p['source_id'] == 'v12_original':
            continue                                   # calibration set, never blind
        if p['material'] not in CALIB_MATERIALS:
            continue                                   # needs material transfer, not frozen
        if not cv.is_as_built(p):
            continue                                   # heat-treated -> out of domain
        if p['source_id'] in HEADLINE_SOURCES:
            headline.append(p)
        elif p['source_id'] in SUPPLEMENT_SOURCES:
            suppl.append(p)
    return headline, suppl


def predict_points(cache, pts, log):
    """Score each point at its material's FROZEN S0 (no refit). Runs come from the cache."""
    rows = []
    for p in pts:
        s0 = FROZEN_S0[p['material']]
        rec = cv.get_near(cache, p['material'], p['orientation_deg'], s0)
        if rec is None:
            # Should not happen: the frozen (mat,psi,S0) runs are already cached. If a new
            # material/orientation were added, run it once at the frozen S0 (no search).
            cv.ensure_runs([(p['material'], p['orientation_deg'], s0)],
                           cache, log, workers=1)
            rec = cv.get_near(cache, p['material'], p['orientation_deg'], s0)
        ef_err = abs(rec['ef'] - p['ef']) / p['ef'] * 100.0
        signed = (rec['ef'] - p['ef']) / p['ef'] * 100.0
        uts_err = (abs(rec['uts'] - p['uts_MPa']) / p['uts_MPa'] * 100.0
                   if p.get('uts_MPa') else None)
        rows.append({
            'point_id': p['point_id'],
            'material': p['material'],
            'source_id': p['source_id'],
            'orientation_deg': p['orientation_deg'],
            'S0_frozen_MPa': s0,
            'ef_exp': round(p['ef'], 5),
            'ef_pred': round(rec['ef'], 5),
            'ef_signed_err_pct': round(signed, 2),
            'ef_abs_err_pct': round(ef_err, 2),
            'uts_exp_MPa': p.get('uts_MPa'),
            'uts_pred_MPa': round(rec['uts'], 1),
            'uts_abs_err_pct': round(uts_err, 2) if uts_err is not None else None,
            'truncated': rec.get('truncated', False),
        })
    return rows


def summarise(rows):
    ef = np.array([r['ef_abs_err_pct'] for r in rows])
    signed = np.array([r['ef_signed_err_pct'] for r in rows])
    uts = np.array([r['uts_abs_err_pct'] for r in rows
                    if r['uts_abs_err_pct'] is not None], dtype=float)
    return {
        'n_points': len(rows),
        'ef_mean_abs_err_pct': round(float(ef.mean()), 2),
        'ef_median_abs_err_pct': round(float(np.median(ef)), 2),
        'ef_max_abs_err_pct': round(float(ef.max()), 2),
        'ef_mean_signed_pct': round(float(signed.mean()), 2),
        'ef_within_10pct': int(np.sum(ef <= 10.0)),
        'ef_within_20pct': int(np.sum(ef <= 20.0)),
        'ef_within_30pct': int(np.sum(ef <= 30.0)),
        'uts_mean_abs_err_pct': round(float(uts.mean()), 2) if uts.size else None,
        'uts_within_10pct': int(np.sum(uts <= 10.0)) if uts.size else None,
    }


def by_source(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r['source_id'], []).append(r)
    return {src: summarise(rs) for src, rs in sorted(groups.items())}


def by_material(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r['material'], []).append(r)
    return {m: summarise(rs) for m, rs in sorted(groups.items())}


def main():
    log = cv.Log(LOG_PATH, echo=True)
    t0 = time.time()
    log('FROZEN-PARAMETER BLIND PREDICTION (external literature, no refit)')
    log(f'frozen S0 from the 9-case v12 calibration: {FROZEN_S0}')

    points = cv.load_extended_dataset()
    cache = cv.load_cache()
    headline, suppl = select_external(points)
    log(f'external in-domain as-built: headline={len(headline)} '
        f'supplementary={len(suppl)}; cache entries={len(cache)}')

    hrows = predict_points(cache, headline, log)
    srows = predict_points(cache, suppl, log)

    log('')
    log('=== HEADLINE blind points (frozen S0, one full trio per material) ===')
    hdr = ('point_id', 'mat', 'ori', 'S0', 'ef_exp', 'ef_pred', 'signed%', 'abs%',
           'uts_exp', 'uts_pred', 'uts%')
    log('{:<22}{:<7}{:>4}{:>6}{:>8}{:>8}{:>8}{:>7}{:>8}{:>9}{:>7}'.format(*hdr))
    for r in sorted(hrows, key=lambda x: (x['material'], x['orientation_deg'])):
        log('{:<22}{:<7}{:>4}{:>6.2f}{:>8.4f}{:>8.4f}{:>8.1f}{:>7.1f}{:>8}{:>9.1f}{:>7}'.format(
            r['point_id'], r['material'], r['orientation_deg'], r['S0_frozen_MPa'],
            r['ef_exp'], r['ef_pred'], r['ef_signed_err_pct'], r['ef_abs_err_pct'],
            r['uts_exp_MPa'], r['uts_pred_MPa'],
            f"{r['uts_abs_err_pct']:.1f}" if r['uts_abs_err_pct'] is not None else 'NA'))

    payload = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'protocol': (
            'Parameters frozen at the nine-case v12 calibration (source_id=v12_original): '
            'one adjustable S0 per material, no grid search / refinement / per-fold refit. '
            'External as-built points from OTHER papers are predicted DIRECTLY.'),
        'frozen_S0_MPa': FROZEN_S0,
        'calibration_source': 'v12_original (9 points: Ti64/316L/AlSi10Mg x 0/45/90)',
        'excluded': {
            'v12_original': 'these ARE the calibration set (never blind)',
            'hovig2018in718': 'solution-treated and IN718 not among the 9 cases',
            'watring2020addma': 'IN718 not among the 9 cases',
            'nist HT1..HT14': 'heat-treated (HIP/anneal/temper) -> out of as-built domain',
        },
        'headline': {
            'description': ('in-domain external as-built trios: sun2024adem (Ti64), '
                            'hitzler2017materials (316L), awd2018metals (AlSi10Mg)'),
            'summary': summarise(hrows),
            'by_source': by_source(hrows),
            'by_material': by_material(hrows),
            'points': sorted(hrows, key=lambda x: (x['material'], x['orientation_deg'])),
        },
        'supplementary': {
            'description': ('Ti-6Al-4V as-built (no post-build treatment) NIST HT0 h/v pair; '
                            'UTS out-of-band (1294-1303 MPa) -> reported separately'),
            'summary': summarise(srows) if srows else {},
            'points': sorted(srows, key=lambda x: x['orientation_deg']),
        },
        'run': {'wall_seconds': round(time.time() - t0, 1),
                'n_grains': cv.N_GRAINS, 'cache_reused': True},
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    hs = payload['headline']['summary']
    log('')
    log(f"HEADLINE n={hs['n_points']}  ef mean/med/max={hs['ef_mean_abs_err_pct']}/"
        f"{hs['ef_median_abs_err_pct']}/{hs['ef_max_abs_err_pct']}%  "
        f"pass@20%={hs['ef_within_20pct']}/{hs['n_points']}  "
        f"UTS mean={hs['uts_mean_abs_err_pct']}%")
    log(f'DONE -> {RESULT_PATH}  ({time.time()-t0:.1f}s)')
    log.close()


if __name__ == '__main__':
    main()
