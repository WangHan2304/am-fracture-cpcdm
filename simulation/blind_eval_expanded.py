# -*- coding: utf-8 -*-
"""blind_eval_expanded.py — Item 2: expand the paired (descriptor vs Lemaitre) blind set.

The headline paired comparison in the manuscript currently rests on n=9 leave-one-
orientation-out folds (paired_test_bootstrap.py). This script lifts that to a larger set
of INDEPENDENT, leakage-free blind points drawn from the 54-point extended literature
dataset, WITHOUT modifying any locked result:

  * Re-uses the R13 CV machinery (experiment_v13_cv_variants as `cv`) verbatim: the same
    load_extended_dataset / build_loso / build_lomo / decide_s0 / grid, the same
    (material, orientation, S0, n_grains) cache, and the same gs(psi) directional rule.
  * Modified model predictions at each fold S0 come from the SHARED cache (already run for
    the locked LOSO/LOMO statistics -> effectively free).
  * The Lemaitre baseline (f_AM == 1) is given its OWN refit S0 on the SAME training set,
    exactly as v12's paired protocol does, so the paired contrast has an identical fitted
    parameter count and identical data separation. Baseline runs are cached separately.

Headline blind unit set:
  * every as-built held point of a FEASIBLE leave-one-source fold (316L, AlSi10Mg, Ti64:
    the external-source folds and the v12_original-held folds, all leakage-free);
  * plus the as-built IN718 points, which cannot form a LOSO fold (single as-built IN718
    source) and are therefore scored under the strict leave-one-material-out transfer S0
    (geometric mean of the other materials), the manuscript's 0.504 MPa.
  The 30 NIST mds2 rows are heat-treated/out-of-domain and are EXCLUDED from the headline
  paired set (reported separately as a transfer diagnostic), consistent with the main text.

One measured fracture-strain value = one blind unit (deduplicated by point_id), each with
a paired (modified, baseline) prediction under a fold-calibrated S0.

Output: simulation/output/blind_eval_expanded_results.json
"""
import argparse
import json
import multiprocessing
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import experiment_v13_cv_variants as cv                       # noqa: E402

OUT = os.path.join(HERE, 'output')
LEM_CACHE_PATH = os.path.join(OUT, 'blind_expanded_lem_cache.json')
RESULT_PATH = os.path.join(OUT, 'blind_eval_expanded_results.json')


# ============================================================
# Lemaitre baseline runner: identical to cv.run_taylor except force_homogeneous=True
# (f_AM == 1), same gs(psi) directional correction, same S0 grid + refinement.
# ============================================================
def run_taylor_lem(mat, psi, S0, n_grains=cv.N_GRAINS):
    from taylor_cpcdm import TaylorCPCDM
    max_strain = cv._max_strain(mat, psi)
    ov = dict(cv.MATERIAL_OVERRIDES.get(mat, {}))
    ov.update(cv._gs_directional(mat, psi))
    ov['S0'] = S0
    base_mat = cv.IN718_BASE_MATERIAL if mat in cv.MATERIAL_OVERRIDES else mat
    model = TaylorCPCDM(
        base_mat, psi, S0_override=S0, n_grains=n_grains,
        eps_step=cv._eps_step(mat), n_sub=30, strain_rate=cv.STRAIN_RATE,
        model_version='v4', force_homogeneous=True,           # f_AM == 1
        param_overrides=ov, seed=42,
    )
    r = model.run_uniaxial(max_strain)
    ef = float(r['fracture_strain'])
    truncated = bool(ef >= max_strain - 1e-12)
    return ef, float(r['uts'] / 1e6), truncated


def _lem_worker(task):
    mat, psi, S0, ng = task
    t0 = time.time()
    try:
        ef, uts, trunc = run_taylor_lem(mat, psi, S0, ng)
        return {'key': cv.ckey(mat, psi, S0, ng), 'ok': True, 'ef': ef, 'uts': uts,
                'truncated': trunc, 'wall_s': round(time.time() - t0, 1)}
    except Exception as exc:                                   # noqa: BLE001
        return {'key': cv.ckey(mat, psi, S0, ng), 'ok': False, 'error': repr(exc),
                'wall_s': round(time.time() - t0, 1)}


def load_lem_cache():
    if os.path.exists(LEM_CACHE_PATH):
        try:
            with open(LEM_CACHE_PATH, encoding='utf-8') as f:
                blob = json.load(f)
            return {k: v for k, v in blob.get('entries', {}).items() if v.get('ok')}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_lem_cache(entries):
    os.makedirs(OUT, exist_ok=True)
    tmp = LEM_CACHE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'meta': {'last_saved': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'note': 'f_AM==1 Lemaitre baseline, own refit S0 per fold',
                            'n_entries': len(entries)},
                   'entries': entries}, f, indent=1, sort_keys=True)
    os.replace(tmp, LEM_CACHE_PATH)


def ensure_lem_runs(tasks, cache, log, workers=1, n_grains=cv.N_GRAINS):
    todo, seen = [], set()
    for mat, psi, S0 in tasks:
        k = cv.ckey(mat, psi, S0, n_grains)
        if k in cache or k in seen:
            continue
        seen.add(k)
        todo.append((mat, psi, S0, n_grains))
    if not todo:
        return 0, 0
    log(f'  [lem-run] {len(todo)} new baseline run(s), workers={workers}')
    n_new, n_fail, t0 = 0, 0, time.time()
    if workers <= 1:
        results = [_lem_worker(t) for t in todo]
    else:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(_lem_worker, todo))
    for res in results:
        if res['ok']:
            cache[res['key']] = {'ef': res['ef'], 'uts': res['uts'],
                                 'truncated': res['truncated'], 'ok': True}
            n_new += 1
        else:
            n_fail += 1
            log(f'    lem FAILED {res["key"]} {res["error"]}')
        save_lem_cache(cache)
    log(f'  [lem-run] done: {n_new} new, {n_fail} failed, {time.time()-t0:.0f}s')
    return n_new, n_fail


def _score_s0_lem(cache, mat, pts, s0):
    errs = []
    for pt in pts:
        rec = cache.get(cv.ckey(mat, pt['orientation_deg'], s0, cv.N_GRAINS))
        if rec is None:
            raise RuntimeError(f'uncached baseline run {cv.ckey(mat, pt["orientation_deg"], s0)}')
        errs.append(abs(rec['ef'] - pt['ef']) / pt['ef'] * 100.0)
    return float(np.mean(errs))


# Provably-equivalent reduced grid for the BASELINE argmin.
# With f_AM == 1 the damage rate is (Y/S0)^s * pdot, so the predicted fracture strain is
# monotonically INCREASING in S0 and saturates at max_strain; the two top v12 grid points
# (12.8, 50.0) are therefore strictly the worst fits (largest ef error) and can never be
# the minimiser. The baseline optimum for every material (locked ~0.2-2.0 MPa region) is
# bracketed by [0.05,0.2,0.8,3.2] plus the same best/1.6 & best*1.6 refinement, so this
# returns the identical S0 as the full grid at a fraction of the cost.
LEM_S0_GRID = [0.05, 0.2, 0.8, 3.2]


def calibrate_lem(cache, mat, train_pts, log, workers=1):
    """v12-identical S0 search (grid + one refinement round) under f_AM == 1."""
    def need(s0):
        ensure_lem_runs([(mat, p['orientation_deg'], s0) for p in train_pts],
                        cache, log, workers=workers)
    best = (None, None)
    for s0 in LEM_S0_GRID:
        need(s0)
        e = _score_s0_lem(cache, mat, train_pts, s0)
        if best[1] is None or e < best[1]:
            best = (s0, e)
    lo, hi = best[0] / 1.6, best[0] * 1.6
    for s0 in (lo, hi):
        need(s0)
        e = _score_s0_lem(cache, mat, train_pts, s0)
        if e < best[1]:
            best = (s0, e)
    return float(best[0]), float(best[1])


def lem_pred(cache, mat, psi, S0):
    return cache.get(cv.ckey(mat, psi, S0, cv.N_GRAINS))


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    log = cv.Log(os.path.join(OUT, 'blind_expanded_log.txt'), echo=True)
    t_start = time.time()
    log('Item 2  Expanded paired blind set (descriptor vs Lemaitre)')
    log(f'n_grains={cv.N_GRAINS}  S0_grid={cv.S0_GRID}  workers={args.workers}')

    points = cv.load_extended_dataset()
    by_id = {p['point_id']: p for p in points}
    cache = cv.load_cache()
    lemcache = load_lem_cache()
    log(f'dataset {len(points)} points, {len({p["source_id"] for p in points})} sources; '
        f'mod cache {len(cache)}, lem cache {len(lemcache)}')

    # ---- LOSO: fold S0 for modified (shared cache) and for baseline (own refit) ----
    loso_specs, loso_skipped = cv.build_loso(points)
    log(f'LOSO feasible folds: {len(loso_specs)}')
    for sp in loso_specs:
        sp['S0'], sp['train_err'], sp['method'], _ = cv.decide_s0(
            cache, sp['material'], sp['train_points'], sp['prefer_locked'],
            log=log, workers=args.workers)
        sp['S0_lem'], sp['train_err_lem'] = calibrate_lem(
            lemcache, sp['material'], sp['train_points'], log, workers=args.workers)
        log(f'  {sp["material"]:9s} held={sp["held_source"]:22s} '
            f'S0mod={sp["S0"]:<9.4g} S0lem={sp["S0_lem"]:<9.4g} '
            f'(train_err mod={sp["train_err"]:.1f}% lem={sp["train_err_lem"]:.1f}%)')

    # ---- LOMO transfer S0 for IN718 (modified from locked values, baseline refit) ----
    lomo_specs = cv.build_lomo(points)
    in718 = next((s for s in lomo_specs if s['held_material'] == 'IN718'), None)
    S0_in718_mod = S0_in718_lem = None
    if in718 is not None:
        mods, lems = [], []
        for m, mpts in in718['train_points'].items():
            s0m, _, _, _ = cv.decide_s0(cache, m, mpts, prefer_locked=True,
                                       log=log, workers=args.workers)
            s0l, _ = calibrate_lem(lemcache, m, mpts, log, workers=args.workers)
            mods.append(s0m)
            lems.append(s0l)
        S0_in718_mod = float(np.exp(np.mean(np.log(mods))))
        S0_in718_lem = float(np.exp(np.mean(np.log(lems))))
        log(f'IN718 strict LOMO transfer S0: mod={S0_in718_mod:.4g} lem={S0_in718_lem:.4g}')

    # ---- assemble unique blind units (dedup by point_id) ----
    units = {}

    def add_unit(pt, s0_mod, s0_lem, fold, src, prefer_in718=False):
        pid = pt['point_id']
        if pid in units and not prefer_in718:
            return
        units[pid] = {'point_id': pid, 'material': pt['material'],
                      'source_id': pt['source_id'], 'orientation_deg': pt['orientation_deg'],
                      'ef_exp': pt['ef'], 'uts_exp_MPa': pt['uts_MPa'],
                      'S0_mod': s0_mod, 'S0_lem': s0_lem, 'fold': fold, 'held_src': src}

    for sp in loso_specs:
        for p in sp['held_points']:
            if cv.is_as_built(p):
                add_unit(p, sp['S0'], sp['S0_lem'], 'LOSO', sp['held_source'])
    if in718 is not None:
        for p in in718['held_points']:
            if cv.is_as_built(p):
                add_unit(p, S0_in718_mod, S0_in718_lem, 'LOMO-IN718', 'watring/hovig',
                         prefer_in718=True)

    log(f'headline blind units (as-built, dedup): n={len(units)}')

    # ---- ensure predictions exist; score both models at each unit's fold S0 ----
    mod_tasks = [(u['material'], u['orientation_deg'], u['S0_mod']) for u in units.values()]
    lem_tasks = [(u['material'], u['orientation_deg'], u['S0_lem']) for u in units.values()]
    cv.ensure_runs(mod_tasks, cache, log, workers=args.workers)
    ensure_lem_runs(lem_tasks, lemcache, log, workers=args.workers)

    rows = []
    for u in units.values():
        mrec = cv.get_near(cache, u['material'], u['orientation_deg'], u['S0_mod'])
        lrec = lem_pred(lemcache, u['material'], u['orientation_deg'], u['S0_lem'])
        if mrec is None or lrec is None:
            log(f'  SKIP {u["point_id"]}: missing run (mod={mrec is None}, lem={lrec is None})')
            continue
        exp = u['ef_exp']
        em = (mrec['ef'] - exp) / exp * 100.0
        el = (lrec['ef'] - exp) / exp * 100.0
        rows.append({**u,
                     'ef_pred_mod': round(mrec['ef'], 5), 'ef_pred_lem': round(lrec['ef'], 5),
                     'signed_err_mod_pct': round(em, 2), 'signed_err_lem_pct': round(el, 2),
                     'abs_err_mod_pct': round(abs(em), 2), 'abs_err_lem_pct': round(abs(el), 2),
                     'uts_pred_mod_MPa': round(mrec['uts'], 1),
                     'uts_pred_lem_MPa': round(lrec['uts'], 1)})

    if not rows:
        raise SystemExit('no paired blind units produced')

    amod = np.array([r['abs_err_mod_pct'] for r in rows])
    alem = np.array([r['abs_err_lem_pct'] for r in rows])
    d = amod - alem
    rng = np.random.default_rng(42)
    boot = rng.choice(d, size=(100000, d.size)).mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    from scipy import stats
    w_stat, w_p = stats.wilcoxon(d)
    n_pos, n_neg = int(np.sum(d > 0)), int(np.sum(d < 0))
    s_p = float(2 * min(stats.binom.cdf(min(n_pos, n_neg), n_pos + n_neg, 0.5),
                        stats.binom.sf(min(n_pos, n_neg) - 1, n_pos + n_neg, 0.5)))
    dz = float(d.mean() / (d.std(ddof=1) + 1e-12)) if d.std(ddof=1) > 0 else 0.0

    payload = {
        'generated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'purpose': ('Expanded paired blind set: descriptor model vs Lemaitre baseline, '
                    'each calibrating a single S0 on the identical leakage-free training '
                    'set. Lifts the headline paired test from n=9 (v12 LOOCV) to n='
                    f'{len(rows)} independent blind points.'),
        'protocol': ('LOSO held as-built points across 316L/AlSi10Mg/Ti64 (external-source '
                     'and v12_original-held folds) + IN718 as-built points under strict '
                     'LOMO transfer S0; NIST mds2 heat-treated rows excluded as '
                     'out-of-domain. Baseline = f_AM==1 with its own refit S0 per fold.'),
        'baseline_grid_deviation': ('baseline S0 search uses the reduced grid [0.05,0.2,'
                                   '0.8,3.2]+refinement; the top two v12 grid points (12.8,'
                                   '50.0) are provably non-optimal under f_AM==1 (ef is '
                                   'monotone-increasing in S0), so the argmin is identical '
                                   'to the full grid. Modified model keeps the full v12 '
                                   'grid from the shared cache.'),
        'n_blind_points': len(rows),
        'units': sorted(rows, key=lambda r: (r['material'], r['source_id'],
                                             r['orientation_deg'])),
        'summary': {
            'modified_mean_abs_err_pct': round(float(amod.mean()), 2),
            'lemaitre_mean_abs_err_pct': round(float(alem.mean()), 2),
            'modified_median_abs_err_pct': round(float(np.median(amod)), 2),
            'lemaitre_median_abs_err_pct': round(float(np.median(alem)), 2),
            'modified_pass10': int(np.sum(amod <= 10.0)),
            'lemaitre_pass10': int(np.sum(alem <= 10.0)),
        },
        'paired': {
            'n': int(d.size),
            'mean_diff_mod_minus_lem_pct': round(float(d.mean()), 2),
            'wilcoxon': {'stat': float(w_stat), 'p_value': round(float(w_p), 4)},
            'sign_test': {'n_mod_better': n_neg, 'n_lem_better': n_pos,
                          'p_value_two_sided': round(s_p, 4)},
            'bootstrap_100k_mean_diff_95ci': [round(float(ci_lo), 2), round(float(ci_hi), 2)],
            'effect_size_dz': round(dz, 3),
        },
        'run': {'wall_seconds': round(time.time() - t_start, 1),
                'workers': args.workers, 'n_lem_cache': len(lemcache)},
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    log('')
    log('=' * 78)
    log(f'n={len(rows)}  mod mean/med={amod.mean():.2f}/{np.median(amod):.2f}  '
        f'lem mean/med={alem.mean():.2f}/{np.median(alem):.2f}')
    log(f'paired mean diff={d.mean():+.2f}pt  wilcoxon p={w_p:.4f}  sign p={s_p:.4f}  '
        f'bootstrap CI=[{ci_lo:.2f},{ci_hi:.2f}]  dz={dz:.2f}')
    log(f'DONE -> {RESULT_PATH}  ({time.time()-t_start:.1f}s)')
    log.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
