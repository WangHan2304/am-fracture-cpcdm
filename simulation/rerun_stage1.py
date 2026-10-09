# -*- coding: utf-8 -*-
"""
rerun_stage1.py — Stage-1 downstream regeneration driver (post xi_eff correction)
==================================================================================
Runs the affected pipeline scripts in dependency-safe waves after the corrected
grain-morphology projection (eq:xi_eff) and the refreshed frozen S0
(Ti64 0.32 / 316L 3.2 / AlSi10Mg 0.20) propagated through every hardcoded constant.
The prediction caches were backed up (*.prefixi.bak) and cleared because their keys
(material, psi, S0, n_grains) omit xi_eff.

Waves
-----
A1  independent single-process consumers (no internal pool) run CONCURRENTLY:
    cross-source blind tests, IN718 blind variants, beta6 slow scans, frozen blind
    prediction, standalone sensitivity studies, single-crystal bounds.
A2  scripts that spawn their OWN ProcessPoolExecutor run SEQUENTIALLY (one per wave)
    so they can each use the whole box without oversubscription.
B   blind_eval_expanded.py (n=23 headline table; internal pool) — feeds the Bayes and
    model-simplification / Sobol consumers in wave C.
C   consumers of blind_eval_expanded_results.json: statistical_upgrade_bayes (single),
    directional_bayes (single), model_simplification (pool), sensitivity_sobol_beta4
    (pool; also needs sensitivity_r5_beta.json from A1).

Nothing here edits results; each script overwrites its own output JSON. A non-zero
return code is logged and the run continues so one slow fold cannot block the rest.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, 'output', 'rerun_stage1_log.txt')

# A1: independent, single-process -> safe to run concurrently.
A1 = [
    'blind_test_ti64_cross_source.py',
    'blind_test_316l_cross_source.py',
    'blind_alsi_second_source.py',
    'blind_test_in718_fast.py',
    'blind_test_in718.py',
    'blind_test_in718_v2.py',
    'blind_test_in718_mode2.py',
    'blind_ti64_beta6_slow.py',
    'blind_316l_beta6_slow.py',
    'blind_predict_frozen.py',
    'sensitivity_Dc.py',
    'sensitivity_k_g_90.py',
    'sensitivity_r5_beta.py',
    'sensitivity_v9.py',
    'taylor_single_crystal_bounds.py',
    'beta6_scan.py',
]

# A2: internal-pool scripts -> sequential, full box each.
A2 = [
    'experiment_v11_loocv_fix.py',
    'experiment_v11_gs_directional.py',
    'experiment_v14_uncertainty.py',
    'experiment_v15_lambda_rel.py',
    'wiso_scan.py',
    'beta5_scan.py',
]

# B: headline expanded blind set (internal pool).
B = ['blind_eval_expanded.py']

# C: consumers of blind_eval_expanded_results.json.
C_single = ['statistical_upgrade_bayes.py', 'directional_bayes.py']
C_pool = ['model_simplification.py', 'sensitivity_sobol_beta4.py']

_pool_args = {'blind_eval_expanded.py': ['--workers', '14'],
              'model_simplification.py': ['--workers', '14'],
              'sensitivity_sobol_beta4.py': ['--workers', '14']}


def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def run(script, extra=None):
    extra = extra or _pool_args.get(script, [])
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, script] + extra,
                           cwd=HERE, capture_output=True, text=True)
        rc = p.returncode
    except Exception as e:  # noqa: BLE001
        rc = -999
        p = None
    dt = time.time() - t0
    tail = ''
    if p is not None:
        blob = (p.stdout or '') + (('\n' + p.stderr) if p.stderr else '')
        tail = ' | '.join(blob.strip().splitlines()[-2:])
    log('%-38s rc=%d  %.0fs  %s' % (script, rc, dt, tail[:220]))
    return script, rc, dt


def main():
    open(LOG, 'a', encoding='utf-8').write(
        '\n==== rerun_stage1 %s ====\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
    results = []

    log('WAVE A1: %d independent single-process scripts (concurrent)' % len(A1))
    with ThreadPoolExecutor(max_workers=12) as ex:
        for fut in as_completed([ex.submit(run, s) for s in A1]):
            results.append(fut.result())

    log('WAVE A2: %d internal-pool scripts (sequential)' % len(A2))
    for s in A2:
        results.append(run(s))

    log('WAVE B: expanded blind set (sequential, full box)')
    for s in B:
        results.append(run(s))

    log('WAVE C single-process (concurrent)')
    with ThreadPoolExecutor(max_workers=len(C_single)) as ex:
        for fut in as_completed([ex.submit(run, s) for s in C_single]):
            results.append(fut.result())

    log('WAVE C pool (sequential)')
    for s in C_pool:
        results.append(run(s))

    bad = [r for r in results if r[1] != 0]
    log('==== DONE: %d/%d ok, %d failed ====' %
        (len(results) - len(bad), len(results), len(bad)))
    for s, rc, dt in sorted(results, key=lambda x: -x[2]):
        log('   rc=%d %6.0fs  %s' % (rc, dt, s))
    if bad:
        sys.exit(1)


if __name__ == '__main__':
    main()
