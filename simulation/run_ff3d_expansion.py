# -*- coding: utf-8 -*-
"""
R13 3D CPFEM expansion runner.
Runs Ti64 45deg, Ti64 90deg, AlSi10Mg 0deg sequentially.
Ti64 0deg already completed (ef_ff=0.1389, bias +24.3%).
Saves to fullfield_3d_cpfem_expansion_results.json (does NOT overwrite existing).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from fullfield_3d_cpfem import run_case, OUT

CASES = [('Ti64', 45), ('Ti64', 90), ('AlSi10Mg', 0)]
OUT_PATH = os.path.join(OUT, 'fullfield_3d_cpfem_expansion_results.json')
LOG_PATH = os.path.join(OUT, 'ff3d_expansion_log.txt')

def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def main():
    log('=== R13 3D CPFEM expansion start ===')
    results = []
    # Load existing if resuming
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, 'r', encoding='utf-8') as f:
            results = json.load(f)
        done = {(r['material'], r['orientation']) for r in results}
        log(f'Resuming: {len(results)} cases already done: {done}')
    else:
        done = set()

    for mat, psi in CASES:
        if (mat, psi) in done:
            log(f'SKIP {mat} {psi}deg (already done)')
            continue
        log(f'--- START {mat} {psi}deg ---')
        t0 = time.time()
        try:
            r = run_case(mat, psi, verbose=True)
            r['wall_clock_total_sec'] = round(time.time() - t0, 1)
            results.append(r)
            log(f'DONE {mat} {psi}deg: ef_ff={r["ef_fullfield"]} '
                f'ef_taylor={r["ef_taylor"]} bias={r["bias_pct"]}% '
                f'({r["wall_clock_total_sec"]}s)')
        except Exception as e:
            log(f'FAILED {mat} {psi}deg: {type(e).__name__}: {e}')
            results.append({'material': mat, 'orientation': psi,
                            'error': f'{type(e).__name__}: {e}',
                            'wall_clock_total_sec': round(time.time() - t0, 1)})
        # Save incrementally
        with open(OUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    log('=== R13 3D CPFEM expansion complete ===')
    log(json.dumps(results, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
