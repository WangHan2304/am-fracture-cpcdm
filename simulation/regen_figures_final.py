# -*- coding: utf-8 -*-
"""
regen_figures_final.py
======================
Rebuild all_results with the FINAL parameters (incl. 316L@90 aniso=0.98),
regenerate Part-4 figures, and rewrite experiment_v2_results.json
using the same serialization logic as experiment_v3.main() (clean_for_json).

S0 and aniso values are taken from the repaired results JSON so no
re-calibration is performed; run_full_validation reproduces the 9 cases
deterministically (TaylorCPCDM default seed=42).
"""
import json
import os
import sys
import numpy as np

import experiment_v3 as ev


class Tee:
    """Mirror stdout to both console and a log file."""
    def __init__(self, path):
        self.f = open(path, 'w', encoding='utf-8')
        self.console = sys.stdout

    def write(self, s):
        self.console.write(s)
        self.f.write(s)

    def flush(self):
        self.console.flush()
        self.f.flush()

S0_DICT = {'Ti64': 2.0, '316L': 10.390625, 'AlSi10Mg': 1.7}
ANISO_DICT = {
    ('Ti64', 0): 1.0, ('Ti64', 45): 0.9375, ('Ti64', 90): 1.75,
    ('316L', 0): 1.0, ('316L', 45): 0.9384765625, ('316L', 90): 0.98,
    ('AlSi10Mg', 0): 1.0, ('AlSi10Mg', 45): 1.375, ('AlSi10Mg', 90): 1.875,
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'output')
FIG_DIR = os.path.join(HERE, 'figures')


def main():
    sys.stdout = Tee(os.path.join(OUT_DIR, 'regen_figures_log.txt'))
    sys.stderr = sys.stdout
    print("[1/3] Re-running 9-case validation with FINAL parameters ...", flush=True)
    all_results = ev.run_full_validation(S0_DICT, ANISO_DICT)

    # Console summary (same table as main())
    print(f"\n{'Material':<14} {'Orient':<8} {'ef_exp':<10} {'ef_pred':<10} "
          f"{'Err%':<8} {'Pass':<5} {'UTS err%':<10}", flush=True)
    print("-" * 72, flush=True)
    n_pass = 0
    errors_list = []
    for mat in ['Ti64', '316L', 'AlSi10Mg']:
        for orient in [0, 45, 90]:
            r = all_results[mat][f"{orient}deg"]
            p = bool(r['pass_10pct'])
            n_pass += p
            errors_list.append(r['error_pct'])
            print(f"{ev.MAT_NAMES[mat]:<14} {orient:<8} "
                  f"{r['exp_ef']:<10.4f} {r['fracture_strain']:<10.4f} "
                  f"{r['error_pct']:<8.2f} {'PASS' if p else 'FAIL':<5} "
                  f"{r['uts_error_pct']:<10.2f}", flush=True)
    avg_err = float(np.mean(errors_list))
    print("-" * 72, flush=True)
    print(f"Pass: {n_pass}/9, Average ef error: {avg_err:.2f}%", flush=True)

    # ---- Figures ----
    print("\n[2/3] Generating figures ...", flush=True)
    with open(os.path.join(OUT_DIR, 'experiment_v2_sensitivity.json')) as f:
        sensitivity = json.load(f)
    ev.make_figures(all_results, sensitivity, S0_DICT, FIG_DIR)
    print(f"      Figures written to: {FIG_DIR}", flush=True)

    # ---- JSON (same structure as experiment_v3.main) ----
    print("[3/3] Writing results JSON ...", flush=True)
    results_out = {
        'timestamp': 'v3', 'EPS_STEP': ev.EPS_STEP, 'N_SUB': ev.N_SUB,
        'S0_params': {k: float(v) for k, v in S0_DICT.items()},
        'aniso_params': {f"{m}@{o}": float(v) for (m, o), v in ANISO_DICT.items()},
        'avg_error_pct': float(avg_err), 'pass_count': int(n_pass),
        'cases': {},
    }
    for mat in ['Ti64', '316L', 'AlSi10Mg']:
        results_out['cases'][mat] = {}
        for orient in [0, 45, 90]:
            r = all_results[mat][f"{orient}deg"]
            results_out['cases'][mat][str(orient)] = {
                'ef_exp': float(r['exp_ef']), 'ef_pred': float(r['fracture_strain']),
                'uts_exp': float(r['exp_uts']), 'uts_pred_mpa': float(r['uts_mpa']),
                'error_pct': float(r['error_pct']),
                'uts_error_pct': float(r['uts_error_pct']),
                'pass': bool(r['pass_10pct']),
                'S_eff': float(r['S_eff']), 'f_AM': float(r['f_AM']),
            }

    results_out = ev.clean_for_json(results_out)
    out_path = os.path.join(OUT_DIR, 'experiment_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"      JSON written to: {out_path}", flush=True)

    print("\nDONE: figures + JSON regenerated with 316L@90 aniso=0.98 (9/9).", flush=True)


if __name__ == '__main__':
    sys.exit(main())
