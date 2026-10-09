# -*- coding: utf-8 -*-
"""
rerun_tail.py — sequential tail of Stage-1 regeneration after stopping the
v11_gs_directional serial bottleneck (protocol-identical to the already-complete
authoritative v12 run). Each job below owns the whole box in turn (v14 uses 8,
v15 uses cpu_count, wiso/beta5 internal pools), so they run one at a time.

Jobs still pending that feed locked manuscript / SI numbers:
  experiment_v14_uncertainty.py  -> sec:uncertainty (LHS MC prediction intervals)
  experiment_v15_lambda_rel.py   -> relative-normalization paragraph + SI table
  wiso_scan.py                   -> w_iso sensitivity (SI)
  beta5_scan.py                  -> beta_5 marginal span (SI/factorial)

model_simplification.py is launched separately (already running) and its
consumer statistics are handled by the SI S12 invariance table.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, 'output', 'rerun_tail_log.txt')
JOBS = [
    'experiment_v14_uncertainty.py',
    'experiment_v15_lambda_rel.py',
    'wiso_scan.py',
    'beta5_scan.py',
]


def log(msg):
    line = '[%s] %s' % (time.strftime('%H:%M:%S'), msg)
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def main():
    open(LOG, 'a', encoding='utf-8').write(
        '\n==== rerun_tail %s ====\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
    for s in JOBS:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, s], cwd=HERE,
                               capture_output=True, text=True)
            rc = p.returncode
            blob = (p.stdout or '') + (('\n' + p.stderr) if p.stderr else '')
            tail = ' | '.join(blob.strip().splitlines()[-2:])
        except Exception as e:  # noqa: BLE001
            rc, tail = -999, repr(e)
        log('%-36s rc=%d  %.0fs  %s' % (s, rc, time.time() - t0, tail[:240]))
    log('==== rerun_tail DONE ====')


if __name__ == '__main__':
    main()
