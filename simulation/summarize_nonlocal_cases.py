# -*- coding: utf-8 -*-
"""Aggregate existing 3D nonlocal/local cases into a convergence table."""
import json, glob, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, 'output', 'nonlocal_cases')
rows = []
for p in sorted(glob.glob(os.path.join(CASE, '*.json'))):
    name = os.path.basename(p)
    if name.startswith('_'):
        continue
    try:
        d = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        print('SKIP', name, e); continue
    if 'fracture_strain' not in d:
        continue
    cid = d.get('case_id', name)
    m = re.match(r'ng(\d+)', cid)
    ng = int(m.group(1)) if m else d.get('ng', None)
    rows.append(dict(
        cid=cid, ng=ng, angle=d.get('angle'), lc=d.get('lc'),
        eps=d.get('eps_step'),
        ef=d.get('fracture_strain'), ftype=d.get('fracture_type'),
        eft=d.get('ef_taylor_eps_matched'),
        bias=d.get('bias_vs_taylor_eps_matched_pct'),
        wall=d.get('wall_sec_total')))
hdr = f"{'case_id':32s} {'ng':>3s} {'ang':>4s} {'lc':>7s} {'eps':>7s} {'ef':>8s} {'efT':>8s} {'bias%':>7s} {'wall':>6s} ftype"
print(hdr); print('-'*len(hdr))
for r in sorted(rows, key=lambda x: (str(x['ng']), str(x['angle']), str(x['lc']))):
    lc = '' if r['lc'] is None else f"{r['lc']:.3f}"
    bias = '' if r['bias'] is None else f"{r['bias']:.2f}"
    print(f"{r['cid']:32s} {str(r['ng']):>3s} {str(r['angle']):>4s} {lc:>7s} "
          f"{str(r['eps']):>7s} {r['ef']:8.5f} {'' if r['eft'] is None else format(r['eft'],'.5f'):>8s} "
          f"{bias:>7s} {str(r['wall']):>6s} {r['ftype']}")
print(f"\ntotal cases = {len(rows)}")
