# -*- coding: utf-8 -*-
"""derive_numbers.py — recompute every manuscript-cited derived statistic from
the authoritative post-correction result JSONs, so the tex backfill uses exact
values instead of hand arithmetic. Read-only; prints a report."""
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), 'output')


def load(name):
    with open(os.path.join(OUT, name), encoding='utf-8') as f:
        return json.load(f)


def med(x):
    return float(np.median(x))


v12 = load('experiment_v12_fxi_transverse_results.json')
mc = v12['main_calibration']
mod = mc['modified_model']
lem = mc['lemaitre_baseline']

print('==== CALIBRATION (nine cases, all-data S0) ====')
for tag, blk in (('MOD', mod), ('LEM', lem)):
    allerr = [c['error_pct'] for m in blk['materials'].values() for c in m['cases']]
    print(f'{tag} avg={blk["avg_error_pct"]:.2f}% median={med(allerr):.2f}% '
          f'pass<=10%={sum(1 for e in allerr if e <= 10)} UTSavg={blk["avg_uts_error_pct"]:.2f}%')
    for mat, mm in blk['materials'].items():
        errs = [c['error_pct'] for c in mm['cases']]
        print(f'   {mat:9s} S0={mm["S0"]:.3f} avg={np.mean(errs):.2f}% errs={[round(e,2) for e in errs]}')

print('\n==== PREDICTED 0/90 EF RATIOS (calibration) ====')
for mat in ('Ti64', '316L', 'AlSi10Mg'):
    cs = {c['orientation']: c for c in mod['materials'][mat]['cases']}
    rmod = cs[0]['ef_pred'] / cs[90]['ef_pred']
    rexp = cs[0]['ef_exp'] / cs[90]['ef_exp']
    print(f'   {mat:9s} pred={rmod:.3f} exp={rexp:.3f} ratio_err={(rmod-rexp)/rexp*100:+.1f}%  '
          f'ef_pred 0/45/90={[round(cs[o]["ef_pred"],4) for o in (0,45,90)]}')

print('\n==== LOOCV (blind folds) ====')
for tag, key in (('MOD', 'modified_model'), ('LEM', 'lemaitre_baseline')):
    b = v12['loocv'][key]
    ef = [f['blind_ef_error_pct'] for f in b['folds']]
    uts = [f['blind_uts_error_pct'] for f in b['folds']]
    tr = [f['train_err_pct'] for f in b['folds']]
    print(f'{tag} blind_ef_avg={np.mean(ef):.2f}% median={med(ef):.2f}% pass@10={b["pass_10pct"]} '
          f'blindUTSavg={np.mean(uts):.2f}% train_avg={np.mean(tr):.2f}%')
    for mat in ('Ti64', '316L', 'AlSi10Mg'):
        fe = [f['blind_ef_error_pct'] for f in b['folds'] if f['material'] == mat]
        print(f'   {mat:9s} folds(0/45/90)={[round(x,2) for x in fe]} mean={np.mean(fe):.2f}%')

print('\n==== AIC/BIC (k=3) and k=7 bracket ====')
ab = v12['aic_bic']
print(f"MOD AIC={ab['modified_model']['AIC']:.2f} BIC={ab['modified_model']['BIC']:.2f} "
      f"RMS={ab['modified_model']['RMS']:.4f}")
print(f"LEM AIC={ab['lemaitre_baseline']['AIC']:.2f} BIC={ab['lemaitre_baseline']['BIC']:.2f} "
      f"RMS={ab['lemaitre_baseline']['RMS']:.4f}")
d3 = ab['dAIC']
n = 9
dAIC7 = d3 + 2 * (7 - 3)
dBIC7 = ab['dBIC'] + (7 - 3) * np.log(n)
print(f"dAIC(k=3)={d3:.2f} dBIC={ab['dBIC']:.2f} | dAIC(k=7)={dAIC7:.2f} dBIC(k=7)={dBIC7:.2f}")

print('\n==== n=9 PAIRED (Wilcoxon/bootstrap/MDE, normal-approx formula) ====')
pr = load('paired_test_bootstrap_results.json')
d = np.array(pr['paired_diff_pct'])
sd = d.std(ddof=1)
mde = (1.959964 + 0.841621) * sd / np.sqrt(d.size)
print(f"mean_diff={pr['mean_diff_pct']:.2f}pt sd_diff={sd:.2f} dz={pr['mean_diff_pct']/sd:.3f} "
      f"wilcoxon_p={pr['wilcoxon_signed_rank']['p_value']:.4f} "
      f"sign pos/neg={pr['sign_test']['n_positive']}/{pr['sign_test']['n_negative']} "
      f"CI=[{pr['bootstrap_100k_mean_diff_95ci']['lo']:.2f},{pr['bootstrap_100k_mean_diff_95ci']['hi']:.2f}] "
      f"MDE80={mde:.1f}pt")

print('\n==== SUB-DOMAIN split ====')
mm = v12['loocv']['modified_model']['folds']
ll = v12['loocv']['lemaitre_baseline']['folds']
for mat in ('Ti64', '316L', 'AlSi10Mg'):
    me = [f['blind_ef_error_pct'] for f in mm if f['material'] == mat]
    le = [f['blind_ef_error_pct'] for f in ll if f['material'] == mat]
    wins = sum(1 for a, b in zip(me, le) if a < b)
    print(f'   {mat:9s} mod_blind={np.mean(me):.2f}% lem_blind={np.mean(le):.2f}% mod_wins={wins}/3')
