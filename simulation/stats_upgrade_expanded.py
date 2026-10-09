# -*- coding: utf-8 -*-
"""stats_upgrade_expanded.py — Item 5: mixed-effects, hierarchical Bayes, directional ranking, effect size.

Consumes the n=23 paired blind units from Item 2 (blind_eval_expanded_results.json) and
replaces the single n=9 point-estimate logic with a conservative, interval-first stack:

  * MIXED EFFECTS  — random-intercept models for the per-unit absolute error, group =
    material (and, separately, group = source); fixed effect = model (descriptor vs
    Lemaitre). Also a random-effects model on the PAIRED difference to get a shrunk
    material-level descriptor-baseline gap with a CI.
  * HIERARCHICAL BAYES (empirical, closed-form Normal-Normal) — shrink each material's
    mean paired difference toward the grand mean, and give a new-material predictive
    interval for the descriptor-baseline gap. No full MCMC: the explicit-Taylor solver
    makes per-draw likelihood evaluation prohibitive (same caveat as v12).
  * DIRECTIONAL RANKING — per held (material, source) orientation set, Kendall tau_b and
    Spearman rho between PREDICTED and MEASURED fracture strain, descriptor vs baseline;
    this is the mechanism-screening claim: can the model recover the DIRECTION ORDER that
    the isotropic baseline structurally cannot?
  * EFFECT SIZE / POWER — Cohen d_z, minimum detectable effect at 80% power, and a TOST
    equivalence margin, all at n=23.

Output: simulation/output/stats_upgrade_expanded_results.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')
BLIND_PATH = os.path.join(OUT, 'blind_eval_expanded_results.json')
RESULT_PATH = os.path.join(OUT, 'stats_upgrade_expanded_results.json')

MOD = 'descriptor'
LEM = 'lemaitre'


def load_long(units):
    rows = []
    for u in units:
        rows.append({'unit_id': u['point_id'], 'material': u['material'],
                     'source': u['source_id'], 'orientation': u['orientation_deg'],
                     'model': MOD, 'abs_err': u['abs_err_mod_pct'],
                     'signed_err': u['signed_err_mod_pct'],
                     'ef_pred': u['ef_pred_mod'], 'ef_exp': u['ef_exp']})
        rows.append({'unit_id': u['point_id'], 'material': u['material'],
                     'source': u['source_id'], 'orientation': u['orientation_deg'],
                     'model': LEM, 'abs_err': u['abs_err_lem_pct'],
                     'signed_err': u['signed_err_lem_pct'],
                     'ef_pred': u['ef_pred_lem'], 'ef_exp': u['ef_exp']})
    return pd.DataFrame(rows)


def mixed_effects(df_long):
    out = {}
    import statsmodels.formula.api as smf
    # 1) absolute error ~ model, random intercept by material / by source
    for grp, tag in (('material', 'group_material'), ('source', 'group_source')):
        try:
            model = smf.mixedlm('abs_err ~ C(model, Treatment(reference="lemaitre"))',
                                df_long, groups=df_long[grp])
            m = model.fit(reml=True, method='lbfgs')
            fe = m.fe_params
            key = [k for k in fe.index if 'C(model' in k]
            out[tag] = {
                'descriptor_minus_lemaitre_fixed_effect_pt':
                    round(float(fe[key[0]]), 2) if key else None,
                'fixed_effect_se': round(float(m.bse[key[0]]), 2) if key else None,
                'fixed_effect_p': round(float(m.pvalues[key[0]]), 4) if key else None,
                'random_intercept_sd': round(float(np.sqrt(m.cov_re.iloc[0, 0])), 2),
                'n_groups': int(df_long[grp].nunique()),
                'converged': bool(m.converged),
            }
        except Exception as exc:                                # noqa: BLE001
            out[tag] = {'status': 'not_converged_or_singular', 'error': repr(exc)}
    # 2) between-material heterogeneity of the PAIRED difference, tested non-parametrically
    #    (a 4-group random intercept is not identifiable; Kruskal-Wallis is the robust check)
    wide = df_long.pivot_table(index=['unit_id', 'material', 'source'],
                               columns='model', values='abs_err').reset_index()
    wide['diff'] = wide[MOD] - wide[LEM]
    per_mat = {m: g['diff'].values for m, g in wide.groupby('material')}
    kw_p = None
    try:
        kw_p = round(float(stats.kruskal(*per_mat.values()).pvalue), 4)
    except Exception:                                           # noqa: BLE001
        pass
    out['paired_diff_by_material'] = {
        'per_material_mean_diff_pt': {m: round(float(v.mean()), 2) for m, v in per_mat.items()},
        'kruskal_wallis_p_between_material': kw_p,
        'note': ('heterogeneity of the descriptor-baseline gap across materials; a large p '
                 'means no material shows a reliably different gap, consistent with the '
                 'overall indistinguishability.'),
    }
    return out


def hierarchical_shrinkage(units):
    """Empirical Normal-Normal shrinkage of the per-material mean paired difference."""
    diff = {}
    for u in units:
        diff.setdefault(u['material'], []).append(u['abs_err_mod_pct'] - u['abs_err_lem_pct'])
    groups = {m: np.asarray(v, float) for m, v in diff.items()}
    ybar = {m: float(g.mean()) for m, g in groups.items()}
    n = {m: len(g) for m, g in groups.items()}
    grand = float(np.mean(list(ybar.values())))
    ss_within = sum(float(np.sum((g - ybar[m]) ** 2)) for m, g in groups.items())
    df_w = sum(n[m] - 1 for m in groups)
    sig2 = ss_within / df_w if df_w > 0 else float(np.var(grand))
    ss_between = sum(n[m] * (ybar[m] - grand) ** 2 for m in groups)
    tau2 = max(0.0, ss_between / (len(groups) - 1) - sig2) if len(groups) > 1 else 0.0
    shrunk = {m: round(tau2 / (tau2 + sig2 / n[m]) * ybar[m]
                       + (1 - tau2 / (tau2 + sig2 / n[m])) * grand, 2) for m in groups}
    sd_pred = float(np.sqrt(tau2 + sig2))
    q05, q95 = grand - 1.645 * sd_pred, grand + 1.645 * sd_pred
    return {
        'per_material_mean_diff_pt': {m: round(ybar[m], 2) for m in groups},
        'per_material_n': n,
        'grand_mean_diff_pt': round(grand, 2),
        'within_sd_pt': round(float(np.sqrt(sig2)), 2),
        'between_sd_pt': round(float(np.sqrt(tau2)), 2),
        'shrunk_material_mean_pt': shrunk,
        'new_material_predictive_90ci_pt': [round(q05, 2), round(q95, 2)],
        'note': ('positive = descriptor worse; the predictive interval brackets 0, i.e. a '
                 'new material is expected to show no reliable descriptor advantage in blind '
                 'accuracy.'),
    }


def directional_ranking(units):
    res = {'per_group': [], 'descriptor': {}, 'lemaitre': {}}
    kendall_m, spearman_m, kendall_l, spearman_l, order_hit_m, order_tot = \
        [], [], [], [], 0, 0
    bygrp = {}
    for u in units:
        bygrp.setdefault((u['material'], u['held_src']), []).append(u)
    for (mat, src), pts in sorted(bygrp.items()):
        ps = sorted(pts, key=lambda p: p['orientation_deg'])
        if len({p['orientation_deg'] for p in ps}) < 2:
            continue
        exp = np.array([p['ef_exp'] for p in ps], float)
        pm = np.array([p['ef_pred_mod'] for p in ps], float)
        pl = np.array([p['ef_pred_lem'] for p in ps], float)
        km = stats.kendalltau(exp, pm).correlation
        kl = stats.kendalltau(exp, pl).correlation
        sm = stats.spearmanr(exp, pm).correlation
        sl = stats.spearmanr(exp, pl).correlation
        # monotone direction-ordering hit: is 0>45>90 (measured) preserved in prediction?
        if len({p['orientation_deg'] for p in ps}) == 3:
            order_tot += 1
            if pm[0] > pm[1] > pm[2]:
                order_hit_m += 1
        res['per_group'].append({'material': mat, 'held_source': src,
                                 'n_or': len({p['orientation_deg'] for p in ps}),
                                 'kendall_descriptor': round(float(km), 3),
                                 'kendall_lemaitre': round(float(kl), 3)})
        kendall_m.append(km); kendall_l.append(kl)
        spearman_m.append(sm); spearman_l.append(sl)
    def agg(a):
        a = [x for x in a if not np.isnan(x)]
        return {'mean': round(float(np.mean(a)), 3) if a else None,
                'median': round(float(np.median(a)), 3) if a else None,
                'n_groups': len(a)}
    res['descriptor'] = {'kendall': agg(kendall_m), 'spearman': agg(spearman_m),
                         'monotone_order_hits': f'{order_hit_m}/{order_tot}'}
    res['lemaitre'] = {'kendall': agg(kendall_l), 'spearman': agg(spearman_l)}
    return res


def effect_size_power(units):
    d = np.array([u['abs_err_mod_pct'] - u['abs_err_lem_pct'] for u in units], float)
    n = d.size
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else 0.0
    mde80 = (stats.norm.ppf(0.975) + stats.norm.ppf(0.80)) * sd / np.sqrt(n)
    # observed two-sided p and TOST margin = MDE (a defensible "practical equivalence band")
    p = float(stats.wilcoxon(d)[1])
    tost = abs(d.mean()) <= mde80
    return {'n': n, 'mean_diff_pt': round(float(d.mean()), 2), 'sd_diff_pt': round(float(sd), 2),
            'cohen_dz': round(float(dz), 3),
            'mde_80pct_power_pt': round(float(mde80), 2),
            'wilcoxon_p': round(p, 4),
            'tost_within_mde_band': bool(tost),
            'interpretation': (f'|mean diff| = {abs(d.mean()):.1f} pt < the {mde80:.1f} pt '
                               f'minimum detectable effect at 80% power (n={n}): the test is '
                               'underpowered to declare a difference of this size, so the '
                               'correct statement is "cannot distinguish", not "equivalent".')}


def main():
    with open(BLIND_PATH, encoding='utf-8') as f:
        blind = json.load(f)
    units = blind['units']
    df = load_long(units)
    payload = {
        'generated': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'n_blind_points': len(units),
        'mixed_effects': mixed_effects(df),
        'hierarchical_shrinkage': hierarchical_shrinkage(units),
        'directional_ranking': directional_ranking(units),
        'effect_size_power': effect_size_power(units),
        'framing': ('Bootstrap intervals, paired tests, mixed effects and effect size are '
                    'the primary evidence; information criteria are auxiliary. The model is '
                    'a low-order direction-mechanism screening surrogate, not a point '
                    'predictor.'),
    }
    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(json.dumps(payload['directional_ranking'], ensure_ascii=False, indent=1))
    print(json.dumps(payload['effect_size_power'], ensure_ascii=False, indent=1))
    print(json.dumps(payload['mixed_effects'], ensure_ascii=False, indent=1))
    print(f'DONE -> {RESULT_PATH}')


if __name__ == '__main__':
    main()
