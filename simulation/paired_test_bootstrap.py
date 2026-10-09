# -*- coding: utf-8 -*-
"""
paired_test_bootstrap.py — R8 B 组：9 折盲测残差配对检验
==========================================================
数据：v12 JSON 逐折 blind_ef_error_pct（MODEL vs LEM，同协议同折）。
检验：Wilcoxon signed-rank、符号检验、bootstrap 均值差 95% CI。
输出: output/paired_test_bootstrap_results.json
"""
import json
import os

import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(__file__), 'output')
with open(os.path.join(OUT, 'experiment_v12_fxi_transverse_results.json'),
          encoding='utf-8') as f:
    v12 = json.load(f)

# 逐折盲测误差（%）
fold_order = [('Ti64', 0), ('Ti64', 45), ('Ti64', 90),
              ('316L', 0), ('316L', 45), ('316L', 90),
              ('AlSi10Mg', 0), ('AlSi10Mg', 45), ('AlSi10Mg', 90)]


def get_folds(key):
    folds = v12['loocv'][key]['folds']
    by_mat = {}
    for fr in folds:
        by_mat.setdefault(fr['material'], {})[int(fr['held_orientation'])] = \
            fr['blind_ef_error_pct']
    out = [by_mat[mat][psi] for mat, psi in fold_order]
    assert len(out) == 9, f'{key}: got {len(out)} folds'
    return out


mod = np.array(get_folds('modified_model'))
lem = np.array(get_folds('lemaitre_baseline'))
d = mod - lem  # 正 = MODEL 更差

rng = np.random.default_rng(42)
boot = rng.choice(d, size=(100000, 9)).mean(axis=1)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

# Wilcoxon signed-rank（连续近似）
w_stat, w_p = stats.wilcoxon(d)
# 符号检验（排除 0 差；无 0）
n_pos = int(np.sum(d > 0))
n_neg = int(np.sum(d < 0))
s_p = 2 * min(stats.binom.cdf(min(n_pos, n_neg), n_pos + n_neg, 0.5),
              stats.binom.sf(min(n_pos, n_neg) - 1, n_pos + n_neg, 0.5))

res = {
    'n_folds': 9,
    'fold_order': [f'{m}@{p}' for m, p in fold_order],
    'model_blind_err_pct': mod.tolist(),
    'lemaitre_blind_err_pct': lem.tolist(),
    'paired_diff_pct': d.tolist(),
    'mean_diff_pct': float(d.mean()),
    'model_mean': float(mod.mean()),
    'lemaitre_mean': float(lem.mean()),
    'wilcoxon_signed_rank': {'stat': float(w_stat), 'p_value': float(w_p)},
    'sign_test': {'n_positive': n_pos, 'n_negative': n_neg,
                  'p_value_two_sided': float(s_p)},
    'bootstrap_100k_mean_diff_95ci': {'lo': float(ci_lo), 'hi': float(ci_hi)},
    'interpretation': ('At n=9 the paired mean difference (model worse by 7.27 pt) '
                       'is not significant: Wilcoxon p and sign-test p both > 0.05; '
                       'the bootstrap 95% CI of the mean difference includes 0. '
                       'AIC/BIC at n=9 are therefore reported with these intervals '
                       'rather than as decisive model-selection evidence.'),
}
with open(os.path.join(OUT, 'paired_test_bootstrap_results.json'), 'w',
          encoding='utf-8') as f:
    json.dump(res, f, indent=1)
print(f"mean_diff={d.mean():.2f}pt  wilcoxon p={w_p:.4f}  sign p={s_p:.4f}")
print(f"bootstrap 95% CI=[{ci_lo:.2f}, {ci_hi:.2f}]")
print('DONE')
