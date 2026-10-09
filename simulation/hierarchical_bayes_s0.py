# -*- coding: utf-8 -*-
"""
hierarchical_bayes_s0.py — S0 分层贝叶斯（Normal-Normal 经验贝叶斯）
=====================================================================
观测：v12 LOOCV 每折 S0（9 个，3 材料 × 3 折）——每折 S0 是网格+细化后的真实最优。
模型：ln S0_{mk} ~ N(theta_m, sigma^2)   （折内噪声）
      theta_m ~ N(mu, tau^2)             （材料间随机效应）
目标：mu/tau/sigma 经验贝叶斯估计；材料收缩估计 theta_m；新材料 S0 后验预测分布。
声明：轻量解析分层（无全 MCMC——explicit-Taylor 求解器使全 MCMC 计算不可行）。
输出: output/hierarchical_bayes_s0_results.json
"""
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), 'output')

# v12 JSON LOOCV 每折 S0（modified model）
FOLD_S0 = {
    'Ti64': [0.32, 0.32, 0.32],
    '316L': [3.2, 3.2, 2.0],
    'AlSi10Mg': [0.32, 0.2, 0.2],
}
MATERIALS = ['Ti64', '316L', 'AlSi10Mg']


def main():
    y = {m: np.log(FOLD_S0[m]) for m in MATERIALS}
    n = {m: len(y[m]) for m in MATERIALS}

    # 组均值与 pooled 组内方差（经验贝叶斯第一步）
    theta_hat = {m: float(np.mean(y[m])) for m in MATERIALS}
    ss_within = sum(float(np.sum((y[m] - theta_hat[m]) ** 2)) for m in MATERIALS)
    df_within = sum(n[m] - 1 for m in MATERIALS)
    sigma2_hat = ss_within / df_within if df_within > 0 else 0.05 ** 2

    # 材料间方差（REML 型，扣掉采样方差）
    grand = float(np.mean([theta_hat[m] for m in MATERIALS]))
    ss_between = sum(n[m] * (theta_hat[m] - grand) ** 2 for m in MATERIALS)
    tau2_hat = max(0.0, (ss_between / (len(MATERIALS) - 1) - sigma2_hat) / 1.0)

    # 收缩因子与收缩估计
    shrink = {m: tau2_hat / (tau2_hat + sigma2_hat / n[m]) for m in MATERIALS}
    theta_shrunk = {m: shrink[m] * grand + (1 - shrink[m]) * theta_hat[m]
                    for m in MATERIALS}

    # 新材料后验预测分布（ln S0 ~ N(mu, tau^2 + sigma^2)）
    mu_pred, sd_pred = grand, float(np.sqrt(tau2_hat + sigma2_hat))
    q05, q50, q95 = (float(np.exp(mu_pred + sd_pred * z)) for z in (-1.645, 0.0, 1.645))

    # 盲测材料实际最优 S0 对照（文献/标定）
    blind_ref = {
        'IN718 mode1 (geo-mean)': 0.589445,
        'IN718 mode2 (cal on 0deg)': 6.0,
        'Ti64-2 (S0 transfer)': 0.32,
        '316L-2 (S0 transfer)': 3.2,
        'AlSi-2 (S0 transfer)': 0.20,
    }

    res = {
        'model': 'Normal-Normal hierarchical (empirical Bayes, closed-form)',
        'observations': FOLD_S0,
        'logS0_group_means': {m: round(theta_hat[m], 3) for m in MATERIALS},
        'within_group_sd': float(np.sqrt(sigma2_hat)),
        'between_group_sd': float(np.sqrt(tau2_hat)),
        'grand_mean_logS0': round(grand, 3),
        'shrinkage': {m: round(shrink[m], 3) for m in MATERIALS},
        'shrunk_theta': {m: round(theta_shrunk[m], 3) for m in MATERIALS},
        'new_material_predictive': {
            'mean_S0': float(np.exp(mu_pred + sd_pred ** 2 / 2)),
            'median_S0': float(np.exp(mu_pred)),
            '95pct_interval': [round(q05, 3), round(q95, 3)],
        },
        'blind_material_reference_S0': blind_ref,
        'note': ('S0 stability across LOOCV folds: Ti64 identical (3/3 folds), '
                 '316L 3.2->2.0 (fold-sensitive), AlSi10Mg 0.32->0.2; the '
                 'within-material spread is the Bayesian counterpart of the '
                 'cross-source non-transferability finding.'),
    }
    with open(os.path.join(OUT, 'hierarchical_bayes_s0_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
