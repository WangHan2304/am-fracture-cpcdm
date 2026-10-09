# -*- coding: utf-8 -*-
"""
statistical_upgrade_bayes.py — (2) 强化统计证据（纯事后，seed 42）
==================================================================
不新增盲数据点（无泄漏、as-built、完整取向三元组的合法上限已达）；改为把
"failure to distinguish" 升级为**肯定性推断**：

(A) 配对差异的 JZS 式贝叶斯因子 BF01（Rouder 尺度，δ/σ 先验），把
    "均值精度不可区分" 从频率派的"未拒绝"转成"支持等效"的证据；
(B) TOST 等效性检验：在明确预设的等效界值下能否**拒绝不等效**；
(C) 方向信息 Kendall τ̄ 的自助置信区间与显著性（按组自助 + 符号翻转置换检验），
    描述符模型 vs Lemaitre 基线各给 τ̄ 的 95% CI 与 p 值。

数据：output/blind_eval_expanded_results.json units（abs_err_mod_pct /
abs_err_lem_pct；及 ef_exp / ef_pred_mod / ef_pred_lem 的组内取向排序）。
输出 output/statistical_upgrade_bayes_results.json。
"""
import json
import os

import numpy as np
from scipy.stats import cauchy, kendalltau, t as student_t

OUT = os.path.join(os.path.dirname(__file__), 'output')
SEED = 42
rng = np.random.default_rng(SEED)


def bf01_paired(d, delta_prior_scale):
    """Rouder JZS one-sample Bayes factor BF01 on paired diffs d (H0: delta=0).

    H0: d_i ~ N(0, sigma^2);  H1: d_i ~ N(delta, sigma^2), delta = sigma*z,
    z ~ Cauchy(0, delta_prior_scale).  sigma carries the improper Jeffreys prior
    1/sigma.  BOTH marginals are integrated on the SAME log-sigma grid with the
    SAME 1/sigma weight and the common (2*pi)^(-n/2) factor, so all constants
    cancel in the ratio and the result is independent of the grid range.
    """
    d = np.asarray(d, float)
    n = len(d)
    ss = float(np.sum(d ** 2))
    scale0 = np.sqrt(ss / n)
    sig_grid = np.exp(np.linspace(np.log(scale0 * 1e-3), np.log(scale0 * 1e3), 600))
    dsg = np.log(sig_grid[1] / sig_grid[0])  # uniform in log sigma
    z_grid = np.linspace(-12.0, 12.0, 1201)
    # H0: L0(sigma) = sigma^-n exp(-ss/(2 sigma^2)); with prior 1/sigma and dsigma
    m0 = np.sum(np.exp(-n * np.log(sig_grid) - ss / (2.0 * sig_grid ** 2)) * (1.0 / sig_grid) * sig_grid * dsg)
    # H1: for each sigma, marginalise delta=sigma*z over the Cauchy prior
    m1 = 0.0
    for sg in sig_grid:
        resid = d[:, None] - sg * z_grid[None, :]
        rss = np.sum(resid ** 2, axis=0)
        inner = np.trapezoid(np.exp(-rss / (2.0 * sg ** 2)) * cauchy.pdf(z_grid, scale=delta_prior_scale), z_grid)
        # sigma^-n * (1/sigma) * (sigma dlog) = sigma^-n dlog
        m1 += inner * (sg ** (-n))
    m1 *= dsg
    return float(m0 / (m1 + 1e-300))  # BF01


def tost(d, margin):
    """Two-one-sided equivalence test on mean paired diff, equivalence +/- margin (same units as d)."""
    n = len(d)
    m = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    t_lo = (m - (-margin)) / se
    t_hi = (m - margin) / se
    p_lo = float(student_t.sf(t_lo, n - 1))   # H: mean <= -margin -> reject if small
    p_hi = float(student_t.cdf(t_hi, n - 1))  # H: mean >= +margin -> reject if small
    return {'mean': m, 'sd': sd, 'se': se, 'p_lower': p_lo, 'p_upper': p_hi,
            'p_tost': max(p_lo, p_hi), 'equivalence_rejected': bool(max(p_lo, p_hi) < 0.05)}


def kendall_groups(units):
    by = {}
    for u in units:
        by.setdefault((u['source_id'], u['material']), []).append(u)
    tau_mod, tau_lem = [], []
    for key, rows in sorted(by.items()):
        if len(rows) < 2:
            continue
        exp = [r['ef_exp'] for r in rows]
        pm = [r['ef_pred_mod'] for r in rows]
        pl = [r['ef_pred_lem'] for r in rows]
        tm = kendalltau(exp, pm).correlation
        tl = kendalltau(exp, pl).correlation
        if np.isfinite(tm):
            tau_mod.append(tm)
        if np.isfinite(tl):
            tau_lem.append(tl)
    return np.array(tau_mod), np.array(tau_lem)


def bootstrap_tau(taus, nboot=20000):
    """Resample groups with replacement; return mean-tau 95% CI and two-sided p vs 0 (sign-flip)."""
    k = len(taus)
    boots = np.array([taus[rng.integers(0, k, k)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # sign-flip permutation null for mean tau == 0
    perm = np.array([(taus * rng.choice([-1, 1], k)).mean() for _ in range(nboot)])
    obs = taus.mean()
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (nboot + 1)
    return float(obs), (float(lo), float(hi)), float(p)


def main():
    np.random.seed(SEED)
    with open(os.path.join(OUT, 'blind_eval_expanded_results.json'), encoding='utf-8') as f:
        units = json.load(f)['units']
    dm = np.array([u['abs_err_mod_pct'] for u in units], float)
    dl = np.array([u['abs_err_lem_pct'] for u in units], float)
    d = dm - dl  # paired difference in percentage points (n=23)
    n = len(d)

    res = {'model': 'Bayes-factor + TOST + bootstrap-Kendall upgrade (post-hoc, seed 42)',
           'n_paired': n,
           'paired_diff_pp': {'mean': round(float(d.mean()), 3), 'sd': round(float(d.std(ddof=1)), 3),
                              'se': round(float(d.std(ddof=1) / np.sqrt(n)), 3)}}

    # (A) Bayes factors for several alternative prior scales on delta/sigma
    sd_ratio = float(d.std(ddof=1) / np.sqrt(np.mean(d ** 2) + 1e-9))
    bf = {}
    for scale in (0.2, 0.35, 0.5, 1.0):
        bf01 = bf01_paired(d, scale)
        bf[f'BF01_delta_over_sigma~Cauchy(0,{scale})'] = round(bf01, 3)
    res['bayes_factor_BF01'] = bf

    # (B) TOST at the pre-registered MDE-based margins
    tost_out = {}
    for margin in (10.0, 15.0, 20.0, 25.0):
        tost_out[f'margin_{margin:.0f}pp'] = {k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in tost(d, margin).items()}
    res['tost'] = tost_out

    # (C) Kendall tau bootstrap CI + significance, both models
    tau_m, tau_l = kendall_groups(units)
    km = bootstrap_tau(tau_m)
    kl = bootstrap_tau(tau_l)
    res['kendall_descriptor'] = {'mean_tau': round(km[0], 3), 'ci95': [round(km[1][0], 3), round(km[1][1], 3)],
                                 'p_vs_0': round(km[2], 4), 'n_groups': int(len(tau_m))}
    res['kendall_lemaitre'] = {'mean_tau': round(kl[0], 3), 'ci95': [round(kl[1][0], 3), round(kl[1][1], 3)],
                               'p_vs_0': round(kl[2], 4), 'n_groups': int(len(tau_l))}
    # descriptor-minus-baseline tau difference (paired by group, same group order)
    if len(tau_m) == len(tau_l):
        dd = tau_m - tau_l
        kd = bootstrap_tau(dd)
        res['kendall_diff_mod_minus_lem'] = {'mean': round(kd[0], 3),
                                             'ci95': [round(kd[1][0], 3), round(kd[1][1], 3)],
                                             'p_vs_0': round(kd[2], 4)}

    with open(os.path.join(OUT, 'statistical_upgrade_bayes_results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print('WROTE statistical_upgrade_bayes_results.json')


if __name__ == '__main__':
    main()
