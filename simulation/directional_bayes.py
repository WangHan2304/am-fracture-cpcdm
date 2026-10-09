# -*- coding: utf-8 -*-
"""
directional_bayes.py — "方向信息"结论的层次贝叶斯 MCMC 后验 (emcee)
====================================================================
论文 item ②：把"描述符不提升绝对精度、却恢复了基线无法给出的方向信息"从单材料
点估计升级为**跨组部分池化的层次贝叶斯后验**（每组一个方向增益 beta_g），并以确定性
方向捕获率(R^2)对照 Lemaitre 基线。纯事后分析，不调生产参数，seed 42。

数据：output/blind_eval_expanded_results.json units（每 source/material/orientation 的
ef_exp / ef_pred_mod / ef_pred_lem），仅取含完整 {0,45,90} 的 (source,material) 组。

方向对比（组内去均值=去水平/去 S0，只留方向形状）：
  y=ln ef_exp; x^P=ln ef_pred^P;  o_g(y)-mean; c^P=x^P-mean(x^P)
层次模型（对每组 beta_g 显式采样）：
  o_i = beta_{g(i)} c_i + eps_i,  eps_i ~ N(0, s^2)
  beta_g ~ N(mu, tau^2);  mu ~ N(0,3^2);  tau ~ HN(2);  s ~ HN(1)
  => 报告每组 beta_g 后验（方向增益；beta~1=方向幅值近定量恢复，beta~0=无方向信息）
参数向量 = [beta_0..beta_{ng-1}, mu, log tau, log s]（ng+3 维）。

Lemaitre 方向增益：其预测对比 c^lem ~ 0（方向盲），增益对 0 回归除方差爆炸不可辨识，
故基线的方向缺失用确定性 R^2 与对比幅度直接呈现（beta_lem 仍报告但注明不可辨识）。

输出 output/directional_bayes_results.json。
"""
import json
import os

import numpy as np
import emcee

OUT = os.path.join(os.path.dirname(__file__), 'output')
SEED = 42
rng = np.random.default_rng(SEED)


def hpd(samples, prob=0.89):
    lo, hi = np.quantile(samples, [(1 - prob) / 2, (1 + prob) / 2])
    return float(lo), float(hi)


def build_groups(units):
    by = {}
    for u in units:
        by.setdefault((u['source_id'], u['material']), {})[int(u['orientation_deg'])] = u
    groups = []
    for (src, mat), d in sorted(by.items()):
        if {0, 45, 90} <= set(d.keys()):
            rows = [{'orientation': ori,
                     'y': float(np.log(d[ori]['ef_exp'])),
                     'x_mod': float(np.log(d[ori]['ef_pred_mod'])),
                     'x_lem': float(np.log(d[ori]['ef_pred_lem']))}
                    for ori in (0, 45, 90)]
            groups.append({'source_id': src, 'material': mat, 'rows': rows})
    return groups


def contrasts(groups, xkey):
    o, c, gidx = [], [], []
    for gi, g in enumerate(groups):
        ys = np.array([r['y'] for r in g['rows']])
        xs = np.array([r[xkey] for r in g['rows']])
        o.extend(ys - ys.mean())
        c.extend(xs - xs.mean())
        gidx.extend([gi] * len(ys))
    return np.array(o), np.array(c), np.array(gidx)


def make_logpost(o, c, gidx, ng):
    def logpost(th):
        beta = th[:ng]
        mu, log_tau, log_s = th[ng], th[ng + 1], th[ng + 2]
        tau, s = np.exp(log_tau), np.exp(log_s)
        if not (1e-6 < tau < 50 and 1e-6 < s < 50):
            return -np.inf
        bg = beta[gidx]
        ll = float(np.sum(-0.5 * (o - bg * c) ** 2 / s ** 2 - np.log(s)))
        lp = float(np.sum(-0.5 * (beta - mu) ** 2 / tau ** 2 - log_tau))  # beta_g~N(mu,tau^2)
        lp += -0.5 * (mu / 3.0) ** 2
        lp += -0.5 * (tau / 2.0) ** 2 + log_tau
        lp += -0.5 * (s / 1.0) ** 2 + log_s
        return ll + lp
    return logpost


def run_hier(o, c, gidx, ng):
    logpost = make_logpost(o, c, gidx, ng)
    slopes = [np.dot(o[gidx == gi], c[gidx == gi]) /
              (np.dot(c[gidx == gi], c[gidx == gi]) + 1e-6) for gi in range(ng)]
    mu0 = float(np.mean(np.clip(slopes, -3, 3)))
    ndim = ng + 3
    nwalk, nstep, nburn = 64, 12000, 3000
    p0 = np.concatenate([np.clip(slopes, -3, 3), [mu0, np.log(0.8), np.log(0.25)]])
    pos = p0 + 0.1 * rng.standard_normal((nwalk, ndim))
    pos[:, ng + 1:] = np.clip(pos[:, ng + 1:], np.log(1e-3), np.log(1.0))
    sampler = emcee.EnsembleSampler(nwalk, ndim, logpost)
    sampler.run_mcmc(pos, nstep, progress=False)
    chain = sampler.get_chain(discard=nburn, thin=15, flat=True)
    return chain, sampler.acceptance_fraction.mean()


def main():
    np.random.seed(SEED)
    with open(os.path.join(OUT, 'blind_eval_expanded_results.json'), encoding='utf-8') as f:
        data = json.load(f)
    groups = build_groups(data['units'])
    ng = len(groups)
    print(f'complete triples: {ng} groups x 3 = {ng*3} obs')
    labels = [f"{g['material']}·{g['source_id']}" for g in groups]

    res = {'model': 'Hierarchical Bayesian per-group directional-gain MCMC (emcee), seed 42',
           'structure': 'o_i=beta_g c_i+e; beta_g~N(mu,tau^2); mu~N(0,3), tau~HN(2), s~HN(1)',
           'data': 'blind_eval_expanded_results.json complete 0/45/90 triples',
           'n_groups': ng, 'n_obs': ng * 3, 'groups': labels}

    block = {}
    for xkey, tag in (('x_mod', 'modified'), ('x_lem', 'lemaitre')):
        o, c, gidx = contrasts(groups, xkey)
        chain, acc = run_hier(o, c, gidx, ng)
        per = []
        for gi in range(ng):
            b = chain[:, gi]
            lo, hi = hpd(b, 0.89)
            per.append({'group': labels[gi],
                        'beta_median': round(float(np.median(b)), 3),
                        'beta_hpd89': [round(lo, 3), round(hi, 3)],
                        'P(beta>0.5)': round(float(np.mean(b > 0.5)), 3)})
        mu = chain[:, ng]
        mlo, mhi = hpd(mu, 0.89)
        # 确定性方向捕获率 R^2（c=0 => R^2=0）
        cap = []
        for gi in range(ng):
            m = gidx == gi
            ss_o = float(np.sum(o[m] ** 2))
            r2 = 1.0 - float(np.sum((o[m] - c[m]) ** 2)) / ss_o if ss_o > 0 else float('nan')
            cap.append({'group': labels[gi], 'R2_direction': round(r2, 3),
                        'contrast_sd_model': round(float(np.std(c[m])), 3),
                        'contrast_sd_obs': round(float(np.std(o[m])), 3)})
        block[tag] = {
            'per_group_beta': per,
            'pooled_mu_median': round(float(np.median(mu)), 3),
            'pooled_mu_hpd89': [round(mlo, 3), round(mhi, 3)],
            'tau_median': round(float(np.median(np.exp(chain[:, ng + 1]))), 3),
            'acceptance_frac': round(float(acc), 3),
            'deterministic_capture': cap,
        }

    res['hierarchical'] = block
    res['lemaitre_note'] = ('Lemaitre per-group beta is numerically unstable because its '
                            'predicted within-group contrast is ~0 (direction-blind): its '
                            'R2_direction ~ 0 and contrast_sd_model << contrast_sd_obs.')

    # headline Ti64 0/90 ratio (calibration-matched v12 source)
    for g in groups:
        if g['material'] == 'Ti64' and 'v12' in g['source_id']:
            r = {x['orientation']: x for x in g['rows']}
            res['ti64_v12_ratio_0_90'] = {
                'exp': round(float(np.exp(r[0]['y'] - r[90]['y'])), 3),
                'modified': round(float(np.exp(r[0]['x_mod'] - r[90]['x_mod'])), 3),
                'lemaitre': round(float(np.exp(r[0]['x_lem'] - r[90]['x_lem'])), 3)}
            break

    # 绝对水平(log-ef) RMSE：精度不可区分
    ya = np.concatenate([np.array([x['y'] for x in g['rows']]) for g in groups])
    xm = np.concatenate([np.array([x['x_mod'] for x in g['rows']]) for g in groups])
    xl = np.concatenate([np.array([x['x_lem'] for x in g['rows']]) for g in groups])
    res['level_rmse_log'] = {'modified': round(float(np.sqrt(np.mean((ya - xm) ** 2))), 4),
                             'lemaitre': round(float(np.sqrt(np.mean((ya - xl) ** 2))), 4)}

    with open(os.path.join(OUT, 'directional_bayes_results.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    print('--- MODIFIED per-group beta ---')
    for row in block['modified']['per_group_beta']:
        print('  ', row)
    print('  pooled mu:', block['modified']['pooled_mu_median'], block['modified']['pooled_mu_hpd89'])
    print('--- LEMAITRE capture (R2, model vs obs contrast sd) ---')
    for row in block['lemaitre']['deterministic_capture']:
        print('  ', row)
    print('MOD capture:')
    for row in block['modified']['deterministic_capture']:
        print('  ', row)
    print('ti64_v12_ratio:', res.get('ti64_v12_ratio_0_90'))
    print('level_rmse:', res['level_rmse_log'])
    print('WROTE directional_bayes_results.json')


if __name__ == '__main__':
    main()
