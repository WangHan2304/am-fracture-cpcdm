# -*- coding: utf-8 -*-
"""重算手稿 AIC/BIC 段落所需统计量（v7 修正模型 vs Lemaitre 基线）"""
import json
import math

v7 = json.load(open(r"D:\20260618断裂模型论文\simulation\output\experiment_v7_final_results.json", encoding="utf-8"))
base = json.load(open(r"D:\20260618断裂模型论文\simulation\output\lemaitre_baseline_results.json", encoding="utf-8"))

res_mod = [abs(c['ef_pred'] - c['ef_exp']) for c in v7['cases']]
res_base = []
for m in ['Ti64', '316L', 'AlSi10Mg']:
    for o in ['0', '45', '90']:
        c = base['cases'][m][o]
        res_base.append(abs(c['ef_pred'] - c['ef_exp']))


def aic_bic(res, k):
    n = len(res)
    rss = sum(r * r for r in res)
    sig2 = rss / n
    lnL_const = n * (math.log(2 * math.pi * sig2) + 1.0)
    AIC = 2 * k + lnL_const
    BIC = k * math.log(n) + lnL_const
    rms = math.sqrt(rss / n)
    return AIC, BIC, rms, rss


for name, res, k in [("modified v7", res_mod, 9), ("Lemaitre baseline", res_base, 3)]:
    AIC, BIC, rms, rss = aic_bic(res, k)
    print(f"{name}: n={len(res)} k={k} RSS={rss:.6e} RMS={rms:.4e}")
    print(f"  AIC={AIC:.2f}  BIC={BIC:.2f}")
    print(f"  residuals: {[f'{r:.5f}' for r in res]}")

AICm, BICm, _, _ = aic_bic(res_mod, 9)
AICb, BICb, _, _ = aic_bic(res_base, 3)
print(f"\ndAIC = {AICm - AICb:.2f}   dBIC = {BICm - BICb:.2f}")
