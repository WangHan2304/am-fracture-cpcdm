"""分层贝叶斯标定（bayes_calibration.py）
===========================================
用 Goodman--Weare 仿射不变集合 MCMC（emcee）对修正 CDM 模型的关键损伤参数
做分层贝叶斯标定，输出参数后验与断裂应变的后验预测区间。

统计模型（层级结构）
---------------------
材料层（每种材料 m）：
    log S0_m  ~ N(log S0_hat_m, 0.4)        S0_hat = 确定性二分标定值（弱信息先验）
    s_m       ~ N(mu_s, sigma_s)            损伤指数，材料间部分汇集
    log a45_m, log a90_m ~ N(0, tau_a)      各向异性乘数，跨材料共享尺度
    log sigma_m ~ log HalfNormal(0, 0.05*ef_bar_m)   观测/数值噪声
超参数层：
    mu_s    ~ N(1.2, 0.6)
    sigma_s ~ HalfNormal(0.5)
    tau_a   ~ HalfNormal(0.5)

似然：
    ef_obs[case] ~ N(ef_emulator(theta; case), sigma_m)

数值一致性（勿改）：所有仿真评估的 max_strain 公式与 experiment_v3.py Part 2
一致，保证标定网格与验证网格严格同构。斜向网格为容纳 aniso<1 的延迟断裂，对
该取向统一放宽 max_strain（同一 (material,orientation) 网格内所有点共用）。

用法：
    python bayes_calibration.py grid --material Ti64   # 构建某材料插值网格
    python bayes_calibration.py grid --material all    # 构建全部材料网格（可后台）
    python bayes_calibration.py sample                 # 组装 emulator + MCMC + 出图

输出：
    output/bayes/grid_<mat>.json                     仿真网格（带缓存，可增量续跑）
    output/bayes_calibration_results.json            后验汇总 + 后验预测区间
    figures/Fig_bayes_corner_<mat>.png                每材料参数后验角图
    figures/Fig_bayes_ppc.png                         后验预测 vs 观测
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import emcee

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
OUT = os.path.join(HERE, 'output')
FIG = os.path.join(HERE, 'figures')
BAYES_DIR = os.path.join(OUT, 'bayes')
os.makedirs(BAYES_DIR, exist_ok=True)

from taylor_cpcdm import TaylorCPCDM                      # noqa: E402
from materials import EXPERIMENTAL                          # noqa: E402

MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]
N_GRAINS, EPS_STEP, N_SUB = 20, 5e-4, 30                    # 与 experiment_v3.py 一致
SEED = 42

# 网格设计
S0_RATIOS = [0.5, 0.63, 0.79, 1.0, 1.26, 1.58, 2.0]         # S0 相对中心的比例
S_GRID = [0.5, 0.875, 1.25, 1.625, 2.0]                     # 损伤指数 s
ANCHOR_RATIOS = [0.79, 1.0, 1.26]                           # S0 锚点（⊂ S0_RATIOS）
ANCHOR_S = [0.875, 1.625]                                   # s 锚点（⊂ S_GRID）
A_GRID = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]               # 各向异性乘数
A_MIN = min(A_GRID)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def evaluate(mat, orient, S0, s, aniso=None):
    """单次确定性仿真（seed 固定 → 参数组合唯一结果）。

    数值一致性：max_strain 公式与 experiment_v3 Part 2 相同（ef_exp*2+0.01）；
    斜向网格统一放宽到 ef_exp*2/A_MIN+0.01，同一 (mat,orient) 内保持一致。
    """
    ef_exp = EXPERIMENTAL[mat]['ef'][orient]
    if orient == 0:
        max_strain = ef_exp * 2.0 + 0.01
    else:
        max_strain = ef_exp * 2.0 / A_MIN + 0.01

    overrides = ()
    if s is not None:
        overrides = {'s_damage': s}
    model = TaylorCPCDM(mat, orient, S0_override=S0, n_grains=N_GRAINS,
                        eps_step=EPS_STEP, n_sub=N_SUB, aniso_override=aniso,
                        seed=SEED, param_overrides=overrides)
    return model.run_uniaxial(max_strain)['fracture_strain']


def s0_center(mat):
    """从正典结果 JSON 读取 S0 标定中心。"""
    path = os.path.join(OUT, 'experiment_v2_results.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        return float(d['S0_params'][mat])
    # 回退：无结果文件时用经验值（应避免）
    return {'Ti64': 2.0, '316L': 11.0, 'AlSi10Mg': 1.7}[mat]


def build_grid(mat):
    """构建某材料插值网格，缓存到 JSON（增量续跑）。"""
    c = s0_center(mat)
    ef0, c0 = EXPERIMENTAL[mat]['ef'][0], None
    path = os.path.join(BAYES_DIR, f'grid_{mat}.json')

    work = []                 # (kind, orient, S0_ratio, s, aniso)
    for r in S0_RATIOS:
        for s in S_GRID:
            work.append(('zero', 0, r, s, None))
    for orient in (45, 90):
        for r in ANCHOR_RATIOS:
            for s in ANCHOR_S:
                for a in A_GRID:
                    work.append(('oblique', orient, r, s, a))

    results = {}
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            results = json.load(f).get('points', {})

    idx = 0
    total = len(work)
    for kind, orient, r, s, a in work:
        key = f"{kind}|{orient}|{r}|{s}|{a}"
        if key in results:
            continue
        S0 = c * r
        ef = evaluate(mat, orient, S0, s, a)
        results[key] = ef
        idx += 1
        if idx % 10 == 0 or idx == total:
            print(f"  {mat}: {idx}/{total}")
            _dump_results(path, mat, c, results)

    _dump_results(path, mat, c, results)
    print(f"完成 {mat}: {path} ({total} 个网格点)")
    return np.nan


def _dump_results(path, mat, center, results):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'material': mat, 'S0_center': center,
                   'points': results}, f, indent=2)


class GridModel:
    """每材料 3 取向插值模型（0°: (S0,s) 二维；45/90°: (S0,s,a) 锚点+线性）。"""

    def __init__(self, mat):
        path = os.path.join(BAYES_DIR, f'grid_{mat}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"缺少网格 {path}，请先运行 grid --material {mat}")
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.mat = mat
        self.center = float(raw['S0_center'])
        self.points = raw['points']
        # 预取 0° 网格
        zero = np.zeros((len(S0_RATIOS), len(S_GRID)))
        for j, r in enumerate(S0_RATIOS):
            for k, s in enumerate(S_GRID):
                key = f"zero|0|{r}|{s}|None"
                zero[j, k] = self.points[key]
        self.zero = zero
        # 预取斜向锚点曲线
        oblique = {}
        for orient in (45, 90):
            arr = np.zeros((len(ANCHOR_RATIOS), len(ANCHOR_S), len(A_GRID)))
            for jr, r in enumerate(ANCHOR_RATIOS):
                for js, s in enumerate(ANCHOR_S):
                    for ja, a in enumerate(A_GRID):
                        key = f"oblique|{orient}|{r}|{s}|{a}"
                        arr[jr, js, ja] = self.points[key]
            oblique[orient] = arr
        self.oblique = oblique

    def _interp2d(self, grid_x, grid_y, values, x, y):
        return _interp_axis(_interp_axis(values, 0, np.searchsorted(grid_x, x)),
                            0, np.searchsorted(grid_y, y)) if False else \
            _bilinear(grid_x, grid_y, values, x, y)

    def ef(self, logS0, s, log_a45, log_a90):
        S0 = np.exp(logS0)
        r = S0 / self.center
        ef0 = _bilinear(np.log(S0_RATIOS), S_GRID, self.zero, np.log(r), s)
        return [ef0,
                self._oblique(45, r, s, np.exp(log_a45)),
                self._oblique(90, r, s, np.exp(log_a90))]

    def _oblique(self, orient, r, s, a):
        logr = np.log(r)
        return _trilinear(np.log(ANCHOR_RATIOS), ANCHOR_S, np.log(A_GRID),
                          self.oblique[orient], logr, s, np.log(a))


def _bilinear(gr, gs, values, r, s):
    ir = int(np.searchsorted(gr, r, side='right') - 1)
    js = int(np.searchsorted(gs, s, side='right') - 1)
    ir = np.clip(ir, 0, len(gr) - 2)
    js = np.clip(js, 0, len(gs) - 2)
    tr = (r - gr[ir]) / (gr[ir + 1] - gr[ir])
    ts = (s - gs[js]) / (gs[js + 1] - gs[js])
    v00 = values[ir, js]
    v10 = values[ir + 1, js]
    v01 = values[ir, js + 1]
    v11 = values[ir + 1, js + 1]
    return (v00 * (1 - tr) * (1 - ts) + v10 * tr * (1 - ts) +
            v01 * (1 - tr) * ts + v11 * tr * ts)


def _trilinear(gr, gs, ga, values, r, s, a):
    ir = int(np.clip(np.searchsorted(gr, r, side='right') - 1, 0, len(gr) - 2))
    js = int(np.clip(np.searchsorted(gs, s, side='right') - 1, 0, len(gs) - 2))
    ia = int(np.clip(np.searchsorted(ga, a, side='right') - 1, 0, len(ga) - 2))
    tr = (r - gr[ir]) / (gr[ir + 1] - gr[ir])
    ts = (s - gs[js]) / (gs[js + 1] - gs[js])
    ta = (a - ga[ia]) / (ga[ia + 1] - ga[ia])
    v = 0.0
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                w = (tr if di else 1 - tr) * (ts if dj else 1 - ts) * \
                    (ta if dk else 1 - ta)
                v += w * values[ir + di, js + dj, ia + dk]
    return v


class HierarchicalModel:
    """分层贝叶斯模型（多材料 + 超参数），供 emcee 采样。"""

    NDIM = 3 * 4 + 3 + 3          # 材料层 12 + 噪声 3 + 超参数 3

    def __init__(self, grids):
        self.grids = grids
        self.mats = list(grids.keys())
        # 每材料观测量
        self.obs = []
        self.case = []
        for m in self.mats:
            for o in ORIENTS:
                self.obs.append(EXPERIMENTAL[m]['ef'][o])
                self.case.append((m, o))
        # 先验中心
        self.s0_hat = {m: g.center for m, g in grids.items()}
        self.ef_bar = {m: float(np.mean(
            [EXPERIMENTAL[m]['ef'][o] for o in ORIENTS])) for m in self.mats}

    def unpack(self, th):
        p = {}
        k = 0
        for m in self.mats:
            p[m] = {'logS0': th[k], 's': th[k + 1],
                    'log_a45': th[k + 2], 'log_a90': th[k + 3]}
            k += 4
        p['log_sig'] = th[k:k + 3]
        p['mu_s'], p['sig_s'], p['tau_a'] = th[k + 3:k + 6]
        return p

    def log_prior(self, th):
        p = self.unpack(th)
        lp = 0.0
        # 超参数先验
        lp += -0.5 * ((p['mu_s'] - 1.2) / 0.6) ** 2
        lp += _log_halfnorm(p['sig_s'], 0.5)
        lp += _log_halfnorm(p['tau_a'], 0.5)
        # 材料层
        for m in self.mats:
            pm = p[m]
            lp += -0.5 * ((pm['logS0'] - np.log(self.s0_hat[m])) / 0.4) ** 2
            lp += -0.5 * (((pm['s'] - p['mu_s']) / p['sig_s']) ** 2) - \
                np.log(p['sig_s'])
            lp += -0.5 * (pm['log_a45'] / p['tau_a']) ** 2 - np.log(p['tau_a'])
            lp += -0.5 * (pm['log_a90'] / p['tau_a']) ** 2 - np.log(p['tau_a'])
            if pm['s'] < 0.1:                     # s 物理正约束
                return -np.inf
        for m, lsig in zip(self.mats, p['log_sig']):
            sig = np.exp(lsig)
            lp += _log_halfnorm(sig, 0.05 * self.ef_bar[m]) + lsig
        return lp

    def log_like(self, th):
        """仅似然项（不含先验）。"""
        p = self.unpack(th)
        tot = 0.0
        for (m, o), y in zip(self.case, self.obs):
            pm = p[m]
            ef = self.grids[m].ef(pm['logS0'], pm['s'],
                                  pm['log_a45'], pm['log_a90'])
            sig = np.exp(p['log_sig'][self.mats.index(m)])
            tot += -0.5 * (((y - ef[ORIENTS.index(o)]) / sig) ** 2) - \
                np.log(sig)
        return tot

    def log_prob(self, th):
        """后验对数密度（emcee 目标）。"""
        lp = self.log_prior(th)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_like(th)


def _log_halfnorm(x, scale):
    if x < 0:
        return -np.inf
    return -0.5 * (x / scale) ** 2 - np.log(scale) - 0.5 * np.log(np.pi / 2)


def sample():
    """组装 emulator，运行 emcee，保存结果与图。"""
    grids = {m: GridModel(m) for m in MATERIALS}
    model = HierarchicalModel(grids)

    nwalkers, nburn, nsteps = 40, 400, 900
    ndim = model.NDIM
    rng = np.random.default_rng(SEED)
    p0 = _init_walkers(model, nwalkers, ndim, rng)

    np.seterr(divide='ignore', invalid='ignore')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, model.log_prob)
    state = sampler.run_mcmc(p0, nburn, progress=True)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=True)
    chain = sampler.get_chain(flat=True)

    # 汇总后验 + 后验预测
    results = _summarize(model, chain)
    _save_and_plot(model, chain, results)
    return results


def _init_walkers(model, nwalkers, ndim, rng):
    """围绕先验中心向量化初始化 walker 位置（保证各列随机独立）。"""
    p0 = np.zeros((nwalkers, ndim))
    scale = dict()
    center = dict()
    k = 0
    for m in MATERIALS:
        center[k], scale[k] = np.log(model.s0_hat[m]), 0.12
        center[k + 1], scale[k + 1] = 1.2, 0.40
        center[k + 2], scale[k + 2] = 0.0, 0.30
        center[k + 3], scale[k + 3] = 0.0, 0.30
        k += 4
    # log sigma（三材料观测噪声）
    for j, m in enumerate(MATERIALS):
        center[k + j] = np.log(0.05 * model.ef_bar[m])
        scale[k + j] = 0.60
    # 超参数
    center[k + 3], scale[k + 3] = 1.2, 0.60
    center[k + 4], scale[k + 4] = 0.5, 0.40
    center[k + 5], scale[k + 5] = 0.5, 0.40

    for i in range(ndim):
        p0[:, i] = rng.normal(center[i], scale[i], size=nwalkers)
    # 额外 jitter，彻底消除任何列间共线
    p0 += rng.normal(0, 1e-3, size=(nwalkers, ndim))
    return p0


def _summarize(model, chain):
    """后验分位数 + 后验预测区间。"""
    lo, med, hi = np.quantile(chain, [0.025, 0.5, 0.975], axis=0)
    params = {}
    k = 0
    for m in MATERIALS:
        params[m] = {
            'S0_median': float(np.exp(med[k])),
            'S0_95CI': [float(np.exp(lo[k])), float(np.exp(hi[k]))],
            's_median': float(med[k + 1]),
            's_95CI': [float(lo[k + 1]), float(hi[k + 1])],
            'a45_median': float(np.exp(med[k + 2])),
            'a45_95CI': [float(np.exp(lo[k + 2])), float(np.exp(hi[k + 2]))],
            'a90_median': float(np.exp(med[k + 3])),
            'a90_95CI': [float(np.exp(lo[k + 3])), float(np.exp(hi[k + 3]))],
        }
        k += 4
    hyper = {
        'mu_s': [float(lo[k + 3]), float(med[k + 3]), float(hi[k + 3])],
        'sigma_s': [float(lo[k + 4]), float(med[k + 4]), float(hi[k + 4])],
        'tau_a': [float(lo[k + 5]), float(med[k + 5]), float(hi[k + 5])],
    }

    pred = {}
    idxs = np.random.default_rng(SEED).integers(0, len(chain), size=2000)
    for (m, o), y in zip(model.case, model.obs):
        vals = []
        for i in idxs:
            p = model.unpack(chain[i])
            ef = model.grids[m].ef(p[m]['logS0'], p[m]['s'],
                                   p[m]['log_a45'], p[m]['log_a90'])
            vals.append(ef[ORIENTS.index(o)])
        vals = np.asarray(vals)
        pred[f"{m}_{o}"] = {
            'obs': y,
            'median': float(np.median(vals)),
            '95CI': [float(np.quantile(vals, 0.025)),
                     float(np.quantile(vals, 0.975))],
        }

    return {'params': params, 'hyper': hyper, 'predictive': pred,
            'sampler': 'emcee',
            'n_samples': int(len(chain))}


def _save_and_plot(model, chain, results):
    """保存 JSON + 每材料角图 + 后验预测图。"""
    out = os.path.join(OUT, 'bayes_calibration_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    for m in MATERIALS:
        g = model.grids[m]
        idx = MATERIALS.index(m)
        cols = chain[:, idx * 4:idx * 4 + 4]
        _corner(cols, [r'$\log S_0$', r'$s$',
                       r'$\log a_{45}$', r'$\log a_{90}$'],
                f'Fig_bayes_corner_{m}.png')
    _ppc(results['predictive'], 'Fig_bayes_ppc.png')


def _corner(cols, names, fname):
    d = cols.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(2 * d, 2 * d))
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
            elif i == j:
                ax.hist(cols[:, i], bins=30, density=True, alpha=0.8)
                ax.set_yticks([])
            else:
                ax.hist2d(cols[:, j], cols[:, i], bins=40,
                          cmap='viridis')
                ax.set_facecolor('white')
            if j == 0:
                ax.set_ylabel(names[i])
            if i == d - 1:
                ax.set_xlabel(names[j])
            if i != j and j < i:
                ax.set_xticks([])
                ax.set_yticks([])
    fig.suptitle(fname.split('_')[-2])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def _ppc(pred, fname):
    labels = list(pred.keys())
    obs = [pred[k]['obs'] for k in labels]
    med = np.asarray([pred[k]['median'] for k in labels])
    ci = np.asarray([pred[k]['95CI'] for k in labels])
    err = np.abs(ci - med[:, None]).T
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.errorbar(x, med, yerr=err, fmt='o', capsize=3, color='#457B9D',
                label='Posterior predictive median (95% CI)')
    ax.plot(x, obs, 'x', color='#E63946', ms=8, label='Observed')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Fracture strain')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    g = sub.add_parser('grid', help='构建仿真插值网格')
    g.add_argument('--material', default='all',
                   choices=MATERIALS + ['all'])
    sub.add_parser('sample', help='MCMC 采样 + 出图')
    args = ap.parse_args()

    if args.cmd == 'grid':
        mats = MATERIALS if args.material == 'all' else [args.material]
        for m in mats:
            build_grid(m)
    elif args.cmd == 'sample':
        sample()
    else:
        print("用法: bayes_calibration.py {grid --material Ti64|sample}")


if __name__ == '__main__':
    main()