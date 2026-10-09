"""分层贝叶斯标定 v7（bayes_calibration_v7.py）
=====================================================
在 experiment_v7_final_results.json 全局参数基础上重建贝叶斯网格与后验，
保证"参数表、贝叶斯后验、收敛性、LOOCV"使用同一套最终参数（v7 正典）：

- 模型结构：v4（S≡S0，f_AM 含物理投影 + 分段 aniso 乘数），与主模型一致；
- S0 网格中心：v7 全局标定值（Ti64 0.7838 / 316L 3.1544 / AlSi10Mg 0.1848）；
- Voce 饱和应力：v7 gs_mult 覆盖（Ti64 2.1996 / 316L 1.2987 / AlSi10Mg 2.1996）；
- aniso 网格：对数均匀扩展至 [0.4, 3.4]，覆盖 v7 的 a45/a90（Ti64 a90=2.740）；
- max_strain：与 v7 主模型一致（316L: ef*1.6+0.02，其余: ef*2.0+0.01）。

输出：output/bayes_v7/grid_<mat>.json + output/bayes_calibration_v7_results.json
用法：
    python bayes_calibration_v7.py grid --material all
    python bayes_calibration_v7.py sample
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
BAYES_DIR = os.path.join(OUT, 'bayes_v7')
os.makedirs(BAYES_DIR, exist_ok=True)

from taylor_cpcdm import TaylorCPCDM                      # noqa: E402
from materials import EXPERIMENTAL                          # noqa: E402

MATERIALS = ['Ti64', '316L', 'AlSi10Mg']
ORIENTS = [0, 45, 90]
N_GRAINS, EPS_STEP, N_SUB = 20, 5e-4, 30
SEED = 42

# v7 全局参数（来自 experiment_v7_final_results.json，硬编码为权威值）
V7_S0 = {'Ti64': 0.7838, '316L': 3.1544, 'AlSi10Mg': 0.1848}
V7_GS = {'Ti64': 2.1996, '316L': 1.2987, 'AlSi10Mg': 2.1996}
V7_A45 = {'Ti64': 1.2455, '316L': 1.0483, 'AlSi10Mg': 0.5584}
V7_A90 = {'Ti64': 2.7402, '316L': 1.2310, 'AlSi10Mg': 0.5630}

# 网格设计（对数均匀覆盖 v7 a 值）
S0_RATIOS = [0.5, 0.63, 0.79, 1.0, 1.26, 1.58, 2.0]
S_GRID = [0.5, 0.875, 1.25, 1.625, 2.0]
ANCHOR_RATIOS = [0.79, 1.0, 1.26]
ANCHOR_S = [0.875, 1.625]
A_GRID = [0.4, 0.56, 0.8, 1.12, 1.6, 2.24, 3.2]           # 对数均匀，覆盖 0.4–3.2
A_MIN = min(A_GRID)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def _gs_override(mat, gs_mult):
    if mat == 'Ti64':
        return {'gs_basal': 700e6 * gs_mult,
                'gs_prism': 720e6 * gs_mult,
                'gs_pyr': 800e6 * gs_mult}
    return {'gs': {'316L': 500e6, 'AlSi10Mg': 280e6}[mat] * gs_mult}


def evaluate(mat, orient, S0, s, aniso=None):
    """单次确定性仿真（v4 模型 + v7 gs 覆盖），max_strain 与 v7 主模型一致。"""
    ef_exp = EXPERIMENTAL[mat]['ef'][orient]
    if mat == '316L':
        max_strain = ef_exp * 1.6 + 0.02
    else:
        max_strain = ef_exp * 2.0 + 0.01

    overrides = _gs_override(mat, V7_GS[mat])
    if s is not None:
        overrides['s_damage'] = s
    model = TaylorCPCDM(mat, orient, S0_override=S0, n_grains=N_GRAINS,
                        eps_step=EPS_STEP, n_sub=N_SUB, aniso_override=aniso,
                        seed=SEED, model_version='v4',
                        param_overrides=overrides)
    return model.run_uniaxial(max_strain)['fracture_strain']


def build_grid(mat):
    """构建 v7 材料网格，缓存到 JSON（增量续跑 + 并行）。"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    c = V7_S0[mat]
    path = os.path.join(BAYES_DIR, f'grid_{mat}.json')

    work = []
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

    pending = []
    for kind, orient, r, s, a in work:
        key = f"{kind}|{orient}|{r}|{s}|{a}"
        if key not in results:
            pending.append((key, kind, orient, r, s, a))

    total = len(pending)
    print(f"{mat}: 共 {len(work)} 点，待算 {total}")
    if total == 0:
        print(f"完成 {mat}: {path}")
        return

    n_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(evaluate, mat, orient, c * r, s, a): key
                for key, kind, orient, r, s, a in pending}
        done = 0
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                results[key] = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  失败 {key}: {e}")
                continue
            done += 1
            if done % 15 == 0 or done == total:
                print(f"  {mat}: {done}/{total}")
                _dump_results(path, mat, c, results)

    _dump_results(path, mat, c, results)
    print(f"完成 {mat}: {path}")


def _dump_results(path, mat, center, results):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'material': mat, 'S0_center': center,
                   'v7_gs': V7_GS[mat], 'points': results}, f, indent=2)


class GridModel:
    """每材料 3 取向插值模型（0°: (S0,s)；45/90°: (S0,s,a)）。"""

    def __init__(self, mat):
        path = os.path.join(BAYES_DIR, f'grid_{mat}.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少网格 {path}，请先运行 grid --material {mat}")
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self.mat = mat
        self.center = float(raw['S0_center'])
        self.points = raw['points']
        zero = np.zeros((len(S0_RATIOS), len(S_GRID)))
        for j, r in enumerate(S0_RATIOS):
            for k, s in enumerate(S_GRID):
                zero[j, k] = self.points[f"zero|0|{r}|{s}|None"]
        self.zero = zero
        oblique = {}
        for orient in (45, 90):
            arr = np.zeros((len(ANCHOR_RATIOS), len(ANCHOR_S), len(A_GRID)))
            for jr, r in enumerate(ANCHOR_RATIOS):
                for js, s in enumerate(ANCHOR_S):
                    for ja, a in enumerate(A_GRID):
                        arr[jr, js, ja] = self.points[f"oblique|{orient}|{r}|{s}|{a}"]
            oblique[orient] = arr
        self.oblique = oblique

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
    ir = int(np.clip(np.searchsorted(gr, r, side='right') - 1, 0, len(gr) - 2))
    js = int(np.clip(np.searchsorted(gs, s, side='right') - 1, 0, len(gs) - 2))
    tr = (r - gr[ir]) / (gr[ir + 1] - gr[ir])
    ts = (s - gs[js]) / (gs[js + 1] - gs[js])
    return (values[ir, js] * (1 - tr) * (1 - ts) + values[ir + 1, js] * tr * (1 - ts) +
            values[ir, js + 1] * (1 - tr) * ts + values[ir + 1, js + 1] * tr * ts)


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
    NDIM = 3 * 4 + 3 + 3

    def __init__(self, grids):
        self.grids = grids
        self.mats = list(grids.keys())
        self.obs, self.case = [], []
        for m in self.mats:
            for o in ORIENTS:
                self.obs.append(EXPERIMENTAL[m]['ef'][o])
                self.case.append((m, o))
        self.s0_hat = {m: g.center for m, g in grids.items()}
        self.ef_bar = {m: float(np.mean([EXPERIMENTAL[m]['ef'][o] for o in ORIENTS]))
                       for m in self.mats}

    def unpack(self, th):
        p, k = {}, 0
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
        lp += -0.5 * ((p['mu_s'] - 1.2) / 0.6) ** 2
        lp += _log_halfnorm(p['sig_s'], 0.5)
        lp += _log_halfnorm(p['tau_a'], 0.5)
        for m in self.mats:
            pm = p[m]
            # v7 中心先验：log S0 以 v7 值为中心（σ=0.25），a 以 v7 值为中心（σ=0.30）
            lp += -0.5 * ((pm['logS0'] - np.log(self.s0_hat[m])) / 0.25) ** 2
            lp += -0.5 * (((pm['s'] - p['mu_s']) / p['sig_s']) ** 2) - np.log(p['sig_s'])
            lp += -0.5 * ((pm['log_a45'] - np.log(V7_A45[m])) / p['tau_a']) ** 2 \
                - np.log(p['tau_a'])
            lp += -0.5 * ((pm['log_a90'] - np.log(V7_A90[m])) / p['tau_a']) ** 2 \
                - np.log(p['tau_a'])
            if pm['s'] < 0.1:
                return -np.inf
        for m, lsig in zip(self.mats, p['log_sig']):
            sig = np.exp(lsig)
            lp += _log_halfnorm(sig, 0.05 * self.ef_bar[m]) + lsig
        return lp

    def log_like(self, th):
        p = self.unpack(th)
        tot = 0.0
        for (m, o), y in zip(self.case, self.obs):
            pm = p[m]
            ef = self.grids[m].ef(pm['logS0'], pm['s'],
                                  pm['log_a45'], pm['log_a90'])
            sig = np.exp(p['log_sig'][self.mats.index(m)])
            tot += -0.5 * (((y - ef[ORIENTS.index(o)]) / sig) ** 2) - np.log(sig)
        return tot

    def log_prob(self, th):
        lp = self.log_prior(th)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_like(th)


def _log_halfnorm(x, scale):
    if x < 0:
        return -np.inf
    return -0.5 * (x / scale) ** 2 - np.log(scale) - 0.5 * np.log(np.pi / 2)


def sample():
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

    results = _summarize(model, chain)
    _save_and_plot(model, chain, results)
    return results


def _init_walkers(model, nwalkers, ndim, rng):
    p0 = np.zeros((nwalkers, ndim))
    scale, center = dict(), dict()
    k = 0
    for m in MATERIALS:
        center[k], scale[k] = np.log(model.s0_hat[m]), 0.12
        center[k + 1], scale[k + 1] = 1.2, 0.40
        center[k + 2], scale[k + 2] = np.log(V7_A45[m]), 0.30
        center[k + 3], scale[k + 3] = np.log(V7_A90[m]), 0.30
        k += 4
    for j, m in enumerate(MATERIALS):
        center[k + j] = np.log(0.05 * model.ef_bar[m])
        scale[k + j] = 0.60
    center[k + 3], scale[k + 3] = 1.2, 0.60
    center[k + 4], scale[k + 4] = 0.5, 0.40
    center[k + 5], scale[k + 5] = 0.5, 0.40

    for i in range(ndim):
        p0[:, i] = rng.normal(center[i], scale[i], size=nwalkers)
    p0 += rng.normal(0, 1e-3, size=(nwalkers, ndim))
    return p0


def _summarize(model, chain):
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
        pred[f"{m}_{o}"] = {'obs': y, 'median': float(np.median(vals)),
                            '95CI': [float(np.quantile(vals, 0.025)),
                                     float(np.quantile(vals, 0.975))]}

    # 参数相关性（可辨识性）：后验链相关系数
    corr = {}
    k = 0
    for m in MATERIALS:
        sub = chain[:, k:k + 4]
        R = np.corrcoef(sub.T)
        corr[m] = {
            'logS0_vs_s': float(R[0, 1]),
            'logS0_vs_loga45': float(R[0, 2]),
            'logS0_vs_loga90': float(R[0, 3]),
            's_vs_loga45': float(R[1, 2]),
            's_vs_loga90': float(R[1, 3]),
            'loga45_vs_loga90': float(R[2, 3]),
        }
        k += 4
    results = {'params': params, 'hyper': hyper, 'predictive': pred,
               'correlations': corr, 'sampler': 'emcee',
               'n_samples': int(len(chain))}
    return results


def _save_and_plot(model, chain, results):
    out = os.path.join(OUT, 'bayes_calibration_v7_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    for m in MATERIALS:
        idx = MATERIALS.index(m)
        cols = chain[:, idx * 4:idx * 4 + 4]
        _corner(cols, [r'$\log S_0$', r'$s$', r'$\log a_{45}$', r'$\log a_{90}$'],
                f'Fig_bayes_corner_v7_{m}.png')
    _ppc(results['predictive'], 'Fig_bayes_ppc_v7.png')


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
                ax.hist2d(cols[:, j], cols[:, i], bins=40, cmap='viridis')
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
    g = sub.add_parser('grid', help='构建 v7 仿真插值网格')
    g.add_argument('--material', default='all', choices=MATERIALS + ['all'])
    sub.add_parser('sample', help='MCMC 采样 + 出图')
    args = ap.parse_args()

    if args.cmd == 'grid':
        mats = MATERIALS if args.material == 'all' else [args.material]
        for m in mats:
            build_grid(m)
    elif args.cmd == 'sample':
        sample()
    else:
        print("用法: bayes_calibration_v7.py {grid --material Ti64|sample}")


if __name__ == '__main__':
    main()
