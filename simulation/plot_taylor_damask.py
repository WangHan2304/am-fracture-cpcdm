"""TaylorCPCDM 无损伤 vs DAMASK FFT 全场对比图（论文图件）。
三材料子图：宏观 Cauchy 应力-对数应变曲线，Taylor 上界偏差可视化。
输出: Fig_taylor_vs_damask.png"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base_wsl = r'\\wsl$\Ubuntu-24.04\home\zj'
out_dir = r'D:\20260618断裂模型论文\simulation\output'
fig_dir = r'D:\20260618断裂模型论文\simulation\figures'

def load_curve(path):
    d = np.loadtxt(path, skiprows=1)
    return d[:, 0], d[:, 1]

materials = [('316L', '316L SS', '#1f77b4'),
             ('Ti64', 'Ti-6Al-4V', '#d62728'),
             ('AlSi10Mg', 'AlSi10Mg', '#2ca02c')]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
res_all = {}

for ax, (mat, label, color) in zip(axes, materials):
    # Taylor（Pa → MPa；工程应变 → 对数应变）
    te, ts_pa = load_curve(os.path.join(out_dir, f'taylor_nodamage_{mat}_0deg.csv'))
    te_ln = np.log(1.0 + te)
    ts = ts_pa / 1e6
    # DAMASK FFT (local staged copy, sigma already in MPa)
    de, ds = load_curve(os.path.join(out_dir, f'damask_{mat}_macro.csv'))

    ax.plot(de, ds, '-', color=color, lw=2.2, label='DAMASK FFT (full-field)')
    ax.plot(te_ln, ts, '--', color='black', lw=2.0, label='Taylor CPCDM (iso-strain)')

    # UTS 标注
    iu = int(np.argmax(ds)); it = int(np.argmax(ts))
    ax.plot(de[iu], ds[iu], 'o', color=color, ms=6)
    ax.plot(te_ln[it], ts[it], 's', color='black', ms=6)
    uts_rel = (ts[it] - ds[iu]) / ds[iu] * 100
    t_txt = f'UTS {ts[it]:.0f} MPa\n(Δ{uts_rel:+.0f}%)'
    d_txt = f'UTS {ds[iu]:.0f} MPa'
    # put the higher UTS marker's label above it and the lower one's below, so the
    # two never collide (Taylor over-predicts the FCC alloys but under-predicts Ti-6Al-4V)
    # soft white halo so a label never reads as touching the curve behind it
    bb = dict(boxstyle='square,pad=0.18', fc='white', ec='none', alpha=0.85)
    if ts[it] >= ds[iu]:
        ax.annotate(t_txt, (te_ln[it], ts[it]), textcoords='offset points', xytext=(-2, 10), ha='right', va='bottom', fontsize=8, color='black', bbox=bb)
        ax.annotate(d_txt, (de[iu], ds[iu]), textcoords='offset points', xytext=(-2, -10), ha='right', va='top', fontsize=8, color=color, bbox=bb)
    else:
        ax.annotate(d_txt, (de[iu], ds[iu]), textcoords='offset points', xytext=(-2, 10), ha='right', va='bottom', fontsize=8, color=color, bbox=bb)
        ax.annotate(t_txt, (te_ln[it], ts[it]), textcoords='offset points', xytext=(-2, -10), ha='right', va='top', fontsize=8, color='black', bbox=bb)
    ax.set_ylim(0, max(ts[it], ds[iu]) * 1.22)
    ax.margins(x=0.08)

    ax.set_xlabel('Logarithmic strain $\\varepsilon_{11}$')
    ax.set_ylabel('Cauchy stress $\\sigma_{11}$ [MPa]')
    ax.set_title(label, fontsize=11)
    ax.grid(alpha=0.3)
    res_all[mat] = {'taylor_uts': float(ts[it]), 'damask_uts': float(ds[iu]),
                    'rel_pct': float(uts_rel)}

axes[0].legend(fontsize=8, loc='lower right')
fig.suptitle('Taylor iso-strain vs full-field FFT (DAMASK): damage-free CP, 32$^3$ RVE, 30 grains, identical orientations',
             fontsize=11, y=1.02)
fig.tight_layout()
out_png = os.path.join(fig_dir, 'Fig_taylor_vs_damask.png')
fig.savefig(out_png, dpi=300, bbox_inches='tight')
print('saved', out_png)
print(json.dumps(res_all, indent=1))
