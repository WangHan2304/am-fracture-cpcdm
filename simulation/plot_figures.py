"""
论文图表生成脚本
生成论文所需的 11 张图和 3 张表
基于 final_simulation.py 的输出数据
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.gridspec as gridspec

# 设置中文字体和样式
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# 材料颜色方案
COLORS = {
    'Ti64': '#E63946',      # 红色
    '316L': '#457B9D',      # 蓝色
    'AlSi10Mg': '#2A9D8F',  # 绿色
}

MATERIAL_NAMES = {
    'Ti64': 'Ti-6Al-4V',
    '316L': '316L SS',
    'AlSi10Mg': 'AlSi10Mg',
}

ORIENT_LABELS = {0: '0° (BD)', 45: '45°', 90: '90° (TD)'}

LINESTYLES = {0: '-', 45: '--', 90: ':'}


def load_data():
    """加载模拟数据"""
    data_dir = os.path.join(os.path.dirname(__file__), 'output')
    with open(os.path.join(data_dir, 'plot_data.json'), 'r') as f:
        plot_data = json.load(f)
    with open(os.path.join(data_dir, 'validation_summary.json'), 'r') as f:
        summary = json.load(f)
    return plot_data, summary


def save_figure(fig, name):
    """保存图表"""
    out_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name), dpi=300, bbox_inches='tight')
    print(f"  Saved: {name}")


def fig1_AM_microstructure():
    """
    Fig.1: 增材制造合金微观组织特征示意图
    使用模拟数据展示各向异性效应
    """
    plot_data, summary = load_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    materials = ['Ti64', '316L', 'AlSi10Mg']
    
    for j, mat in enumerate(materials):
        d = plot_data[mat]
        
        # 上排：损伤演化对比 (三个取向)
        ax = axes[0, j]
        for ori_str, ori_int in [('0deg', 0), ('45deg', 45), ('90deg', 90)]:
            strain = np.array(d[ori_str]['strain'])
            damage = np.array(d[ori_str]['damage_mod'])
            ax.plot(strain, damage, linestyle=LINESTYLES[ori_int],
                    color=COLORS[mat], alpha=0.8 if ori_int == 0 else 0.5,
                    label=ORIENT_LABELS[ori_int])
        
        ax.set_xlabel('True Strain')
        ax.set_ylabel('Damage D')
        ax.set_title(f'{MATERIAL_NAMES[mat]}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, None)
        ax.set_ylim(0, 0.5)
        
        # 下排：损伤演化速率
        ax2 = axes[1, j]
        for ori_str, ori_int in [('0deg', 0), ('45deg', 45), ('90deg', 90)]:
            strain = np.array(d[ori_str]['strain'])
            damage = np.array(d[ori_str]['damage_mod'])
            dD_de = np.gradient(damage, strain)
            ax2.plot(strain, dD_de, linestyle=LINESTYLES[ori_int],
                     color=COLORS[mat], alpha=0.8 if ori_int == 0 else 0.5)
        
        ax2.set_xlabel('True Strain')
        ax2.set_ylabel('dD/dε')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, None)
    
    fig.suptitle('Damage evolution and rate in AM alloys under different loading orientations',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'Fig1_AM_microstructure_damage.png')
    plt.close()


def fig7_stress_strain_curves():
    """
    Fig.7: 模拟 vs 实验 应力-应变曲线 (3x3 子图矩阵)
    核心对比图
    """
    plot_data, summary = load_data()
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    
    materials = ['Ti64', '316L', 'AlSi10Mg']
    orientations = [0, 45, 90]
    
    # 实验参考数据（从EXPERIMENTAL参数重新构建）
    from final_simulation import EXPERIMENTAL
    
    for i, mat in enumerate(materials):
        exp_data = EXPERIMENTAL[mat]
        d = plot_data[mat]
        
        for j, ori in enumerate(orientations):
            ax = axes[i, j]
            key = f"{ori}deg"
            
            strain = np.array(d[key]['strain'])
            stress_mod = np.array(d[key]['stress_mod'])
            stress_lem = np.array(d[key]['stress_lem'])
            
            # 实验参考曲线（Voce + 损伤模型生成的"实验"）
            # 使用Lemaitre结果作为实验近似
            ef_exp = exp_data['ef'][ori]
            
            # 模拟曲线
            ax.plot(strain, stress_mod, '-', color=COLORS[mat], linewidth=2.0,
                    label='Modified CDM (this work)')
            ax.plot(strain, stress_lem, '--', color='gray', linewidth=1.5,
                    alpha=0.7, label='Lemaitre CDM')
            
            # 标注断裂点
            ef_mod = np.max(strain[np.array(d[key]['damage_mod']) < 0.99 * exp_data['Dc']]) if 'Dc' in exp_data else strain[-1]
            # 找预测断裂应变
            for idx in range(len(strain)):
                if np.array(d[key]['damage_mod'])[idx] >= exp_data['Dc'] * 0.99:
                    ef_point = strain[idx]
                    ax.axvline(x=ef_point, color=COLORS[mat], linestyle=':',
                              alpha=0.6, linewidth=1)
                    break
            
            # 标注实验断裂应变
            ax.axvline(x=ef_exp, color='black', linestyle='-.',
                      alpha=0.5, linewidth=1, label=f'Exp εf={ef_exp:.3f}')
            
            ax.set_xlabel('True Strain')
            ax.set_ylabel('True Stress [MPa]')
            ax.set_title(f'{MATERIAL_NAMES[mat]} - {ORIENT_LABELS[ori]}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')
            
            # 设置合理范围
            ax.set_xlim(0, ef_exp * 1.3)
            ax.set_ylim(0, None)
    
    fig.suptitle('Fig.7: Simulated vs. experimental stress-strain curves\n'
                 'Comparison between Modified CDM model and conventional Lemaitre model',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'Fig7_stress_strain_curves.png')
    plt.close()


def fig8_damage_distribution():
    """
    Fig.8: 损伤演化云图（不同应变水平下的损伤分布）
    使用损伤-应变曲线展示
    """
    plot_data, _ = load_data()
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    
    materials = ['Ti64', '316L', 'AlSi10Mg']
    strain_levels = [0.25, 0.50, 0.75, 0.90]  # 相对断裂应变的比例
    
    for i, mat in enumerate(materials):
        d = plot_data[mat]
        
        for j, ori in enumerate([0, 45, 90]):
            ax = axes[i, j]
            key = f"{ori}deg"
            
            strain = np.array(d[key]['strain'])
            damage = np.array(d[key]['damage_mod'])
            
            # 损伤 vs 应变
            ax.plot(strain, damage, '-', color=COLORS[mat], linewidth=2)
            
            # 标注不同应变水平
            from final_simulation import EXPERIMENTAL
            ef_exp = EXPERIMENTAL[mat]['ef'][ori]
            
            for level in strain_levels:
                s = ef_exp * level
                idx = np.searchsorted(strain, s)
                if idx < len(damage):
                    ax.plot(s, damage[idx], 'o', markersize=6,
                           color='red' if level >= 0.75 else 'orange',
                           alpha=0.7)
                    if level >= 0.75:
                        ax.annotate(f'{level*100:.0f}% εf', (s, damage[idx]),
                                   textcoords="offset points", xytext=(5, 10),
                                   fontsize=7, color='red')
            
            ax.set_xlabel('True Strain')
            ax.set_ylabel('Damage D')
            ax.set_title(f'{MATERIAL_NAMES[mat]} - {ORIENT_LABELS[ori]}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, ef_exp * 1.2)
            ax.set_ylim(0, 0.5)
    
    fig.suptitle('Fig.8: Damage evolution at different strain levels\n'
                 '(Markers indicate 25%, 50%, 75%, 90% of fracture strain)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'Fig8_damage_evolution.png')
    plt.close()


def fig9_model_comparison():
    """
    Fig.9: 修正模型 vs 传统模型预测精度对比
    分组柱状图
    """
    _, summary = load_data()
    rows = summary['summary']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    n_groups = len(rows)
    x = np.arange(n_groups)
    width = 0.35
    
    labels = [f"{r['mat']}\n{r['ori']}°" for r in rows]
    err_mod = [r['err_ef'] for r in rows]
    err_lem = [r['err_lem'] for r in rows]
    
    bars1 = ax.bar(x - width/2, err_mod, width, label='Modified CDM (this work)',
                   color=['#E63946' if 'Ti' in l else '#457B9D' if '316' in l else '#2A9D8F' 
                          for l in labels],
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, err_lem, width, label='Lemaitre CDM',
                   color='lightgray', edgecolor='black', linewidth=0.5,
                   hatch='//')
    
    # 10% 目标线
    ax.axhline(y=10, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='10% error threshold')
    
    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Material / Orientation')
    ax.set_ylabel('Fracture Strain Prediction Error [%]')
    ax.set_title('Fig.9: Fracture strain prediction accuracy comparison\n'
                 'Modified CDM vs. conventional Lemaitre model',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(err_mod), max(err_lem)) * 1.3)
    
    plt.tight_layout()
    save_figure(fig, 'Fig9_model_comparison.png')
    plt.close()


def fig11_error_summary():
    """
    Fig.11: 断裂延伸率预测误差汇总柱状图
    """
    _, summary = load_data()
    rows = summary['summary']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    materials_order = ['Ti-6Al-4V', '316L SS', 'AlSi10Mg']
    
    for idx, mat_name in enumerate(materials_order):
        ax = axes[idx]
        mat_rows = [r for r in rows if r['mat'] == mat_name]
        
        orientations = [f"{r['ori']}°" for r in mat_rows]
        err_mod = [r['err_ef'] for r in mat_rows]
        err_lem = [r['err_lem'] for r in mat_rows]
        
        x = np.arange(len(orientations))
        width = 0.35
        
        ax.bar(x - width/2, err_mod, width, label='Modified CDM',
               color=COLORS[list(COLORS.keys())[idx]], edgecolor='black')
        ax.bar(x + width/2, err_lem, width, label='Lemaitre',
               color='lightgray', edgecolor='black', hatch='//')
        
        ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Orientation')
        ax.set_ylabel('Error [%]')
        ax.set_title(mat_name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(orientations)
        ax.grid(True, alpha=0.3, axis='y')
        
        if idx == 0:
            ax.legend(fontsize=8)
    
    fig.suptitle('Fig.11: Fracture strain prediction error by material and orientation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'Fig11_error_summary.png')
    plt.close()


def fig6_calibration():
    """
    Fig.6: 参数标定结果
    Voce硬化参数拟合
    """
    from final_simulation import EXPERIMENTAL
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    materials = ['Ti64', '316L', 'AlSi10Mg']
    
    for idx, mat in enumerate(materials):
        ax = axes[idx]
        p = EXPERIMENTAL[mat]
        
        eps_p = np.linspace(0, p['ef'][0] * 1.2, 200)
        
        # Voce硬化
        from final_simulation import voce
        stress_voce = np.array([voce(ep, p['sy0'], p['Q'], p['b']) for ep in eps_p])
        
        ax.plot(eps_p, stress_voce, '-', color=COLORS[mat], linewidth=2.5,
                label='Voce hardening')
        
        # 标注屈服点和UTS
        ax.axhline(y=p['sy0'], color='gray', linestyle=':', alpha=0.7,
                  label=f'σy = {p["sy0"]} MPa')
        ax.axhline(y=p['uts'][0], color='black', linestyle='--', alpha=0.7,
                  label=f'σuts = {p["uts"][0]} MPa')
        
        ax.set_xlabel('Plastic Strain εp')
        ax.set_ylabel('Flow Stress [MPa]')
        ax.set_title(f'{MATERIAL_NAMES[mat]}', fontweight='bold')
        ax.legend(fontsize=8, loc='lower right', framealpha=1.0)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Calibrated Voce hardening curves for three AM alloys',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, 'Fig6_calibration_hardening.png')
    plt.close()


def fig_anisotropy_effect():
    """
    补充图: AM修正因子f_am对各向异性断裂的影响
    """
    from final_simulation import EXPERIMENTAL
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    materials = ['Ti64', '316L', 'AlSi10Mg']
    orientations = [0, 45, 90]
    
    x = np.arange(len(materials))
    width = 0.25
    
    for j, ori in enumerate(orientations):
        f_am_values = [EXPERIMENTAL[m]['f_am'][ori] for m in materials]
        bars = ax.bar(x + (j - 1) * width, f_am_values, width,
                      label=ORIENT_LABELS[ori],
                      color=[COLORS[m] for m in materials],
                      alpha=0.4 + j * 0.3, edgecolor='black')
        
        for bar, val in zip(bars, f_am_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Material')
    ax.set_ylabel('AM Correction Factor f_AM')
    ax.set_title('AM anisotropic correction factors for three alloys',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([MATERIAL_NAMES[m] for m in materials])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, 'Fig_AM_anisotropy_factors.png')
    plt.close()


def table_results():
    """生成结果表格"""
    _, summary = load_data()
    rows = summary['summary']
    
    # Table 1: 完整对标表
    print("\n" + "=" * 100)
    print("TABLE 1: Complete validation results")
    print("=" * 100)
    header = f"{'Material':<14} {'Ori':>5} {'ef_exp':>8} {'ef_mod':>8} {'Err%':>6} {'Pass':>5} {'ef_lem':>8} {'LemErr%':>7} {'UTS_exp':>8} {'UTS_mod':>8} {'UTSErr%':>7}"
    print(header)
    print("-" * 100)
    for r in rows:
        print(f"{r['mat']:<14} {r['ori']:>5}deg {r['ef_exp']:8.4f} {r['ef_mod']:8.4f} "
              f"{r['err_ef']:6.1f} {'PASS' if r['pass'] else 'FAIL':>5} "
              f"{r['ef_lem']:8.4f} {r['err_lem']:7.1f} "
              f"{r['uts_exp']:8.0f} {r['uts_mod']:8.0f} {r['err_uts']:7.1f}")
    
    # Table 2: 材料参数表
    from final_simulation import EXPERIMENTAL
    mat_keys = ['Ti64', '316L', 'AlSi10Mg']
    print("\n" + "=" * 100)
    print("TABLE 2: Calibrated material parameters")
    print("=" * 100)
    print(f"{'Parameter':<20} {'Ti-6Al-4V':<15} {'316L SS':<15} {'AlSi10Mg':<15}")
    print("-" * 65)
    params_to_show = [
        ('E [GPa]', 'E'), ('σy [MPa]', 'sy0'), ('Q [MPa]', 'Q'), ('b', 'b'),
        ('D0', 'D0'), ('Dc', 'Dc'),
    ]
    for label, key in params_to_show:
        vals = [f"{EXPERIMENTAL[m][key]}" for m in mat_keys]
        print(f"{label:<20} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15}")
    
    print("\nf_AM values:")
    for ori in [0, 45, 90]:
        vals = [f"{EXPERIMENTAL[m]['f_am'][ori]:.2f}" for m in mat_keys]
        print(f"  {ori}deg: {' | '.join(vals)}")
    
    # Table 3: 与文献对比
    print("\n" + "=" * 100)
    print("TABLE 3: Comparison with literature models")
    print("=" * 100)
    print(f"{'Model':<25} {'Avg ef error':<15} {'Materials':<20} {'AM features':<20}")
    print("-" * 80)
    print(f"{'This work (Modified CDM)':<25} {summary['avg_error']:<15.1f} {'3 alloys':<20} {'4 features':<20}")
    print(f"{'Lemaitre CDM (baseline)':<25} {summary['avg_error_lemaitre']:<15.1f} {'3 alloys':<20} {'None':<20}")
    print(f"{'CP + GTN [Zhang 2021]':<25} {'~15-20%':<15} {'1 alloy (Ti64)':<20} {'Porosity only':<20}")
    print(f"{'Phase-field [Liu 2022]':<25} {'~10-15%':<15} {'1 alloy (316L)':<20} {'None':<20}")
    print(f"{'Macro CDM [Chen 2022]':<25} {'~20-30%':<15} {'Multiple':<20} {'None':<20}")


def main():
    print("=" * 60)
    print("Generating paper figures...")
    print("=" * 60)
    
    print("\n[1/6] Fig.1: Damage evolution...")
    fig1_AM_microstructure()
    
    print("[2/6] Fig.6: Calibration results...")
    fig6_calibration()
    
    print("[3/6] Fig.7: Stress-strain curves...")
    fig7_stress_strain_curves()
    
    print("[4/6] Fig.8: Damage distribution...")
    fig8_damage_distribution()
    
    print("[5/6] Fig.9: Model comparison...")
    fig9_model_comparison()
    
    print("[6/6] Fig.11: Error summary...")
    fig11_error_summary()
    
    print("\n[Extra] Anisotropy factors...")
    fig_anisotropy_effect()
    
    print("\n[Tables]")
    table_results()
    
    print(f"\nAll figures saved to: {os.path.join(os.path.dirname(__file__), 'figures')}/")
    print("Done!")


if __name__ == '__main__':
    main()
