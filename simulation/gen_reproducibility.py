# -*- coding: utf-8 -*-
"""
gen_reproducibility.py — 可复现性版本清单生成器 (Round 14, 任务2)
================================================================
本脚本把论文中每一个会被审稿人复算的关键数字，绑定到"产生它的脚本 + 结果 JSON +
脚本/数据/环境 SHA-256"，生成两个工件：

  1) output/version_manifest.json   —— 机器可读：环境 + 文件 SHA-256 清单
  2) REPRODUCIBILITY.md             —— 人类可读：清单 + 关键数字→来源 绑定表

无 git 仓库，故以文件内容 SHA-256 作为版本指纹（内容不变→哈希不变）。
所有随机性由固定 seed=42 控制；并行度不影响确定性结果。

用法: python gen_reproducibility.py
"""
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output')

# ---------------------------------------------------------------------------
# 1. 环境（版本锁定）
# ---------------------------------------------------------------------------
def env_info():
    import numpy
    info = {
        'python': sys.version.split()[0],
        'numpy': numpy.__version__,
        'platform': platform.platform(),
        'seed': 42,
        'solver': 'TaylorCPCDM (in-house Taylor-type crystal plasticity, DAMASK interface conventions)',
        'determinism': 'fixed seed=42; ProcessPoolExecutor parallelism does not alter per-case results',
    }
    try:
        import scipy
        info['scipy'] = scipy.__version__
    except Exception:
        info['scipy'] = 'not installed (unused in headline paths)'
    return info


# ---------------------------------------------------------------------------
# 2. 需指纹化的文件清单（分组；路径相对 simulation/）
# ---------------------------------------------------------------------------
MANIFEST_FILES = {
    'core_model': [
        'src/taylor_cpcdm.py',
        'src/materials.py',
        'src/calibrated_model.py',
        'src/cp_cdm_model.py',
        'src/rve_simulator.py',
        'src/am_correction_v4.py',
    ],
    'data': [
        'data/extended_literature_dataset.csv',
        'data/extended_process_groups.json',
        'data/extended_source_groups.json',
        'data/extraction_log.md',
    ],
    'blind_and_cross_source_scripts': [
        'blind_test_in718_fast.py',
        'blind_test_in718_mode2.py',
        'blind_test_ti64_cross_source.py',
        'blind_alsi_second_source.py',
        'blind_test_316l_cross_source.py',
        'blind_316l_beta6_slow.py',
    ],
    'blind_and_cross_source_results': [
        'output/blind_test_in718_fast_results.json',
        'output/blind_test_in718_mode2_results.json',
        'output/blind_test_ti64_cross_source_results.json',
        'output/blind_test_316l_cross_source_results.json',
        'output/blind_316l_beta6_slow_results.json',
        'output/blind_alsi_second_source_results.json',
    ],
    'headline_statistics_scripts': [
        'experiment_v12_fxi_transverse.py',
        'experiment_v11_loocv_fix.py',
        'lemaitre_baseline.py',
        'paired_test_bootstrap.py',
        'recompute_aic.py',
        'experiment_v14_uncertainty.py',
        'bayes_calibration_v7.py',
    ],
    'headline_statistics_results': [
        'output/experiment_v12_fxi_transverse_results.json',
        'output/experiment_v11_gs_directional_results.json',
        'output/lemaitre_baseline_results.json',
        'output/paired_test_bootstrap_results.json',
        'output/experiment_v14_uncertainty_results.json',
        'output/bayes_calibration_v7_results.json',
    ],
    'fullfield_scripts': [
        'fullfield_3d_cpfem.py',
        'fullfield_2d_cpfem.py',
        'fullfield_3d_cpfem_nonlocal.py',
    ],
    'fullfield_results': [
        'output/fullfield_3d_cpfem_r13_merged.json',
        'output/fullfield_2d_cpfem_results.json',
        'output/nonlocal_damage_results.json',
    ],
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 3. 关键数字 → 来源绑定（脚本 / 结果 JSON 字段 / 值）
#    值以重跑后的结果 JSON 为准；此表提供审稿人定位路径。
# ---------------------------------------------------------------------------
HEADLINE_BINDINGS = [
    {'id': 'cal_mean_ef_error_pct', 'value': 20.35, 'unit': '%',
     'claim': 'mean calibration fracture-strain error (nine cases)',
     'tex': 'Abstract; Table tab:results', 'script': 'experiment_v12_fxi_transverse.py',
     'result_json': 'output/experiment_v12_fxi_transverse_results.json',
     'json_field': 'main_calibration.modified_model.avg_error_pct'},
    {'id': 'loocv_blind_ef_error_pct', 'value': 31.22, 'unit': '%',
     'claim': 'leakage-free leave-one-orientation-out blind mean error',
     'tex': 'Abstract; Section sec:loocv; Table tab:loocv',
     'script': 'experiment_v12_fxi_transverse.py',
     'result_json': 'output/experiment_v12_fxi_transverse_results.json',
     'json_field': 'loocv.modified_model.avg_ef_error_pct'},
    {'id': 'lemaitre_baseline_error_pct', 'value': 24.34, 'unit': '%',
     'claim': 'Lemaitre baseline blind error at equal single-parameter count',
     'tex': 'Abstract; Section sec:validation', 'script': 'experiment_v12_fxi_transverse.py',
     'result_json': 'output/experiment_v12_fxi_transverse_results.json',
     'json_field': 'loocv.lemaitre_baseline.avg_ef_error_pct'},
    {'id': 'paired_test_n9', 'value': 'not significant', 'unit': '',
     'claim': 'paired bootstrap/Wilcoxon/sign test: no significant separation at n=9',
     'tex': 'Abstract; Section sec:validation', 'script': 'paired_test_bootstrap.py',
     'result_json': 'output/paired_test_bootstrap_results.json'},
    {'id': 'blind_ti64_sun_avg_err_pct', 'value': 22.8, 'unit': '%',
     'claim': 'Ti-6Al-4V cross-source (Sun) transferred-S0 mean error, 0/45/90 = 22.6/1.5/44.3',
     'tex': 'Section sec:cross_source; Table tab:blind_failures',
     'script': 'blind_test_ti64_cross_source.py',
     'result_json': 'output/blind_test_ti64_cross_source_results.json',
     'json_field': 'avg_error_pct'},
    {'id': 'blind_in718_mode2_avg_err_pct', 'value': 71.5, 'unit': '%',
     'claim': 'IN718 mode-2 in-source blind avg (S0*=8.0 on 0deg), ratio 90/0 pred 0.17 vs exp 0.67 (same side)',
     'tex': 'Section sec:in718_blind; Table tab:blind_failures',
     'script': 'blind_test_in718_fast.py',
     'result_json': 'output/blind_test_in718_fast_results.json',
     'json_field': 'mode2_avg_blind_ef_error_pct'},
    {'id': 'blind_alsi_awd_avg_err_pct', 'value': 38.8, 'unit': '%',
     'claim': 'AlSi10Mg cross-source (Awd) mean error, 0/45/90 = 17.2/34.7/64.7, V-shape',
     'tex': 'Section sec:cross_source; Table tab:blind_failures',
     'script': 'blind_alsi_second_source.py',
     'result_json': 'output/blind_alsi_second_source_results.json',
     'json_field': 'avg_error_pct'},
    {'id': 'blind_316l_hitzler_avg_err_pct', 'value': 80.6, 'unit': '%',
     'claim': '316L (Hitzler) transferred-S0 blind avg, ratio 90/0 pred 0.29 vs exp 2.83 (genuine reversal)',
     'tex': 'Section sec:cross_source; Table tab:blind_failures',
     'script': 'blind_test_316l_cross_source.py',
     'result_json': 'output/blind_test_316l_cross_source_results.json',
     'json_field': 'avg_blind_ef_error_pct'},
    {'id': 'beta6_hitzler_repair', 'value': '32.7 -> 12.9', 'unit': '%',
     'claim': 'geometric shielding repairs the 316L 90deg reversal from 32.7% (b6=0) to 12.9% (b6=0.3)',
     'tex': 'Section sec:cross_source (b6 paragraph); Table tab:blind_failures',
     'script': 'blind_316l_beta6_slow.py',
     'result_json': 'output/blind_316l_beta6_slow_results.json',
     'json_field': "['0.0'].err_pct -> ['0.3'].err_pct"},
]


def main():
    manifest = {'generated_by': 'gen_reproducibility.py', 'environment': env_info(),
                'files': {}}
    missing = []
    for group, files in MANIFEST_FILES.items():
        rows = []
        for rel in files:
            abspath = os.path.join(HERE, rel.replace('/', os.sep))
            if os.path.isfile(abspath):
                rows.append({'path': rel, 'sha256': sha256_of(abspath),
                             'bytes': os.path.getsize(abspath)})
            else:
                missing.append(rel)
                rows.append({'path': rel, 'sha256': None, 'bytes': None})
        manifest['files'][group] = rows
    manifest['headline_bindings'] = HEADLINE_BINDINGS

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'version_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # human-readable markdown
    lines = []
    lines.append('# Reproducibility Version Manifest\n')
    lines.append('This file is generated by `simulation/gen_reproducibility.py`. Because the project is '
                 'not under git, every input/output that backs a reported number is fingerprinted by its '
                 'content SHA-256; re-running a bound script must reproduce both the archived result JSON '
                 'and the bound headline value.\n')
    e = manifest['environment']
    lines.append('## Environment (version-locked)\n')
    lines.append(f'- Python **{e["python"]}**, NumPy **{e["numpy"]}**, SciPy {e.get("scipy","-")}')
    lines.append(f'- Platform: {e["platform"]}')
    lines.append(f'- Random seed: **{e["seed"]}** ({e["determinism"]})')
    lines.append(f'- Solver: {e["solver"]}\n')
    lines.append('## File SHA-256 manifest\n')
    for group, rows in manifest['files'].items():
        lines.append(f'### {group}\n')
        lines.append('| file | bytes | sha256 |')
        lines.append('|---|---:|---|')
        for r in rows:
            b = '' if r['bytes'] is None else f"{r['bytes']:,}"
            h = '_MISSING_' if r['sha256'] is None else f"`{r['sha256']}`"
            lines.append(f'| `{r["path"]}` | {b} | {h} |')
        lines.append('')
    lines.append('## Headline number -> source binding\n')
    lines.append('| id | value | manuscript location | producing script | result JSON (field) |')
    lines.append('|---|---|---|---|---|')
    for hb in HEADLINE_BINDINGS:
        val = f"{hb['value']}{hb['unit']}"
        fld = hb.get('json_field', '')
        lines.append(f"| `{hb['id']}` | {val} | {hb['tex']} | `{hb['script']}` | "
                     f"`{hb['result_json']}`{' ('+fld+')' if fld else ''} |")
    lines.append('')
    with open(os.path.join(HERE, 'REPRODUCIBILITY.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"wrote output/version_manifest.json and REPRODUCIBILITY.md")
    if missing:
        print('MISSING FILES (hash as null):')
        for m in missing:
            print('  -', m)
    else:
        print('all manifest files present')


if __name__ == '__main__':
    main()
