"""导出 TaylorCPCDM 同 seed 同织构的 30 晶粒取向（旋转矩阵 + 四元数），
供 DAMASK RVE 使用，保证 Taylor 与全场对比的微结构完全一致。"""
import sys, io, json
sys.path.insert(0, r'D:\20260618断裂模型论文\simulation\src')
import numpy as np
from taylor_cpcdm import TaylorCPCDM

def R_to_quat(R):
    """旋转矩阵 → 四元数 [qw, qx, qy, qz]（DAMASK 约定，需验证符号）"""
    q = np.zeros(4)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2,1] - R[1,2]) / s
        q[2] = (R[0,2] - R[2,0]) / s
        q[3] = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = np.sqrt(max(1.0 + R[0,0] - R[1,1] - R[2,2], 1e-12)) * 2.0
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(max(1.0 + R[1,1] - R[0,0] - R[2,2], 1e-12)) * 2.0
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(max(1.0 + R[2,2] - R[0,0] - R[1,1], 1e-12)) * 2.0
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s
    return q / np.linalg.norm(q)

out = {}
for mat in ['316L', 'Ti64', 'AlSi10Mg']:
    # 与实验脚本一致：seed=42, n_grains=30, v4 模型
    model = TaylorCPCDM(mat, orientation_deg=0, n_grains=30, seed=42, model_version='v4')
    Rs = model.R  # (30, 3, 3)
    quats = np.array([R_to_quat(R) for R in Rs])
    out[mat] = {
        'quats': quats.tolist(),
        'Rs': Rs.tolist(),
        'theta_tex': model.params.get('theta_tex', 0.0),
    }
    print(f"{mat}: theta_tex={out[mat]['theta_tex']}, quat[0]={np.round(quats[0],4)}")

with open(r'D:\20260618断裂模型论文\simulation\output\damask_orientations.json', 'w') as f:
    json.dump(out, f, indent=1)
print("saved damask_orientations.json")
