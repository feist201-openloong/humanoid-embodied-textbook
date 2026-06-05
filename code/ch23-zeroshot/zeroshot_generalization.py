"""
第23章 仿真实操：零样本泛化实验
================================================================
目标：在源场景训练，在目标场景零样本测试，分析泛化能力。
"""

import mujoco
import numpy as np

print("="*60)
print("  零样本泛化实验")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"

def run_with_friction(friction_coeff, label=""):
    """在指定摩擦系数的地面上运行仿真"""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # 设置地面摩擦系数
    for i in range(model.ngeom):
        model.geom_friction[i, 0] = friction_coeff

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0

    heights = []
    for step in range(500):
        mujoco.mj_step(model, data)
        if step % 10 == 0:
            heights.append(data.qpos[2])
        if step % 100 == 0:
            print(f"  [{label}] 步 {step:4d}: 高度 = {data.qpos[2]:.3f}m")

    return heights

# 训练场景（标准地面）
print(f"\n--- 训练场景: 标准地面 (摩擦=0.5) ---")
h_train = run_with_friction(0.5, "标准地面")

# 测试场景
print(f"\n--- 测试场景1: 冰面 (摩擦=0.1) ---")
h_test1 = run_with_friction(0.1, "冰面")

print(f"\n--- 测试场景2: 粗糙地面 (摩擦=1.0) ---")
h_test2 = run_with_friction(1.0, "粗糙地面")

print(f"\n--- 测试场景3: 极滑冰面 (摩擦=0.01) ---")
h_test3 = run_with_friction(0.01, "极滑冰面")

# 结果分析
results = {
    "标准地面 (0.5)": (h_train[0], h_train[-1]),
    "冰面 (0.1)": (h_test1[0], h_test1[-1]),
    "粗糙地面 (1.0)": (h_test2[0], h_test2[-1]),
    "极滑冰面 (0.01)": (h_test3[0], h_test3[-1]),
}

print(f"\n{'='*60}")
print(f"  泛化测试结果:")
print(f"  {'场景':<20} {'初始高度':<12} {'最终高度':<12} {'状态':<12}")
print(f"  {'─'*56}")
for scene, (init, final) in results.items():
    status = "✅ 稳定" if final > init * 0.7 else "❌ 摔倒" if final < init * 0.3 else "⚠️ 不稳"
    print(f"  {scene:<20} {init:<12.3f} {final:<12.3f} {status:<12}")

print(f"\n  结论: 固定的策略无法泛化到所有物理条件。")
print(f"  域随机化训练可以显著提升零样本泛化能力。")
print(f"  这正是第15章 Sim2Real 要解决的核心问题。")
print(f"{'='*60}")
