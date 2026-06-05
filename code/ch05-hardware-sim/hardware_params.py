"""
第5章 仿真实操：硬件参数对动力的影响
================================================================
目标：修改 MuJoCo 人形机器人关节的物理参数，
观察参数变化对行走稳定性和姿态的影响。
"""

import mujoco
import numpy as np

print("="*60)
print("  硬件参数仿真实验")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"\n默认关节参数 (部分):")
for i in range(min(10, model.njnt)):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
        # 打印关节的阻尼和摩擦力参数
        jnt_id = i
        damping = model.damping[jnt_id] if jnt_id < len(model.damping) else 'N/A'
        print(f"  关节 {name}: 阻尼={damping}")

print(f"\n--- 实验1: 增加关节阻尼（从默认增加到3倍）---")
model2 = mujoco.MjModel.from_xml_path(str(xml_path))
data2 = mujoco.MjData(model2)
for i in range(model2.njnt):
    model2.damping[i] = model.damping[i] * 3 if i < len(model.damping) else model.damping[i]

def run_simulation(model, data, steps=500, label=""):
    """运行仿真并返回高度轨迹"""
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0
    heights = []
    for step in range(steps):
        mujoco.mj_step(model, data)
        if step % 10 == 0:
            heights.append(data.qpos[2])
    return heights

# 对比实验
print("  运行默认参数仿真（500步）...")
h_default = run_simulation(model, data, 500, "默认")
print(f"    初始高度: {h_default[0]:.3f}m, 最终高度: {h_default[-1]:.3f}m")

print("  运行3倍阻尼仿真（500步）...")
h_damped = run_simulation(model2, data2, 500, "3倍阻尼")
print(f"    初始高度: {h_damped[0]:.3f}m, 最终高度: {h_damped[-1]:.3f}m")

print(f"\n--- 实验2: 修改关节扭矩极限 ---")
model3 = mujoco.MjModel.from_xml_path(str(xml_path))
data3 = mujoco.MjData(model3)
if model3.nu > 0:
    # 放大执行器扭矩上限
    old_ctrl_range = model3.actuator_ctrlrange.copy()
    for i in range(model3.nu):
        new_max = old_ctrl_range[i, 1] * 2
        model3.actuator_ctrlrange[i, 1] = new_max
    print(f"  执行器扭矩上限放大为2倍")

# 验证模型参数
print(f"\n  模型参数汇总:")
print(f"  关节数: {model.njnt}")
print(f"  刚体数: {model.nbody}")
print(f"  执行器数: {model.nu}")
print(f"  默认阻尼范围: {model.damping.min():.4f} ~ {model.damping.max():.4f}")

print(f"\n{'='*60}")
print(f"  实验完成！观察默认 vs 3倍阻尼的高度变化差异。")
print(f"  结论: 增大阻尼 → 稳定性提升但响应变慢。")
print(f"{'='*60}")
