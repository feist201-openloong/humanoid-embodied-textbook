"""
第1章 仿真实操：MuJoCo 第一课 —— 加载人形机器人模型
================================================================
目标：学习加载 MuJoCo 模型，观察关节层次结构，打印自由度信息。
"""

import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# 1. 加载模型
# ============================================================
# 使用 MuJoCo 内置的人形机器人模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"{'='*60}")
print(f"  模型名称: {model.name}")
print(f"  文件路径: {xml_path}")
print(f"{'='*60}")

# ============================================================
# 2. 探索模型结构
# ============================================================
print(f"\n--- 基本信息 ---")
print(f"  自由度 (nq):        {model.nq}")
print(f"  关节数 (njnt):      {model.njnt}")
print(f"  刚体数 (nbody):     {model.nbody}")
print(f"  几何体数 (ngeom):   {model.ngeom}")
print(f"  执行器数 (nu):      {model.nu}")
print(f"  传感器数 (nsensor): {model.nsensor}")

print(f"\n--- 关节列表 ---")
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    type_names = ["自由 (free)", "球铰 (ball)", "旋转 (hinge)", "滑动 (slide)"]
    jnt_type = model.jnt_type[i]
    jnt_type_name = type_names[jnt_type] if jnt_type < len(type_names) else "未知"
    parent_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.jnt_bodyid[i])
    print(f"  关节 {i:2d}: {name or '(unnamed)':20s}  类型: {jnt_type_name}  "
          f"  父刚体: {parent_body or '(世界)'}")

print(f"\n--- 刚体列表 (仅显示有名称的) ---")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name:
        parent = model.body_parentid[i]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent) or "世界"
        print(f"  刚体 {i:2d}: {name:20s}  父节点: {parent_name}")

# ============================================================
# 3. 运行被动仿真（无控制）
# ============================================================
print(f"\n--- 运行被动仿真（500步）---")
mujoco.mj_resetData(model, data)

for step in range(500):
    mujoco.mj_step(model, data)
    if step % 100 == 0:
        torso_height = data.qpos[2] if model.nq > 2 else 0
        print(f"  步 {step:4d}: 躯干高度 = {torso_height:.3f} m")

print(f"\n  最终躯干高度: {data.qpos[2]:.3f} m")
print(f"  说明: 在无控制输入下，机器人因重力倒地——控制系统的重要性！")
print(f"{'='*60}")

# ============================================================
# 4. 交互式可视化（可选 - 无显示环境下自动跳过）
# ============================================================
print(f"\n--- 交互式查看器 ---")
print(f"即将打开 MuJoCo 查看器... (无显示环境会自动跳过)")
print(f"操作提示:")
print(f"  - 鼠标左键拖拽: 旋转视角")
print(f"  - 鼠标滚轮: 缩放")
print(f"  - 按 ESC 关闭查看器")

try:
    mujoco.mj_resetData(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(2000):
            mujoco.mj_step(model, data)
            viewer.sync()
    print(f"  查看器已关闭。")
except Exception as e:
    print(f"  查看器不可用 (当前环境无显示支持): {e}")

print(f"\n实验完成！请对照第1章内容理解人形机器人的层级结构。")
