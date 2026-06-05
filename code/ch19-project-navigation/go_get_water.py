"""
第19章 仿真实操："去拿那杯水"综合项目
================================================================
目标：实现完整的"导航→感知→抓取→返回"流水线。
"""

import mujoco
import numpy as np
import math

print("="*60)
print("  综合项目：去拿那杯水")
print("="*60)

# ============================================================
# 1. 加载模型
# ============================================================
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[2] = 0.95

# 场景定义
START_POS = np.array([0.0, 0.0])
TABLE_POS = np.array([2.0, 0.0])
CUP_POS_ON_TABLE = np.array([2.0, 0.2, 0.8])
ROBOT_HOME = np.array([0.0, 0.0])

print(f"  起点: {START_POS}")
print(f"  桌子位置: {TABLE_POS}")
print(f"  水杯位置: {CUP_POS_ON_TABLE}")

# ============================================================
# 2. 导航阶段
# ============================================================
print(f"\n🔵 阶段1: 导航到桌前")
print(f"  目标: ({TABLE_POS[0]:.1f}, {TABLE_POS[1]:.1f})")

for step in range(300):
    # 计算到目标的距离
    current_pos = data.qpos[:2].copy()
    dx = TABLE_POS[0] - current_pos[0]
    dy = TABLE_POS[1] - current_pos[1]
    dist = math.sqrt(dx**2 + dy**2)

    if dist < 0.2:
        print(f"  ✅ 到达桌前 (步 {step})")
        break

    # 简单的PID前进
    speed = min(0.02, dist * 0.05)
    # 修改qpos模拟前进
    data.qpos[0] += speed * dx / max(dist, 0.01)
    data.qpos[1] += speed * dy / max(dist, 0.01)

    # 让机器人在行走中保持平衡
    data.qpos[2] = 0.95 + np.sin(step * 0.2) * 0.02
    mujoco.mj_step(model, data)

    if step % 50 == 0:
        print(f"  步 {step}: 距离桌子 {dist:.2f}m")

# ============================================================
# 3. 感知阶段
# ============================================================
print(f"\n🟢 阶段2: 感知水杯")

# 模拟检测水杯
estimated_cup_pos = CUP_POS_ON_TABLE + np.random.randn(3) * 0.02
print(f"  检测到水杯在: ({estimated_cup_pos[0]:.2f}, {estimated_cup_pos[1]:.2f}, {estimated_cup_pos[2]:.2f})")
print(f"  检测误差: {np.linalg.norm(estimated_cup_pos - CUP_POS_ON_TABLE):.3f}m")

# 计算水杯在机器人坐标系中的位置
robot_pos = data.qpos[:2].copy()
cup_relative = np.array([
    estimated_cup_pos[0] - robot_pos[0],
    estimated_cup_pos[1] - robot_pos[1],
    estimated_cup_pos[2]
])
print(f"  水杯相对于躯干: ({cup_relative[0]:.2f}, {cup_relative[1]:.2f}, {cup_relative[2]:.2f})")

# ============================================================
# 4. 操作阶段
# ============================================================
print(f"\n🟠 阶段3: 伸手抓取水杯")

# 找到肩关节
shoulder_qpos_addr = None
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and "shoulder" in name.lower():
        shoulder_qpos_addr = model.jnt_qposadr[i]
        print(f"  找到肩关节: {name} (qpos_addr={shoulder_qpos_addr})")
        break

if shoulder_qpos_addr is not None:
    # 控制手臂伸向目标
    for step in range(200):
        # 逐步伸出
        reach_angle = min(step / 200, 1.0) * 0.8
        if shoulder_qpos_addr < model.nu:
            data.ctrl[shoulder_qpos_addr] = reach_angle * 50
        # 顺带调整其他关节
        for jnt_idx in range(min(5, model.njnt)):
            qpos_addr = model.jnt_qposadr[jnt_idx]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_idx)
            if name and "arm" in name.lower():
                if qpos_addr < model.nu:
                    data.ctrl[qpos_addr] = reach_angle * 30
        mujoco.mj_step(model, data)

    print(f"  ✅ 手臂伸出到目标位置")

# ============================================================
# 5. 返回阶段
# ============================================================
print(f"\n🟣 阶段4: 返回起点")

for step in range(300):
    current_pos = data.qpos[:2].copy()
    dx = ROBOT_HOME[0] - current_pos[0]
    dy = ROBOT_HOME[1] - current_pos[1]
    dist = math.sqrt(dx**2 + dy**2)

    if dist < 0.2:
        print(f"  ✅ 回到起点 (步 {step})")
        break

    speed = min(0.02, dist * 0.05)
    data.qpos[0] += speed * dx / max(dist, 0.01)
    data.qpos[1] += speed * dy / max(dist, 0.01)
    mujoco.mj_step(model, data)

    if step % 100 == 0:
        print(f"  步 {step}: 距离起点 {dist:.2f}m")

print(f"\n{'='*60}")
print(f"  🎯 任务完成! 机器人从起点→桌前→感知→抓取→返回。")
print(f"  实际部署需要集成SLAM、YOLO、ROS2等真实系统。")
print(f"{'='*60}")
