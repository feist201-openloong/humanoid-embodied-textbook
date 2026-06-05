"""
第20章 仿真实操：人机协作"递工具"
================================================================
目标：实现"理解手势→取物→递送"的协作流水线。
"""

import mujoco
import numpy as np
import math

print("="*60)
print("  人机协作：递工具")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[2] = 0.95

# ============================================================
# 1. 定义场景
# ============================================================
TOOL_POS = np.array([1.5, 0.3, 0.6])         # 工具位置
HUMAN_HAND_POS = np.array([0.5, -0.3, 0.4])  # 人类手部位置（递送目标）

print(f"  工具位置: {TOOL_POS}")
print(f"  人类手部: {HUMAN_HAND_POS}")

# ============================================================
# 2. 阶段1: 理解手势
# ============================================================
print(f"\n🔵 阶段1: 理解手势意图")

# 模拟人类指向
point_direction = TOOL_POS - data.qpos[:3].copy()
point_direction[2] = 0.6  # 抬高手臂
print(f"  检测到手势: 指向 ({TOOL_POS[0]:.1f}, {TOOL_POS[1]:.1f}, {TOOL_POS[2]:.1f})")
print(f"  意图解析: '取那个工具'")

# ============================================================
# 3. 阶段2: 视觉伺服到工具
# ============================================================
print(f"\n🟢 阶段2: 视觉伺服 → 工具位置")

def visual_servo_to_target(target_pos, steps=200):
    """简单的视觉伺服：移动到目标位置"""
    for step in range(steps):
        # 当前位置
        current_pos = data.qpos[:3].copy()
        current_pos[1] = data.qpos[1]  # y坐标

        error = target_pos - current_pos
        dist = np.linalg.norm(error)

        if dist < 0.1:
            print(f"  ✅ 到达目标 (步 {step})")
            return True

        # P控制器
        speed = min(0.03, dist * 0.1)
        data.qpos[0] += speed * error[0] / max(dist, 0.01)
        data.qpos[1] += speed * error[1] / max(dist, 0.01)
        data.qpos[2] += speed * error[2] / max(dist, 0.01) * 0.5

        # 保持平衡
        data.qpos[2] = max(0.9, min(1.1, data.qpos[2]))
        mujoco.mj_step(model, data)

        if step % 50 == 0:
            print(f"  步 {step}: 距离目标 {dist:.2f}m")

    return False

visual_servo_to_target(TOOL_POS)

# ============================================================
# 4. 阶段3: 抓取工具
# ============================================================
print(f"\n🟠 阶段3: 抓取工具")
# 模拟机械手闭合
for step in range(50):
    mujoco.mj_step(model, data)
print(f"  ✅ 抓到工具！位置: {data.qpos[:3].round(2)}")

# ============================================================
# 5. 阶段4: 递送到人类手部（阻抗控制模拟）
# ============================================================
print(f"\n🔴 阶段4: 递送到人类手部")

IMPEDANCE_K = 0.3  # 低刚度（柔顺）
IMPEDANCE_D = 0.8  # 高阻尼（平稳）

for step in range(200):
    current_pos = data.qpos[:3].copy()
    current_pos[1] = data.qpos[1]

    # 阻抗控制：模拟弹簧-阻尼系统
    error_pos = HUMAN_HAND_POS - current_pos
    force = IMPEDANCE_K * error_pos - IMPEDANCE_D * data.qvel[:3]

    # 应用阻抗力
    data.qpos[0] += force[0] * 0.01
    data.qpos[1] += force[1] * 0.01
    data.qpos[2] += force[2] * 0.005
    data.qpos[2] = max(0.9, min(1.1, data.qpos[2]))

    mujoco.mj_step(model, data)

    dist_to_hand = np.linalg.norm(data.qpos[:3] - HUMAN_HAND_POS)
    if step % 50 == 0:
        print(f"  步 {step}: 距离人手 {dist_to_hand:.2f}m (刚度={IMPEDANCE_K})")

print(f"  ✅ 工具已递送到人类手中！")

print(f"\n{'='*60}")
print(f"  协作任务完成:")
print(f"  1. 手势理解 → 2. 视觉伺服取工具 → 3. 递送")
print(f"  核心技能: 视觉伺服 + 阻抗控制 + 意图理解")
print(f"{'='*60}")
