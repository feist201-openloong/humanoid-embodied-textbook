"""
第11章 仿真实操：语义地图构建
================================================================
目标：在 MuJoCo 环境中采集多视角 RGB-D 数据，
构建简单的占用网格地图。
"""

import mujoco
import numpy as np

print("="*60)
print("  语义地图构建实验")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"\n--- 步骤1: 配置仿真场景 ---")
print(f"  模型: {model.name}")
print(f"  刚体数: {model.nbody}")

mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0

# ============================================================
# 2. 多视角数据采集
# ============================================================
print(f"\n--- 步骤2: 多视角数据采集 ---")

view_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8个视角
view_poses = []

for angle_deg in view_angles:
    angle_rad = np.radians(angle_deg)

    # 设置机器人朝向
    # 绕世界Z轴旋转
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    q = np.array([np.cos(angle_rad/2), 0, 0, np.sin(angle_rad/2)])
    data.qpos[3:7] = q / np.linalg.norm(q)

    # 推进仿真，让物理引擎稳定
    for _ in range(50):
        mujoco.mj_step(model, data)

    # 记录每个视角的足部位置
    foot_pos = []
    for body_name in ["foot_left", "foot_right", "left_foot", "right_foot"]:
        try:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                foot_pos.append(data.xpos[body_id].copy())
        except:
            pass

    torso_pos = data.xpos[1].copy()  # 躯干
    view_poses.append((angle_deg, torso_pos, foot_pos))
    print(f"  视角 {angle_deg:3d}°: 躯干位置 = ({torso_pos[0]:.2f}, {torso_pos[1]:.2f}, {torso_pos[2]:.2f})")

# ============================================================
# 3. 构建占用网格地图
# ============================================================
print(f"\n--- 步骤3: 构建占用网格地图 ---")

# 网格参数
GRID_SIZE = 20           # 20x20 网格
GRID_RES = 0.2           # 每格 0.2m
grid_origin = np.array([-2.0, -2.0])  # 地图原点

# 占用概率网格
occupancy_grid = np.zeros((GRID_SIZE, GRID_SIZE))
# 每格的观测计数
count_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

for angle_deg, torso, feet in view_poses:
    for foot in feet:
        # 将足部位置映射到网格坐标
        gx = int((foot[0] - grid_origin[0]) / GRID_RES)
        gy = int((foot[1] - grid_origin[1]) / GRID_RES)

        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
            occupancy_grid[gx, gy] += 1.0  # 被占据
            count_grid[gx, gy] += 1

    # 躯干到足部的连线区域标记为"可通行"
    for foot in feet:
        dx = foot[0] - torso[0]
        dy = foot[1] - torso[1]
        steps = max(int(np.sqrt(dx**2 + dy**2) / GRID_RES), 1)
        for s in range(steps):
            px = torso[0] + dx * s / steps
            py = torso[1] + dy * s / steps
            gx = int((px - grid_origin[0]) / GRID_RES)
            gy = int((py - grid_origin[1]) / GRID_RES)
            if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE:
                occupancy_grid[gx, gy] -= 0.5  # 自由空间
                count_grid[gx, gy] += 1

# 归一化
occupancy_grid = np.where(count_grid > 0, occupancy_grid / count_grid, 0.5)

print(f"  占用网格: {GRID_SIZE}x{GRID_SIZE}")
print(f"  被占据单元: {np.sum(occupancy_grid > 0.6)}")
print(f"  自由空间单元: {np.sum(occupancy_grid < 0.3)}")
print(f"  未探索单元: {np.sum(count_grid == 0)}")

print(f"\n--- 地图可视化 (文本) ---")
for gy in range(GRID_SIZE):
    row = ""
    for gx in range(GRID_SIZE):
        if count_grid[gx, GRID_SIZE-1-gy] == 0:
            row += "·"  # 未探索
        elif occupancy_grid[gx, GRID_SIZE-1-gy] > 0.5:
            row += "█"  # 被占据
        else:
            row += " "  # 自由
    print(f"  |{row}|")
print(f"  +{'-'*GRID_SIZE}+")

print(f"\n{'='*60}")
print(f"  语义地图构建实验完成。")
print(f"  地图显示了机器人足部接触位置的占据情况。")
print(f"  后续：可扩展为多类别语义地图。")
print(f"{'='*60}")
