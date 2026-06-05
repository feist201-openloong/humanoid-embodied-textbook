"""
第9章 仿真实操：RGB-D感知实验
================================================================
目标：在 MuJoCo 仿真中挂载 RGB-D 相机，
使用 OpenCV 实现物体的检测和三维定位。
"""

import mujoco
import numpy as np

print("="*60)
print("  RGB-D 感知实验")
print("="*60)

# 加载模型（使用 MuJoCo 内置场景）
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"\n--- 步骤1: 获取仿真相机图像 ---")
# 获取仿真中可用的相机
print(f"  模型相机数: {model.ncam}")

# 查找相机位置和名称
for i in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    print(f"  相机 {i}: {name or '(unnamed)'}")

print(f"\n--- 步骤2: 渲染 RGB-D 图像（无可视化环境） ---")
# 创建渲染缓冲区
width, height = 640, 480
rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
depth_buffer = np.zeros((height, width), dtype=np.float32)

# 使用 MuJoCo 渲染
mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0
for _ in range(100):
    mujoco.mj_step(model, data)

# 渲染 RGB 图像
mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height),
                   rgb_buffer, None)  # 简化渲染

print(f"  渲染分辨率: {width}x{height}")
print(f"  RGB 缓冲区形状: {rgb_buffer.shape}")
print(f"  RGB 值范围: {rgb_buffer.min()} ~ {rgb_buffer.max()}")

print(f"\n--- 步骤3: 模拟物体检测 ---")
# 在 RGB 图像中搜索特定颜色区域
# 这里使用模拟方式：假设检测到一个位于 (x=0.3, y=0.0, z=0.5) 的物体
simulated_object_pos = np.array([0.3, 0.0, 0.5])
print(f"  检测到物体: 位置 = {simulated_object_pos.round(3)}")

# 计算物体在机器人坐标系中的位置
torso_pos = data.xpos[1]  # 躯干位置
object_rel_pos = simulated_object_pos - torso_pos
print(f"  相对于躯干的位置: {object_rel_pos.round(3)}")

print(f"\n--- 步骤4: 生成控制指令 ---")
# 根据物体位置计算机器人需要旋转的角度
import math
angle_to_object = math.atan2(object_rel_pos[0], object_rel_pos[2])
print(f"  机器人需要旋转: {math.degrees(angle_to_object):.1f}°")
print(f"  动作: {'右转' if angle_to_object > 0 else '左转'} 以面对物体")

print(f"\n{'='*60}")
print(f"  说明：实际运行时需要 OpenCV 和显示环境支持。")
print(f"  建议安装: pip install opencv-python")
print(f"  然后用 cv2.imshow() 可视化 MuJoCo 渲染结果。")
print(f"{'='*60}")
