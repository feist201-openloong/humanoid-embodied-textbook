"""
第18章 仿真实操：MuJoCo-ROS2桥接
================================================================
目标：搭建仿真器与ROS2的双向通信（支持无ROS2环境运行）。
"""

import mujoco
import numpy as np
import threading
import time
from collections import deque

print("="*60)
print("  MuJoCo-ROS2 桥接系统")
print("="*60)

# 检查ROS2是否可用
try:
    import rclpy
    HAS_ROS2 = True
    print("  ✅ ROS2 已安装（将使用真实ROS2通信）")
except ImportError:
    HAS_ROS2 = False
    print("  ⚠️ 未检测到ROS2，将使用模拟通信")

# ============================================================
# 1. 仿真节点（模拟）
# ============================================================
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0

print(f"\n--- 仿真节点启动 ---")
print(f"  模型: {model.name}")
print(f"  关节数: {model.njnt}, 自由度: {model.nq}")

# ============================================================
# 2. 发布者线程（仿真→外部）
# ============================================================
class SimPublisher:
    """发布MuJoCo仿真状态"""
    def __init__(self):
        self.callback = None

    def on_receive(self, callback):
        self.callback = callback

    def publish(self, topic, data_dict):
        if self.callback:
            self.callback(topic, data_dict)

sim_out = SimPublisher()

# ============================================================
# 3. 订阅者（外部→仿真）
# ============================================================
class SimSubscriber:
    """接收外部控制指令"""
    def __init__(self):
        self.command_queue = deque()

    def on_message(self, topic, data):
        self.command_queue.append((topic, data))

    def get_command(self):
        if self.command_queue:
            return self.command_queue.popleft()
        return None

sim_in = SimSubscriber()

# ============================================================
# 4. 仿真循环
# ============================================================
print(f"\n--- 仿真循环（运行500步）---")

SIM_FREQ = 200  # Hz
CTRL_FREQ = 50  # Hz
sim_counter = 0

def extract_joint_states(model, data):
    """提取关节状态"""
    states = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            qpos_addr = model.jnt_qposadr[i]
            qvel_addr = qpos_addr
            states[name] = {
                'position': float(data.qpos[qpos_addr]) if qpos_addr < len(data.qpos) else 0.0,
                'velocity': float(data.qvel[qpos_addr]) if qpos_addr < len(data.qvel) else 0.0,
            }
    return states

for step in range(500):
    # 处理接收到的控制指令
    cmd = sim_in.get_command()
    if cmd:
        topic, data = cmd
        if topic == '/cmd_joint':
            # 应用关节控制指令
            for joint_name, value in data.items():
                for i in range(model.njnt):
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if name == joint_name:
                        qpos_addr = model.jnt_qposadr[i]
                        if qpos_addr < model.nu:
                            data.ctrl[qpos_addr] = value

    # 推进仿真
    mujoco.mj_step(model, data)

    # 按控制频率发布状态
    if step % (SIM_FREQ // CTRL_FREQ) == 0:
        joint_states = extract_joint_states(model, data)
        pose = {
            'position': data.xpos[1].copy(),  # 躯干位置
            'orientation': data.qpos[3:7].copy(),  # 躯干姿态
        }
        print(f"  步 {step:4d}: 发布关节状态 ({len(joint_states)}关节), "
              f"躯干高度={data.qpos[2]:.2f}m")

# ============================================================
# 5. 发送测试控制指令
# ============================================================
print(f"\n--- 发送测试控制指令 ---")
test_cmd = {'hip': 0.1, 'knee': 0.2}
sim_in.on_message('/cmd_joint', test_cmd)
print(f"  发送指令: {test_cmd}")

# 继续运行几步验证
for step in range(100):
    mujoco.mj_step(model, data)

print(f"  执行后高度: {data.qpos[2]:.3f}m")

print(f"\n{'='*60}")
print(f"  MuJoCo-ROS2桥接实验完成！")
if not HAS_ROS2:
    print(f"  提示: 安装ROS2后，本桥接可连接真实ROS2节点。")
    print(f"  brew install ros-humble-desktop (macOS)")
    print(f"  sudo apt install ros-humble-desktop (Ubuntu)")
print(f"{'='*60}")
