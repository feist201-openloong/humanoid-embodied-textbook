"""
第24章 仿真实操：安全控制器设计
================================================================
目标：实现安全监控系统，在 MuJoCo 中验证三级安全响应。
"""

import mujoco
import numpy as np

print("="*60)
print("  安全控制器设计实验")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0

# ============================================================
# 安全监控器
# ============================================================
class SafetyMonitor:
    """三级安全监控器"""
    # 安全阈值
    MAX_JOINT_VEL = 15.0      # rad/s - 关节速度
    MAX_HEIGHT_DROP = 0.4     # m - 最大高度下降（摔倒检测）
    MAX_CONTROL_FORCE = 200.0 # N - 最大控制力

    # 安全状态
    SAFE = 0
    WARNING = 1
    SLOW_DOWN = 2
    EMERGENCY_STOP = 3

    def __init__(self):
        self.state = self.SAFE
        self.initial_height = 1.0

    def check(self, data, step):
        """检测安全状态"""
        status_msgs = []

        # 1. 关节速度检测
        max_vel = np.max(np.abs(data.qvel))
        if max_vel > self.MAX_JOINT_VEL * 1.5:
            self.state = self.EMERGENCY_STOP
            status_msgs.append(f"🚨 紧急停止: 关节速度异常 ({max_vel:.1f} rad/s)")
        elif max_vel > self.MAX_JOINT_VEL:
            self.state = self.SLOW_DOWN
            status_msgs.append(f"⚠️ 降速: 关节速度过高 ({max_vel:.1f} rad/s)")
        else:
            self.state = self.SAFE

        # 2. 高度检测（摔倒检测）
        height_drop = self.initial_height - data.qpos[2]
        if height_drop > self.MAX_HEIGHT_DROP:
            self.state = self.EMERGENCY_STOP
            status_msgs.append(f"🚨 紧急停止: 检测到摔倒 (高度下降{height_drop:.2f}m)")

        # 3. 控制力检测
        if model.nu > 0:
            max_ctrl = np.max(np.abs(data.ctrl[:min(10, model.nu)]))
            if max_ctrl > self.MAX_CONTROL_FORCE * 0.9 and self.state == self.SAFE:
                self.state = self.WARNING
                status_msgs.append(f"⚡ 警告: 控制力接近上限 ({max_ctrl:.0f}N)")

        return self.state, status_msgs

    def reset(self, height=1.0):
        self.state = self.SAFE
        self.initial_height = height

# ============================================================
# 仿真测试
# ============================================================
print(f"\n--- 场景1: 正常运行 ---")
safety = SafetyMonitor()

for step in range(200):
    # 模拟正常控制
    if model.nu > 0:
        data.ctrl[:model.nu] = 0.0

    mujoco.mj_step(model, data)
    state, msgs = safety.check(data, step)
    if state >= safety.WARNING:
        for m in msgs:
            print(f"  {m}")
        break
else:
    print(f"  ✅ 正常运行200步，无安全事件")

print(f"\n--- 场景2: 模拟失控（异常控制力）---")
safety.reset(1.0)

for step in range(100):
    # 模拟失控：施加超大控制力
    if model.nu > 0:
        data.ctrl[:min(5, model.nu)] = 500.0

    mujoco.mj_step(model, data)
    state, msgs = safety.check(data, step)
    if state >= safety.EMERGENCY_STOP:
        for m in msgs:
            print(f"  {m}")
        # 执行紧急停止
        if model.nu > 0:
            data.ctrl[:model.nu] = 0.0
        print(f"  🔒 紧急停止已触发，控制力归零")
        break

print(f"\n--- 安全系统总结 ---")
print(f"  三级安全响应:")
print(f"    1. 警告 (WARNING): 参数接近阈值，记录日志")
print(f"    2. 降速 (SLOW_DOWN): 限制速度/力矩，防止恶化")
print(f"    3. 紧急停止 (EMERGENCY_STOP): 立即切断动力")
print(f"  ✅ 安全监控器验证完成")

print(f"\n{'='*60}")
print(f"  实验完成！安全控制器是机器人系统最重要的组件之一。")
print(f"  建议在实际系统中增加硬件冗余（双通道急停）。")
print(f"{'='*60}")
