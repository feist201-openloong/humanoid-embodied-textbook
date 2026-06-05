"""
第4章 仿真实操：MuJoCo 完整入门 —— 机器人站立仿真
================================================================
目标：从零开始，跑通一个完整的人形机器人站立仿真。
涵盖：模型加载 → 初始化 → 控制循环 → 可视化 → 数据记录。
"""

import mujoco
import numpy as np

print("="*60)
print("  MuJoCo 完整入门：机器人站立仿真")
print("="*60)

# ============================================================
# 步骤1：加载模型
# ============================================================
print("\n[步骤1/5] 加载模型...")

xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
print(f"  模型路径: {xml_path}")

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"  模型名称: {model.name}")
print(f"  自由度 (nq): {model.nq}, 执行器 (nu): {model.nu}")

# ============================================================
# 步骤2：初始化状态
# ============================================================
print("\n[步骤2/5] 初始化状态...")
mujoco.mj_resetData(model, data)
data.qpos[2] = 1.2  # 设置躯干高度

print(f"  初始躯干高度: {data.qpos[2]:.3f}m")
print(f"  初始姿态 (前6个qpos): {data.qpos[:6].round(3)}")

# ============================================================
# 步骤3：配置控制参数
# ============================================================
print("\n[步骤3/5] 配置控制参数...")
kp = 100.0   # 比例增益
kd = 10.0    # 微分增益
stand_qpos = data.qpos.copy()

print(f"  Kp = {kp}, Kd = {kd}")
print(f"  目标姿态已记录 (保持初始站立姿态)")

# 查找关节位置控制的执行器映射
joint_names = []
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_names.append((i, name))

# ============================================================
# 步骤4：运行仿真循环
# ============================================================
print("\n[步骤4/5] 运行仿真（500步 = 约2.5秒）...")

SIMULATION_STEPS = 500
height_log = []
time_log = []

mujoco.mj_resetData(model, data)
data.qpos[2] = 1.2

for step in range(SIMULATION_STEPS):
    # 应用简单的 PD 控制：保持站立姿态
    # 对每个可控制的位置，计算位置误差并施加力
    for jnt_idx, jnt_name in joint_names:
        qpos_addr = model.jnt_qposadr[jnt_idx]
        qvel_addr = model.jnt_dofadr[jnt_idx] if jnt_idx < model.nv else qpos_addr

        if qpos_addr < len(stand_qpos) and qpos_addr < len(data.qpos):
            # 位置误差
            pos_err = stand_qpos[qpos_addr] - data.qpos[qpos_addr]
            # 速度 (近似：用 qvel)
            vel = data.qvel[qpos_addr] if qpos_addr < len(data.qvel) else 0
            # PD 控制力 (近似作用)
            control_force = kp * pos_err - kd * vel

            # 如果有执行器，通过执行器施加控制
            if qpos_addr < model.nu:
                data.ctrl[qpos_addr] = np.clip(control_force, -100, 100)

    # 仿真步进
    mujoco.mj_step(model, data)

    # 记录数据
    if step % 10 == 0:
        height_log.append(data.qpos[2])
        time_log.append(step * model.opt.timestep)

    # 每100步打印进度
    if step % 100 == 0:
        print(f"  步 {step:4d} | 高度: {data.qpos[2]:.3f}m | "
              f"躯干旋转 (w): {data.qpos[3]:.3f}")

# 结果分析
if height_log:
    final_height = height_log[-1]
    height_drop = 1.2 - final_height
    print(f"\n  最终高度: {final_height:.3f}m")
    print(f"  高度变化: {height_drop:.3f}m")
    print(f"  站立稳定性: {'✅ 稳定站立中' if height_drop < 0.5 else '⚠ 机器人倒地了'}")

print(f"\n{'-'*40}")
print(f"  提示：将代码中的 kp/kd 参数调整后重新运行，观察变化")
print(f"  kp 越大 → 恢复力越强，但可能震荡")
print(f"  kd 越大 → 阻尼越强，但响应变慢")
print(f"{'-'*40}")

# ============================================================
# 步骤5：交互式可视化（可选）
# ============================================================
print("\n[步骤5/5] 交互式可视化 (可选)...")
print(f"  按 ESC 退出查看器")

try:
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.2

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(2000):
            mujoco.mj_step(model, data)
            viewer.sync()
except Exception as e:
    print(f"  查看器不可用: {e}")

print(f"\n{'='*60}")
print(f"  第4章仿真实操完成！")
print(f"  你已经成功跑通了 MuJoCo 完整仿真流程。")
print(f"  接下来可以前往 code/ch05-hardware-sim/ 继续下一章。")
print(f"{'='*60}")
