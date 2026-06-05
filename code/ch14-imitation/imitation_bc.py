"""
第14章 仿真实操：行为克隆与演示数据采集
================================================================
目标：采集MuJoCo中的示范轨迹，用行为克隆训练策略。
"""

import mujoco
import numpy as np

print("="*60)
print("  模仿学习：行为克隆实验")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"\n--- 步骤1: 生成专家演示数据 ---")
# 使用规则控制器生成"专家"轨迹
# 简单规则：保持站立 + 轻微左右摆动

def expert_controller(model, data, step):
    """简单的专家控制器：保持姿态并产生摆动"""
    # 保持初始姿态
    action = np.zeros(model.nu)
    for i in range(model.nu):
        # 轻微正弦摆动
        action[i] = np.sin(step * 0.05 + i) * 0.1
    return action

DEMO_EPISODES = 5
DEMO_LENGTH = 200

demonstrations = []
for ep in range(DEMO_EPISODES):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0  # 初始高度
    states = []
    actions = []
    for step in range(DEMO_LENGTH):
        # 记录状态（关节角度）
        state = data.qpos[:10].copy()  # 前10个自由度
        # 获取专家动作
        action = expert_controller(model, data, step)
        states.append(state)
        actions.append(action)
        # 执行动作
        for i in range(min(model.nu, len(action))):
            data.ctrl[i] = action[i]
        mujoco.mj_step(model, data)
    demonstrations.append((np.array(states), np.array(actions)))
    print(f"  演示 {ep+1}/{DEMO_EPISODES}: 完成 {DEMO_LENGTH} 步")

print(f"\n--- 步骤2: 训练行为克隆策略 ---")
# 将演示数据展平训练简单的行为克隆
all_states = np.concatenate([d[0] for d in demonstrations])
all_actions = np.concatenate([d[1] for d in demonstrations])

print(f"  训练数据: {all_states.shape[0]}个样本")
print(f"  状态维度: {all_states.shape[1]}")
print(f"  动作维度: {all_actions.shape[1]}")

# 简单的线性BC策略: a = W @ s + b
# 最小二乘解
X = np.column_stack([all_states, np.ones(all_states.shape[0])])
Y = all_actions
W = np.linalg.lstsq(X, Y, rcond=None)[0]
W_bc, b_bc = W[:-1], W[-1]

def bc_policy(state):
    """行为克隆策略"""
    return W_bc @ state + b_bc

# 在训练集上验证
train_pred = X @ W
train_error = np.mean((train_pred - Y) ** 2)
print(f"  训练集MSE: {train_error:.6f}")

print(f"\n--- 步骤3: 测试BC策略（验证分布偏移）---")
mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0

for step in range(DEMO_LENGTH):
    state = data.qpos[:10].copy()
    action = bc_policy(state)
    for i in range(min(model.nu, len(action))):
        data.ctrl[i] = action[i]
    mujoco.mj_step(model, data)
    if step % 50 == 0:
        print(f"  步 {step}: 高度 = {data.qpos[2]:.3f}m")

print(f"\n  最终高度: {data.qpos[2]:.3f}m")
print(f"  说明: 如果高度明显下降，说明BC存在分布偏移")

print(f"\n{'='*60}")
print(f"  实验完成！BC是最简单但也是最有限的模仿学习方法。")
print(f"  分布偏移是其核心瓶颈——DAgger和扩散策略可以缓解。")
print(f"{'='*60}")
