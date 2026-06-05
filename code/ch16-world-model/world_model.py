"""
第16章 仿真实操：训练一个极简世界模型
================================================================
目标：用简单的VAE+RNN结构构建世界模型，在隐空间做推演。
"""

import numpy as np

print("="*60)
print("  极简世界模型实验")
print("="*60)

# ============================================================
# 1. 生成训练数据（简单动力学系统）
# ============================================================
print(f"\n--- 步骤1: 生成训练数据 ---")
# 简化：模拟一个2D阻尼弹簧系统
# 状态: [位置, 速度]，动作: 外力

def simulate_spring_mass(x0, v0, force, steps=100, dt=0.01, k=1.0, d=0.1):
    """模拟阻尼弹簧系统"""
    x, v = x0, v0
    states = []
    for _ in range(steps):
        states.append([x, v])
        a = -k * x - d * v + force
        v += a * dt
        x += v * dt
    return np.array(states)

# 生成数据集
np.random.seed(42)
dataset = []
for _ in range(200):
    x0 = np.random.uniform(-2, 2)
    v0 = np.random.uniform(-2, 2)
    force = np.random.uniform(-5, 5)
    traj = simulate_spring_mass(x0, v0, force)
    for t in range(len(traj)-1):
        dataset.append({
            'state': traj[t],
            'action': np.array([force]),
            'next_state': traj[t+1]
        })
print(f"  生成 {len(dataset)} 个训练样本")

# ============================================================
# 2. 极简世界模型（用NumPy模拟）
# ============================================================
print(f"\n--- 步骤2: 训练极简世界模型 ---")
# 线性世界模型: next_state = W @ [state, action] + b

X_data = np.array([np.concatenate([d['state'], d['action']]) for d in dataset])
Y_data = np.array([d['next_state'] for d in dataset])

# 用最小二乘法学习线性世界模型
X_with_bias = np.column_stack([X_data, np.ones(X_data.shape[0])])
W_wm = np.linalg.lstsq(X_with_bias, Y_data, rcond=None)[0]

def world_model(state, action):
    """学习到的世界模型"""
    x = np.concatenate([state, action, [1.0]])
    return W_wm @ x

# 验证模型
errors = []
for d in dataset:
    pred = world_model(d['state'], d['action'])
    error = np.linalg.norm(pred - d['next_state'])
    errors.append(error)
print(f"  预测误差: 均值={np.mean(errors):.4f}, 最大={np.max(errors):.4f}")

# ============================================================
# 3. 隐空间推演规划
# ============================================================
print(f"\n--- 步骤3: 用世界模型做推演规划 ---")
# 目标: 从x0=1.0, v0=0 开始，找到最优外力使系统在10步后稳定到x=0, v=0

x0, v0 = 1.0, 0.0
PLAN_HORIZON = 20

# 随机采样多个候选动作序列（暴力搜索）
best_cost = float('inf')
best_forces = None

for trial in range(1000):
    forces = np.random.uniform(-5, 5, PLAN_HORIZON)
    x, v = x0, v0
    cost = 0
    for f in forces:
        state = np.array([x, v])
        next_state = world_model(state, np.array([f]))
        x, v = next_state
        cost += x**2 + v**2 + 0.01 * f**2  # 位置+速度+控制代价
    if cost < best_cost:
        best_cost = cost
        best_forces = forces

print(f"  最优规划代价: {best_cost:.4f}")
print(f"  最优动作序列 (前5步): {best_forces[:5].round(3)}")

# 用最优动作序列执行
x, v = x0, v0
for t, f in enumerate(best_forces):
    state = np.array([x, v])
    next_state = world_model(state, np.array([f]))
    x, v = next_state
    if t % 5 == 0:
        print(f"  步 {t}: 位置={x:.3f}, 速度={v:.3f}, 外力={f:.2f}")

print(f"\n  最终状态: 位置={x:.3f}, 速度={v:.3f}")
print(f"  目标状态: 位置=0.0, 速度=0.0")

print(f"\n{'='*60}")
print(f"  实验完成！世界模型的核心思想：学习动力学，在隐空间推演。")
print(f"  此处使用线性模型做演示，实际场景需使用神经网络(Dreamer)。")
print(f"{'='*60}")
