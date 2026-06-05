"""
第13章 仿真实操：PPO人形机器人行走训练
================================================================
目标：使用PPO在 MuJoCo Humanoid 环境中训练行走策略。
前置依赖：pip install stable-baselines3 gymnasium[mujoco]
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

print("="*60)
print("  PPO 人形机器人行走训练")
print("="*60)

# 检查依赖
try:
    import mujoco
    print("  ✅ MuJoCo 已安装")
except:
    print("  ⚠️ 请安装: pip install mujoco gymnasium")

try:
    from stable_baselines3 import PPO
    print("  ✅ Stable-Baselines3 已安装")
except:
    print("  ⚠️ 请安装: pip install stable-baselines3")

# ============================================================
# 1. 创建环境
# ============================================================
print(f"\n--- 1. 创建训练环境 ---")

env = gym.make("Humanoid-v5", render_mode=None)
env = Monitor(env)

# 检查观测和动作空间
print(f"  观测空间: {env.observation_space.shape}")
print(f"  动作空间: {env.action_space.shape}")

# ============================================================
# 2. 配置PPO模型
# ============================================================
print(f"\n--- 2. 配置PPO模型 ---")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log="./rl_walk_tb/",
)

print(f"  ✅ PPO模型配置完成")

# ============================================================
# 3. 训练
# ============================================================
TOTAL_TIMESTEPS = 200_000  # 完整训练建议 1_000_000+
print(f"\n--- 3. 训练 ({TOTAL_TIMESTEPS}步) ---")

model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
model.save("ppo_humanoid_walk")
print(f"  ✅ 模型已保存: ppo_humanoid_walk.zip")

# ============================================================
# 4. 测试
# ============================================================
print(f"\n--- 4. 测试策略 ---")

test_env = gym.make("Humanoid-v5", render_mode=None)
obs, _ = test_env.reset()
total_reward = 0
episode_length = 0

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward
    episode_length += 1
    if terminated or truncated:
        break

print(f"  测试奖励: {total_reward:.1f}")
print(f"  存活步数: {episode_length}")
test_env.close()
env.close()

print(f"\n{'='*60}")
print(f"  🎯 训练完成！运行 tensorboard --logdir ./rl_walk_tb 查看曲线")
print(f"  快速测试参数: TOTAL_TIMESTEPS={TOTAL_TIMESTEPS}")
print(f"  建议最终训练: TOTAL_TIMESTEPS=1_000_000")
print(f"{'='*60}")
