"""
第8章 仿真实操：PPO站立策略训练
================================================================
目标：使用PPO算法在MuJoCo环境中训练一个抗扰动站立策略。
前置依赖：pip install stable-baselines3
"""

import gymnasium as gym
import numpy as np

print("="*60)
print("  PPO站立策略训练 — 准备阶段")
print("="*60)

# 检查依赖
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback
    print("  ✅ Stable-Baselines3 已安装")
except ImportError:
    print("  ⚠️ 需要安装 Stable-Baselines3: pip install stable-baselines3")
    print("  安装后重新运行本脚本。")
    exit(1)

try:
    import gymnasium.envs.mujoco
    print("  ✅ MuJoCo Gymnasium 环境可用")
except:
    print("  ⚠️ MuJoCo Gymnasium 环境可能需要安装: pip install gymnasium[mujoco]")

# ============================================================
# 环境配置
# ============================================================
ENV_NAME = "HumanoidStandup-v5"  # MuJoCo 的站立环境

print(f"\n--- 配置训练环境 ---")
print(f"  环境: {ENV_NAME}")

# 创建矢量环境
try:
    env = make_vec_env(ENV_NAME, n_envs=4, seed=42)
    print(f"  ✅ 环境创建成功 (4个并行环境)")

    # 检查环境
    obs = env.reset()
    print(f"  观测空间: {env.observation_space.shape}")
    print(f"  动作空间: {env.action_space.shape}")

    # ============================================================
    # PPO模型配置
    # ============================================================
    print(f"\n--- 配置PPO模型 ---")
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
        tensorboard_log="./ppo_standing_tensorboard/",
    )
    print(f"  ✅ PPO模型配置完成")

    # ============================================================
    # 训练
    # ============================================================
    print(f"\n--- 开始训练 ---")
    print(f"  计划训练步数: 200,000")
    print(f"  (注: 完整训练需要较长时间，可将步数减少进行快速测试)")

    TOTAL_TIMESTEPS = 200_000  # 完整训练建议 500_000 - 1_000_000
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    print(f"  ✅ 训练完成 ({TOTAL_TIMESTEPS}步)")

    # ============================================================
    # 保存模型
    # ============================================================
    model.save("ppo_standing_policy")
    print(f"  ✅ 策略已保存至 ppo_standing_policy.zip")

    # ============================================================
    # 测试策略
    # ============================================================
    print(f"\n--- 测试训练好的策略 ---")
    test_env = gym.make(ENV_NAME, render_mode=None)
    obs, _ = test_env.reset()
    total_reward = 0
    for i in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"  测试奖励: {total_reward:.1f}")
    print(f"  存活步数: {i+1}")
    test_env.close()

    print(f"\n  🎯 训练完成! 可以使用 ppo_standing_policy.zip 进行后续测试。")
    print(f"  提示: 可尝试在策略中添加域随机化，提高抗扰动能力。")

except Exception as e:
    print(f"  ❌ 环境创建失败: {e}")
    print(f"  请确保安装了所有依赖:")
    print(f"    pip install stable-baselines3 gymnasium[mujoco]")

print(f"\n{'='*60}")
print(f"  第8章仿真实操说明:")
print(f"  本脚本实现了PPO训练站立策略的完整流水线。")
print(f"  实际训练（50万步）建议在具有CUDA的机器上运行。")
print(f"  快速测试可将 TOTAL_TIMESTEPS 减少至 10,000。")
print(f"{'='*60}")
