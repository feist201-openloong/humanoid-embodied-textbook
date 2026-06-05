---
sidebar_position: 1
---

# 第13章：强化学习与机器人

> **本章导语**：强化学习（RL）是具身智能的核心技术之一，它使机器人能够通过与环境的试错交互自主学会复杂行为。本章从MDP建模出发，深入PPO算法原理，并在 MuJoCo 中训练人形机器人行走策略。

---

## 13.1 强化学习基础

### 13.1.1 马尔可夫决策过程（MDP）

强化学习问题通常形式化为马尔可夫决策过程（Markov Decision Process, MDP），由五元组 $(S, A, P, R, \gamma)$ 定义：

- $S$：状态空间（机器人能观测到的一切）
- $A$：动作空间（机器人能执行的动作）
- $P(s' \mid s, a)$：状态转移概率（环境动力学）
- $R(s, a)$：奖励函数（告诉我们什么是"好"的行为）
- $\gamma \in [0, 1]$：折扣因子（平衡即时和未来奖励）

**智能体（Agent）**的目标是找到最优策略 $\pi^*(a \mid s)$，使得期望累计折扣奖励最大化：

$$ J(\pi) = \mathbb&#123;E&#125;_&#123;\tau \sim \pi&#125; \left[ \sum_&#123;t=0&#125;^\infty \gamma^t R(s_t, a_t) \right] $$

其中 $\tau = (s_0, a_0, s_1, a_1, \ldots)$ 是一条轨迹。

### 13.1.2 价值函数与策略

**状态价值函数** $V^\pi(s)$：从状态 $s$ 开始，遵循策略 $\pi$ 能获得的期望回报。

$$ V^\pi(s) = \mathbb&#123;E&#125;_&#123;a \sim \pi(\cdot \mid s)&#125; \left[ R(s, a) + \gamma \mathbb&#123;E&#125;_&#123;s' \sim P(\cdot \mid s, a)&#125; [V^\pi(s')] \right] $$

**动作价值函数** $Q^\pi(s, a)$：从状态 $s$ 采取动作 $a$ 后，遵循策略 $\pi$ 的期望回报。

**优势函数** $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$：衡量某个动作相对于平均水平的优劣。

### 13.1.3 强化学习分类

| 分类 | 方法 | 特点 |
|:----|:----|:------|
| **基于价值** | DQN, Double DQN | 学习 $Q$ 函数，策略从 $Q$ 导出 |
| **基于策略** | REINFORCE, PPO | 直接优化策略 $\pi$ |
| **Actor-Critic** | A2C, SAC, TD3 | 同时学习策略（Actor）和价值（Critic）|
| **基于模型** | Dreamer, MuZero | 学习环境模型，在模型中规划 |
| **无模型** | PPO, SAC, TD3 | 不学习模型，直接学习策略 |

## 13.2 PPO算法

PPO（Proximal Policy Optimization）是目前最广泛使用的强化学习算法，因其稳定性和性能的平衡而成为机器人RL的事实标准。

### 13.2.1 策略梯度

策略梯度方法直接优化策略 $\pi_\theta$ 的参数 $\theta$，梯度为：

$$ \nabla_\theta J(\theta) = \mathbb&#123;E&#125;_&#123;\tau \sim \pi_\theta&#125; \left[ \sum_&#123;t=0&#125;^T \nabla_\theta \log \pi_\theta(a_t \mid s_t) A_t \right] $$

直觉：增加带来高优势的动作的概率，降低带来低优势的动作的概率。

### 13.2.2 PPO的核心创新：裁剪

PPO的关键创新是**裁剪目标函数**（Clipped Objective），防止策略更新过大导致训练崩溃：

$$ L^&#123;\text&#123;CLIP&#125;&#125;(\theta) = \mathbb&#123;E&#125;_t \left[ \min\left( r_t(\theta) A_t, \ \text&#123;clip&#125;(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] $$

其中 $r_t(\theta) = \frac&#123;\pi_\theta(a_t \mid s_t)&#125;&#123;\pi_&#123;\theta_&#123;\text&#123;old&#125;&#125;&#125;(a_t \mid s_t)&#125;$ 是新旧策略的概率比。

当概率比超出 $[1-\epsilon, 1+\epsilon]$ 范围时，梯度被裁剪，防止一步更新过大。$\epsilon$ 通常取 0.2。

### 13.2.3 PPO超参数

| 参数 | 典型值 | 作用 |
|:----|:------:|:-----|
| 学习率 | $3 \times 10^&#123;-4&#125;$ | 更新步长 |
| clip范围 ($\epsilon$) | 0.2 | 限制策略更新幅度 |
| GAE $\lambda$ | 0.95 | 优势估计的衰减 |
| 折扣因子 $\gamma$ | 0.99 | 未来奖励的衰减 |
| 每批步数 $n_&#123;\text&#123;steps&#125;&#125;$ | 2048 | 收集数据的长度 |
| 训练轮数 $n_&#123;\text&#123;epochs&#125;&#125;$ | 10 | 每批数据的利用次数 |

## 13.3 奖励函数设计

奖励函数是RL中最关键的"艺术"。对于人形机器人行走，典型奖励函数包括：

$$ R = w_1 R_&#123;\text&#123;forward&#125;&#125; + w_2 R_&#123;\text&#123;alive&#125;&#125; + w_3 R_&#123;\text&#123;energy&#125;&#125; + w_4 R_&#123;\text&#123;smooth&#125;&#125; + w_5 R_&#123;\text&#123;orientation&#125;&#125; $$

- $R_&#123;\text&#123;forward&#125;&#125;$：鼓励前进速度（正值随速度增加）
- $R_&#123;\text&#123;alive&#125;&#125;$：每存活一步给一个小奖励（鼓励不摔倒）
- $R_&#123;\text&#123;energy&#125;&#125;$：惩罚过大的关节力矩（鼓励节能）
- $R_&#123;\text&#123;smooth&#125;&#125;$：惩罚关节角速度突变（鼓励平滑运动）
- $R_&#123;\text&#123;orientation&#125;&#125;$：鼓励躯干保持竖直

## 13.4 MuJoCo RL生态

MuJoCo与强化学习的集成非常成熟：

- **Gymnasium**：标准的RL环境接口，提供 `Humanoid-v5`、`HumanoidStandup-v5` 等环境
- **Stable-Baselines3**：PPO/SAC/TD3的标准实现
- **Isaac Gym**：NVIDIA的高性能RL训练平台（支持GPU并行仿真）

---

## 🛠️ 仿真实操：PPO人形机器人行走训练

> 🕐 预计时长：60-120分钟 · 📁 代码位置：`code/ch13-rl-walk/rl_walk.py`

### 实验目标
使用PPO在 MuJoCo Humanoid 环境中训练行走策略。

### 实验步骤
1. 创建 Gymnasium 的 Humanoid-v5 环境
2. 配置 PPO 模型（Stable-Baselines3）
3. 训练 200,000 步
4. 测试训练效果
5. 可视化训练曲线

### 延伸思考
1. 尝试修改奖励函数中的权重，观察行为变化
2. 增加训练步数到 1,000,000，行走效果会更好吗？
3. 将 PPO 换为 SAC，比较两种算法的效果差异

---

## 本章小结

- RL通过试错交互最大化累计奖励
- PPO用裁剪目标函数保证训练稳定性
- 奖励函数设计是RL成功的关键
- MuJoCo + Gymnasium + SB3 是人形机器人RL的标准工具链

## 习题

1. **选择题**：PPO的核心创新是什么？
   A. 经验回放  B. 裁剪目标函数  C. 双Q网络  D. 优先采样

2. **简答题**：为什么强化学习需要"探索"？探索太多或太少会有什么后果？

3. **简答题**：稀疏奖励和密集奖励的区别是什么？各有什么优缺点？

4. **计算题**：GAE的 $\lambda = 0$ 和 $\lambda = 1$ 分别对应什么情况？

## 参考文献
1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347.*
2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction.* MIT Press.
3. Raffin, A., et al. (2021). "Stable-Baselines3: Reliable Reinforcement Learning Implementations." *JMLR.*
