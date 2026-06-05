---
sidebar_position: 4
---

# 第4章：开发环境与工具链

> **本章导语**：在开始学习和动手之前，我们需要搭建一套完整的开发环境。本章将引导你安装和配置本书所需的所有工具——从 Python 科学计算栈到 MuJoCo 仿真器，从 Git 版本控制到深度学习框架。请跟随本章的指导一步步操作，如果你已经熟悉某些工具，可以跳过对应的章节。

---

## 4.1 Python 科学计算栈

### 4.1.1 安装 Python

本书要求 Python **3.10 或更高版本**。

**推荐安装方式**：
- **macOS/Linux**：使用 Homebrew 或 apt 安装，推荐使用 pyenv 管理多个版本
- **Windows**：从 [python.org](https://python.org) 下载安装包，**务必勾选"Add Python to PATH"**

验证安装：
```bash
python --version
# 输出: Python 3.10.x 或更高
```

### 4.1.2 NumPy 基础

NumPy 是 Python 科学计算的基础库，提供了高效的数组操作和线性代数功能。

```python
import numpy as np

# 创建数组
a = np.array([[1, 2], [3, 4]])          # 从列表创建
b = np.zeros((3, 3))                     # 全零矩阵
c = np.eye(3)                            # 单位矩阵
d = np.linspace(0, 1, 5)                 # 0到1等间距取5个点
e = np.random.randn(3, 3)               # 标准正态分布随机数

# 索引与切片
a[0, 1]       # 第0行第1列 → 2
a[:, 0]       # 第0列所有行 → [1, 3]
a[0, :]       # 第0行的所有列 → [1, 2]

# 数组运算 —— 广播机制
x = np.array([[1, 2], [3, 4]])
y = np.array([10, 20])                    # 形状 (2,)
print(x + y)                             # 广播: [[11, 22], [13, 24]]

# 线性代数
R = np.array([[0, -1], [1, 0]])          # 旋转矩阵
v = np.array([1, 0])
v_rotated = R @ v                        # 矩阵乘法
print(f"旋转后: {v_rotated}")            # [0, 1]

# 特征分解
eigvals, eigvecs = np.linalg.eig(R)
```

**NumPy 是仿真实操代码的基础依赖**。后续所有章节的代码都依赖 NumPy，请务必熟悉最常用的操作。

### 4.1.3 Matplotlib 可视化

```python
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 2*np.pi, 100)
plt.plot(t, np.sin(t), label='sin')
plt.plot(t, np.cos(t), label='cos')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('正弦和余弦函数')
plt.grid(True)
plt.show()
```

在后续章节中，我们将使用 Matplotlib 绘制机器人运动轨迹、训练曲线、传感器数据等。

### 4.1.4 Jupyter Notebook

Jupyter Notebook 是交互式开发的利器，特别适合教学和实验。安装方式：

```bash
pip install jupyter
jupyter notebook    # 启动
```

使用 Jupyter 的好处：
1. **逐单元格执行**：可以逐步理解代码
2. **图文并茂**：代码、结果、注释在一个页面
3. **交互式调试**：随时修改变量重新计算

本书的所有实操代码也可以通过 Jupyter 来逐步学习和调试。

### 4.1.5 虚拟环境管理

为不同项目隔离依赖是良好实践。推荐使用 venv（Python 内置）：

```bash
# 创建虚拟环境
python -m venv .venv

# 激活
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 退出
deactivate

# 导出依赖
pip freeze > requirements.txt
```

本书每个章节的实操代码目录都包含独立的 `requirements.txt` 文件。建议为每个章节创建独立的虚拟环境。

## 4.2 MuJoCo 安装与配置

### 4.2.1 安装

MuJoCo（Multi-Joint dynamics with Contact）是一个开源物理引擎，由 Google DeepMind 维护。安装非常简单：

```bash
pip install mujoco
```

验证安装：
```python
import mujoco
print(mujoco.__version__)       # 应输出 3.x

# 尝试加载内置模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
print(f"加载成功! 模型自由度: {model.nq}")
```

**常见问题排错**：

| 问题 | 原因 | 解决方案 |
|:----|:-----|:---------|
| `pip install` 报错 | 网络问题 | 使用国内镜像 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mujoco` |
| `libGL.so.1` 错误 | 缺少 OpenGL 库 | `sudo apt install libgl1-mesa-glx`（Linux） |
| 查看器黑屏 | 无显示驱动 | macOS/Linux 桌面环境需要安装显卡驱动 |
| 仿真卡顿 | 模型复杂度过高 | 减小 `model.opt.timestep` 或简化模型 |

### 4.2.2 第一个仿真程序

```python
import mujoco
import mujoco.viewer
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path(
    str(mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"))
data = mujoco.MjData(model)

# 仿真循环
for step in range(1000):
    mujoco.mj_step(model, data)
    if step % 100 == 0:
        print(f"步 {step}: 高度 = {data.qpos[2]:.3f}m")
```

这个程序加载人形机器人模型，运行被动仿真。因为没有任何控制，机器人从初始姿态开始，在重力作用下下落并倒地。

### 4.2.3 MJCF 模型格式入门

MuJoCo 使用 XML 格式的模型文件（称为 MJCF），描述机器人的物理属性和几何外观。一个最简单的模型结构：

```xml
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom type="plane" size="1 1 0.1" material="ground"/>
    
    <body name="torso" pos="0 0 1">
      <freejoint/>                     <!-- 自由关节：6自由度 -->
      <geom type="box" size="0.2 0.1 0.3" rgba="0 0 1 1"/>
      
      <body name="arm" pos="0.25 0 0">
        <joint type="hinge" axis="0 1 0" name="shoulder"/>  <!-- 旋转关节 -->
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
```

关键元素：
- `<body>`：刚体（rigid body），是构成机器人的基本单元
- `<joint>`：关节，定义了刚体之间的运动自由度
- `<geom>`：几何体，定义了碰撞和视觉外形
- `<freejoint>`：自由关节，允许刚体在空间中自由运动（6自由度）
- `<actuator>`：执行器，定义了如何向关节施加力和力矩

本书不要求读者精通 MJCF 格式，但理解基本结构有助于调试和修改模型。

## 4.3 Git 与版本控制

### 4.3.1 基础操作

Git 是现代软件开发的基础设施。以下是基本操作流程：

```bash
# 配置身份（仅首次）
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"

# 初始化仓库
git init

# 查看状态
git status

# 添加与提交
git add filename.py           # 暂存单个文件
git add .                     # 暂存所有变更
git commit -m "feat: add rotation visualizer"

# 查看历史
git log --oneline             # 简略历史
git log --graph --all         # 分支图
```

### 4.3.2 分支管理

分支是 Git 最强大的特性之一，允许你在不影响主线的情况下进行试验性开发。

```bash
# 创建并切换分支
git checkout -b new-feature

# 切换分支
git checkout main

# 合并分支
git merge new-feature

# 删除分支
git branch -d new-feature
```

### 4.3.3 GitHub 开源协作

本书采用"主笔 + 开源社区审校"的编著方式，这意味着 GitHub 是协作的核心平台。

**标准协作流程**：

1. **Fork** 本书的仓库到你的 GitHub 账号
2. **Clone** 你的 Fork 到本地：`git clone https://github.com/你的用户名/humanoid-embodied-textbook.git`
3. **创建分支**：`git checkout -b fix-typo-ch1`
4. **修改并提交**
5. **推送**到你的 Fork：`git push origin fix-typo-ch1`
6. **创建 Pull Request（PR）**：在 GitHub 上向原仓库提交 PR

**Issues**：使用 Issue 提交错误报告、功能建议、内容讨论。每章末尾的习题答案也可以在 Issues 中讨论。

## 4.4 项目结构与规范

### 4.4.1 本书的代码组织结构

```
humanoid-embodied-textbook/
├── website/                # 教材在线阅读网站 (Docusaurus)
│   └── docs/               # 各章正文 Markdown
├── code/                   # 仿真实操代码
│   ├── ch01-load-model/    # 第1章代码
│   ├── ch02-perception-action/
│   ├── ch03-rotation-math/
│   ├── ch04-env-setup/
│   └── ...                 # 后续章节
├── assets/                 # 模型文件、图片
└── docs/                   # 设计文档与计划
```

### 4.4.2 编码规范

遵循 PEP 8 编码规范：
- 使用4个空格缩进（不用Tab）
- 类名：`CamelCase`（如 `HumanoidController`）
- 函数/变量名：`小写_下划线`（如 `detect_ground_contact()`）
- 常量：`大写_下划线`（如 `SIMULATION_STEPS`）

### 4.4.3 README 与文档规范

每个代码目录都包含 README.md，说明该目录的用途和运行方式。在提交代码时，请确保 README 与代码保持一致。

### 4.4.4 可复现性

每个代码目录都包含 `requirements.txt`，列出了所需依赖和版本。这确保了代码在不同机器上的可复现性。

## 4.5 深度学习环境配置

### 4.5.1 PyTorch 安装

PyTorch 是本书使用的深度学习框架（第13-17章）。

```bash
# CPU 版本（推荐入门）
pip install torch torchvision

# GPU 版本（CUDA 12.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证：
```python
import torch
print(torch.__version__)    # 版本号
print(torch.cuda.is_available())  # GPU 是否可用
```

### 4.5.2 Stable-Baselines3 安装

Stable-Baselines3 是基于 PyTorch 的强化学习库，提供了 PPO、SAC、TD3 等经典算法的标准实现。

```bash
pip install stable-baselines3
```

验证：
```python
from stable_baselines3 import PPO
print("SB3 安装成功!")
```

### 4.5.3 硬件配置建议

| 配置 | CPU | GPU | RAM | 适用场景 |
|:----|:---:|:---:|:---:|:--------|
| 最低 | 4核 | 无 | 8GB | 第1-12章感知与控制（无需训练）|
| 推荐 | 8核 | 8GB显存 | 16GB | 第13-17章 RL/模仿学习训练 |
| 最佳 | 16核 | 24GB显存 | 32GB | 第17章 VLA 模型部署/微调 |

**注意**：如果本地没有 GPU，可以使用 Google Colab（免费 GPU）进行训练。

### 4.5.4 Docker / Colab 云端方案

**Docker**：本书提供 Docker 镜像，包含完整开发环境。

```bash
docker pull humanoid-textbook/env:latest
docker run -it --rm -v $(pwd):/workspace humanoid-textbook/env:latest
```

**Google Colab**：对于强化学习训练章节，Colab 的免费 GPU 已经足够使用。

## 4.6 实用开发工具推荐

### 4.6.1 VS Code 配置

推荐安装以下 VS Code 插件：
- **Python**（Microsoft）：Python 语言支持
- **Jupyter**：在 VS Code 中直接运行 Notebook
- **GitLens**：强大的 Git 可视化工具
- **Markdown All in One**：Markdown 编辑增强
- **indent-rainbow**：缩进可视化

### 4.6.2 调试工具

```python
# pdb 调试
import pdb; pdb.set_trace()    # 设置断点
# 常用命令: n(ext), s(tep), c(ontinue), p(rint), l(ist)

# ipdb（增强版）
pip install ipdb
import ipdb; ipdb.set_trace()
```

### 4.6.3 效率工具

```bash
# 代码格式化 (black)
pip install black
black your_script.py           # 一键格式化

# 代码检查 (ruff，比pylint更快)
pip install ruff
ruff check your_script.py

# 类型检查 (mypy)
pip install mypy
mypy your_script.py
```

---

## 🛠️ 仿真实操：MuJoCo 完整入门

> 🕐 预计时长：45分钟 · 📁 代码位置：`code/ch04-env-setup/standing_demo.py`

### 实验目标

从零开始，跑通一个完整的人形机器人站立仿真。涵盖：
1. 模型加载与初始化
2. PD 控制器配置
3. 仿真数据记录
4. 结果分析与可视化

### 实验步骤

1. **加载模型**：使用 `mujoco.models.MODEL_DIR` 中的 humanoid 模型
2. **初始化**：设置机器人的初始高度和姿态
3. **PD 控制**：实现一个简单的比例-微分控制器，让机器人保持站立姿态
4. **数据记录**：每10步记录一次机器人躯干高度
5. **结果分析**：输出最终高度和稳定性判断

### 延伸思考

1. 调节 KP（比例增益）和 KD（微分增益），观察机器人站立行为的变化
2. KP 过大时会发生什么？KD 过大时呢？
3. 尝试给机器人一个初始扰动（如修改初始姿态），它能否恢复？

---

## 本章小结

- Python + NumPy + Matplotlib 是机器人编程的基础工具栈
- MuJoCo 是本书使用的仿真引擎，安装简单、Python 绑定优雅
- MJCF XML 是 MuJoCo 的模型描述语言，理解基本元素即可动手
- Git + GitHub 是协作写作和版本控制的基础设施
- PyTorch + Stable-Baselines3 是深度学习和大模型章节的前置依赖

## 习题

1. **选择题**：MuJoCo 中描述机器人模型的 XML 文件格式叫做？
   A. URDF  B. MJCF  C. SDF  D. XACRO

2. **选择题**：以下哪个不是 NumPy 的核心功能？
   A. 数组运算  B. 线性代数  C. 网络通信  D. 随机数生成

3. **简答题**：为什么在机器人开发中使用虚拟环境是一个好习惯？

4. **简答题**：本书采用 Docusaurus 网站的目录结构是什么样的？code/ 和 website/ 各自的作用是什么？

5. **操作题**：按照本章引导完成 MuJoCo 的安装，运行 `standing_demo.py`，并截图输出结果。

## 参考文献

1. Todorov, E., Erez, T., & Tassa, Y. (2012). "MuJoCo: A physics engine for model-based control." *IEEE/RSJ IROS.*
2. DeepMind. (2022). "MuJoCo: Open-source physics engine." [GitHub: google-deepmind/mujoco]
3. Chacon, S., & Straub, B. (2014). *Pro Git.* Apress. (免费在线版: https://git-scm.com/book/zh/v2)
4. Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS.*
5. Raffin, A., et al. (2021). "Stable-Baselines3: Reliable Reinforcement Learning Implementations." *JMLR.*
