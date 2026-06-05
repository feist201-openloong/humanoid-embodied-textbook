# 仿真实操代码

本目录包含各章的 MuJoCo 仿真实操代码。

## 环境要求

- Python 3.10+
- MuJoCo 3.x

安装基础依赖：
```bash
pip install mujoco numpy matplotlib
```

## 目录结构

| 目录 | 对应章节 | 实验内容 |
|------|---------|---------|
| ch01-load-model/ | 第1章 | MuJoCo第一课：加载人形机器人模型 |
| ch02-perception-action/ | 第2章 | 感知→行动闭环 |
| ch03-rotation-math/ | 第3章 | 旋转与变换可视化 |
| ch04-env-setup/ | 第4章 | MuJoCo完整入门：机器人站立仿真 |

## 使用方式

每个目录均为独立可运行的 Python 项目。进入目录后：
```bash
cd ch01-load-model
pip install -r requirements.txt
python load_humanoid.py
```
