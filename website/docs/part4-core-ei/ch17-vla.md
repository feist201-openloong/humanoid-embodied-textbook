---
sidebar_position: 5
---

# 第17章：VLA大模型与机器人

> **本章导语**：大语言模型（LLM）和多模态大模型（VLM）正在深刻改变机器人领域。视觉-语言-动作模型（VLA）将视觉感知、语言理解和动作生成融合为统一的端到端框架——这是具身智能的最新前沿。

---

## 17.1 从LLM到VLA

### 17.1.1 技术演进

```
LLM（纯文本） → VLM（文本+图像） → VLA（文本+图像+动作）
```

| 模型 | 时间 | 输入 | 输出 | 意义 |
|:----|:----:|:----|:----|:------|
| GPT-4 | 2023.03 | 文本 | 文本 | 通用推理能力 |
| RT-2 | 2023.07 | 图像+文本 | 动作token | 首个VLA |
| GPT-4V | 2023.10 | 图像+文本 | 文本 | 多模态理解 |
| Gemini Robotics | 2024.12 | 图像+文本 | 动作+文本 | 原生多模态机器人 |

### 17.1.2 为什么VLA是范式转变

传统机器人方法需要为每个任务单独训练模型。VLA将Internet规模预训练的知识直接迁移到机器人控制中：

- RT-2在互联网图文数据上预训练，再在机器人数据上微调
- 这意味着机器人不仅学会了"如何抓取杯子"（来自机器人数据），还学会了"杯子是什么"（来自互联网数据）

## 17.2 RT-2架构

### 17.2.1 核心思路

RT-2（Brohan et al., 2023）的核心创新：**将机器人动作表示为一种新的"语言"token**。

具体来说：
1. 将机器人动作 $(a_x, a_y, a_z, \ldots)$ 离散化为有限个值
2. 将这些离散值编码为文本token（如"1 138 241 ..."）
3. 在图文数据和机器人数据的混合上训练模型

### 17.2.2 架构

RT-2基于预训练的PaLI-X或PaLM-E视觉-语言模型，输出动作token：

```
图像 + 文本指令 → 视觉编码器 → LLM → 动作token
```

```python
# RT-2的推理过程（概念性）
# 输入: 当前图像 + "把杯子拿到桌上"
# 输出: "动作: 1 138 241 87 322 55" ← 离散化的关节角度
```

### 17.2.3 局限性

- **动作空间离散化限制了精度**
- **推理延迟较高**（LLM的序列生成）
- **不能泛化到训练分布之外的长尾场景**

## 17.3 开源VLA：OpenVLA

### 17.3.1 简介

OpenVLA（2024）是第一个开源的VLA模型，基于Prismatic（融合SigLIP和DinoV2视觉编码器）和Llama 2语言模型构建。提供了7B参数的完整VLA模型。

### 17.3.2 使用方式

```python
# OpenVLA推理示例（概念性）
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("openvla/openvla-7b")
model = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")

# 输入: 图像 + 任务描述
inputs = processor(
    images=rgb_image,
    text="What action should the robot take to pick up the red cup?"
)
outputs = model.generate(**inputs)
action = processor.decode(outputs[0])
```

## 17.4 Gemini Robotics

Gemini Robotics（2024.12）是Google DeepMind的最新VLA模型，基于Gemini多模态模型构建。关键特点：

- **原生多模态**：视觉、语言、动作共享同一组参数
- **泛化能力**：无需微调即可泛化到新物体和新场景
- **对话能力**：可在执行过程中解释自己的行为
- **安全性**：内置安全过滤机制

## 17.5 VLA面临的挑战

| 挑战 | 说明 | 当前进展 |
|:----|:------|:---------|
| **动作精度** | 离散化动作损失精度 | 连续动作VLA研究中 |
| **推理速度** | 大模型生成动作慢 | 小模型蒸馏 + 缓存 |
| **数据效率** | 机器人数据采集成本极高 | 仿真 → 预训练 → 少量微调 |
| **端侧部署** | 7B+模型在机器人上部署困难 | 量化 + 剪枝 + 边缘计算 |
| **安全性** | 大模型行为不可预测 | 安全过滤器 + 控制约束 |

---

## 🛠️ 仿真实操：连接VLM与MuJoCo

> 🕐 预计时长：60分钟 · 📁 代码位置：`code/ch17-openvla/openvla_demo.py`

### 实验目标
部署一个开源的VLM（如OpenVLA或Qwen-VL），通过自然语言指令控制 MuJoCo 机器人。

### 实验原理
1. 从 MuJoCo 获取当前渲染图像
2. 将图像 + 自然语言指令发送给VLM
3. VLM输出文本或动作token
4. 将VLM输出映射为 MuJoCo 控制指令
5. 执行指令，观察结果

### 注意事项
OpenVLA需要较大的GPU显存（7B模型约16GB）。如果硬件有限，可以使用Qwen-VL或直接通过API调用LLM。

### 延伸思考
1. 如何让VLM理解"左转30°"这种精确指令？
2. 如果VLM输出错误指令，如何保证安全？
3. 与第12章的规则引擎相比，VLM方法有什么优势和劣势？

---

## 本章小结

- VLA将视觉、语言和动作统一到一个模型中
- RT-2通过将动作离散化为token实现端到端控制
- OpenVLA提供了开源的VLA模型
- Gemini Robotics实现了原生多模态机器人控制
- 动作精度、推理速度和安全性是VLA面临的主要挑战

## 习题

1. **选择题**：VLA中的"A"代表什么？
   A. Artificial  B. Action  C. Algorithm  D. Agent

2. **简答题**：RT-2将机器人动作表示为token的核心思路是什么？

3. **简答题**：VLA模型相比传统机器人方法（感知+规划+控制）的优势是什么？

4. **开放题**：如果你要设计一个VLA模型，你会选择LLM优先还是专门为机器人设计的架构？为什么？

## 参考文献
1. Brohan, A., et al. (2023). "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." *arXiv:2307.15818.*
2. Kim, M. J., et al. (2024). "OpenVLA: An Open-Source Vision-Language-Action Model." *arXiv:2406.09246.*
3. Gemini Robotics Team. (2024). "Gemini Robotics." *Google DeepMind.*
4. Driess, D., et al. (2023). "PaLM-E: An Embodied Multimodal Language Model." *arXiv:2303.03378.*
