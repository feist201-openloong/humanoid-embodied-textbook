---
sidebar_position: 4
---

# 第21章：综合项目Ⅲ——部署与端到端

> **本章导语**：仿真中的策略最终要部署到真实机器人上。本章讲解模型压缩、边缘部署、实时性优化的完整流程，实现从 MuJoCo 仿真到真实硬件（Jetson）的端到端迁移。

---

## 21.1 项目概要

### 21.1.1 目标

将前几章训练好的策略（PPO行走、VLA控制等）从仿真部署到边缘计算设备（NVIDIA Jetson），实现：

```
MuJoCo 仿真训练 → 模型导出 → Jetson 部署 → 真实机器人控制
```

### 21.1.2 挑战

| 挑战 | 说明 |
|:-----|:------|
| **计算资源** | Jetson 算力远小于训练服务器 |
| **推理延迟** | 策略推理需在 &lt;10ms 内完成 |
| **精度损失** | 量化/剪枝可能影响策略表现 |
| **传感器切换** | 从仿真传感器切换到真实传感器 |

## 21.2 模型压缩

### 21.2.1 量化（Quantization）

将FP32权重转为INT8，模型大小减少4倍，推理速度提升2-4倍：

```python
# PyTorch 量化示例
import torch

model = load_pretrained_policy()
model.eval()

# 训练后动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), 'policy_quantized.pth')
```

### 21.2.2 剪枝（Pruning）

去除不重要的网络连接或通道，减少计算量：

```python
# 结构化剪枝：移除重要性低的通道
import torch.nn.utils.prune as prune

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, 'weight')  # 使剪枝永久化
```

### 21.2.3 蒸馏（Distillation）

用大模型（教师）指导学生小模型（学生）学习：

```
教师模型（大，高精度）→ 生成软标签 → 训练学生模型（小，快速）
```

## 21.3 ONNX导出

ONNX（Open Neural Network Exchange）是通用的模型交换格式，可以在不同框架间迁移模型。

```python
import torch.onnx

dummy_input = torch.randn(1, observation_dim)
torch.onnx.export(
    model,
    dummy_input,
    'policy.onnx',
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={'observation': {0: 'batch_size'}},
)
```

## 21.4 Jetson部署

### 21.4.1 NVIDIA Jetson系列

| 型号 | AI算力 | RAM | 适用场景 |
|:----|:------:|:---:|:---------|
| Jetson Nano | 0.5 TOPS | 4GB | 入门级感知 |
| Jetson TX2 | 1.3 TOPS | 8GB | 中等负载 |
| Jetson Xavier NX | 21 TOPS | 8GB | 实时推理 |
| Jetson Orin NX | 70 TOPS | 16GB | 复杂策略 |
| Jetson AGX Orin | 275 TOPS | 64GB | 端侧大模型 |

### 21.4.2 TensorRT优化

利用TensorRT对模型进行图优化和层融合：

```python
import tensorrt as trt

# ONNX → TensorRT 引擎
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()

parser = trt.OnnxParser(network, logger)
with open('policy.onnx', 'rb') as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
engine = builder.build_serialized_network(network, config)
```

## 21.5 实时性优化

### 21.5.1 整体延迟预算

端到端策略推理的延迟预算（目标：10ms内完成）：

```
传感器采集: 1ms → 预处理: 1ms → 模型推理: 3ms → 后处理: 1ms → 控制输出: 1ms = 7ms
```

### 21.5.2 优化策略

| 方法 | 效果 | 实现难度 |
|:-----|:----:|:--------:|
| 模型量化（INT8） | 推理加速2-4x | 低 |
| TensorRT优化 | 推理加速3-5x | 中 |
| 减少观测维度 | 输入减半，加速~2x | 低 |
| C++部署（vs Python） | 开销减半 | 高 |
| 流水线并行 | 延迟隐藏 | 高 |

---

## 🛠️ 仿真实操：策略导出与部署

> 🕐 预计时长：60分钟 · 📁 代码位置：`code/ch21-project-deploy/deploy_policy.py`

### 实验目标
将训练好的策略导出为ONNX，在 Jetson（或本地CPU）上部署推理。

### 实验步骤
1. 加载预训练的PPO策略
2. 将策略转换为ONNX格式
3. （可选）使用ONNX Runtime进行推理
4. 在MuJoCo中测试导出的策略
5. 对比原始PyTorch和ONNX Runtime的推理速度

### 延伸思考
1. ONNX导出的策略和原始PyTorch策略的表现一致吗？精度损失多少？
2. 量化到INT8后策略表现是否会明显下降？什么情况下下降最严重？
3. 如果部署硬件是CPU而不是GPU，优化策略会有什么不同？

---

## 本章小结

- 模型压缩（量化、剪枝、蒸馏）使大模型适合边缘部署
- ONNX提供跨框架的模型交换格式
- TensorRT进一步优化推理性能
- Jetson系列提供不同算力级别的边缘AI计算平台
- 端到端延迟预算和优化是实时机器人控制的工程基础

## 习题

1. **选择题**：模型量化将FP32转为INT8后，模型大小约变为原来的？
   A. 1/2  B. 1/4  C. 1/8  D. 不变

2. **简答题**：知识蒸馏的基本原理是什么？

3. **简答题**：为什么部署时选择C++通常比Python快？

4. **操作题**：在本章代码中，尝试使用不同精度的ONNX Runtime（FP32 vs INT8），对比推理速度差异。

## 参考文献
1. NVIDIA. "TensorRT Documentation." *developer.nvidia.com/tensorrt.*
2. Jacob, B., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference." *CVPR.*
3. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *arXiv:1503.02531.*
4. Bai, J., et al. (2019). "ONNX: Open Neural Network Exchange." *GitHub: onnx/onnx.*
