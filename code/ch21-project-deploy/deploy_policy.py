"""
第21章 仿真实操：策略导出与部署
================================================================
目标：将策略导出为ONNX，在MuJoCo中测试部署推理。
"""

import numpy as np
import time

print("="*60)
print("  策略导出与部署实验")
print("="*60)

# ============================================================
# 1. 检查ONNX运行时可用性
# ============================================================
try:
    import onnxruntime as ort
    HAS_ONNX = True
    print(f"  ✅ ONNX Runtime 已安装 (v{ort.__version__ if hasattr(ort, '__version__') else '?'})")
except ImportError:
    HAS_ONNX = False
    print("  ⚠️ ONNX Runtime 未安装，使用NumPy模拟")

try:
    import torch
    HAS_TORCH = True
    print(f"  ✅ PyTorch 已安装 (v{torch.__version__})")
except ImportError:
    HAS_TORCH = False
    print("  ⚠️ PyTorch 未安装，使用NumPy模拟")

# ============================================================
# 2. 创建模拟策略
# ============================================================
print(f"\n--- 创建简化的行走策略 ---")
OBS_DIM = 32
ACT_DIM = 12

# 简单的线性策略: action = W @ observation + b
np.random.seed(42)
W = np.random.randn(ACT_DIM, OBS_DIM) * 0.5
b = np.zeros(ACT_DIM)

def policy_numpy(obs):
    """NumPy策略"""
    return W @ obs + b

print(f"  观测维度: {OBS_DIM}")
print(f"  动作维度: {ACT_DIM}")

# ============================================================
# 3. 性能基准测试
# ============================================================
print(f"\n--- 性能基准测试 ---")
NUM_BENCHMARK = 10000
obs = np.random.randn(OBS_DIM).astype(np.float32)

# NumPy推理
start = time.perf_counter()
for _ in range(NUM_BENCHMARK):
    action = policy_numpy(obs)
numpy_time = (time.perf_counter() - start) / NUM_BENCHMARK * 1000

print(f"  NumPy 推理: {numpy_time:.2f} ms/步")

# PyTorch推理（如果可用）
if HAS_TORCH:
    model_torch = torch.nn.Linear(OBS_DIM, ACT_DIM)
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model_torch(torch.from_numpy(obs))
        start = time.perf_counter()
        for _ in range(NUM_BENCHMARK):
            _ = model_torch(torch.from_numpy(obs))
        torch_time = (time.perf_counter() - start) / NUM_BENCHMARK * 1000
    print(f"  PyTorch 推理: {torch_time:.2f} ms/步")

# ONNX推理（如果可用）
if HAS_ONNX:
    print(f"  提示: ONNX推理需先导出模型为.onnx格式")
    print(f"  运行: python -c \"import torch; m=torch.nn.Linear(32,12); torch.onnx.export(m, torch.randn(1,32), 'policy.onnx')\"")

# ============================================================
# 4. 仿真中部署
# ============================================================
print(f"\n--- 仿真中部署策略 ---")

# 模拟机器人接收观测、推理、执行动作的循环
TOTAL_STEPS = 100
sim_heights = [1.0]

for step in range(TOTAL_STEPS):
    # 模拟接收观测（传感器数据）
    simulated_obs = np.random.randn(OBS_DIM).astype(np.float32)

    # 策略推理
    action = policy_numpy(simulated_obs)

    # 模拟在仿真环境中执行动作
    height_change = np.mean(action[:3]) * 0.01
    new_height = max(0.5, min(1.5, sim_heights[-1] + height_change))
    sim_heights.append(new_height)

    if step % 25 == 0:
        print(f"  步 {step:3d}: 推理延迟 = {numpy_time:.1f}ms, "
              f"高度 = {new_height:.2f}m")

print(f"\n{'='*60}")
print(f"  策略部署实验完成。")
print(f"  💡 在真实Jetson上部署时:")
print(f"    1. torch.onnx.export() 导出ONNX")
print(f"    2. TensorRT优化: trtexec --onnx=policy.onnx --saveEngine=policy.trt")
print(f"    3. C++部署: 用Jetson的GPIO/I2C接口控制实物")
print(f"{'='*60}")
