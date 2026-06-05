"""
第17章 仿真实操：连接VLM与MuJoCo
================================================================
目标：通过自然语言指令控制 MuJoCo 机器人。
使用OpenAI API（或其他LLM）作为VLM替代方案。
"""

import mujoco
import numpy as np
import base64
import json

print("="*60)
print("  VLM + MuJoCo 自然语言控制")
print("="*60)

# ============================================================
# 1. 加载模型
# ============================================================
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0

print(f"  模型: {model.name}")
print(f"  自由度: {model.nq}")

# ============================================================
# 2. 指令→动作映射（作为VLM替代方案）
# ============================================================
def parse_natural_language(command):
    """将自然语言指令解析为MuJoCo控制命令"""
    cmd = command.lower().strip()

    # 动作参数
    duration = 100  # 执行步数

    if any(w in cmd for w in ["前", "前进", "走", "walk", "forward"]):
        amount = 0.3
        # 模拟前进：前倾 + 抬腿
        for t in range(duration):
            phase = 2 * np.pi * t / duration
            data.qpos[2] = 1.0 + np.sin(phase) * 0.05  # 轻微上下
            # 让身体前倾
            data.qpos[4] = np.sin(phase) * 0.1  # 前后俯仰
            mujoco.mj_step(model, data)
        return f"前进了（幅度: {amount}）"

    elif any(w in cmd for w in ["后", "后退", "back"]):
        for t in range(duration):
            phase = 2 * np.pi * t / duration
            data.qpos[4] = -np.sin(phase) * 0.1
            mujoco.mj_step(model, data)
        return "后退了"

    elif any(w in cmd for w in ["左", "left"]):
        for t in range(duration):
            # 绕Z轴旋转
            angle = np.radians(30) * min(1, t / duration)
            data.qpos[3] = np.cos(angle / 2)
            data.qpos[6] = -np.sin(angle / 2)
            mujoco.mj_step(model, data)
        return "左转了 30°"

    elif any(w in cmd for w in ["右", "right"]):
        for t in range(duration):
            angle = np.radians(30) * min(1, t / duration)
            data.qpos[3] = np.cos(angle / 2)
            data.qpos[6] = np.sin(angle / 2)
            mujoco.mj_step(model, data)
        return "右转了 30°"

    elif any(w in cmd for w in ["抬", "举起", "raise", "lift"]):
        for t in range(duration):
            # 模拟抬手
            for jnt_idx in range(model.njnt):
                qpos_addr = model.jnt_qposadr[jnt_idx]
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_idx)
                if name and ("shoulder" in name.lower()):
                    data.qpos[qpos_addr] = np.radians(45) * min(1, t / duration)
            mujoco.mj_step(model, data)
        return "抬起了手臂"

    elif any(w in cmd for w in ["状态", "status", "高度", "height"]):
        height = data.qpos[2]
        return f"当前高度: {height:.2f}m, 姿态正常"

    else:
        return f"无法理解指令: '{command}'"

# ============================================================
# 3. 主循环
# ============================================================
print(f"\n--- 可用指令示例 ---")
examples = [
    "前进", "后退",
    "左转", "右转",
    "抬起手臂",
    "查看状态",
]
for ex in examples:
    print(f"  • {ex}")

print(f"\n--- 执行指令: '前进并抬起手臂' ---")
# 多指令处理
commands = ["前进", "抬起手臂"]
for cmd in commands:
    result = parse_natural_language(cmd)
    print(f"  指令 '{cmd}': {result}")

print(f"\n--- 执行指令: '左转' ---")
result = parse_natural_language("左转")
print(f"  结果: {result}")

print(f"\n--- 执行指令: '查看状态' ---")
result = parse_natural_language("状态")
print(f"  结果: {result}")

print(f"\n{'='*60}")
print(f"  实验说明:")
print(f"  在实际部署中，可以使用OpenVLA/Qwen-VL等VLM模型")
print(f"  替代上述 rule-based 的自然语言解析。")
print(f"  运行VLM需要GPU支持，建议: pip install transformers torch")
print(f"{'='*60}")
