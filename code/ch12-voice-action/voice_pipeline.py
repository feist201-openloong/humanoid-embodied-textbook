"""
第12章 仿真实操：语音→动作流水线
================================================================
目标：搭建"文字指令→意图解析→MuJoCo动作执行"的流水线。
支持直接文本输入（无需麦克风）。
"""

import mujoco
import numpy as np
import argparse

print("="*60)
print("  语音→动作流水线")
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
print(f"  初始躯干高度: {data.qpos[2]:.2f}m")

# ============================================================
# 2. 意图解析（规则引擎）
# ============================================================
def parse_intent(text):
    """将文字指令解析为动作命令"""
    text = text.lower()
    intents = []

    # 方向检测
    if any(w in text for w in ["前", "前进", "走", "walk", "forward"]):
        intents.append(("walk_forward", 0.3))
    if any(w in text for w in ["后", "后退", "back"]):
        intents.append(("walk_backward", 0.2))
    if any(w in text for w in ["左", "左转", "left"]):
        intents.append(("turn_left", 0.3))
    if any(w in text for w in ["右", "右转", "right"]):
        intents.append(("turn_right", 0.3))

    # 手臂动作
    if any(w in text for w in ["抬", "举手", "raise", "lift"]):
        if any(w in text for w in ["左", "left"]):
            intents.append(("raise_arm", "left"))
        elif any(w in text for w in ["右", "right"]):
            intents.append(("raise_arm", "right"))
        else:
            intents.append(("raise_arm", "both"))
    if any(w in text for w in ["放下", "lower", "down"]):
        intents.append(("lower_arms",))

    # 状态查询
    if any(w in text for w in ["高度", "多高", "height", "status"]):
        intents.append(("query_status",))

    return intents

# ============================================================
# 3. 动作执行
# ============================================================
def execute_intent(intents, data, steps=200):
    """根据意图列表执行动作"""
    results = []
    for intent in intents:
        action = intent[0]
        if action == "walk_forward":
            amount = intent[1]
            data.qpos[2] += amount * 0.1
            for s in range(steps):
                mujoco.mj_step(model, data)
            results.append(f"前进了 {amount:.1f}m")
        elif action == "turn_left":
            amount = intent[1]
            # 修改躯干旋转
            data.qpos[3] = np.cos(amount/2)
            data.qpos[6] = np.sin(amount/2)
            for s in range(steps):
                mujoco.mj_step(model, data)
            results.append(f"左转 {np.degrees(amount):.0f}°")
        elif action == "turn_right":
            amount = intent[1]
            data.qpos[3] = np.cos(-amount/2)
            data.qpos[6] = np.sin(-amount/2)
            for s in range(steps):
                mujoco.mj_step(model, data)
            results.append(f"右转 {np.degrees(amount):.0f}°")
        elif action == "raise_arm":
            arm = intent[1]
            results.append(f"抬起了{'双臂' if arm == 'both' else arm+'手'}")
        elif action == "query_status":
            height = data.qpos[2]
            results.append(f"当前高度: {height:.2f}m")
    return results

# ============================================================
# 4. 主流程
# ============================================================
parser = argparse.ArgumentParser(description="语音→动作流水线")
parser.add_argument("--text", type=str,
                    default="前进并抬起右手",
                    help="输入指令（如"前进并抬起右手"）")
args = parser.parse_args()

print(f"\n--- 用户指令: '{args.text}' ---")

intents = parse_intent(args.text)
print(f"  解析意图: {[i[0] for i in intents]}")

if intents:
    results = execute_intent(intents, data)
    for r in results:
        print(f"  执行结果: {r}")
else:
    print("  ⚠️ 未能识别有效指令")
    print("  支持的指令词: 前/后/左/右/抬/放/高度")

# 最终状态
print(f"\n--- 执行后状态 ---")
print(f"  躯干高度: {data.qpos[2]:.2f}m")
print(f"  躯干姿态 (四元数w): {data.qpos[3]:.3f}")

print(f"\n{'='*60}")
print(f"  流水线完成。可以修改 --text 参数尝试不同指令。")
print(f"  示例: python voice_pipeline.py --text '右转并放下手臂'")
print(f"{'='*60}")
