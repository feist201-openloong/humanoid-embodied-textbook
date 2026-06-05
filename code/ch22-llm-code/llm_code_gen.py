"""
第22章 仿真实操：LLM生成MuJoCo控制代码
================================================================
目标：用LLM（或模板引擎）根据自然语言生成MuJoCo控制代码。
"""

import mujoco
import numpy as np
import math

print("="*60)
print("  LLM 代码生成控制")
print("="*60)

# ============================================================
# 1. 机器人控制API
# ============================================================
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0

class RobotAPI:
    """LLM可调用的机器人控制API"""
    @staticmethod
    def move_forward(steps=50):
        """前进指定步数"""
        print(f"  [API] 前进 {steps} 步")
        for _ in range(steps):
            data.qpos[0] += 0.01
            mujoco.mj_step(model, data)

    @staticmethod
    def turn_left(angle_deg=30):
        """左转指定角度"""
        print(f"  [API] 左转 {angle_deg}°")
        angle = math.radians(angle_deg)
        for _ in range(100):
            data.qpos[3] = math.cos(angle/2 * _/100)
            data.qpos[6] = -math.sin(angle/2 * _/100)
            mujoco.mj_step(model, data)

    @staticmethod
    def turn_right(angle_deg=30):
        """右转指定角度"""
        print(f"  [API] 右转 {angle_deg}°")
        angle = math.radians(angle_deg)
        for _ in range(100):
            data.qpos[3] = math.cos(angle/2 * _/100)
            data.qpos[6] = math.sin(angle/2 * _/100)
            mujoco.mj_step(model, data)

    @staticmethod
    def raise_arm(steps=50):
        """抬起手臂"""
        print(f"  [API] 抬起手臂")
        for jnt_idx in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_idx)
            if name and ("shoulder" in name.lower()):
                qpos_addr = model.jnt_qposadr[jnt_idx]
                data.qpos[qpos_addr] = 0.5
        for _ in range(steps):
            mujoco.mj_step(model, data)

    @staticmethod
    def lower_arm(steps=50):
        """放下手臂"""
        print(f"  [API] 放下手臂")
        for jnt_idx in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_idx)
            if name and ("shoulder" in name.lower()):
                qpos_addr = model.jnt_qposadr[jnt_idx]
                data.qpos[qpos_addr] = 0.0
        for _ in range(steps):
            mujoco.mj_step(model, data)

    @staticmethod
    def get_status():
        """获取机器人当前状态"""
        height = data.qpos[2]
        print(f"  [API] 当前高度: {height:.2f}m")
        return height

# ============================================================
# 2. LLM代码生成（使用模板匹配模拟LLM）
# ============================================================
def llm_generate_code(task_description):
    """模拟LLM根据任务描述生成控制代码"""
    desc = task_description.lower()

    code_lines = []

    # 分解任务描述为步骤
    steps = []

    if "前进" in desc or "向前" in desc or "forward" in desc:
        steps.append("api.move_forward(100)")

    if "左转" in desc or "turn_left" in desc or "向左" in desc:
        steps.append("api.turn_left(45)")

    if "右转" in desc or "turn_right" in desc or "向右" in desc:
        steps.append("api.turn_right(45)")

    if "抬" in desc or "举手" in desc or "raise" in desc:
        steps.append("api.raise_arm(50)")
    elif "放" in desc or "lower" in desc:
        steps.append("api.lower_arm(50)")

    if "状态" in desc or "status" in desc or "高度" in desc:
        steps.append("api.get_status()")

    if not steps:
        steps.append(f"# 无法理解指令: {task_description}")
        steps.append("print('⚠️ Unknown command')")

    code = "def execute(api):\n"
    for s in steps:
        code += f"    {s}\n"
    return code, steps

# ============================================================
# 3. 主流程
# ============================================================
api = RobotAPI()
tasks = [
    "前进并抬起手臂",
    "右转后检查状态",
    "放下手臂",
]

for task in tasks:
    print(f"\n{'─'*50}")
    print(f"  📝 用户指令: '{task}'")
    print(f"{'─'*50}")

    code, steps = llm_generate_code(task)
    print(f"\n  💻 LLM 生成代码:")
    for s in steps:
        print(f"    {s}")

    print(f"\n  ▶️  执行:")
    try:
        for s in steps:
            exec(s)
    except Exception as e:
        print(f"  ❌ 执行错误: {e}")

    print(f"\n  ✅ 指令完成。高度 = {data.qpos[2]:.2f}m")

print(f"\n{'='*60}")
print(f"  实验完成！LLM代码生成控制的核心流程：")
print(f"  自然语言 → LLM生成代码 → 执行 → 验证")
print(f"  提示: 使用真实LLM API可实现更复杂的任务规划。")
print(f"{'='*60}")
