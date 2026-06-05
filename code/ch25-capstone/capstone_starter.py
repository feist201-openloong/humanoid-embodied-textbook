"""
第25章 综合大作业：通用机器人流水线（起始代码）
================================================================
目标：设计一个"能完成任意文字指令"的机器人通用pipeline。
此文件为起始模板，最终实现需要综合全书所学知识。
"""

import mujoco
import numpy as np
import math

print("="*60)
print("  综合大作业：通用机器人流水线")
print("="*60)
print("  这是一个开放命题。以下提供基础框架和API。")
print("  你需要完成的是让机器人能够理解并执行任意文字指令。")

# ============================================================
# 基础框架
# ============================================================
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0

print(f"\n  模型: {model.name}, 自由度: {model.nq}")
print(f"  初始高度: {data.qpos[2]:.2f}m")

# ============================================================
# 可用技能库
# ============================================================
class Skills:
    """机器人技能库——各章积累的可复用技能"""
    @staticmethod
    def walk_to(target_x, target_y, steps=300):
        """导航到目标位置（第8/19章）"""
        print(f"  [行走] 前往 ({target_x:.1f}, {target_y:.1f})")
        for s in range(steps):
            dx = target_x - data.qpos[0]
            dy = target_y - data.qpos[1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 0.2:
                break
            speed = min(0.02, dist * 0.03)
            data.qpos[0] += speed * dx / max(dist, 0.01)
            data.qpos[1] += speed * dy / max(dist, 0.01)
            mujoco.mj_step(model, data)
        return f"到达 ({data.qpos[0]:.2f}, {data.qpos[1]:.2f})"

    @staticmethod
    def detect_object(object_name="cup"):
        """感知物体（第9/11章）"""
        # 模拟物体检测
        objects = {
            "cup": np.array([2.0, 0.2, 0.8]),
            "book": np.array([1.5, -0.3, 0.6]),
            "screwdriver": np.array([1.0, 0.5, 0.4]),
        }
        if object_name in objects:
            pos = objects[object_name]
            print(f"  [感知] 检测到 '{object_name}' 在 ({pos[0]:.1f}, {pos[1]:.1f})")
            return pos
        print(f"  [感知] 未找到 '{object_name}'")
        return None

    @staticmethod
    def pick(target_pos):
        """抓取物体（第6/19章）"""
        print(f"  [操作] 抓取 ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
        for jnt_idx in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_idx)
            if name and "shoulder" in name.lower():
                data.ctrl[model.jnt_qposadr[jnt_idx]] = 50
        for _ in range(100):
            mujoco.mj_step(model, data)
        return "抓取完成"

    @staticmethod
    def speak(message):
        """语音输出（第12章）"""
        print(f"  [语音] 🤖: '{message}'")
        return f"已输出: {message}"

# ============================================================
# 任务规划器（你需要实现的部分）
# ============================================================
def plan_and_execute(task_description):
    """
    任务规划与执行的核心函数。
    输入：自然语言指令
    输出：执行结果
    """
    print(f"\n{'='*50}")
    print(f"  任务: '{task_description}'")
    print(f"{'='*50}")

    skills = Skills()
    results = []

    # --------------------------------------------------
    # 以下是需要你实现的部分：
    # 1. 用LLM/规则引擎解析任务
    # 2. 分解为子任务序列
    # 3. 调度技能执行
    # 4. 处理异常
    # --------------------------------------------------
    desc = task_description.lower()

    if "拿" in desc and ("杯" in desc or "水" in desc):
        # 示例："去厨房拿一杯水" → 导航+感知+抓取+返回
        skills.speak("好的，我去拿水")
        results.append(skills.walk_to(2.0, 0.0))
        cup_pos = skills.detect_object("cup")
        if cup_pos is not None:
            results.append(skills.walk_to(cup_pos[0], cup_pos[1], 100))
            results.append(skills.pick(cup_pos))
            results.append(skills.walk_to(0.0, 0.0))
            skills.speak("水拿来了")
        else:
            skills.speak("我找不到水杯")
    elif "状态" in desc or "高度" in desc:
        results.append(f"高度: {data.qpos[2]:.2f}m")
    elif "跳" in desc or "dance" in desc:
        skills.speak("跳舞还不支持，但我可以走两步")
        results.append(skills.walk_to(0.5, 0.0))
        results.append(skills.walk_to(-0.5, 0.0))
    else:
        skills.speak(f"抱歉，我还没学会'{task_description}'")
        results.append("未识别的指令")

    return results

# ============================================================
# 主循环
# ============================================================
test_tasks = [
    "去拿一杯水",
    "查看当前状态",
]

for task in test_tasks:
    results = plan_and_execute(task)
    for r in results:
        print(f"  → {r}")

print(f"\n{'='*60}")
print(f"  🎯 综合大作业基础框架已就绪！")
print(f"  你需要完成:")
print(f"  1. 扩展任务规划器（支持更多指令）")
print(f"  2. 集成LLM做意图解析（第22章）")
print(f"  3. 增加更多技能（各章实操代码）")
print(f"  4. 处理多任务组合")
print(f"  5. 处理异常和失败恢复")
print(f"{'='*60}")
