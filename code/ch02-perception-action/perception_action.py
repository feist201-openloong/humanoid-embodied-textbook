"""
第2章 仿真实操：感知→行动闭环
================================================================
目标：实现一个最简单的"感知→行动"闭环——
机器人检测到"脚触地"信号后，产生"抬腿"反应。
"""

import mujoco
import mujoco.viewer
import numpy as np

# ============================================================
# 1. 加载模型
# ============================================================
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print("="*60)
print("  感知→行动闭环实验")
print("="*60)

# ============================================================
# 2. 感知函数：检测脚部触地
# ============================================================
# 查找脚部刚体
foot_body_ids = []
for name_pattern in ["foot_left", "foot_right", "left_foot", "right_foot",
                       "lfoot", "rfoot", "left_ankle", "right_ankle",
                       "left leg", "right leg"]:
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name_pattern)
        if body_id >= 0:
            foot_body_ids.append(body_id)
            print(f"  发现脚部刚体: {name_pattern} (id={body_id})")
    except:
        pass

if not foot_body_ids:
    # 回退：使用末端刚体作为"脚部"
    print("  未找到命名脚部，使用末端刚体模拟")
    foot_body_ids = [model.nbody - 1, model.nbody - 2]

def detect_ground_contact(data, body_ids, geom_model=None):
    """检测指定的刚体是否与地面接触"""
    if geom_model is None:
        geom_model = data.geom_xpos.shape[0]
    for i in range(data.ncon):
        c = data.contact[i]
        # 检查两个接触几何体是否涉及脚部
        for foot_id in body_ids:
            # 获取接触涉及的刚体 ID
            if c.geom[0] == foot_id or c.geom[1] == foot_id:
                return True
    return False

# ============================================================
# 3. 行动参数
# ============================================================
LIFT_DURATION = 50      # 抬腿持续步数
LIFT_AMOUNT = 0.3       # 抬腿幅度 (弧度)

# 找髋关节相关关节
hip_joints = []
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and ("hip" in name.lower() or ("thigh" in name.lower())):
        addr = model.jnt_qposadr[i]
        hip_joints.append((i, name, addr))
        print(f"  发现髋关节: {name} (id={i}, qpos_addr={addr})")

# ============================================================
# 4. 闭环仿真
# ============================================================
print(f"\n--- 运行感知→行动闭环仿真（1000步）---")
mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0  # 抬高一点初始位置

lift_counter = 0
is_lifting = False

for step in range(1000):
    # 感知
    on_ground = detect_ground_contact(data, foot_body_ids)

    # 决策
    if on_ground and not is_lifting:
        print(f"  步 {step:4d}: ⚡ 检测到触地! 启动抬腿反应")
        is_lifting = True
        lift_counter = LIFT_DURATION

    # 行动
    if is_lifting:
        lift_counter -= 1
        for jid, jname, addr in hip_joints:
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                # 旋转关节：修改角度实现抬腿
                data.qpos[addr] -= LIFT_AMOUNT * (LIFT_DURATION - lift_counter) / LIFT_DURATION
        if lift_counter <= 0:
            is_lifting = False
            print(f"  步 {step:4d}: 抬腿结束，恢复姿态")

    # 推进仿真
    mujoco.mj_step(model, data)

    if step % 200 == 0:
        height = data.qpos[2]
        status = "🦶 触地" if on_ground else "🦶 悬空"
        lift_status = "⬆ 抬腿中" if is_lifting else "➖ 常态"
        print(f"  步 {step:4d}: 高度={height:.2f}m  {status}  {lift_status}")

print(f"\n  ✅ 仿真实操完成！")
print(f"  验证：观察控制台中 ⚡ 触地反应是否触发")
print(f"="*60)

# ============================================================
# 5. 交互式可视化（可选）
# ============================================================
print(f"\n--- 交互式查看器 ---")
try:
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(1000):
            mujoco.mj_step(model, data)
            viewer.sync()
except Exception as e:
    print(f"  查看器不可用: {e}")

print(f"\n实验完成！这个感知→行动闭环展示了具身智能的基本框架。")
