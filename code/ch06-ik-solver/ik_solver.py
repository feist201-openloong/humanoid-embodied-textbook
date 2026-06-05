"""
第6章 仿真实操：IK求解器实现
================================================================
目标：用Python从零实现数值IK求解器，
控制MuJoCo机器人手臂末端到达指定位置。
"""

import mujoco
import numpy as np

print("="*60)
print("  逆运动学（IK）求解器 — 雅可比转置法")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# 寻找手臂末端刚体（手部）
hand_body_ids = []
for name in ["hand_left", "hand_right", "left_hand", "right_hand",
             "lhand", "rhand", "left_palm", "right_palm"]:
    try:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            hand_body_ids.append(body_id)
            print(f"  发现手部刚体: {name} (id={body_id})")
    except:
        pass

if not hand_body_ids:
    # 使用右臂末端
    print("  未找到命名的手部，使用右臂末端")
    hand_body_ids = [model.nbody - 2]  # 倒数第二个刚体通常是末端

# 寻找肩关节和肘关节
arm_joints = []
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and ("shoulder" in name.lower() or "elbow" in name.lower()
                 or "upper_arm" in name.lower() or "forearm" in name.lower()):
        arm_joints.append((i, name, model.jnt_qposadr[i]))
        print(f"  发现手臂关节: {name} (id={i}, qpos_addr={model.jnt_qposadr[i]})")

def get_endeffector_pos(data, body_id):
    """获取末端执行器的三维位置"""
    return data.xpos[body_id].copy()

def numerical_jacobian(model, data, body_id, eps=1e-6):
    """数值计算雅可比矩阵（位置部分）"""
    nq = model.nq
    jac = np.zeros((3, nq))
    x0 = get_endeffector_pos(data, body_id)
    for i in range(nq):
        # 微小扰动第i个自由度
        data.qpos[i] += eps
        mujoco.mj_forward(model, data)
        x_plus = get_endeffector_pos(data, body_id)
        data.qpos[i] -= eps  # 恢复
        jac[:, i] = (x_plus - x0) / eps
    mujoco.mj_forward(model, data)  # 恢复原始状态
    return jac

def ik_solve(model, data, target_pos, body_id, max_iter=1000, alpha=0.5, tol=1e-3):
    """雅可比转置法求解逆运动学"""
    for iteration in range(max_iter):
        mujoco.mj_forward(model, data)
        current_pos = get_endeffector_pos(data, body_id)
        error = target_pos - current_pos
        err_norm = np.linalg.norm(error)

        if err_norm < tol:
            return iteration, True

        jac = numerical_jacobian(model, data, body_id, 1e-4)
        dq = jac.T @ error
        dq_norm = np.linalg.norm(dq)
        if dq_norm > 1.0:
            dq = dq / dq_norm  # 限制步长

        data.qpos[:model.nq] += alpha * dq

        if iteration % 100 == 0:
            print(f"  IK迭代 {iteration:4d}: 误差 = {err_norm:.4f}")

    return max_iter, False

# 设置初始姿态
mujoco.mj_resetData(model, data)
data.qpos[2] = 1.0

if hand_body_ids and arm_joints:
    target = np.array([0.3, 0.0, 0.5])
    print(f"\n目标位置: {target}")

    iterations, success = ik_solve(model, data, target, hand_body_ids[0])
    final_pos = get_endeffector_pos(data, hand_body_ids[0])
    final_error = np.linalg.norm(target - final_pos)

    print(f"\n  IK {'✅ 收敛' if success else '⚠️ 未完全收敛'} (迭代 {iterations}次)")
    print(f"  最终位置: {final_pos.round(3)}")
    print(f"  最终误差: {final_error:.4f}")

print(f"\n{'='*60}")
print(f"  实验完成！从零实现的IK求解器已运行。")
print(f"{'='*60}")
