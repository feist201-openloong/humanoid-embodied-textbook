"""
第3章 仿真实操：旋转与变换可视化
================================================================
目标：用 NumPy 实现旋转矩阵↔四元数↔轴角转换，
并在 MuJoCo 中验证关节控制。
"""

import mujoco
import numpy as np

print("="*60)
print("  旋转表示法转换与 MuJoCo 验证")
print("="*60)

# ============================================================
# 第一部分：数学演示 (NumPy)
# ============================================================

def rotation_matrix_from_axis_angle(axis, angle):
    """轴角 → 旋转矩阵 (罗德里格斯公式)"""
    axis = np.asarray(axis) / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c + x*x*(1-c),    x*y*(1-c) - z*s,  x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,  c + y*y*(1-c),    y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,  z*y*(1-c) + x*s,  c + z*z*(1-c)]
    ])

def rotation_matrix_to_quaternion(R):
    """旋转矩阵 → 四元数 (w,x,y,z)"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def quaternion_to_rotation_matrix(q):
    """四元数 (w,x,y,z) → 旋转矩阵"""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qw*qz,     2*qx*qz + 2*qw*qy],
        [2*qx*qy + 2*qw*qz,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qw*qx],
        [2*qx*qz - 2*qw*qy,       2*qy*qz + 2*qw*qx,     1 - 2*qx*qx - 2*qy*qy]
    ])

print(f"\n--- 基础测试：绕 Z 轴旋转 90° ---")
axis_z = np.array([0, 0, 1])
angle_90 = np.pi / 2
R_z90 = rotation_matrix_from_axis_angle(axis_z, angle_90)
q_z90 = rotation_matrix_to_quaternion(R_z90)
R_back = quaternion_to_rotation_matrix(q_z90)

print(f"  旋转矩阵:\n{R_z90.round(4)}")
print(f"  四元数 (w,x,y,z): [{q_z90[0]:.4f}, {q_z90[1]:.4f}, {q_z90[2]:.4f}, {q_z90[3]:.4f}]")
print(f"  还原验证: {'✅ 一致' if np.allclose(R_z90, R_back) else '❌ 不一致'}")

# 复合旋转
print(f"\n--- 复合旋转：先绕 X 转 45° 再绕 Y 转 45° ---")
R_x45 = rotation_matrix_from_axis_angle(np.array([1,0,0]), np.pi/4)
R_y45 = rotation_matrix_from_axis_angle(np.array([0,1,0]), np.pi/4)
R_comp = R_y45 @ R_x45
q_comp = rotation_matrix_to_quaternion(R_comp)
print(f"  复合旋转矩阵:\n{R_comp.round(4)}")
print(f"  复合四元数: [{q_comp[0]:.4f}, {q_comp[1]:.4f}, {q_comp[2]:.4f}, {q_comp[3]:.4f}]")

# ============================================================
# 第二部分：MuJoCo 验证
# ============================================================
print(f"\n{'='*60}")
print(f"  MuJoCo 关节控制验证")
print(f"{'='*60}")

model = mujoco.MjModel.from_xml_path(
    str(mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"))
data = mujoco.MjData(model)

# 查找肩关节
shoulder_joints = []
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name and ("shoulder" in name.lower() or "upper_arm" in name.lower()):
        addr = model.jnt_qposadr[i]
        shoulder_joints.append((i, name, addr, model.jnt_type[i]))
        print(f"  发现关节: {name} (id={i}, qpos_addr={addr})")

if shoulder_joints:
    jid, jname, addr, jtype = shoulder_joints[0]
    target_angle = np.pi / 6  # 30°

    if jtype == mujoco.mjtJoint.mjJNT_HINGE:
        data.qpos[addr] = target_angle
        print(f"\n  设置关节 [{jname}] 角度: {target_angle:.3f} rad ({np.degrees(target_angle):.1f}°)")
    elif jtype == mujoco.mjtJoint.mjJNT_BALL:
        q = rotation_matrix_to_quaternion(
            rotation_matrix_from_axis_angle(np.array([0,0,1]), target_angle))
        data.qpos[addr:addr+4] = q
        print(f"\n  设置球铰 [{jname}] 四元数: {q.round(4)}")

    mujoco.mj_forward(model, data)
    print(f"  前向运动学计算完成")

    # 打印该关节在世界坐标系中的位置
    body_id = model.jnt_bodyid[jid]
    print(f"  刚体 [{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)}] 位置: "
          f"{data.xpos[body_id].round(3)}")
else:
    print("  未找到肩关节，尝试使用躯干验证")
    # 直接验证自由关节：绕 Z 轴旋转机器人
    q_home = rotation_matrix_to_quaternion(np.eye(3))
    data.qpos[3:7] = q_home  # 设置躯干旋转
    mujoco.mj_forward(model, data)
    print(f"  躯干姿态四元数: {data.qpos[3:7].round(4)}")

print(f"\n{'='*60}")
print(f"  实验完成！第3章旋转与变换可视化实验通过。")
print(f"{'='*60}")
