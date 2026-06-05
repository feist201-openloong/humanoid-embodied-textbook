"""
第7章 仿真实操：PD控制调节实验
================================================================
目标：在MuJoCo中实现关节PD控制器，
调节Kp/Kd参数观察从振荡→稳定→过阻尼的变化。
"""

import mujoco
import numpy as np

print("="*60)
print("  PD控制器参数调节实验")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"  模型自由度: {model.nq}, 执行器: {model.nu}")

def run_pd_control(kp, kd, steps=500, label=""):
    """用给定的PD参数运行仿真并返回高度轨迹"""
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 1.2  # 初始高度
    stand_qpos = data.qpos.copy()

    heights = []
    for step in range(steps):
        # 关节空间的PD控制
        for jnt_idx in range(model.njnt):
            qpos_addr = model.jnt_qposadr[jnt_idx]
            if qpos_addr < len(data.qpos) and qpos_addr < len(stand_qpos):
                pos_error = stand_qpos[qpos_addr] - data.qpos[qpos_addr]
                vel = data.qvel[qpos_addr] if qpos_addr < len(data.qvel) else 0.0
                control_force = kp * pos_error - kd * vel
                if qpos_addr < model.nu:
                    data.ctrl[qpos_addr] = np.clip(control_force, -100, 100)

        mujoco.mj_step(model, data)

        if step % 10 == 0:
            heights.append(data.qpos[2])

        # 每100步输出状态
        if step % 100 == 0:
            print(f"  [{label}] 步 {step:4d}: 高度 = {data.qpos[2]:.3f}m")

    return heights

# 参数对比实验
print(f"\n--- 实验1: 固定Kp=100，变化Kd ---")
for kd in [1, 10, 50]:
    print(f"\n  Kp=100, Kd={kd}:")
    heights = run_pd_control(100, kd, 300, f"Kd={kd}")
    final_h = heights[-1] if heights else 0
    print(f"  → 最终高度: {final_h:.3f}m")
    if kd <= 1:
        print(f"  → 低阻尼: 可能振荡")
    elif kd >= 50:
        print(f"  → 高阻尼: 响应慢但稳定")
    else:
        print(f"  → 中等阻尼: 较稳定的响应")

print(f"\n--- 实验2: 固定Kd=10，变化Kp ---")
for kp in [10, 100, 500]:
    print(f"\n  Kp={kp}, Kd=10:")
    heights = run_pd_control(kp, 10, 300, f"Kp={kp}")
    final_h = heights[-1] if heights else 0
    print(f"  → 最终高度: {final_h:.3f}m")
    if kp < 50:
        print(f"  → 低增益: 恢复慢")
    elif kp > 300:
        print(f"  → 高增益: 恢复快但可能振荡")
    else:
        print(f"  → 适中增益: 稳定的恢复")

print(f"\n{'='*60}")
print(f"  PD控制实验结果总结:")
print(f"  - Kp 决定恢复力的大小，Kp 越大恢复越快但可能振荡")
print(f"  - Kd 决定阻尼大小，Kd 越大越稳定但响应变慢")
print(f"  理想情况下 Kd 与 sqrt(Kp) 成正比。")
print(f"{'='*60}")
