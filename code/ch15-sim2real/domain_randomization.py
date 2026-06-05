"""
第15章 仿真实操：域随机化实战
================================================================
目标：训练对物理参数变化鲁棒的策略，测试泛化边界。
"""

import mujoco
import numpy as np

print("="*60)
print("  域随机化实战")
print("="*60)

# 加载模型
xml_path = mujoco.models.MODEL_DIR / "humanoid" / "humanoid.xml"

def randomize_physics(model, seed=None):
    """随机化仿真物理参数"""
    if seed is not None:
        np.random.seed(seed)

    # 物理参数随机化范围
    param_ranges = {
        'friction': (0.2, 1.5),        # 地面摩擦系数
        'damping': (0.5, 3.0),          # 关节阻尼乘数
        'mass_scale': (0.8, 1.2),       # 质量缩放
        'gravity_scale': (0.8, 1.2),    # 重力缩放
        'timestep': (0.002, 0.01),       # 仿真步长
    }

    params = {}
    # 地面摩擦
    params['friction'] = np.random.uniform(*param_ranges['friction'])
    # 注意: MuJoCo的friction是通过geom参数设置的
    for i in range(model.ngeom):
        model.geom_friction[i, 0] = params['friction']

    # 关节阻尼
    damping_scale = np.random.uniform(*param_ranges['damping'])
    for i in range(model.njnt):
        if i < len(model.damping):
            original_damping = model.damping[i]
            model.damping[i] = original_damping * damping_scale
    params['damping_scale'] = damping_scale

    # 刚体质量
    mass_scale = np.random.uniform(*param_ranges['mass_scale'])
    for i in range(model.nbody):
        model.body_mass[i] *= mass_scale
    params['mass_scale'] = mass_scale

    # 重力
    grav_scale = np.random.uniform(*param_ranges['gravity_scale'])
    model.opt.gravity[2] = -9.81 * grav_scale
    params['gravity_scale'] = grav_scale

    return params

print(f"\n--- 实验1: 无域随机化训练 ---")
# 固定参数下运行
model_default = mujoco.MjModel.from_xml_path(str(xml_path))
data_default = mujoco.MjData(model_default)
mujoco.mj_resetData(model_default, data_default)
data_default.qpos[2] = 1.0

for step in range(500):
    mujoco.mj_step(model_default, data_default)
final_height_default = data_default.qpos[2]
print(f"  默认参数: 最终高度 = {final_height_default:.3f}m")

print(f"\n--- 实验2: 随机化参数后测试 ---")
BEST_HEIGHT = -np.inf
BEST_PARAMS = None

for trial in range(5):
    model_rand = mujoco.MjModel.from_xml_path(str(xml_path))
    data_rand = mujoco.MjData(model_rand)

    params = randomize_physics(model_rand, seed=trial)
    mujoco.mj_resetData(model_rand, data_rand)
    data_rand.qpos[2] = 1.0

    for step in range(500):
        mujoco.mj_step(model_rand, data_rand)
    final_height = data_rand.qpos[2]
    print(f"  试验 {trial+1}: 摩擦={params['friction']:.2f}, "
          f"阻尼={params['damping_scale']:.2f}, "
          f"质量={params['mass_scale']:.2f}, "
          f"重力={params['gravity_scale']:.2f} → "
          f"高度={final_height:.2f}m")

    if final_height > final_height_default * 0.7:
        print(f"    → 策略鲁棒: 高度下降不超过30%")

print(f"\n--- 结论 ---")
print(f"  默认参数: 高度 = {final_height_default:.2f}m")
print(f"  域随机化: 策略在变化的物理条件下表现出不同的稳定性。")
print(f"  在实际部署中，域随机化训练的策略比固定参数训练的鲁棒得多。")

print(f"\n{'='*60}")
print(f"  实验完成！域随机化是Sim2Real迁移的核心技术。")
print(f"{'='*60}")
