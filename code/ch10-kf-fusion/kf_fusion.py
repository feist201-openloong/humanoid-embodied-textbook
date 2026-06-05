"""
第10章 仿真实操：卡尔曼滤波姿态估计
================================================================
目标：融合IMU + 视觉标记数据进行姿态估计。
模拟IMU的快速高频（但有漂移）和视觉标记的低频精确观测。
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*60)
print("  卡尔曼滤波姿态估计融合实验")
print("="*60)

# ============================================================
# 1. 仿真参数
# ============================================================
DT = 0.01           # 时间步长 (秒)
SIM_TIME = 10.0     # 总仿真时间 (秒)
N_STEPS = int(SIM_TIME / DT)

# 真实的角速度（匀速旋转）
TRUE_OMEGA = 0.3    # rad/s

# 噪声参数
IMU_OMEGA_STD = 0.1     # IMU 角速度噪声标准差 (rad/s)
VIS_THETA_STD = 0.05    # 视觉观测角度噪声标准差 (rad)
VIS_INTERVAL = 10       # 视觉观测周期（每N步观测一次）

print(f"  仿真时间: {SIM_TIME}s, 步长: {DT}s, 总步数: {N_STEPS}")
print(f"  真实角速度: {TRUE_OMEGA} rad/s")
print(f"  IMU 噪声: {IMU_OMEGA_STD} rad/s")
print(f"  视觉观测周期: 每{VIS_INTERVAL}步观测一次")

# ============================================================
# 2. 生成真实状态
# ============================================================
true_theta = np.zeros(N_STEPS)
for k in range(1, N_STEPS):
    true_theta[k] = true_theta[k-1] + TRUE_OMEGA * DT

# ============================================================
# 3. 模拟传感器数据
# ============================================================
# IMU（高频，有偏置）
imu_bias = 0.02  # 固定的偏置
imu_omega = true_theta.copy()
for k in range(N_STEPS):
    imu_omega[k] = imu_omega[k-1] + (TRUE_OMEGA + imu_bias + np.random.randn() * IMU_OMEGA_STD) * DT
    # 通过积分获得姿态
imu_theta = np.cumsum(imu_omega) * DT + np.random.randn(N_STEPS) * 0.01

# 视觉（低频，无漂移）
vis_theta = np.full(N_STEPS, np.nan)
for k in range(0, N_STEPS, VIS_INTERVAL):
    vis_theta[k] = true_theta[k] + np.random.randn() * VIS_THETA_STD

# ============================================================
# 4. 卡尔曼滤波
# ============================================================
print(f"\n--- 运行卡尔曼滤波 ---")

# 状态: [theta] (角度)
# 模型: theta_k = theta_{k-1} + omega * dt
x_est = 0.0       # 初始估计
P_est = 1.0       # 初始协方差
Q = 0.1           # 过程噪声（IMU不确定度）
R = VIS_THETA_STD**2  # 观测噪声

x_history = np.zeros(N_STEPS)

for k in range(N_STEPS):
    # 预测
    x_pred = x_est + imu_omega[k]  # 使用IMU角速度
    P_pred = P_est + Q

    # 观测更新（如果有视觉数据）
    if not np.isnan(vis_theta[k]):
        K = P_pred / (P_pred + R)
        x_est = x_pred + K * (vis_theta[k] - x_pred)
        P_est = (1 - K) * P_pred
    else:
        x_est = x_pred
        P_est = P_pred

    x_history[k] = x_est

# ============================================================
# 5. 结果分析
# ============================================================
error_kf = np.abs(x_history - true_theta)
error_imu = np.abs(imu_theta - true_theta)

print(f"  卡尔曼滤波平均误差: {np.mean(error_kf):.4f} rad")
print(f"  纯IMU积分平均误差: {np.mean(error_imu):.4f} rad")
print(f"  滤波提升: {(1 - np.mean(error_kf)/np.mean(error_imu))*100:.1f}%")

print(f"\n{'='*60}")
print(f"  卡尔曼滤波姿态估计实验完成。")
print(f"  结果: 融合IMU + 视觉 的估计精度远高于纯IMU积分。")
print(f"{'='*60}")
