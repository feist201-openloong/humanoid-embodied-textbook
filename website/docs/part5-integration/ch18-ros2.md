---
sidebar_position: 1
---

# 第18章：机器人操作系统与集成

> **本章导语**：机器人是一个复杂的分布式系统——感知、规划、控制、人机交互等模块需要高效协同。ROS2（Robot Operating System 2）是当前最广泛使用的机器人软件框架。本章学习ROS2的核心概念，并搭建MuJoCo与ROS2的桥接系统。

---

## 18.1 ROS2 基础

### 18.1.1 核心概念

ROS2是一个分布式通信框架，核心概念包括：

| 概念 | 类比 | 说明 |
|:----|:-----|:------|
| **节点（Node）** | 程序/进程 | 每个节点完成一个特定功能 |
| **话题（Topic）** | 广播频道 | 发布-订阅模式，一对多通信 |
| **服务（Service）** | 远程函数调用 | 请求-响应模式，一对一 |
| **动作（Action）** | 可取消的长任务 | 带反馈的服务调用 |
| **TF（Transform）** | 坐标变换树 | 维护所有坐标系的变换关系 |

### 18.1.2 ROS2 vs ROS1

| 特性 | ROS1 | ROS2 |
|:----|:----|:-----|
| 通信中间件 | 自定义TCPROS | DDS（Data Distribution Service）|
| 实时性 | 不支持 | 支持 |
| 安全性 | 无 | 内置加密+认证 |
| 多平台 | Linux | Linux/Windows/macOS |
| 生命周期管理 | 无 | 有（Unconfigured→Inactive→Active）|

### 18.1.3 节点间通信

**话题（Topic）** 是ROS2中最常用的通信方式：

```python
# 发布者
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')
        self.pub = self.create_publisher(Pose, 'robot_pose', 10)
        self.timer = self.create_timer(0.1, self.publish_pose)

    def publish_pose(self):
        msg = Pose()
        # 填充机器人位姿
        msg.position.x = 1.0
        msg.position.y = 2.0
        msg.position.z = 0.5
        self.pub.publish(msg)
```

```python
# 订阅者
class RobotStateSubscriber(Node):
    def __init__(self):
        super().__init__('robot_state_subscriber')
        self.sub = self.create_subscription(
            Pose, 'robot_pose', self.pose_callback, 10)

    def pose_callback(self, msg):
        self.get_logger().info(f'收到位姿: ({msg.position.x:.2f}, {msg.position.y:.2f})')
```

## 18.2 TF坐标变换

TF（Transform）维护机器人所有坐标系的变换关系：

```
world → odom → base_link → 躯干、头部、左臂、右臂、左腿、右腿…
                            ↓
                        左肩、左上臂、左肘、左前臂、左腕、左手…
```

TF的两种使用方式：
- **TF监听（Lookup）**：查询任意两个坐标系之间的变换
- **TF广播（Broadcast）**：发布坐标系的变换关系

```python
# TF监听示例
from tf2_ros import TransformListener, Buffer

tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer)

# 获取"左手"在"世界坐标系"中的位置
try:
    t = tf_buffer.lookup_transform('world', 'left_hand', rclpy.time.Time())
    pos = t.transform.translation
    print(f'左手位置: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})')
except Exception as e:
    print(f'TF查询失败: {e}')
```

## 18.3 MuJoCo-ROS2桥接

将MuJoCo仿真与ROS2连接，使仿真环境像真实机器人一样通过ROS2接口通信。

### 18.3.1 桥接架构

```
┌──────────────────┐     ROS2话题      ┌──────────────────┐
│   MuJoCo 仿真     │ ◄──────────────► │   ROS2 节点       │
│                  │  /joint_states    │   (控制器/规划器)  │
│  模型 + 数据      │  /robot_pose     │                   │
│                  │  /cmd_vel         │                   │
│                  │  → control        │                   │
└──────────────────┘                   └──────────────────┘
```

### 18.3.2 关键通信接口

| ROS2话题 | 方向 | 数据类型 | 用途 |
|:---------|:----|:---------|:-----|
| `/joint_states` | 仿真→ROS | JointState | 发布关节状态 |
| `/robot_pose` | 仿真→ROS | Pose | 发布机器人位姿 |
| `/cmd_joint` | ROS→仿真 | Float64MultiArray | 接收关节控制指令 |
| `/cmd_vel` | ROS→仿真 | Twist | 接收速度指令 |

---

## 🛠️ 仿真实操：搭建MuJoCo-ROS2桥接

> 🕐 预计时长：60分钟 · 📁 代码位置：`code/ch18-ros2-bridge/mujoco_ros2_bridge.py`

### 实验目标
搭建仿真器与ROS2的双向通信，使MuJoCo成为一个"虚拟机器人"。

### 实验步骤
1. 安装ROS2（若未安装）
2. 实现MuJoCo仿真节点：加载模型、步进仿真、发布关节状态
3. 实现指令订阅节点：接收外部控制指令
4. 启动桥接系统，验证双向通信
5. **无ROS2环境**：使用Python多线程模拟ROS2通信

### 延伸思考
1. 为什么仿真频率和控制频率可能不同？如何处理？
2. 如果ROS2指令的发布频率高于MuJoCo仿真频率，应该怎么办？
3. 从仿真切换到实物时，通信接口需要做哪些修改？

---

## 本章小结

- ROS2的节点/话题/服务/动作是机器人软件开发的核心抽象
- TF坐标变换是机器人空间推理的基础设施
- MuJoCo-ROS2桥接使仿真环境像真实机器人一样工作
- 桥接的关键接口包括关节状态、位姿和控制指令

## 习题

1. **选择题**：ROS2中话题（Topic）的通信模式是？
   A. 请求-响应  B. 发布-订阅  C. 远程过程调用  D. 共享内存

2. **简答题**：ROS2中TF（坐标变换）的作用是什么？

3. **简答题**：MuJoCo-ROS2桥接中，为什么需要将仿真频率和控制频率解耦？

4. **操作题**：在你的机器上安装ROS2（推荐Humble），运行本章的桥接示例。

## 参考文献
1. Quigley, M., et al. (2009). "ROS: an open-source Robot Operating System." *ICRA Workshop.*
2. Macenski, S., et al. (2022). "The Marathon 2: A Navigation System." *IROS.*
3. Open Source Robotics Foundation. "ROS2 Documentation." *docs.ros.org.*
