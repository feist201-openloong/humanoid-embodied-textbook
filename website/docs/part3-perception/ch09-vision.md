---
sidebar_position: 1
---

# 第9章：机器人视觉基础

> **本章导语**：视觉是人形机器人感知世界最重要的通道。本章从相机模型出发，讲解特征提取、立体视觉、深度估计和视觉SLAM，并指导你完成 MuJoCo 仿真中的 RGB-D 感知实验。

---

## 9.1 相机模型

### 9.1.1 针孔相机模型

针孔相机模型描述了三维空间点如何投影到二维图像平面：

$$ s \begin&#123;bmatrix&#125; u \\ v \\ 1 \end&#123;bmatrix&#125; = K \begin&#123;bmatrix&#125; R & t \end&#123;bmatrix&#125; \begin&#123;bmatrix&#125; X \\ Y \\ Z \\ 1 \end&#123;bmatrix&#125; $$

其中 $K$ 是**相机内参矩阵**（焦距 $f_x, f_y$、光心 $c_x, c_y$）：

$$ K = \begin&#123;bmatrix&#125; f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end&#123;bmatrix&#125; $$

$[R \mid t]$ 是**相机外参矩阵**，将世界坐标转换为相机坐标。$s$ 是缩放因子。

### 9.1.2 相机标定

标定（Calibration）的目的是确定内参 $K$ 和畸变参数。常用的方法是拍摄多张棋盘格图像，提取角点后用最小二乘法求解。

```python
# OpenCV 相机标定示例
import cv2
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None)
```

## 9.2 特征提取

### 9.2.1 传统特征

| 特征 | 年份 | 特点 | 用途 |
|:----|:----:|------|:----:|
| SIFT | 1999 | 尺度不变、旋转不变 | 图像匹配、SLAM |
| SURF | 2006 | SIFT的加速版 | 实时匹配 |
| ORB | 2011 | 快速、开源（无专利） | 视觉里程计 |
| SuperPoint | 2018 | 基于学习的特征 | 深度学习SLAM |

### 9.2.2 特征匹配

特征匹配通过描述子距离（如欧氏距离或汉明距离）找到两幅图像间的对应点：

```python
# ORB 特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
```

## 9.3 深度估计

### 9.3.1 双目立体视觉

双目相机通过计算左右图像的**视差**（disparity）来估计深度：

$$ Z = \frac&#123;f \cdot b&#125;&#123;d&#125; $$

其中 $f$ 是焦距，$b$ 是基线距离，$d = u_L - u_R$ 是视差。视差越大，物体越近。

### 9.3.2 RGB-D 相机

RGB-D 相机（如 Intel RealSense、Microsoft Kinect）直接输出每个像素的深度值，省略了立体匹配的计算。深度获取方式包括：

- **结构光**：投射红外图案，根据图案形变计算深度
- **飞行时间（ToF）**：测量红外光往返时间

## 9.4 视觉 SLAM

SLAM（Simultaneous Localization and Mapping）是让机器人在未知环境中同时估计自身位置和构建环境地图的技术。视觉 SLAM 的典型管线：

```
传感器输入（图像）
    ↓
前端：特征提取 + 特征匹配 + 帧间位姿估计（VO）
    ↓
后端：局部BA优化 + 回环检测 + 全局位姿图优化
    ↓
地图构建（稀疏/半稠密/稠密）
```

**ORB-SLAM3** 是目前最成熟的开源视觉SLAM系统，支持单目、双目和RGB-D，以及视觉-惯导融合。

---

## 🛠️ 仿真实操：RGB-D 感知实验

> 🕐 预计时长：45分钟 · 📁 代码位置：`code/ch09-rgbd-perception/rgbd_perception.py`

### 实验目标
在 MuJoCo 仿真环境中挂载 RGB-D 相机，使用 OpenCV 实现物体的检测和定位。

### 实验步骤
1. 在仿真场景中添加物体（球体或立方体）
2. 从 MuJoCo 获取 RGB 图像和深度图像
3. 使用 OpenCV 在 RGB 图像中检测物体（颜色分割）
4. 结合深度图计算物体的三维位置
5. 给机器人发送"看向物体"的控制指令

### 延伸思考
1. 不同光照条件下（修改仿真光照）颜色分割的鲁棒性如何？
2. 尝试检测多个物体并为每个物体计算三维位置
3. 如果物体与背景颜色相近，还有什么方法可以替代颜色分割？

---

## 本章小结

- 针孔相机模型建立了三维世界到二维图像的映射
- 特征提取（SIFT/ORB/SuperPoint）是视觉SLAM的基础
- 双目立体视觉和RGB-D相机提供了深度信息
- 视觉SLAM使机器人能在未知环境中定位和建图

## 习题

1. **选择题**：双目相机深度估计的精度与什么因素有关？
   A. 焦距  B. 基线距离  C. 视差精度  D. 以上都是

2. **简答题**：RGB-D 相机的深度获取原理有哪两种？

3. **简答题**：视觉 SLAM 中的"前端"和"后端"分别负责什么？

4. **计算题**：双目相机基线 $b = 0.1m$，焦距 $f = 500$ 像素，某点的视差 $d = 20$ 像素，求该点的深度 $Z$。

## 参考文献
1. Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision.* Cambridge.
2. Mur-Artal, R., & Tardós, J. D. (2017). "ORB-SLAM2: An Open-Source SLAM System for Monocular, Stereo, and RGB-D Cameras." *IEEE TRO.*
3. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics.* MIT Press.
