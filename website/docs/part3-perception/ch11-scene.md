---
sidebar_position: 3
---

# 第11章：场景理解与建图

> **本章导语**：要让机器人在环境中自主行动，它必须"理解"周围的场景——哪些区域可以通行、哪些物体可以操作、环境结构如何。本章从语义分割、点云处理到占用建图和隐式场景表示，建立完整的场景理解技术栈。

---

## 11.1 语义分割

### 11.1.1 什么是语义分割

语义分割（Semantic Segmentation）为图像的每个像素分配一个类别标签——"地面"、"墙壁"、"人"、"机器人"、"桌子"等。这是场景理解的"第一公里"。

### 11.1.2 主要模型

| 模型 | 年份 | 特点 |
|:----|:----:|------|
| FCN | 2015 | 全卷积网络，端到端语义分割 |
| UNet | 2015 | 编码-解码对称结构，医学图像 |
| DeepLabV3+ | 2018 | 空洞卷积 + ASPP模块 |
| SAM | 2023 | Meta的通用分割模型，零样本分割 |

```python
# SAM 使用示例（伪代码）
import torch
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(
    point_coords=input_points, point_labels=input_labels)
```

### 11.1.3 实例分割 vs 语义分割 vs 全景分割

- **语义分割**：给每个像素一个类别（"所有椅子都标记为'椅子'"）
- **实例分割**：区分同类物体的不同实例（"椅子1"、"椅子2"）
- **全景分割**：语义分割 + 实例分割（给每个像素一个类别+实例ID）

## 11.2 3D点云处理

### 11.2.1 点云获取

点云数据可以从以下来源获得：
- 深度相机 + 内参反投影
- 激光雷达（LiDAR）直接测量
- 多视角图像重建（SfM / MVS）

### 11.2.2 点云处理算法

**PointNet++** 是最有影响力的点云深度学习架构之一。它直接在点云上操作，学习逐点特征和全局特征。

常用的点云处理工具：
- **Open3D**：Python点云处理库（滤波、配准、分割、可视化）
- **PCL**（Point Cloud Library）：C++点云处理库
- **pyntcloud**：轻量级Python点云处理

```python
# Open3D 体素滤波
import open3d as o3d
pcd = o3d.io.read_point_cloud("scene.pcd")
pcd_down = pcd.voxel_down_sample(voxel_size=0.01)  # 体素滤波降采样
pcd.estimate_normals()  # 估计法向量
```

## 11.3 占用建图

### 11.3.1 占用栅格地图

占用栅格地图（Occupancy Grid Map）将环境离散化为均匀网格，每个网格存储该位置被占用的概率。这是最经典的环境建图方法。

占用概率通过**贝叶斯更新**来融合多次观测：

$$ P(occupied \mid z_&#123;1:t&#125;) = \frac&#123;P(z_t \mid occupied)&#125;&#123;P(z_t \mid free)&#125; \cdot \frac&#123;P(occupied \mid z_&#123;1:t-1&#125;)&#125;&#123;1 - P(occupied \mid z_&#123;1:t-1&#125;)&#125; $$

### 11.3.2 OctoMap

OctoMap 使用八叉树（Octree）来表示占用信息，相比统一栅格更加高效——稀疏区域用大节点表示，细节区域用小节点表示。

## 11.4 隐式场景表示

### 11.4.1 NeRF

NeRF（Neural Radiance Fields）用神经网络隐式表示场景的辐射场。输入5D坐标（3D位置 + 2D视角方向），输出颜色和密度。NeRF可以生成高质量的新视角图像。

### 11.4.2 SDF 与 Occupancy Networks

符号距离函数（Signed Distance Function, SDF）和 Occupancy Network 是另一种隐式表示，用神经网络输出每个3D位置到物体表面的距离（SDF）或被占用的概率。这类方法被广泛应用于机器人操作场景的几何建模。

---

## 🛠️ 仿真实操：语义地图构建

> 🕐 预计时长：60分钟 · 📁 代码位置：`code/ch11-scene-mapping/scene_mapping.py`

### 实验目标
在 MuJoCo 环境中采集多视角 RGB-D 数据，构建语义占用网格地图。

### 实验步骤
1. 在 MuJoCo 中搭建包含多种物体的场景
2. 在多个视角拍摄 RGB-D 图像
3. 对 RGB 图像做语义分割（颜色标签）
4. 将语义信息投影到占用网格中
5. 可视化最终的语义地图

### 延伸思考
1. 不同视角的数量对地图质量有什么影响？
2. 如何处理动态物体（如移动的人）？
3. 如何利用语义信息改进机器人的路径规划？

---

## 本章小结

- 语义分割为图像每个像素分配类别标签
- PointNet++ 直接在3D点云上学习特征
- 占用栅格地图和OctoMap是环境建图的标准方法
- NeRF和SDF提供了高质量的隐式场景表示

## 习题

1. **选择题**：语义分割和实例分割的主要区别是？
   A. 语义分割更精确  B. 实例分割区分同类不同个体
   C. 实例分割更快  D. 没有区别

2. **简答题**：占用栅格地图的贝叶斯更新过程的基本原理是什么？

3. **简答题**：OctoMap 相比均匀栅格地图的优势是什么？

4. **开放题**：如果你要让机器人在一个从未见过的房间里找到"杯子"，需要场景理解的哪些能力？

## 参考文献
1. Long, J., Shelhamer, E., & Darrell, T. (2015). "Fully Convolutional Networks for Semantic Segmentation." *CVPR.*
2. Qi, C. R., et al. (2017). "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space." *NeurIPS.*
3. Hornung, A., et al. (2013). "OctoMap: An Efficient Probabilistic 3D Mapping Framework Based on Octrees." *Autonomous Robots.*
4. Mildenhall, B., et al. (2020). "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis." *ECCV.*
