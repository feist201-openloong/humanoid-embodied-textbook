# 人形机器人与具身智能：从入门到精通

> **Humanoid Robots & Embodied Intelligence: From Foundations to Frontiers**

[![CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Website](https://img.shields.io/badge/Website-在线阅读-blue)](https://your-username.github.io/humanoid-embodied-textbook/)
[![Deploy](https://github.com/your-username/humanoid-embodied-textbook/actions/workflows/deploy-website.yml/badge.svg)](https://github.com/your-username/humanoid-embodied-textbook/actions/workflows/deploy-website.yml)

## 📚 关于本书

一本**完全开源**的本科生教材，系统讲解人形机器人与具身智能，从入门到精通。

- **字数**: 75万+，25章 + 5个附录
- **实操**: 每章配有 MuJoCo 仿真实操，共25个实验
- **对象**: 本科生（大三/大四），Python + 线性代数基础
- **开源**: CC BY-NC-SA 4.0 协议

## 🗂️ 目录结构

```
├── website/          # Docusaurus 网站（在线阅读）
│   └── docs/         # 教材正文 Markdown
├── code/             # 各章仿真实操代码
├── assets/           # 模型文件、图片素材
└── docs/             # 设计文档与计划
```

## 🚀 在线阅读

访问 [https://your-username.github.io/humanoid-embodied-textbook/](https://your-username.github.io/humanoid-embodied-textbook/)

## 🛠️ 本地开发

```bash
# 1. 克隆
git clone https://github.com/your-username/humanoid-embodied-textbook.git
cd humanoid-embodied-textbook

# 2. 安装网站依赖
cd website && npm install

# 3. 启动本地预览
npm start

# 4. 安装 Python 依赖（实操代码）
cd ../code
pip install -r ch01-load-model/requirements.txt
```

## 📖 写作进度

| 部分 | 章节 | 状态 |
|:----|:----|:----:|
| 第一部分：基础入门篇 | 第1-4章 | 🖊️ 撰写中 |
| 第二部分：运动系统篇 | 第5-8章 | ⬜ 待撰写 |
| 第三部分：感知与交互篇 | 第9-12章 | ⬜ 待撰写 |
| 第四部分：具身智能核心篇 | 第13-17章 | ⬜ 待撰写 |
| 第五部分：系统集成与实操项目篇 | 第18-21章 | ⬜ 待撰写 |
| 第六部分：前沿拓展篇 | 第22-25章 | ⬜ 待撰写 |
| 附录 | A-E | ⬜ 待撰写 |

## 🤝 参与贡献

欢迎通过以下方式参与：

- **提交勘误**: 在对应章节页面点击 "编辑此页" 提交 PR
- **反馈建议**: 在 GitHub Issues 中提交
- **代码贡献**: 改进仿真实操代码

## 📄 协议

本教材采用 **CC BY-NC-SA 4.0** 协议开源。
商业出版请联系作者。
