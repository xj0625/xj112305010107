# 机器学习实验：基于CNN的手写数字识别

## 1. 学生信息

- **姓名**：谢洁
- **学号**：112305010107
- **班级**：数据1231

## 项目在线链接

- **GitHub 仓库地址**：https://github.com/xj0625/xj112305010107
- **GitHub README / 实验报告**：https://github.com/xj0625/xj112305010107/blob/main/README.md
- **在线演示**：部署中（待完成）

---

## 2. 项目概述

本实验基于 Kaggle Digit Recognizer / MNIST 手写数字识别任务，使用卷积神经网络（CNN）完成手写数字分类。实验主要包括模型训练、超参数调优、Kaggle 提交结果生成，以及后续 Web 应用部署准备。

本次最终提交文件为 sample_submission.csv，Kaggle 提交记录显示得分为 **0.99625**，达到实验要求的 0.98+ 目标。

---

## 3. 仓库目录结构

```
├── code/                      # 代码目录
│   ├── app.py                 # Gradio Web应用
│   ├── dnn_mnist.py           # 模型训练代码
│   ├── requirements.txt       # Python依赖
│   └── render.yaml            # Render部署配置
├── report/                    # 实验报告目录
│   └── CNN手写数字识别实验模板.md  # 完整实验报告
├── results/                    # 实验结果目录
│   ├── loss_curve.png         # 训练损失曲线
│   ├── acc_curve.png          # 训练准确率曲线
│   ├── sample_submission.csv  # Kaggle提交文件
│   └── exp_results.csv        # 对比实验结果
├── .github/                    # GitHub配置
│   └── workflows/
│       └── deploy.yml         # 部署工作流
├── .gitignore                 # Git忽略规则
└── README.md                  # 项目说明文档
```

---

## 4. 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows |
| Python | Python 3.12 |
| 深度学习框架 | PyTorch 2.4.0 + CUDA 12.6 |
| GPU | NVIDIA GeForce RTX |
| 数据集 | Kaggle Digit Recognizer：train.csv、test.csv、sample_submission.csv |
| 主要依赖 | torch、torchvision、numpy、Pillow、gradio、matplotlib |

---

## 5. 实验一：模型训练与超参数调优

### 5.1 实验目标

使用 CNN 对 28×28 灰度手写数字图像进行 0-9 十分类识别。通过对比不同优化器、学习率、Batch Size、数据增强和 Early Stopping 设置，分析超参数对模型收敛速度和泛化能力的影响，并生成 Kaggle 可提交的预测结果文件。

### 5.2 模型结构

最终提交模型使用增强型 CNN 结构：

```
输入(1×28×28)
→ Conv2d(1, 48, 3×3) + BatchNorm + SiLU
→ Conv2d(48, 48, 3×3) + BatchNorm + SiLU + MaxPool + Dropout(0.2)
→ Conv2d(48, 96, 3×3) + BatchNorm + SiLU
→ Conv2d(96, 96, 3×3) + BatchNorm + SiLU + MaxPool + Dropout(0.2)
→ Conv2d(96, 192, 3×3) + BatchNorm + SiLU
→ Conv2d(192, 256, 3×3) + BatchNorm + SiLU + AdaptiveAvgPool + Dropout(0.3)
→ Linear(256, 10)
→ 输出10类数字
```

### 5.3 超参数对比实验

| 实验组 | 优化器 | 学习率 | Batch Size | 数据增强 | Early Stopping | 验证集AUC |
|--------|--------|--------|------------|----------|----------------|-----------|
| 实验1 | SGD | 0.01 | 64 | 无 | 无 | 0.97xx |
| 实验2 | Adam | 0.001 | 128 | 无 | 无 | 0.98xx |
| 实验3 | AdamW | 0.003 | 128 | 有 | 有 | **0.99625** |
| 实验4 | AdamW | 0.003 | 128 | 无 | 有 | 0.99xxx |

### 5.4 最终模型超参数配置

| 类别 | 参数 | 值 |
|------|------|-----|
| 数据 | 训练集大小 | 42000 |
| | 验证集比例 | 20% |
| | 数据增强 | RandomShift, RandomRotation, RandomZoom |
| 模型 | 网络结构 | CNN(48→96→192→256) + AdaptiveAvgPool |
| | 激活函数 | SiLU |
| | BatchNorm | 是 |
| | Dropout | 0.2/0.3 |
| 训练 | 优化器 | AdamW (lr=0.003, weight_decay=1e-4) |
| | 学习率调度器 | OneCycleLR (pct_start=0.3) |
| | Batch Size | 128 |
| | 训练轮数 | 30 (Early Stopping) |
| | 损失函数 | CrossEntropyLoss |
| **Kaggle Score** | **0.99625** |

---

## 6. 实验二：模型封装与Web部署

### 6.1 目标

将训练好的模型封装为 Web 服务，提供在线手写数字识别功能。

### 6.2 技术栈

- **Web框架**：Gradio
- **模型框架**：PyTorch
- **部署平台**：待部署（Render/ModelScope）

### 6.3 功能特性

- [x] 上传图片识别
- [x] 网页手写板识别
- [x] Top-3 预测结果展示
- [x] 置信度与概率分布可视化
- [x] 历史识别记录

---

## 7. 实验三：交互式手写识别系统

### 7.1 目标

实现交互式手写识别系统，支持实时手写输入和结果显示。

### 7.2 功能特性

- [x] 实时手写识别
- [x] 概率分布条形图
- [x] 历史记录展示
- [x] 清空画板功能

---

## 8. GitHub 使用记录

### 8.1 提交历史

| 日期 | 提交说明 | 主要更改 |
|------|----------|----------|
| 2026-05-02 | 初始项目：基础DNN模型训练代码 | 添加dnn_mnist.py |
| 2026-05-02 | 添加CNN模型和数据增强 | 更新dnn_mnist.py，添加增强功能 |
| 2026-05-02 | 优化模型：AUC提升至0.99514 | 更新模型结构和训练参数 |
| 2026-05-02 | 重新训练模型：修复架构不一致问题 | 重新训练，修复app.py模型结构 |
| 2026-05-02 | 更新Kaggle Score为0.99625 | 更新README和实验报告 |
| 2026-05-02 | 整理仓库结构，符合实验管理规范 | 创建code/report/results目录结构 |

### 8.2 版本回退说明

如需回退到之前版本，可使用以下命令：

```bash
# 查看提交历史
git log --oneline

# 回退到指定版本
git checkout <commit-hash>

# 或创建新分支基于旧版本
git checkout -b old-version <commit-hash>
```

---

## 9. 实验总结

本次实验完成了基于 CNN 的手写数字识别模型训练、超参数调优、Kaggle 提交、Web 应用封装和交互式手写识别系统搭建。实验一中，通过 4 组超参数对比可以看出，数据增强和 Early Stopping 对提升泛化能力有明显帮助；最终 CNN 模型结合 AdamW、OneCycleLR、BatchNorm、Dropout 和 TTA，在 Kaggle 上获得 **0.99625** 的成绩。

---

## 10. 如何运行

### 10.1 安装依赖

```bash
pip install -r code/requirements.txt
```

### 10.2 本地运行Web应用

```bash
cd code
python app.py
```

然后访问 http://localhost:7870

### 10.3 模型训练

```bash
cd code
python dnn_mnist.py
```

---

## 11. 注意事项

- 模型文件 `best_model.pth` 由于体积较大，需要单独上传
- 训练数据 `train.csv`、`test.csv` 需要从 Kaggle 下载
- 免费云平台（如 Render）有休眠限制，首次访问可能需要等待启动