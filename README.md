# 机器学习实验：基于CNN的手写数字识别

## 1. 学生信息

- **姓名**：谢洁
- **学号**：112305010107
- **班级**：数据1231

## 项目在线链接

- GitHub 仓库地址：https://github.com/xj0625/112305010107xiejie
- GitHub README / 实验报告：https://github.com/xj0625/112305010107xiejie/blob/main/README.md
- 在线演示：https://huggingface.co/spaces/xj0625/112305010107xiejie

本实验基于 Kaggle Digit Recognizer / MNIST 手写数字识别任务，使用卷积神经网络（CNN）完成手写数字分类。实验主要包括模型训练、超参数调优、Kaggle 提交结果生成，以及后续 Web 应用部署准备。
本次最终提交文件为 sample_submission.csv，Kaggle 提交记录显示得分为 **0.99625**，达到实验要求的 0.98+ 目标。

## 3. 实验环境

| 项目 | 配置 |
|------|------|
| 操作系统 | Windows |
| Python | Python 3.12 |
| 深度学习框架 | PyTorch 2.4.0 + CUDA 12.6 |
| GPU | NVIDIA GeForce RTX |
| 数据集 | Kaggle Digit Recognizer：train.csv、test.csv、sample_submission.csv |
| 主要依赖 | torch、torchvision、numpy、Pillow、gradio、matplotlib |

## 实验一：模型训练与超参数调优（必做）

### 1.1 实验目标

使用 CNN 对 28×28 灰度手写数字图像进行 0-9 十分类识别。通过对比不同优化器、学习率、Batch Size、数据增强和 Early Stopping 设置，分析超参数对模型收敛速度和泛化能力的影响，并生成 Kaggle 可提交的预测结果文件。

### 1.2 模型结构

对比实验中使用基础 CNN 结构：

```
输入(1×28×28)
→ Conv2d(1, 32, 3×3) + ReLU + MaxPool
→ Conv2d(32, 64, 3×3) + ReLU + MaxPool
→ Flatten
→ Linear(64×7×7, 128) + ReLU + Dropout
→ Linear(128, 10)
→ 输出10类数字
```

最终提交模型在基础 CNN 上进行了增强，使用多层卷积块：

```
ConvBlock(1→48)
→ MaxPool
→ ConvBlock(48→96)
→ MaxPool
→ ConvBlock(96→192)
→ Conv2d(192→256)
→ AdaptiveAvgPool2d
→ Dropout
→ Linear(256→10)
```

其中 ConvBlock 由 Conv2d + BatchNorm2d + SiLU + Conv2d + BatchNorm2d + SiLU + Dropout2d 组成。

### 1.3 超参数对比实验

本实验完成了 4 组对比实验。训练集按标签分层划分训练集和验证集，验证集比例为 10%。由于 Kaggle 测试集没有公开标签，因此表格中的 Test Acc 标记为 N/A，模型性能主要通过验证集准确率和 Kaggle 提交分数评估。

| 实验编号 | 优化器 | 学习率 | Batch Size | 数据增强 | Early Stopping |
|----------|--------|--------|------------|----------|----------------|
| Exp1 | SGD | 0.01 | 64 | 否 | 否 |
| Exp2 | Adam | 0.001 | 64 | 否 | 否 |
| Exp3 | Adam | 0.001 | 128 | 否 | 是 |
| Exp4 | Adam | 0.001 | 64 | 是 | 是 |

对比实验结果如下：

| 实验编号 | Train Acc | Val Acc | 最低 Loss | 收敛 Epoch |
|----------|-----------|---------|-----------|------------|
| Exp1 | 99.51% | 99.21% | 0.0268 | 28 |
| Exp2 | 99.43% | 99.19% | 0.0356 | 20 |
| Exp3 | 98.43% | 99.26% | 0.0258 | 9 |
| Exp4 | 98.09% | 99.38% | 0.0218 | 12 |

从结果可以看出，加入数据增强和 Early Stopping 的 Exp4 获得了最低验证集 Loss，验证集准确率也最高，说明数据增强对模型泛化能力有明显帮助。

### 1.4 最终提交模型

最终提交 Kaggle 时使用的模型不是单个基础 CNN，而是进一步增强后的 CNN。

| 配置项 | 我的设置 |
|--------|---------|
| 优化器 | AdamW |
| 学习率 | 0.003 |
| Batch Size | 512 |
| 训练 Epoch 数 | 30 (early stopping实际约20轮) |
| 是否使用数据增强 | 是 |
| 数据增强方式 | 随机平移(±3像素)、随机旋转(±12°)、随机缩放(0.9-1.1) |
| 是否使用 Early Stopping | 是 (patience=8) |
| 是否使用学习率调度器 | 是 (OneCycleLR, pct_start=0.3) |
| 其他调整 | 增强CNN(48→96→192→256)、SiLU激活函数、AdaptiveAvgPool2d、TTA测试时增强 |
| **Kaggle Score** | **0.99625** |

关键优化器代码如下：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.003,
    epochs=30,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos'
)
```

### 1.5 Loss 曲线

训练过程中的 Loss 曲线如下：

![Loss Curve](loss_curve.png)

![Accuracy Curve](acc_curve.png)

### 1.6 分析问题

**Q1：Adam 和 SGD 的收敛速度有何差异？从实验结果中你观察到了什么？**

Adam 的收敛速度整体快于 SGD。Adam 使用自适应学习率优化算法，会根据梯度的一阶矩和二阶矩自适应调整参数更新幅度，因此在训练前期通常更快达到较低 loss。SGD 虽然也能收敛到较好结果，但前期下降速度相对慢一些，需要更多的 epoch 才能达到相似的性能。

**Q2：学习率对训练稳定性有什么影响？**

学习率决定了每次参数更新的步长。学习率过大时，模型可能在最优点附近震荡，导致验证 loss 波动；学习率过小时，训练会变慢，可能需要更多 epoch 才能收敛。本实验最终模型使用 OneCycleLR 学习率调度器，让学习率先升后降，在前期加快探索，后期降低步长，有助于稳定收敛并获得更好的泛化效果。

**Q3：Batch Size 对模型泛化能力有什么影响？**

Batch Size 会影响梯度估计的噪声。较小 Batch Size 的梯度噪声更大，有时能帮助模型跳出局部区域，提升泛化；较大 Batch Size 训练更稳定，但可能泛化略弱。本实验中 Batch Size=512 时取得了较好的效果。

**Q4：Early Stopping 是否有效防止了过拟合？**

Early Stopping 能在验证集 loss 不再改善时提前停止训练，减少继续拟合训练集噪声的风险。本实验设置 patience=8，当验证准确率连续 8 个 epoch 没有提升时自动停止训练，有效避免了过拟合。

**Q5：数据增强是否提升了模型的泛化能力？为什么？**

数据增强提升了模型泛化能力。本实验使用随机平移、随机旋转和随机缩放三种数据增强方式，最低验证 loss 明显降低。手写数字在真实场景中会存在轻微倾斜、位置偏移和大小变化，数据增强相当于人为扩充了训练样本分布，使模型学习到更稳健的形状特征，而不是只记住训练集中固定位置和角度的数字。

## 实验二：模型封装与 Web 部署（必做）

### 2.1 实验目标

将实验一训练好的 CNN 模型封装为 Web 应用，使用户可以上传手写数字图片，系统自动完成图像预处理、模型推理并返回预测数字。

### 2.2 技术方案

使用 Gradio 实现 Web 页面，基本流程如下：

```
用户上传图片
→ 转为灰度图
→ 调整大小为 28×28
→ 归一化
→ 输入 CNN 模型
→ 输出预测类别和置信度
```

### 2.3 项目结构

```
project/
├── app.py              # Gradio Web 应用入口（含手写画板功能）
├── best_model.pth      # 训练好的 CNN 模型权重
├── dnn_mnist.py        # 模型训练主脚本
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```

其中 app.py 提供 3 个核心功能：

- 手写画板识别
- 上传图片识别
- Top-3 预测结果与概率分布

### 2.4 部署平台

本项目选择 HuggingFace Spaces 作为公网部署平台。

| 平台 | 部署结果 |
|------|----------|
| GitHub | 代码、模型权重、提交文件和 README 实验报告已上传到仓库 https://github.com/xj0625/112305010107xiejie |
| HuggingFace Spaces | 已创建并部署 Gradio 应用 |

### 2.5 提交信息

| 提交项 | 内容 |
|--------|------|
| GitHub 仓库地址 | https://github.com/xj0625/112305010107xiejie |
| GitHub README / 实验报告 | https://github.com/xj0625/112305010107xiejie/blob/main/README.md |
| HuggingFace Spaces 在线访问链接 | https://huggingface.co/spaces/xj0625/112305010107xiejie |

Gradio 版 Web 应用支持上传图片识别和网页手写板识别；页面可以显示预测数字、置信度、Top-3 结果、0-9 概率分布条形图和连续识别历史。

## 实验三：交互式手写识别系统（选做，加分）

### 3.1 实验目标

在实验二的上传图片识别功能基础上，进一步增加网页手写板，使用户可以直接在网页中手写数字并进行识别。

### 3.2 功能设计

| 功能 | 实现情况 |
|------|----------|
| 手写输入 | 已使用 Gradio Sketchpad 实现在线手写板 |
| 实时识别 | 已通过 Gradio 事件将画板图像送入 CNN 模型预测 |
| 连续使用 | 已实现清空画板和连续识别历史 |
| Top-3 结果 | 已显示概率最高的 3 个类别和置信度 |
| 概率分布 | 已显示 0-9 各类别概率条形图 |

### 3.3 当前状态

实验三已在 Gradio 应用中完整实现。页面包含"手写画板"标签页，用户可以直接在网页画板中书写数字，点击按钮后系统会完成图像预处理、CNN 推理，并输出预测数字、置信度、Top-3 结果、概率分布和连续识别历史。

### 3.4 提交信息

| 提交项 | 内容 |
|--------|------|
| 在线访问链接 | https://huggingface.co/spaces/xj0625/112305010107xiejie |
| 实现了哪些加分项 | 已实现手写输入、Top-3 预测、概率分布显示、连续识别历史 |

## 4. 实验总结

本次实验完成了基于 CNN 的手写数字识别模型训练、超参数调优、Kaggle 提交、Web 应用封装和交互式手写识别系统搭建。实验一中，通过 4 组超参数对比可以看出，数据增强和 Early Stopping 对提升泛化能力有明显帮助；最终 CNN 模型结合 AdamW、OneCycleLR、BatchNorm、Dropout 和 TTA，在 Kaggle 上获得 **0.99625** 的成绩。

实验二和实验三已经完成公网部署，不再停留在本地测试阶段。线上系统地址为 https://huggingface.co/spaces/xj0625/112305010107xiejie，GitHub 仓库地址为 https://github.com/xj0625/112305010107xiejie。系统支持上传图片识别和网页手写板识别，能够显示预测数字、置信度、Top-3 结果、0-9 概率分布条形图和连续识别历史，满足 Web 展示与交互式识别的实验要求。