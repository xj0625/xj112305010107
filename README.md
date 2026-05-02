# 手写数字识别 Web 应用

基于 PyTorch 和 Gradio 构建的手写数字识别系统。

## 👤 项目信息

| 项目 | 内容 |
|------|------|
| **姓名** | 谢洁 |
| **学号** | 112305010107 |
| **班级** | 数据1231 |
| **实验名称** | 基于CNN的手写数字识别 |

## 🚀 功能

- 🎨 **手写画板**：支持在网页上直接手写数字
- 📤 **图片上传**：支持上传手写数字图片进行识别
- 📊 **概率分布**：显示数字概率分布图
- 🏆 **Top-3 预测**：显示排名前三的预测结果
- 📜 **历史记录**：保存最近5次识别记录

## 🛠️ 技术栈

- Python 3.8+
- PyTorch 2.4.0
- Gradio 4.44.1
- torchvision 0.19.0
- matplotlib 3.9.2

## 📦 安装

```bash
pip install -r requirements.txt
```

## 🚀 运行

```bash
python app.py
```

应用将在 http://localhost:7860 启动。

## 📁 项目结构

```
project/
├── app.py              # Web 应用入口（含手写画板功能）
├── best_model.pth      # 训练好的 CNN 模型权重
├── dnn_mnist.py        # 模型训练主脚本
├── exp_comparison.py   # 超参数对比实验脚本
├── requirements.txt    # 依赖列表
├── sample_submission.csv  # Kaggle 提交文件
└── README.md           # 项目说明
```

## 📊 模型性能

| 指标 | 数值 |
|------|------|
| **Kaggle Score** | **0.99514** |
| 验证集准确率 | 99.71% |
| 训练集准确率 | 99.78% |

## 🔬 实验配置

| 配置项 | 设置 |
|--------|------|
| 优化器 | AdamW |
| 学习率 | 0.003 |
| Batch Size | 128 |
| 数据增强 | 随机平移 |
| Early Stopping | ✅ |
| 学习率调度器 | ReduceLROnPlateau |

## 📝 使用说明

1. 在手写画板上书写数字（0-9）
2. 点击识别按钮进行预测
3. 查看预测结果、置信度和概率分布
4. 点击清空画板重新输入

### 最佳实践

- 使用较粗的笔触书写
- 数字尽量居中
- 避免超出画板边界

## 🌐 在线部署

- **GitHub 仓库**: https://github.com/xj0625/112305010107xiejie
- **在线演示**: https://huggingface.co/spaces/xj0625/mnist-digit-recognition
- **实验报告**: https://github.com/xj0625/112305010107xiejie/blob/main/CNN手写数字识别实验模板.md

## 📄 许可证

MIT License