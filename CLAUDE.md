# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**重要提示：请使用中文与用户交流**

## Repository Overview

这是一个基于深度学习的化学方程式OCR识别系统，能够从图像中准确识别和提取化学方程式文本。系统支持多种先进的OCR架构，并针对化学方程式的特性（如上下标、特殊符号）进行了优化。

## Development Commands

### Environment Setup
```bash
cd "C:\Users\Administrator\Downloads\化学"
conda create -n cer python=3.9
conda activate cer
pip install -r requirements.txt
```

### Training Commands

**TransformerCNN训练 (推荐的主力模型):**
```bash
# BiLSTM版本 (默认50轮早停)
python train_attention_models.py --data-dir dataset --rnn-type lstm --epochs 1000 --batch-size 64 --fast-mode

# BiGRU版本 (默认50轮早停)
python train_attention_models.py --data-dir dataset --rnn-type gru --epochs 1000 --batch-size 64 --fast-mode

# 自定义早停耐心值
python train_attention_models.py --data-dir dataset --rnn-type lstm --epochs 1000 --patience 30 --fast-mode
```

**其他模型训练:**
```bash
# CRNN
python src/train.py --model-type CRNN --data-dir dataset --epochs 50

# LCRNN
python src/train.py --model-type LCRNN --data-dir dataset --epochs 50

# MSF_LCRNN
python src/train.py --model-type MSF_LCRNN --data-dir dataset --epochs 50
```

**批量训练所有模型:**
```bash
python train_all.py --data-dir dataset --epochs 50
```

### Testing Commands

**TransformerCNN预测:**
```bash
# 单张图片
python src/predict_transformer_cnn.py --model-path checkpoints/transformer_cnn/best_model.pth --input dataset/test_images/1881.jpg --visualize

# 批量预测
python src/predict_transformer_cnn.py --model-path checkpoints/transformer_cnn/best_model.pth --input dataset/test_images/ --output_file results/predictions.json
```

**其他模型预测:**
```bash
python src/predict.py --model-path checkpoints/crnn/best_model.pth --image-path dataset/test_images/1881.jpg
```

## Architecture Overview

### Core Components

**支持的模型架构:**
- **TransformerCNN**: 主力模型，结合CNN和Transformer，支持BiLSTM/BiGRU
- **CRNN**: 经典卷积循环神经网络
- **LCRNN**: 轻量级CRNN，基于MobileNetV3
- **MSF_LCRNN**: 多尺度特征融合的LCRNN，适合处理特殊符号

**关键技术特性:**
- 混合精度训练（AMP）
- 模型编译优化（torch.compile）
- 梯度累积
- CTC损失函数
- 自动批量大小调整
- 多种优化器和学习率调度器支持

**创新点 - 语义指导模块:**
- 基于SVTRv2论文思想
- 训练时使用文本语义信息指导视觉模型
- 推理时完全移除，实现零成本性能提升

### Key Features

**数据处理:**
- 自动读取`dataset/labels.txt`构建数据集
- 动态生成字符集
- 支持数据增强
- 训练/验证自动划分

**模型保存机制:**
- 最佳AP模型：`{model_type}_best_ap.pth`
- 阶段性检查点：`{model_type}_{epoch}.pth`
- 最终模型：`{model_type}_final.pth`
- 训练指标可视化：`{model_type}_metrics.png`
- 详细指标数据：`{model_type}_metrics.json`

**性能优化:**
- GPU加速训练
- 支持TF32计算
- 固定内存优化
- 多线程数据加载
- 自动显存管理

## File Structure

```
化学/
├── src/                           # 源代码
│   ├── crnn.py                   # CRNN模型定义
│   ├── lcrnn.py                  # LCRNN模型定义
│   ├── msf_lcrnn.py              # MSF_LCRNN模型定义
│   ├── transformer_cnn_net.py    # TransformerCNN模型定义
│   ├── train.py                  # 通用训练脚本
│   ├── train_attention_models.py # TransformerCNN训练脚本
│   ├── predict.py                # 通用预测脚本
│   ├── predict_transformer_cnn.py # TransformerCNN预测脚本
│   ├── dataset.py                # 数据集类定义
│   └── utils.py                  # 工具函数（CTCLabelConverter等）
├── dataset/                       # 数据集目录
│   ├── images/                   # 训练图像（支持.jpg, .png）
│   ├── labels.txt                # 图像-标签对应关系
│   └── test_images/              # 测试图像
├── checkpoints/                   # 模型保存目录
├── train_all.py                   # 批量训练脚本
└── gaussian_blur.py               # 数据增强工具
```

## Model Storage

**模型文件命名规范:**
- 最佳模型：`checkpoints/{model_type}/{model_type}_best_ap.pth`
- 阶段检查点：`checkpoints/{model_type}/{model_type}_{epoch}.pth`
- 最终模型：`checkpoints/{model_type}/{model_type}_final.pth`

**辅助文件:**
- 训练指标图表：`checkpoints/{model_type}/{model_type}_metrics.png`
- 训练指标数据：`checkpoints/{model_type}/{model_type}_metrics.json`
- 配置文件：`checkpoints/{model_type}/config.json`

## Technology Stack

- **深度学习框架**: PyTorch 2.7.1
- **计算机视觉**: torchvision, OpenCV 4.11.0
- **数据处理**: NumPy 2.3.1, PIL 11.3.0
- **可视化**: matplotlib 3.10.3
- **系统监控**: psutil 7.0.0
- **进度显示**: tqdm 4.67.1

## Important Notes

- **请始终使用中文与用户交流**
- 项目使用PyTorch作为深度学习框架
- 代码第一行不要加`#!/usr/bin/env python3`（Windows平台）
- 不要一次性创建多个文件来实现同一个功能
- 测试文件测试完后应删除
- 训练需要CUDA支持，建议使用GPU加速
- TransformerCNN是主力模型，提供最佳识别精度
- 批量训练脚本会自动根据硬件配置调整参数
- 中文字符显示问题已在代码中处理，自动选择系统字体