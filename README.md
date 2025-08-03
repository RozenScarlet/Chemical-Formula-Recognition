# 化学方程式识别 (Chemical Equation Recognition)

本项目是一个基于深度学习的化学方程式识别系统，能够从图像中准确识别和提取化学方程式文本。系统支持多种先进的OCR架构，并针对化学方程式的特性（如上下标、特殊符号）进行了优化。

## 主要功能

- **多种模型支持**: 内置了多种主流和创新的OCR模型：
  - **`CRNN`**: 经典的卷积循环神经网络。
  - **`LCRNN`**: 轻量级的CRNN，速度更快。
  - **`MSF_LCRNN`**: 融合了多尺度特征的LCRNN，对大小不一的字符更鲁棒。
  - **`TransformerCNNNet`**: **(主力模型)** 结合了CNN的局部特征提取能力和Transformer的全局上下文理解能力，并支持动态配置使用 `BiLSTM` 或 `BiGRU` 作为序列处理层。

- **高性能训练**: 训练脚本支持混合精度（AMP）、模型编译（`torch.compile`）、梯度累积和多种优化策略，以实现高效训练。

- **便捷的批量训练**: 提供 `train_all.py` 脚本，可一键启动对所有或指定模型的训练。

## 项目结构

```
.
├── checkpoints/             # 模型检查点保存目录
├── dataset/                 # 数据集目录
│   ├── images/              # 存放所有训练和验证图像 (.jpg, .png)
│   ├── labels.txt           # 图像文件名与对应化学方程式的列表
│   └── test_images/         # 用于测试的图像
├── src/                     # 源代码
│   ├── crnn.py, lcrnn.py, msf_lcrnn.py # 各模型定义
│   ├── transformer_cnn_net.py # TransformerCNNNet 模型定义
│   ├── dataset.py           # PyTorch 数据集类
│   ├── utils.py             # 工具函数 (如 CTCLabelConverter)
│   ├── train.py             # CRNN, LCRNN, MSF_LCRNN 训练脚本
│   ├── train_attention_models.py # TransformerCNNNet 训练脚本
│   ├── predict.py           # 预测脚本
│   └── ...
├── train_all.py             # 一键训练所有模型的脚本
├── requirements.txt         # 项目依赖
└── README.md                # 本文档
```

## 环境配置

建议使用 `conda` 或 `venv` 创建独立的Python环境。

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd gplan

# 2. (推荐) 创建并激活 Conda 环境
conda create -n cer python=3.9
conda activate cer

# 3. 安装依赖
pip install -r requirements.txt
```

## 数据准备

1.  将所有化学方程式图像（如 `1.jpg`, `2.png`）放入 `dataset/images/` 目录下。
2.  在 `dataset/` 目录下创建一个 `labels.txt` 文件。
3.  `labels.txt` 文件每行包含一个 `图像文件名` 和它对应的 `真实标签`，用空格隔开。示例如下：

```
# dataset/labels.txt

1.jpg 2H2+O2=2H2O
2.jpg CaCO3+2HCl=CaCl2+H2O+CO2^
...
```

训练脚本会自动读取此文件，并动态构建字符集。

## 如何使用

### 训练模型

#### 1. 训练 TransformerCNNNet (推荐)

这是项目的主力模型，可以通过 `--rnn-type` 参数选择使用 `lstm` 或 `gru`。

- **训练BiLSTM版本**:
  ```bash
  python train_attention_models.py \
      --data-dir dataset \
      --rnn-type lstm \
      --epochs 100 \
      --batch-size 64 \
      --fast-mode
  ```

- **训练BiGRU版本**:
  ```bash
  python train_attention_models.py \
      --data-dir dataset \
      --rnn-type gru \
      --epochs 100 \
      --batch-size 64 \
      --fast-mode
  ```

#### 2. 训练其他模型

使用 `src/train.py` 脚本并指定 `--model-type`。

- **训练CRNN**:
  ```bash
  python src/train.py --model-type CRNN --data-dir dataset --epochs 50
  ```

- **训练LCRNN**:
  ```bash
  python src/train.py --model-type LCRNN --data-dir dataset --epochs 50
  ```

#### 3. 批量训练 (使用 train_all.py)

此脚本可以方便地按顺序训练多个模型。

```bash
# 训练所有模型 (CRNN, LCRNN, MSF_LCRNN, TransformerCNN)
python train_all.py --data-dir dataset --epochs 50

# 只训练指定模型
python train_all.py --models CRNN TransformerCNN --data-dir dataset --epochs 50
```

### 执行预测

#### 1. 使用 TransformerCNNNet 模型预测

- **预测单张图片**:
  ```bash
  python src/predict_transformer_cnn.py \
      --model-path checkpoints/transformer_cnn/best_model.pth \
      --input dataset/test_images/1881.jpg \
      --visualize
  ```

- **批量预测整个目录**:
  ```bash
  python src/predict_transformer_cnn.py \
      --model-path checkpoints/transformer_cnn/best_model.pth \
      --input dataset/test_images/ \
      --output_file results/predictions.json
  ```

#### 2. 使用其他模型预测

- **预测单张图片**:
  ```bash
  python src/predict.py \
      --model-path checkpoints/crnn/best_model.pth \
      --image-path dataset/test_images/1881.jpg
  ```

## 模型架构详解

### 1. CRNN (卷积循环神经网络)
基础的OCR模型架构，结合了CNN的特征提取能力和RNN的序列建模能力。
- **特征提取**: 使用VGG风格的卷积网络
- **序列建模**: 双向LSTM
- **解码方式**: CTC (Connectionist Temporal Classification)
- **特点**: 结构简单，易于训练，适合简单化学方程式识别

### 2. LCRNN (轻量级卷积循环神经网络)
- **特征提取**: 基于MobileNetV3的轻量级特征提取器
- **序列建模**: 双向GRU
- **特点**: 参数量大幅减少(约85%)，推理速度更快，保持较好的识别能力

### 3. MSF_LCRNN (多尺度特征融合轻量级卷积循环神经网络)
- **特征提取**: MobileNetV3基础上增加多尺度特征提取模块
- **特征融合**: 并行的不同卷积核尺寸(1x1, 3x3, 5x5, 7x7)提取并融合不同尺度特征
- **注意力机制**: SE (Squeeze-and-Excitation) 通道注意力
- **序列建模**: 双向GRU
- **特点**: 特别适合处理不同大小的字符和特殊符号，如上下标

### 4. TransformerCNN (Transformer卷积神经网络)
- **特征提取**: 轻量级MobileNetV3+多尺度特征融合
- **位置编码**: 标准的正弦余弦位置编码
- **自注意力**: 多头自注意力机制的Transformer编码器
- **序列处理**: 双向LSTM增强序列依赖性建模
- **特点**: 强大的全局建模能力，特别适合长化学方程式和复杂结构

## 参数设置详解

### 通用训练参数

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `--data-dir` | 数据集目录 | `dataset` | 指定训练和测试数据的位置 |
| `--checkpoint-dir` | 检查点保存目录 | `checkpoints` | 指定模型保存位置 |
| `--epochs` | 训练轮数 | `100` | 控制训练时间和模型收敛程度 |
| `--batch-size` | 批次大小 | `64` | 控制每次处理的样本数量，影响训练速度和内存使用 |
| `--lr` | 学习率 | `0.001` | 控制模型参数更新步长，影响收敛速度和稳定性 |
| `--optimizer` | 优化器类型 | `adamw` | 选择不同的优化算法，影响收敛特性 |
| `--scheduler` | 学习率调度器 | `cosine` | 控制学习率变化方式，影响训练稳定性和最终性能 |
| `--seed` | 随机种子 | `42` | 确保实验可重复性 |
| `--val-split` | 验证集比例 | `0.1` | 控制训练/验证数据划分比例 |
| `--gpu` | GPU编号 | `0` | 指定使用的GPU设备，-1表示使用CPU |
| `--num-workers` | 数据加载线程数 | `4` | 控制数据加载并行度，影响数据准备速度 |
| `--generate-labels` | 生成主标签文件 | `False` | 从单个标签文件生成总的`labels.txt` |
| `--create-classes-file` | 生成字符集文件 | `False` | 从`labels.txt`创建`classes.txt` |
| `--save-interval` | 模型保存间隔 | `5` | 每隔多少个epoch保存一次检查点 |
| `--gradient-accumulation` | 梯度累积步数 | `1` | 允许在小批量下模拟大批量训练效果，减少显存需求 |

### 性能优化参数

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `--amp` | 自动混合精度训练 | `True` | 使用FP16和FP32混合精度，加速训练并减少显存占用 |
| `--auto-batch-size` | 自动寻找最优批量大小 | `False` | 根据显存自动调整批量大小，避免OOM错误 |
| `--min-batch-size` | 自动批量大小的最小值 | `16` | 设置批量大小下限，保证足够的随机性 |
| `--max-batch-size` | 自动批量大小的最大值 | `128` | 设置批量大小上限，避免批量过大导致收敛问题 |
| `--pin-memory` | 使用固定内存 | `True` | 提高CPU到GPU的数据传输效率 |
| `--use-tf32` | 启用TF32计算 | `True` | 在支持TF32的GPU上使用TF32精度提高性能 |
| `--compile-model` | 编译模型 | `False` | 使用PyTorch 2.0+的编译功能加速模型，但初次编译较慢 |
| `--memory-fraction` | GPU显存使用比例 | `0.95` | 控制GPU显存使用上限，预留部分显存给系统 |

### TransformerCNN专用参数

| 参数 | 说明 | 默认值 | 影响 |
|------|------|--------|------|
| `--num-heads` | 注意力头数量 | `8` | 控制自注意力的并行度，影响捕捉多种特征关系的能力 |
| `--num-encoder-layers` | Transformer编码器层数 | `4` | 控制Transformer的深度，影响特征提取复杂度 |
| `--fast-mode` | 启用快速模式 | `True` | 加速训练过程，在保持性能的前提下优化内存使用 |
| `--ultra-fast` | 启用超快速模式 | `False` | 极致轻量化模型，大幅提升训练和推理速度 |
| `--use-augmentation` | 使用数据增强 | `True` | 启用数据增强提高模型泛化能力 |
| `--use-msf` | 使用多尺度特征 | `True` | 控制是否使用多尺度特征模块 |
| `--use-bilstm` | 使用双向LSTM | `True` | 控制是否使用BiLSTM处理序列 |

## 文件存储位置

### 模型检查点

模型训练过程中和结束后，会保存以下文件：

1. **最佳模型**：保存**训练AP值**最高的模型权重
   - CRNN: `checkpoints/crnn/crnn_best_ap.pth`
   - LCRNN: `checkpoints/lcrnn/lcrnn_best_ap.pth`
   - MSF_LCRNN: `checkpoints/msf_lcrnn/msf_lcrnn_best_ap.pth`
   - TransformerCNN: `checkpoints/transformer_cnn/transformer_cnn_best_ap.pth`

2. **阶段性检查点**：每隔 `save-interval` 个epoch保存一次
   - 格式: `checkpoints/{model_type}/{model_type}_{epoch}.pth`
   - 例如: `checkpoints/crnn/crnn_50.pth`

3. **最终模型**：训练结束后保存的最后一个模型
   - 格式: `checkpoints/{model_type}/{model_type}_final.pth`

### 训练指标记录

1. **指标图表**：训练过程中损失、AP、APc和字符准确率的可视化图表
   - 格式: `checkpoints/{model_type}/{model_type}_metrics.png`
   - 例如: `checkpoints/crnn/crnn_metrics.png`

2. **指标数据**：详细的每个epoch的指标数据，以JSON格式保存
   - 格式: `checkpoints/{model_type}/{model_type}_metrics.json`
   - 例如: `checkpoints/transformer_cnn/transformer_cnn_metrics.json`

### 预测结果

预测结果默认保存在 `results/` 目录下：
- 单张图像预测：直接在控制台输出
- 批量预测：保存为JSON文件，如 `results/predictions.json`

## 模型选择指南

根据不同场景选择适合的模型：

| 模型 | 参数量 | 速度 | 优势 | 适用场景 |
|------|--------|------|------|----------|
| CRNN | 中等 | 快 | 简单结构，稳定性好 | 简单方程式，资源受限环境 |
| LCRNN | 少 | 很快 | 轻量级，推理速度快 | 移动设备，实时应用 |
| MSF-LCRNN | 中等 | 中等 | 良好的符号识别能力 | 含特殊符号的方程式 |
| TransformerCNN | 多 | 慢 | 最佳识别精度，全局理解能力强 | 复杂长方程式，高精度要求场景 |
| TransformerCNN(ultra-fast) | 少 | 快 | 平衡速度和准确性 | 中等复杂度场景，有速度要求 |

## 性能优化建议

1. **GPU训练**：强烈建议使用GPU加速训练，可显著提高训练速度
2. **混合精度训练**：启用`--amp`参数使用混合精度训练，可减少显存使用并提高速度
3. **自动批量大小**：使用`--auto-batch-size`自动寻找最优批量大小
4. **数据加载优化**：适当调整`--num-workers`和`--prefetch-factor`优化数据加载
5. **模型选择**：资源受限环境下考虑使用LCRNN或TransformerCNN的ultra-fast模式

## 常见问题解答

**Q: 训练中出现"CUDA out of memory"错误怎么办?**  
A: 减小批量大小(--batch-size)，启用混合精度训练(--amp)，或使用梯度累积(--gradient-accumulation 2或更大)。

**Q: 如何提高模型对特殊符号的识别能力?**  
A: 使用MSF_LCRNN或TransformerCNN模型，它们具有更好的多尺度特征提取能力，更适合识别化学方程式中的特殊符号。

**Q: TransformerCNN模型训练时间过长怎么解决?**  
A: 使用--fast-mode或--ultra-fast参数加速训练，同时启用--amp和--use-tf32参数提高计算效率。

**Q: 生成的classes.txt文件可以手动修改吗?**  
A: 可以，但要确保添加的字符在训练数据中出现。修改后需要重新训练模型。

**Q: 训练时无法保存检查点怎么办?**  
A: 检查磁盘空间是否充足，以及checkpoints目录的写入权限。我们已优化模型保存机制，使用临时文件避免意外中断导致文件损坏。

**Q: 生成的图表中，中文字符显示为方框怎么办？**
A: 这是因为`matplotlib`的默认字体不支持中文。脚本已尝试自动根据您的操作系统（Windows、macOS、Linux）设置相应的中文字体（如`SimHei`、`PingFang SC`、`WenQuanYi Zen Hei`）。如果问题仍然存在，您可能需要手动在系统中安装这些字体，或者修改代码中的字体配置，指定为您系统中已安装的其他中文字体。

## 模型评估

使用以下指标评估模型性能：

1. **Loss (损失)**: 训练和验证过程的CTC损失值
2. **AP (Average Precision)**: 完全正确识别的样本比例
3. **APc (Character Average Precision)**: 字符级别的相似度，考虑部分正确识别
4. **字符准确率**: 正确识别的字符数量与总字符数量之比

## 后续开发计划

1. 添加基于注意力机制的端到端模型
2. 优化对化学方程中平衡关系的理解
3. 增加更多数据增强策略提高模型鲁棒性
4. 提供预训练模型下载功能

## 贡献指南

欢迎通过以下方式贡献：
1. 报告问题和建议
2. 提交改进代码
3. 优化文档和示例
4. 分享训练数据和预训练模型 