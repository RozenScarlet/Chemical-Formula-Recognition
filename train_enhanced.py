"""
使用增强配置训练TransformerCNN模型的脚本
包含：
1. 强化数据增强（概率80%，10种增强技术）
2. 提高学习率（1e-4）
3. 增强正则化：dropout=0.5, weight_decay=5e-4
4. 梯度裁剪=1.0
5. 语义权重=0.002
6. 特征噪声=0.02
"""

import subprocess
import sys
import os

def run_training():
    """运行增强训练"""
    
    # 训练命令
    cmd = [
        sys.executable, "train_attention_models.py",
        "--data-dir", "dataset",
        "--rnn-type", "gru",  # 使用GRU可能泛化更好
        "--epochs", "1000",
        "--batch-size", "64",  # 增加批大小
        "--lr", "1e-4",  # 提高学习率
        "--dropout", "0.5",  # 增加dropout
        "--weight-decay", "5e-4",  # 增加权重衰减
        "--grad-clip", "1.0",  # 梯度裁剪
        "--semantic-weight", "0.002",  # 增加语义权重
        "--feature-noise", "0.02",  # 特征噪声
        "--patience", "40",  # 稍微减少早停耐心值
        "--save-dir", "checkpoints/enhanced_training",
        "--fast-mode"
    ]
    
    print("开始增强训练配置...")
    print(f"执行命令: {' '.join(cmd)}")
    
    # 创建保存目录
    os.makedirs("checkpoints/enhanced_training", exist_ok=True)
    
    # 运行训练
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__), 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        return False
    

if __name__ == "__main__":
    print("=" * 60)
    print("增强训练配置")
    print("=" * 60)
    print("主要改进:")
    print("- 强化数据增强: 80%概率，10种增强技术")
    print("- 提高学习率: 5e-5 → 1e-4")
    print("- 增强正则化: dropout 0.3→0.5, weight_decay 1e-4→5e-4")
    print("- 语义权重: 0.0001 → 0.002")
    print("- 特征噪声: 0.02")
    print("- 使用BiGRU替代BiLSTM")
    print("=" * 60)
    
    success = run_training()
    
    if success:
        print("\n✓ 训练完成！")
        print("请检查 checkpoints/enhanced_training/ 目录下的结果")
    else:
        print("\n✗ 训练过程中出现问题")