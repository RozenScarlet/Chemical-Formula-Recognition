"""
工具函数模块
包含训练、验证和预测的辅助函数
"""

import os
import json
import time
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from PIL import Image
import cv2


# 设置中文字体
def set_chinese_font():
    """设置matplotlib中文字体"""
    try:
        # 尝试使用系统字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 如果设置失败，使用默认字体
        print("警告：无法设置中文字体，使用默认字体")


class AverageMeter:
    """计算和存储平均值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        """保存模型"""
        if self.verbose:
            print(f'验证损失降低 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def calculate_accuracy(preds, labels, label_converter):
    """计算字符级和序列级准确率"""
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # CTC解码
    pred_probs = F.log_softmax(preds, dim=2)
    input_lengths = torch.full((pred_probs.size(0),), pred_probs.size(1), dtype=torch.long)
    
    # 获取最大概率的字符序列
    _, pred_indices = pred_probs.max(2)
    
    # 解码预测结果 - 使用统一的CTC解码
    pred_texts = []
    for i in range(pred_indices.size(0)):
        pred_text = label_converter.ctc_greedy_decode(pred_indices[i])
        pred_texts.append(pred_text)
    
    # 解码真实标签 - 使用统一的CTC解码
    true_texts = []
    for i in range(labels.size(0)):
        true_text = label_converter.ctc_greedy_decode(labels[i])
        true_texts.append(true_text)
    
    # 计算准确率
    char_correct = 0
    char_total = 0
    seq_correct = 0
    
    for pred, true in zip(pred_texts, true_texts):
        if pred == true:
            seq_correct += 1
        
        # 字符级准确率
        for p, t in zip(pred, true):
            if p == t:
                char_correct += 1
        char_total += max(len(pred), len(true))
    
    char_accuracy = char_correct / max(char_total, 1)
    seq_accuracy = seq_correct / len(pred_texts)
    
    return char_accuracy, seq_accuracy


def save_training_curves(metrics, save_path):
    """保存训练曲线"""
    set_chinese_font()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(metrics['train_loss'], label='训练损失', color='blue')
    axes[0, 0].plot(metrics['val_loss'], label='验证损失', color='red')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 字符准确率曲线
    axes[0, 1].plot(metrics['train_char_acc'], label='训练字符准确率', color='blue')
    axes[0, 1].plot(metrics['val_char_acc'], label='验证字符准确率', color='red')
    axes[0, 1].set_title('字符准确率曲线')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Character Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 序列准确率曲线
    axes[1, 0].plot(metrics['train_seq_acc'], label='训练序列准确率', color='blue')
    axes[1, 0].plot(metrics['val_seq_acc'], label='验证序列准确率', color='red')
    axes[1, 0].set_title('序列准确率曲线')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Sequence Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 学习率曲线
    if 'learning_rate' in metrics:
        axes[1, 1].plot(metrics['learning_rate'], label='学习率', color='green')
        axes[1, 1].set_title('学习率变化')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_json(metrics, save_path):
    """保存训练指标为JSON文件"""
    # 转换numpy数组为列表
    json_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            json_metrics[key] = value.tolist()
        elif isinstance(value, list):
            json_metrics[key] = value
        else:
            json_metrics[key] = float(value) if isinstance(value, (int, float)) else str(value)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_metrics, f, indent=2, ensure_ascii=False)


def get_optimizer(model, optimizer_name='Adam', lr=0.001, weight_decay=1e-4):
    """获取优化器"""
    if optimizer_name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name='StepLR', **kwargs):
    """获取学习率调度器"""
    if scheduler_name == 'StepLR':
        return StepLR(optimizer, step_size=kwargs.get('step_size', 10), gamma=kwargs.get('gamma', 0.1))
    elif scheduler_name == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50))
    elif scheduler_name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, mode='min', factor=kwargs.get('factor', 0.5), 
                               patience=kwargs.get('patience', 5))
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}")


def preprocess_image(image_path, target_size=(256, 64)):
    """预处理单张图像用于预测"""
    # 加载图像
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # 调整大小
    image = image.resize(target_size, Image.BILINEAR)
    
    # 转换为tensor
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)  # 添加batch维度
    return tensor


def visualize_prediction(image_path, prediction, confidence=None, save_path=None):
    """可视化预测结果"""
    set_chinese_font()
    
    # 加载原始图像
    image = Image.open(image_path)
    
    plt.figure(figsize=(12, 4))
    plt.imshow(image)
    plt.axis('off')
    
    # 添加预测文本
    title = f"预测结果: {prediction}"
    if confidence is not None:
        title += f" (置信度: {confidence:.3f})"
    
    plt.title(title, fontsize=14, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class ModelSize:
    """计算模型大小和参数量"""
    
    @staticmethod
    def get_model_size(model):
        """获取模型大小（MB）"""
        param_size = 0
        param_sum = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        
        buffer_size = 0
        buffer_sum = 0
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        
        all_size = (param_size + buffer_size) / 1024 / 1024
        
        return {
            'param_count': param_sum,
            'buffer_count': buffer_sum,
            'model_size_mb': all_size,
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024
        }


def setup_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # 测试工具函数
    print("工具函数模块测试")
    
    # 测试字体设置
    set_chinese_font()
    print("中文字体设置完成")
    
    # 测试模型大小计算
    import torch.nn as nn
    test_model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),
        nn.Linear(64, 100)
    )
    
    size_info = ModelSize.get_model_size(test_model)
    print(f"测试模型信息: {size_info}")
    
    print("工具函数模块测试完成")