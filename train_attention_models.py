"""
TransformerCNN模型训练脚本
支持BiLSTM和BiGRU两种变体，包含混合精度训练和模型编译优化
"""

import os
import sys
import argparse
import time
import json
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import psutil
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformer_cnn_net import create_transformer_cnn
from dataset import create_dataset, build_character_set
from utils import (
    AverageMeter, EarlyStopping, calculate_accuracy,
    save_training_curves, save_metrics_json,
    get_optimizer, get_scheduler, ModelSize, setup_seed
)


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, label_converter, epoch, semantic_weight=0.05, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    
    # 计算Scheduled Sampling的teacher forcing比例
    teacher_forcing_ratio = max(0.9 * (0.98 ** epoch), 0.1)
    
    loss_meter = AverageMeter()
    char_acc_meter = AverageMeter()
    seq_acc_meter = AverageMeter()
    semantic_loss_meter = AverageMeter()
    teacher_forcing_meter = AverageMeter()  # 记录teacher forcing使用次数
    
    pbar = tqdm(train_loader, desc=f'训练中 (TF:{teacher_forcing_ratio:.3f})')
    
    for batch_idx, (images, targets, target_lengths, raw_texts) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        
        optimizer.zero_grad()
        
        # Scheduled Sampling: 随机决定是否使用teacher forcing
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
        
        # 混合精度前向传播
        with torch.amp.autocast('cuda'):
            # 计算输出序列长度
            batch_size = images.size(0)
            
            # 前向传播 - 根据Scheduled Sampling决定是否传入标签
            if use_teacher_forcing:
                outputs = model(images, targets, target_lengths.squeeze(-1))
            else:
                outputs = model(images)
            
            if isinstance(outputs, tuple):
                preds, semantic_preds = outputs
                use_semantic_guide = True
            else:
                preds = outputs
                use_semantic_guide = False
            
        # 调整维度用于CTC损失: (T, N, C) - 在autocast外面计算避免Half精度问题
        preds = preds.permute(1, 0, 2)
        
        # 动态计算输出序列长度
        seq_len = preds.size(0)  # T维度
        preds_size = torch.IntTensor([seq_len] * batch_size).to(device)
        
        # 计算CTC损失 - 需要log_softmax
        log_probs = F.log_softmax(preds, dim=2).float()
        
        ctc_loss = criterion(log_probs, targets, preds_size, target_lengths)
        
        # 计算语义辅助损失（如果有且使用了teacher forcing）
        if use_semantic_guide and use_teacher_forcing:
            # 语义预测也需要调整维度和精度，并转换为log_softmax
            semantic_preds = semantic_preds.permute(1, 0, 2)
            semantic_log_probs = F.log_softmax(semantic_preds, dim=2).float()
            semantic_loss = criterion(semantic_log_probs, targets, preds_size, target_lengths)
            # 总损失 = CTC损失 + 权重 * 语义损失
            loss = ctc_loss + semantic_weight * semantic_loss
        else:
            loss = ctc_loss
            semantic_loss = torch.tensor(0.0)
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪，防止梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        # 计算准确率
        preds_for_acc = preds.permute(1, 0, 2)  # 转回 (N, T, C)
        char_acc, seq_acc = calculate_accuracy(preds_for_acc, targets, label_converter)
        
        # 更新指标
        loss_meter.update(loss.item(), batch_size)
        char_acc_meter.update(char_acc, batch_size)
        seq_acc_meter.update(seq_acc, batch_size)
        teacher_forcing_meter.update(float(use_teacher_forcing), 1)
        if use_semantic_guide and use_teacher_forcing:
            semantic_loss_meter.update(semantic_loss.item(), batch_size)
        
        # 更新进度条
        postfix = {
            'Loss': f'{loss_meter.avg:.4f}',
            'CharAcc': f'{char_acc_meter.avg:.4f}',
            'SeqAcc': f'{seq_acc_meter.avg:.4f}',
            'TFRate': f'{teacher_forcing_meter.avg:.3f}'
        }
        if semantic_loss_meter.count > 0:
            postfix['SemLoss'] = f'{semantic_loss_meter.avg:.4f}'
        pbar.set_postfix(postfix)
    
    return loss_meter.avg, char_acc_meter.avg, seq_acc_meter.avg


def validate_epoch(model, val_loader, criterion, device, label_converter):
    """验证一个epoch"""
    model.eval()
    
    loss_meter = AverageMeter()
    char_acc_meter = AverageMeter()
    seq_acc_meter = AverageMeter()
    
    with torch.no_grad():
        for images, targets, target_lengths, raw_texts in tqdm(val_loader, desc='验证中'):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            batch_size = images.size(0)
            
            # 前向传播
            preds = model(images)
            
            # 调整维度用于CTC损失
            preds = preds.permute(1, 0, 2)
            
            # 动态计算输出序列长度
            seq_len = preds.size(0)  # T维度
            preds_size = torch.IntTensor([seq_len] * batch_size).to(device)
            
            # 计算损失 - 需要log_softmax
            log_probs = F.log_softmax(preds, dim=2).float()
            loss = criterion(log_probs, targets, preds_size, target_lengths)
            
            # 计算准确率
            preds_for_acc = preds.permute(1, 0, 2)
            char_acc, seq_acc = calculate_accuracy(preds_for_acc, targets, label_converter)
            
            # 更新指标
            loss_meter.update(loss.item(), batch_size)
            char_acc_meter.update(char_acc, batch_size)
            seq_acc_meter.update(seq_acc, batch_size)
    
    return loss_meter.avg, char_acc_meter.avg, seq_acc_meter.avg


def save_test_predictions(model, val_loader, label_converter, device, save_dir):
    """保存验证集图片和对应的预测标签"""
    model.eval()
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告：无法设置中文字体")
    
    # 创建保存目录
    predictions_dir = os.path.join(save_dir, 'test_predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    prediction_results = []
    
    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths, raw_texts) in enumerate(tqdm(val_loader, desc='保存测试预测')):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            batch_size = images.size(0)
            
            # 前向传播
            preds = model(images)
            
            # CTC解码
            pred_probs = F.log_softmax(preds, dim=2)
            _, pred_indices = pred_probs.max(2)
            
            # 解码预测结果
            for i in range(batch_size):
                # 获取原始图片路径（从数据集获取）
                dataset_idx = batch_idx * val_loader.batch_size + i
                if dataset_idx >= len(val_loader.dataset):
                    break
                    
                # 获取原始图片
                original_image = val_loader.dataset.get_original_image(dataset_idx)
                
                # 解码预测序列 - 使用统一的CTC解码
                pred_text = label_converter.ctc_greedy_decode(pred_indices[i])
                
                # 解码真实标签 - 使用统一的CTC解码
                true_text = label_converter.ctc_greedy_decode(targets[i])
                
                # 保存结果
                prediction_results.append({
                    'image_idx': dataset_idx,
                    'true_text': true_text,
                    'pred_text': pred_text,
                    'match': pred_text == true_text
                })
                
                # 创建可视化图片
                plt.figure(figsize=(12, 4))
                plt.imshow(original_image)
                plt.axis('off')
                
                # 添加预测文本和真实标签
                title = f"真实: {true_text}\n预测: {pred_text}"
                if pred_text == true_text:
                    title += " ✓"
                    plt.title(title, fontsize=12, color='green', pad=20)
                else:
                    title += " ✗"
                    plt.title(title, fontsize=12, color='red', pad=20)
                
                # 保存图片
                save_path = os.path.join(predictions_dir, f'test_{dataset_idx:04d}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                # 限制保存数量，避免生成过多文件
                if len(prediction_results) >= 100:
                    break
            
            if len(prediction_results) >= 100:
                break
    
    # 保存预测结果JSON
    results_path = os.path.join(save_dir, 'test_predictions.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_results, f, indent=2, ensure_ascii=False)
    
    # 统计准确率
    total_samples = len(prediction_results)
    correct_samples = sum(1 for r in prediction_results if r['match'])
    accuracy = correct_samples / total_samples if total_samples > 0 else 0
    
    print(f"测试预测完成！")
    print(f"保存了 {total_samples} 个测试样本")
    print(f"准确预测: {correct_samples}/{total_samples} ({accuracy:.2%})")
    print(f"预测图片保存至: {predictions_dir}")
    print(f"预测结果保存至: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='TransformerCNN训练脚本')
    parser.add_argument('--data-dir', type=str, default='dataset', help='数据集目录')
    parser.add_argument('--rnn-type', type=str, default='gru', choices=['lstm', 'gru'], 
                       help='RNN类型')
    parser.add_argument('--epochs', type=int, default=1000, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率（提高到1e-4）')
    parser.add_argument('--hidden-size', type=int, default=256, help='隐藏层大小')
    parser.add_argument('--num-layers', type=int, default=6, help='Transformer层数')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout概率（提高到0.5）')
    parser.add_argument('--optimizer', type=str, default='AdamW', 
                       choices=[ 'AdamW'], help='优化器')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                       choices=['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'], 
                       help='学习率调度器')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='权重衰减（提高到5e-4）')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值')
    parser.add_argument('--save-dir', type=str, default='checkpoints/test', 
                       help='模型保存目录')
    parser.add_argument('--fast-mode', action='store_true', default=True, help='快速模式（减少数据处理线程）')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的模型路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--semantic-weight', type=float, default=0.002, help='语义指导损失权重（提高到0.002）')
    parser.add_argument('--semantic-decay', type=float, default=0.95, help='语义权重衰减因子')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='梯度裁剪阈值')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='标签平滑系数')
    parser.add_argument('--feature-noise', type=float, default=0.02, help='特征噪声标准差')
    
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA版本: {torch.version.cuda}")
        
    # 启用优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # 启用TF32（如果支持）
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 构建字符集
    character_set = build_character_set(args.data_dir)
    num_classes = len(character_set) + 3  # 添加BLANK、空格和未知字符
    print(f"字符集大小: {num_classes}")
    
    # 创建数据加载器
    num_workers = 2 if args.fast_mode else 4
    train_loader, val_loader, label_converter = create_dataset(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    
    print(f"训练样本: {len(train_loader.dataset)}")
    print(f"验证样本: {len(val_loader.dataset)}")
    
    # 创建模型
    model = create_transformer_cnn(
        num_classes=num_classes,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # 模型信息
    model_info = ModelSize.get_model_size(model)
    print(f"模型参数量: {model_info['param_count']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    
    # 模型编译优化（PyTorch 2.0+）
    # 在Windows上禁用编译以避免Triton依赖问题
    if not args.fast_mode and sys.platform != 'win32':
        try:
            model = torch.compile(model)
            print("启用模型编译优化")
        except:
            print("模型编译优化不可用")
    else:
        print("快速模式或Windows平台，跳过模型编译")
    
    model = model.to(device)
    
    # 创建损失函数
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # 添加噪声正则化（在训练过程中向特征添加噪声）
    def add_feature_noise(features, noise_std=args.feature_noise):
        if model.training:
            noise = torch.randn_like(features) * noise_std
            return features + noise
        return features
    
    # 创建优化器
    optimizer = get_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    
    # 创建学习率调度器
    scheduler_kwargs = {}
    if args.scheduler == 'StepLR':
        scheduler_kwargs = {'step_size': 20, 'gamma': 0.5}
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler_kwargs = {'T_max': args.epochs}
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler_kwargs = {'patience': 5, 'factor': 0.5}
    
    scheduler = get_scheduler(optimizer, args.scheduler, **scheduler_kwargs)
    
    # 混合精度训练
    scaler = GradScaler('cuda')
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=args.patience,  # 使用命令行参数设置的耐心值
        verbose=True,
        path=os.path.join(args.save_dir, 'best_model.pth')
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_char_acc': [],
        'val_char_acc': [],
        'train_seq_acc': [],
        'val_seq_acc': [],
        'learning_rate': []
    }
    
    start_epoch = 0
    best_val_acc = 0.0
    current_semantic_weight = args.semantic_weight
    
    # 恢复训练
    if args.resume and os.path.exists(args.resume):
        print(f"从 {args.resume} 恢复训练")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        if 'history' in checkpoint:
            history = checkpoint['history']
    
    print("开始训练...")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print('-' * 50)
        
        # 训练
        train_loss, train_char_acc, train_seq_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, label_converter, epoch, 
            semantic_weight=current_semantic_weight, grad_clip=args.grad_clip
        )
        
        # 验证
        val_loss, val_char_acc, val_seq_acc = validate_epoch(
            model, val_loader, criterion, device, label_converter
        )
        
        # 更新学习率
        if args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_char_acc'].append(train_char_acc)
        history['val_char_acc'].append(val_char_acc)
        history['train_seq_acc'].append(train_seq_acc)
        history['val_seq_acc'].append(val_seq_acc)
        history['learning_rate'].append(current_lr)
        
        # 计算当前epoch的teacher forcing比例用于显示
        current_tf_ratio = max(0.8 * (0.95 ** epoch), 0.1)
        
        # 打印结果
        print(f"训练 - 损失: {train_loss:.4f}, 字符准确率: {train_char_acc:.4f}, 序列准确率: {train_seq_acc:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, 字符准确率: {val_char_acc:.4f}, 序列准确率: {val_seq_acc:.4f}")
        print(f"学习率: {current_lr:.6f}, 语义权重: {current_semantic_weight:.6f}, Teacher Forcing比例: {current_tf_ratio:.3f}")
        
        # 语义权重衰减
        current_semantic_weight *= args.semantic_decay
        
        # 早停检查
        early_stopping(val_loss, model)
        
        # 保存最佳模型
        if val_char_acc > best_val_acc:
            best_val_acc = val_char_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'best_acc_model.pth'))
            print(f"保存最佳准确率模型 (验证字符准确率: {val_char_acc:.4f})")
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # 早停检查
        if early_stopping.early_stop:
            print("早停触发")
            break
    
    # 训练结束
    training_time = time.time() - start_time
    print(f"\n训练完成! 总用时: {training_time/3600:.2f} 小时")
    print(f"最佳验证字符准确率: {best_val_acc:.4f}")
    
    # 保存最终模型
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
        'history': history,
        'args': vars(args),
        'training_time': training_time
    }, os.path.join(args.save_dir, 'final_model.pth'))
    
    # 保存训练曲线
    curves_path = os.path.join(args.save_dir, f'training_curves_{args.rnn_type}.png')
    save_training_curves(history, curves_path)
    
    # 保存训练指标
    metrics_path = os.path.join(args.save_dir, f'training_metrics_{args.rnn_type}.json')
    save_metrics_json(history, metrics_path)
    
    # 保存配置
    config_path = os.path.join(args.save_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    print(f"训练曲线保存至: {curves_path}")
    print(f"训练指标保存至: {metrics_path}")
    print(f"配置文件保存至: {config_path}")
    
    # 保存测试集预测结果
    print("\n开始保存测试集预测结果...")
    save_test_predictions(model, val_loader, label_converter, device, args.save_dir)


if __name__ == "__main__":
    main()