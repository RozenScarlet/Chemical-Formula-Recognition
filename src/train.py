import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import time
import random
import numpy as np
import glob
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import gc
import json
import psutil
import difflib  # 导入difflib库，用于计算字符串相似度
import platform
import matplotlib.pyplot as plt

# 添加项目根目录到路径
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入模型和数据集
from crnn import CRNN
from lcrnn import LCRNN
from msf_lcrnn import MSFLCRNN as MSF_LCRNN  # 使用别名匹配train.py中使用的名称
from dataset import ChemicalEquationDataset, collate_fn, create_data_loaders
from utils import CTCLabelConverter, train_epoch, validate, model_size_analysis, save_checkpoint, load_checkpoint

def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU
    cudnn.deterministic = False  # 为了速度，允许一些非确定性
    cudnn.benchmark = True  # 启用cudnn benchmark以加速卷积操作

def get_optimal_workers():
    """获取最佳的数据加载器worker数量，针对AMD Ryzen 5 3600X (6核12线程)优化"""
    num_cpus = psutil.cpu_count(logical=True)
    # Ryzen 5 3600X有6核12线程，保留2个线程给系统和主进程，其余分配给数据加载
    return max(1, min(num_cpus - 2, 10))  # 默认为10个worker（适合Ryzen 5 3600X）

def generate_label_file(image_dir, label_dir, output_path):
    """生成用于训练的标签文件"""
    print("正在生成标签文件...")
    
    # 获取所有图像文件
    image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    image_filenames = [os.path.basename(f) for f in image_files]
    
    # 读取classes.txt
    classes = []
    # 尝试多种可能的路径找到classes.txt文件
    possible_paths = [
        os.path.join(label_dir, "classes.txt"),
        os.path.join(os.path.dirname(image_dir), "label", "classes.txt"),
        os.path.join(os.path.dirname(os.path.dirname(image_dir)), "label", "classes.txt"),
    ]
    
    classes_path = None
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        print(f"尝试读取字符集文件: {normalized_path}")
        if os.path.exists(normalized_path):
            classes_path = normalized_path
            break
    
    if not classes_path:
        raise FileNotFoundError(f"无法找到classes.txt文件，尝试过的路径: {possible_paths}")
    
    print(f"使用字符集文件: {classes_path}")
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"将生成标签文件: {output_path}")
    
    # 处理每个图像对应的标签
    with open(output_path, "w", encoding="utf-8") as out_file:
        for img_file in tqdm(image_filenames, desc="处理标签"):
            base_name = os.path.splitext(img_file)[0]  # 移除.jpg后缀
            label_file = os.path.normpath(os.path.join(label_dir, f"{base_name}.txt"))
            
            if os.path.exists(label_file):
                # 这部分需要根据实际标签格式进行修改
                # 假设每行是 "类别ID x y 宽 高" 格式，我们按顺序处理
                labels = []
                with open(label_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 1:
                            class_id = int(parts[0])
                            if class_id < len(classes):
                                labels.append(classes[class_id])
                
                if labels:
                    out_file.write(f"{img_file} {''.join(labels)}\n")
    
    print(f"标签文件已生成: {output_path}")
    return output_path

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"GPU {i}: 总内存 {total_memory:.2f} GB, "
                  f"已分配 {allocated_memory:.2f} GB, "
                  f"已缓存 {cached_memory:.2f} GB")

def optimize_gpu_memory():
    """优化GPU内存使用，针对RTX 3060 12GB显存优化"""
    # 清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置内存分配策略
        try:
            # 使用95%的GPU显存，为RTX 3060留出一些余量
            torch.cuda.empty_cache()
            gc.collect()
            # 设置允许TensorFloat32加速
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # 启用cudnn benchmark加速卷积操作
            torch.backends.cudnn.benchmark = True
            # 禁用确定性算法，提高性能
            torch.backends.cudnn.deterministic = False
            # 启用异步执行CUDA操作
            torch.cuda.set_device(torch.cuda.current_device())
        except Exception as e:
            print(f"设置GPU内存分配策略时出现错误: {e}")
            print("继续使用默认内存分配策略")
    # 调用垃圾回收
    gc.collect()

def train_epoch_amp(model, data_loader, optimizer, criterion, device, scaler, accumulation_steps=1, converter=None):
    """使用混合精度训练一个epoch"""
    model.train()
    total_loss = 0
    processed_batches = 0
    
    # 添加指标追踪
    correct_num = 0
    all_samples = 0
    total_similarity = 0.0
    correct_chars = 0
    total_chars = 0
    
    optimizer.zero_grad()  # 初始梯度清零
    
    pbar = tqdm(data_loader, desc="训练批次")
    for batch_idx, (images, targets, target_lengths, texts) in enumerate(pbar):
        images = images.to(device, non_blocking=True)  # 使用non_blocking=True加速数据传输
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)
        
        # 混合精度前向传播
        with autocast():
            outputs = model(images)
            log_probs = outputs.log_softmax(2)
            batch_size, seq_length, _ = log_probs.size()
            
            # 计算输入长度 (CTC需要)
            input_lengths = torch.full((batch_size,), seq_length, device=device)
            
            # 计算损失
            loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
            loss = loss / accumulation_steps  # 梯度累积
        
        # 混合精度反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪，防止梯度爆炸
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # 更新参数
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # 计算准确率指标 (在验证模式下计算，不影响训练)
        if converter is not None:
            with torch.no_grad():
                _, preds = log_probs.max(2)
                preds_list = preds.detach().cpu().numpy()
                
                # 使用模型输出的序列长度作为解码长度
                length_for_pred = torch.IntTensor([seq_length] * batch_size)
                pred_texts = converter.decode(preds_list, length_for_pred)
                
                # 计算各种指标
                for i, (pred_text, gt_text) in enumerate(zip(pred_texts, texts)):
                    # AP: 完全匹配准确率
                    if pred_text == gt_text:
                        correct_num += 1
                    
                    # APc: 使用difflib计算字符串相似度
                    similarity = difflib.SequenceMatcher(None, pred_text, gt_text).ratio()
                    total_similarity += similarity
                    
                    # 计算字符级准确率
                    min_len = min(len(pred_text), len(gt_text))
                    for j in range(min_len):
                        if j < len(pred_text) and j < len(gt_text) and pred_text[j] == gt_text[j]:
                            correct_chars += 1
                    total_chars += len(gt_text)
                
                all_samples += batch_size
        
        total_loss += loss.item() * accumulation_steps
        processed_batches += 1
        
        # 更新进度条
        if converter is not None and all_samples > 0:
            pbar.set_postfix(loss=f"{loss.item()*accumulation_steps:.4f}", 
                            AP=f"{correct_num/all_samples:.4f}",
                            APc=f"{total_similarity/all_samples:.4f}",
                            char_acc=f"{correct_chars/total_chars if total_chars > 0 else 0:.4f}")
        else:
            pbar.set_postfix(loss=f"{loss.item()*accumulation_steps:.4f}")
        
        # 定期释放未使用的内存
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    # 处理剩余的梯度（如果有）
    if (batch_idx + 1) % accumulation_steps != 0:
        # 梯度裁剪，防止梯度爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # 计算平均损失和准确率
    avg_loss = total_loss / processed_batches
    if converter is not None and all_samples > 0:
        ap = correct_num / all_samples
        apc = total_similarity / all_samples
        char_acc = correct_chars / total_chars if total_chars > 0 else 0
        print(f"训练结果 - 损失: {avg_loss:.4f}, AP: {ap:.4f}, APc: {apc:.4f}, 字符准确率: {char_acc:.4f}")
        return avg_loss, ap, apc, char_acc
    
    return avg_loss

def validate_amp(model, data_loader, criterion, device, converter):
    """使用混合精度验证模型"""
    model.eval()
    total_loss = 0
    correct_num = 0
    all_samples = 0
    
    # APc相似度计算总和
    total_similarity = 0.0
    
    # 字符级准确率评估
    correct_chars = 0
    total_chars = 0
    
    pbar = tqdm(data_loader, desc="验证批次")
    with torch.no_grad():
        for images, targets, target_lengths, texts in pbar:
            images = images.to(device, non_blocking=True)  # 使用non_blocking=True加速数据传输
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            
            # 混合精度前向传播
            with autocast():
                outputs = model(images)
                log_probs = outputs.log_softmax(2)
                batch_size, seq_length, _ = log_probs.size()
                
                # 计算输入长度 (CTC需要)
                input_lengths = torch.full((batch_size,), seq_length, device=device)
                
                # 计算损失
                loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
                total_loss += loss.item()
            
            # 解码预测
            _, preds = log_probs.max(2)
            preds_list = preds.detach().cpu().numpy()
            
            # 使用模型输出的序列长度作为解码长度
            length_for_pred = torch.IntTensor([seq_length] * batch_size)
            pred_texts = converter.decode(preds_list, length_for_pred)
            
            # 计算准确率
            for i, (pred_text, gt_text) in enumerate(zip(pred_texts, texts)):
                # AP: 完全匹配准确率
                if pred_text == gt_text:
                    correct_num += 1
                
                # APc: 使用difflib计算字符串相似度
                similarity = difflib.SequenceMatcher(None, pred_text, gt_text).ratio()
                total_similarity += similarity
                
                # 计算字符级准确率（正确字符数/总字符数）
                # 使用较短长度作为基准避免除零错误
                min_len = min(len(pred_text), len(gt_text))
                for j in range(min_len):
                    if j < len(pred_text) and j < len(gt_text) and pred_text[j] == gt_text[j]:
                        correct_chars += 1
                total_chars += len(gt_text)  # 使用真实标签长度作为字符总数
            
            all_samples += batch_size
            
            # 更新进度条
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(data_loader)
    ap = correct_num / all_samples if all_samples > 0 else 0
    apc = total_similarity / all_samples if all_samples > 0 else 0
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    
    print(f"验证结果 - 损失: {avg_loss:.4f}, AP: {ap:.4f}, APc: {apc:.4f}, 字符准确率: {char_acc:.4f}")
    print(f"完全正确样本数: {correct_num}/{all_samples}, 正确字符数: {correct_chars}/{total_chars}")
    
    return avg_loss, ap, apc, char_acc

def find_optimal_batch_size(model, sample_batch, device, max_batch_size=128, start_batch_size=16):
    """找到最大可用的批量大小"""
    print("寻找最优批量大小...")
    batch_size = start_batch_size
    prev_batch_size = start_batch_size
    
    # 获取输入图像数据（从数据加载器返回的元组的第一个元素）
    if isinstance(sample_batch, tuple) and len(sample_batch) >= 1:
        images = sample_batch[0]  # 从元组中提取图像数据
    else:
        images = sample_batch  # 如果不是元组，直接使用
    
    if not isinstance(images, torch.Tensor) or images.size(0) == 0:
        print("警告: 样本批次为空或无效，使用默认批量大小")
        return start_batch_size

    while batch_size <= max_batch_size:
        try:
            # 尝试创建一个更大的批次
            x = torch.cat([images] * (batch_size // images.size(0) + 1), 0)[:batch_size]
            x = x.to(device)
            
            # 尝试前向传播
            with autocast():
                model(x)
            
            # 如果成功，尝试更大的批量
            prev_batch_size = batch_size
            batch_size *= 2
            
            # 清理内存
            del x
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # 内存不足，回到上一个工作的批量大小
                torch.cuda.empty_cache()
                return prev_batch_size
            else:
                # 其他错误
                print(f"寻找最优批量大小时发生错误: {e}")
                return prev_batch_size
    
    # 如果到达最大批量大小仍然工作正常
    return max_batch_size

def save_config(args, filepath):
    """保存训练配置到文件"""
    config = vars(args)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存到: {filepath}")

def main():
    """训练模型的主函数"""
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import numpy as np
    import platform

    # 解决matplotlib中文显示问题
    try:
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei']
        elif system == 'Darwin':
            plt.rcParams['font.sans-serif'] = ['PingFang SC']
        else: # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"警告: 设置中文字体失败，图表中的中文可能无法正确显示。错误: {e}")
    
    # 命令行参数
    parser = argparse.ArgumentParser(description='训练化学方程式识别模型')
    parser.add_argument('--data-dir', type=str, default='dataset', help='数据集目录')
    parser.add_argument('--model-type', type=str, choices=['CRNN', 'LCRNN', 'MSF_LCRNN'], required=True)
    parser.add_argument('--input-channel', type=int, default=1, help='输入通道数')
    parser.add_argument('--output-channel', type=int, default=512, help='特征提取输出通道数')
    parser.add_argument('--hidden-size', type=int, default=256, help='RNN隐藏层大小')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小，适用于RTX 3060 12GB显存')
    parser.add_argument('--auto-batch-size', action='store_true', help='自动寻找最优批量大小')
    parser.add_argument('--min-batch-size', type=int, default=16, help='自动批量大小的最小值')
    parser.add_argument('--max-batch-size', type=int, default=128, help='自动批量大小的最大值')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='最小学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='检查点保存目录')
    parser.add_argument('--checkpoint', type=str, default=None, help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--val-split', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--target-height', type=int, default=32, help='输入图像高度')
    parser.add_argument('--max-width', type=int, default=512, help='输入图像最大宽度')
    parser.add_argument('--num-workers', type=int, default=None, help='数据加载线程数，适用于Ryzen 5 3600X 6核12线程')
    parser.add_argument('--generate-labels', action='store_true', help='生成标签文件')
    parser.add_argument('--label-file', type=str, default='labels.txt', help='标签文件名称')
    parser.add_argument('--amp', action='store_true', help='使用自动混合精度训练')
    parser.add_argument('--scheduler', type=str, default='plateau', choices=['plateau', 'cosine'], help='学习率调度器类型')
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='梯度累积步数')
    parser.add_argument('--save-interval', type=int, default=5, help='模型保存间隔(epochs)')
    parser.add_argument('--prefetch-factor', type=int, default=2, help='数据预取因子，提高CPU和GPU并行效率')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='优化器类型')
    parser.add_argument('--gpu', type=int, default=0, help='要使用的GPU编号，-1表示使用CPU')
    parser.add_argument('--adaptive-batch-size', action='store_true', help='根据内存使用情况自适应调整批次大小')
    parser.add_argument('--grad-clip', type=float, default=5.0, help='梯度裁剪阈值')
    parser.add_argument('--pin-memory', action='store_true', default=True, help='使用固定内存提高CPU到GPU的数据传输速度')
    parser.add_argument('--memory-fraction', type=float, default=0.95, help='GPU显存使用比例上限(0-1)，为系统操作预留余量')
    parser.add_argument('--use-tf32', action='store_true', help='在Ampere及更高架构上启用TF32计算')
    parser.add_argument('--compile-model', action='store_true', help='使用torch.compile编译模型以提高性能(需要PyTorch 2.0+)')
    args = parser.parse_args()
    
    # 检测操作系统类型
    is_windows = os.name == 'nt'
    print(f"检测到操作系统: {'Windows' if is_windows else 'Unix/Linux/MacOS'}")
    
    # 标准化路径
    args.data_dir = os.path.normpath(args.data_dir)
    args.checkpoint_dir = os.path.normpath(args.checkpoint_dir)
    
    print(f"数据目录: {args.data_dir}")
    print(f"检查点目录: {args.checkpoint_dir}")
    
    # 确保目录存在
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 使用CUDA
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    
    # 优化GPU内存使用
    if "cuda" in device.type:
        optimize_gpu_memory()
        # 显示GPU内存信息
        get_gpu_memory_info()
        
        # 设置内存分数
        if args.memory_fraction < 1.0:
            try:
                import torch.cuda
                torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device.index)
                print(f"已设置GPU显存使用上限为总显存的 {args.memory_fraction*100:.1f}%")
            except:
                print("无法设置GPU显存使用上限，使用默认设置")
    
    # 启用TF32精度
    if args.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("已启用TF32精度")
    
    # 处理工作线程数
    if args.num_workers is None:
        args.num_workers = get_optimal_workers()
        print(f"自动设置工作线程数: {args.num_workers}")
    
    # Windows系统调整
    if is_windows and args.num_workers > 0:
        # Windows上DataLoader工作线程限制
        args.num_workers = min(4, args.num_workers)
        print(f"Windows系统: 调整工作线程数为 {args.num_workers}")
    
    # 处理标签文件路径
    label_path = os.path.join(args.data_dir, "labels.txt")
    
    # 生成标签文件(如果需要)
    if args.generate_labels:
        image_dir = os.path.join(args.data_dir, 'images')
        label_dir = os.path.join(args.data_dir, 'label')
        
        # 标准化路径
        image_dir = os.path.normpath(image_dir)
        label_dir = os.path.normpath(label_dir)
        label_path = os.path.normpath(label_path)
        
        print(f"生成标签文件:")
        print(f"- 图像目录: {image_dir}")
        print(f"- 标签目录: {label_dir}")
        print(f"- 输出文件: {label_path}")
        
        generate_label_file(image_dir, label_dir, label_path)
    
    # 检查标签文件是否存在
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"标签文件不存在: {label_path}，请先生成标签文件")
    
    # 创建数据集
    print("正在创建数据集...")
    dataset = ChemicalEquationDataset(
        data_dir=os.path.join(args.data_dir, 'images'),
        label_file=label_path,
        target_height=args.target_height
    )
    
    # 确保数据集中的字符集被正确加载
    if not dataset.char_to_idx:
        raise ValueError("字符集未正确加载，请检查数据集创建过程")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, dataset = create_data_loaders(
        args.data_dir, args.label_file, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, 
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0)
    )
    
    # 选择模型架构
    print(f"创建{args.model_type}模型...")
    if args.model_type == 'CRNN':
        model = CRNN(
            input_channel=args.input_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_classes=dataset.num_classes
        )
    elif args.model_type == 'LCRNN':
        model = LCRNN(
            input_channel=args.input_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_classes=dataset.num_classes
        )
    elif args.model_type == 'MSF_LCRNN':
        model = MSF_LCRNN(
            input_channel=args.input_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_classes=dataset.num_classes
        )
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 分析模型大小
    model_info = model_size_analysis(model)
    
    # 使用torch.compile加速(如果可用)
    if args.compile_model:
        # 检查PyTorch版本是否支持编译
        if hasattr(torch, 'compile'):
            print("使用torch.compile编译模型...")
            try:
                model = torch.compile(model)
                print("模型编译成功!")
            except Exception as e:
                print(f"模型编译失败: {e}")
        else:
            print("当前PyTorch版本不支持compile，跳过编译")
    
    # 将模型移动到设备
    model.to(device)
    
    # 创建CTCLoss
    criterion = nn.CTCLoss(blank=dataset.char_to_idx[''], reduction='mean', zero_infinity=True).to(device)
    
    # 选择优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"不支持的优化器类型: {args.optimizer}")
    
    # 混合精度训练准备
    scaler = GradScaler() if args.amp else None
    
    # 创建检查点目录
    model_checkpoint_dir = os.path.normpath(os.path.join(args.checkpoint_dir, args.model_type.lower()))
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    
    # 自动批量大小搜索
    if args.auto_batch_size:
        print("搜索最优批量大小...")
        # 获取一个样本批次
        for images, targets, target_lengths, _ in train_loader:
            sample_batch = (images, targets, target_lengths)
            break
        
        args.batch_size = find_optimal_batch_size(
            model, sample_batch, device, 
            max_batch_size=args.max_batch_size,
            start_batch_size=args.min_batch_size
        )
        print(f"找到最优批量大小: {args.batch_size}")
        
        # 使用新的批量大小重新创建数据加载器
        train_loader, dataset = create_data_loaders(
            args.data_dir, args.label_file, 
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=args.pin_memory,
            persistent_workers=(args.num_workers > 0)
        )
    
    # 分割训练集和验证集
    train_size = int((1 - args.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")
    
    # 创建训练和验证数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if args.num_workers else 0,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2,
        persistent_workers=args.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if args.num_workers else 0,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else 2,
        persistent_workers=args.num_workers > 0
    )
    
    # 创建学习率调度器
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=args.min_lr)
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=args.min_lr)
    
    # 创建文本转换器
    # 从数据集中获取字符集，确保不包含空白符
    characters = [dataset.idx_to_char[i] for i in sorted(dataset.idx_to_char.keys()) if dataset.idx_to_char[i] != '']
    converter = CTCLabelConverter(characters)
    
    # 加载检查点(如果存在)
    last_checkpoint = None
    best_checkpoint = None
    
    checkpoint_pattern = os.path.normpath(os.path.join(model_checkpoint_dir, f"{args.model_type.lower()}_*.pth"))
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if checkpoint_files:
        print(f"找到 {len(checkpoint_files)} 个检查点文件")
        last_epoch = -1
        best_acc = -1
        
        for cf in checkpoint_files:
            if "best" in cf:
                best_checkpoint = cf
            else:
                try:
                    epoch = int(cf.split("_")[-1].split(".")[0])
                    if epoch > last_epoch:
                        last_epoch = epoch
                        last_checkpoint = cf
                except:
                    pass
        
        if last_checkpoint:
            print(f"加载最新检查点: {last_checkpoint}")
            model, optimizer, start_epoch, loss, acc = load_checkpoint(model, optimizer, last_checkpoint)
        else:
            start_epoch = 0
    else:
        print("未找到检查点文件，从头开始训练")
        start_epoch = 0
    
    # 保存配置信息
    config_file = os.path.normpath(os.path.join(model_checkpoint_dir, "config.json"))
    save_config(args, config_file)
    
    # 创建指标记录列表
    train_losses = []
    train_aps = []
    train_apcs = []
    train_char_accs = []
    
    val_losses = []
    val_aps = []
    val_apcs = []
    val_char_accs = []
    
    best_metrics = {
        'loss': float('inf'),
        'ap': 0,
        'apc': 0,
        'char_acc': 0
    }
    
    # 开始训练
    print(f"\n开始训练{args.model_type}模型...")
    best_accuracy = 0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        epoch_start_time = time.time()
        
        # 训练
        if args.amp:
            train_loss, train_ap, train_apc, train_char_acc = train_epoch_amp(model, train_loader, optimizer, criterion, device, scaler, args.gradient_accumulation, converter)
            train_losses.append(train_loss)
            train_aps.append(train_ap)
            train_apcs.append(train_apc)
            train_char_accs.append(train_char_acc)
        else:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            train_losses.append(train_loss)
        
        # 验证
        if args.amp:
            val_loss, val_ap, val_apc, val_char_acc = validate_amp(model, val_loader, criterion, device, converter)
            val_losses.append(val_loss)
            val_aps.append(val_ap)
            val_apcs.append(val_apc)
            val_char_accs.append(val_char_acc)
        else:
            val_loss, val_ap, val_apc, val_char_acc = validate(model, val_loader, criterion, device, converter)
            val_losses.append(val_loss)
            val_aps.append(val_ap)
            val_apcs.append(val_apc)
            val_char_accs.append(val_char_acc)
        
        # 更新学习率
        if args.scheduler == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 输出当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.8f}")
        
        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_file = os.path.normpath(os.path.join(model_checkpoint_dir, f"{args.model_type.lower()}_{epoch+1}.pth"))
            save_checkpoint(model, optimizer, epoch + 1, train_loss, train_ap, checkpoint_file)
        
        # 更新最佳指标 - 使用训练指标
        if train_loss < best_metrics['loss']:
            best_metrics['loss'] = train_loss
        
        # 保存最佳AP模型 - 使用训练AP
        if train_ap > best_accuracy:
            best_accuracy = train_ap
            best_metrics['ap'] = train_ap
            best_checkpoint_file = os.path.normpath(os.path.join(model_checkpoint_dir, f"{args.model_type.lower()}_best_ap.pth"))
            save_checkpoint(model, optimizer, epoch + 1, train_loss, train_ap, best_checkpoint_file)
            print(f"保存最佳模型，训练AP: {best_accuracy:.4f}")
        
        # 更新其他最佳指标 - 使用训练指标
        if train_apc > best_metrics['apc']:
            best_metrics['apc'] = train_apc
        
        if train_char_acc > best_metrics['char_acc']:
            best_metrics['char_acc'] = train_char_acc
        
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch用时: {epoch_duration:.2f}秒")
        
        # 内存清理
        torch.cuda.empty_cache()
        gc.collect()
    
    if not train_losses:
        print("未执行任何训练轮次 (可能已达到目标轮次). 跳过指标和图表生成。")
        return 0
    
    # 训练完成，绘制收敛曲线并保存
    plt.figure(figsize=(16, 12))
    
    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(val_losses, 'r--', label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title(f'损失曲线 (最佳训练损失: {best_metrics["loss"]:.4f})')
    plt.legend()
    plt.grid(True)
    
    # 2. AP曲线
    plt.subplot(2, 2, 2)
    plt.plot(train_aps, 'g-', label='训练AP')
    plt.plot(val_aps, 'c--', label='验证AP')
    plt.xlabel('Epoch')
    plt.ylabel('AP')
    plt.title(f'AP曲线 (最佳训练AP: {best_metrics["ap"]:.4f})')
    plt.legend()
    plt.grid(True)
    
    # 3. APc曲线
    plt.subplot(2, 2, 3)
    plt.plot(train_apcs, 'r-', label='训练APc')
    plt.plot(val_apcs, 'b--', label='验证APc')
    plt.xlabel('Epoch')
    plt.ylabel('APc')
    plt.title(f'APc曲线 (最佳训练APc: {best_metrics["apc"]:.4f})')
    plt.legend()
    plt.grid(True)
    
    # 4. 字符准确率曲线
    plt.subplot(2, 2, 4)
    plt.plot(train_char_accs, 'm-', label='训练字符准确率')
    plt.plot(val_char_accs, 'y--', label='验证字符准确率')
    plt.xlabel('Epoch')
    plt.ylabel('字符准确率')
    plt.title(f'字符准确率曲线 (最佳训练字符准确率: {best_metrics["char_acc"]:.4f})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图表
    metrics_img_file = os.path.normpath(os.path.join(model_checkpoint_dir, f"{args.model_type.lower()}_metrics.png"))
    plt.savefig(metrics_img_file)
    print(f"保存指标图表到: {metrics_img_file}")
    
    # 保存指标数据
    metrics_data = {
        'train_losses': train_losses,
        'train_aps': train_aps,
        'train_apcs': train_apcs,
        'train_char_accs': train_char_accs,
        'val_losses': val_losses,
        'val_aps': val_aps,
        'val_apcs': val_apcs,
        'val_char_accs': val_char_accs,
        'best_metrics': best_metrics
    }
    
    metrics_file = os.path.normpath(os.path.join(model_checkpoint_dir, f"{args.model_type.lower()}_metrics.json"))
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f"保存指标数据到: {metrics_file}")
    
    print(f"\n{args.model_type}模型训练完成!")
    print(f"最佳损失: {best_metrics['loss']:.4f}")
    print(f"最佳AP: {best_metrics['ap']:.4f}")
    print(f"最佳APc: {best_metrics['apc']:.4f}")
    print(f"最佳字符准确率: {best_metrics['char_acc']:.4f}")
    
    # 寻找最佳指标的轮次
    best_loss_epoch = train_losses.index(min(train_losses)) + 1
    best_ap_epoch = train_aps.index(max(train_aps)) + 1
    best_apc_epoch = train_apcs.index(max(train_apcs)) + 1
    best_char_acc_epoch = train_char_accs.index(max(train_char_accs)) + 1
    
    print(f"最佳损失出现在第 {best_loss_epoch} 轮")
    print(f"最佳AP出现在第 {best_ap_epoch} 轮")
    print(f"最佳APc出现在第 {best_apc_epoch} 轮")
    print(f"最佳字符准确率出现在第 {best_char_acc_epoch} 轮")
    
    return 0

if __name__ == '__main__':
    main() 