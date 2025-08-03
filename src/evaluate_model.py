#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import cv2
import numpy as np
import json
import glob
import difflib
from tqdm import tqdm
from src.models import TransformerCNNNet
from src.utils import CTCLabelConverter, preprocess_image
from src.dataset import ChemicalEquationDataset
from torch.utils.data import DataLoader, Dataset

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestDataset(Dataset):
    """测试数据集类"""
    def __init__(self, image_dir, label_file=None, target_height=32):
        self.image_dir = image_dir
        self.target_height = target_height
        
        # 获取所有图像文件
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        # 如果提供了标签文件，加载标签
        self.labels = {}
        if label_file and os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        label = ' '.join(parts[1:])
                        self.labels[image_name] = label
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        
        # 读取图像并预处理
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取图像: {image_path}")
            img = np.zeros((self.target_height, 100), dtype=np.uint8)
        
        # 调整图像大小
        img = preprocess_image(img, self.target_height)
        
        # 获取标签（如果有）
        label = self.labels.get(image_name, "")
        
        return {
            'image': torch.FloatTensor(img),
            'image_path': image_path,
            'label': label
        }


def collate_fn(batch):
    """自定义的collate_fn函数，处理不同宽度的图像"""
    images = []
    image_paths = []
    labels = []
    
    for item in batch:
        images.append(item['image'])
        image_paths.append(item['image_path'])
        labels.append(item['label'])
    
    # 单独处理每个样本，不做批量处理
    return {
        'image': images,
        'image_path': image_paths,
        'label': labels
    }


def load_model(model_path, config_path=None):
    """加载保存的模型"""
    # 加载配置文件
    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    
    if not os.path.exists(config_path):
        model_dir = os.path.dirname(model_path)
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
    
    # 加载配置
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 创建字符集
    try:
        classes_file = os.path.join(config.get('data_dir', 'dataset'), 'label', 'classes.txt')
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                character_set = [line.strip() for line in f.readlines() if line.strip()]
            print(f"成功加载了 {len(character_set)} 个字符")
        else:
            raise FileNotFoundError(f"找不到字符集文件: {classes_file}")
    except Exception as e:
        raise RuntimeError(f"读取字符集文件出错: {str(e)}")
    
    # 创建CTC解码器
    converter = CTCLabelConverter(character_set)
    num_classes = converter.num_classes
    
    # 从配置中获取rnn_type
    rnn_type = config.get('rnn_type', 'lstm')

    # 创建模型
    model = TransformerCNNNet(
        input_channel=config.get('input_channel', 1),
        output_channel=config.get('output_channel', 64),
        num_classes=num_classes,
        hidden_size=config.get('hidden_size', 256),
        num_heads=config.get('num_heads', 8),
        num_encoder_layers=config.get('num_encoder_layers', 4),
        dropout=config.get('dropout', 0.1),
        rnn_type=rnn_type,
        use_msf=config.get('use_msf', True),
        lightweight=config.get('lightweight', False)
    )
    
    # 加载模型权重
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(model_path, weights_only=False)
        else:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f'已加载模型: {model_path}')
    except Exception as e:
        raise RuntimeError(f"加载模型出错: {str(e)}")
    
    return model, converter, config


def levenshtein_distance_norm(s1, s2):
    """计算规范化的编辑距离（字符准确率）"""
    if len(s2) == 0:
        return 0.0 if len(s1) == 0 else 0.0
    
    # 使用difflib计算相似度
    similarity = difflib.SequenceMatcher(None, s1, s2).ratio()
    return similarity


def evaluate(model, test_loader, converter):
    """评估模型的AP和APc指标"""
    model.eval()
    
    all_samples = 0
    correct_samples = 0  # AP指标的分子
    total_char_accuracy = 0.0  # APc指标的分子
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估"):
            batch_images = batch['image']
            batch_labels = batch['label']
            batch_paths = batch['image_path']
            
            # 单独处理每个样本
            for i, (image, label, path) in enumerate(zip(batch_images, batch_labels, batch_paths)):
                # 将图像放到设备上并添加批次维度
                image = image.unsqueeze(0).to(device)
                
                # 模型推理
                pred = model(image)
                
                # 获取预测的最高概率索引
                _, pred_index = pred.max(2)
                pred_index = pred_index.detach().cpu().numpy()
                
                # 解码预测结果
                length = np.array([pred_index.shape[1]])
                pred_text = converter.decode(pred_index, length)[0]
                
                # 完全匹配准确率
                is_correct = pred_text == label
                if is_correct:
                    correct_samples += 1
                
                # 字符级准确率
                char_accuracy = levenshtein_distance_norm(pred_text, label)
                total_char_accuracy += char_accuracy
                
                # 保存结果
                results.append({
                    'image': os.path.basename(path),
                    'prediction': pred_text,
                    'label': label,
                    'is_correct': bool(is_correct),
                    'char_accuracy': float(char_accuracy)
                })
                
                all_samples += 1
    
    # 计算最终指标
    ap = correct_samples / all_samples if all_samples > 0 else 0
    apc = total_char_accuracy / all_samples if all_samples > 0 else 0
    
    return ap, apc, results


def main():
    parser = argparse.ArgumentParser(description='评估Transformer-CNN化学方程式识别模型')
    parser.add_argument('--model-path', type=str, required=True, help='模型路径')
    parser.add_argument('--config-path', type=str, default=None, help='配置文件路径')
    parser.add_argument('--test-dir', type=str, required=True, help='测试图像目录')
    parser.add_argument('--label-file', type=str, default=None, help='测试标签文件路径')
    parser.add_argument('--output-file', type=str, default=None, help='评估结果输出文件')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--target-height', type=int, default=32, help='调整图像高度')
    parser.add_argument('--gpu', type=int, default=0, help='使用GPU编号，-1表示CPU')
    parser.add_argument('--rnn-type', type=str, default=None, help='手动指定RNN类型 (lstm 或 gru) 以覆盖配置')
    
    args = parser.parse_args()
    
    # 设置设备
    global device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, converter, config = load_model(args.model_path, args.config_path)
    
    # 如果命令行指定了rnn_type，则覆盖配置
    if args.rnn_type:
        config['rnn_type'] = args.rnn_type
    
    # 创建测试数据集
    test_dataset = TestDataset(
        image_dir=args.test_dir,
        label_file=args.label_file,
        target_height=args.target_height
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn  # 使用自定义的collate_fn
    )
    
    print(f"加载了 {len(test_dataset)} 个测试样本")
    
    # 评估模型
    ap, apc, results = evaluate(model, test_loader, converter)
    
    # 打印评估结果
    print(f"\n评估结果:")
    print(f"AP (完全匹配准确率): {ap:.4f}")
    print(f"APc (字符级准确率): {apc:.4f}")
    print(f"总样本数: {len(test_dataset)}")
    print(f"正确预测数: {sum(r['is_correct'] for r in results)}")
    
    # 保存评估结果
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': {
                    'ap': ap,
                    'apc': apc,
                    'total_samples': len(test_dataset),
                    'correct_samples': sum(r['is_correct'] for r in results)
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {args.output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"错误: {str(e)}")
        exit(1) 