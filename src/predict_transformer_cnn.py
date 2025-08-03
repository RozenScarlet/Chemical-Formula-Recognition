"""
TransformerCNN模型预测脚本
支持单张图片和批量预测，包含可视化功能
"""

import os
import sys
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformer_cnn_net import create_transformer_cnn
from dataset import build_character_set, CTCLabelConverter
from utils import preprocess_image, visualize_prediction, ModelSize


def load_model(model_path, device):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数
    if 'args' in checkpoint:
        args = checkpoint['args']
        num_classes = len(build_character_set(args.get('data_dir', 'dataset'))) + 2
        
        model = create_transformer_cnn(
            num_classes=num_classes,
            rnn_type=args.get('rnn_type', 'lstm'),
            hidden_size=args.get('hidden_size', 256),
            nhead=args.get('nhead', 8),
            num_layers=args.get('num_layers', 6),
            dropout=args.get('dropout', 0.1)
        )
    else:
        # 默认参数
        model = create_transformer_cnn(num_classes=100)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint.get('args', {})


def predict_single_image(model, image_path, label_converter, device, target_size=(256, 64)):
    """预测单张图片"""
    # 预处理图像
    image_tensor = preprocess_image(image_path, target_size)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # 前向传播
        outputs = model(image_tensor)
        
        # 如果有多个输出（语义指导），取第一个
        if isinstance(outputs, tuple):
            preds = outputs[0]
        else:
            preds = outputs
        
        # 获取预测序列
        pred_probs = F.log_softmax(preds, dim=2)
        pred_probs = pred_probs.squeeze(0)  # 移除batch维度
        
        # CTC解码 - 贪心解码
        _, pred_indices = pred_probs.max(1)
        pred_indices = pred_indices.cpu().numpy()
        
        # 使用统一的CTC解码
        pred_text = label_converter.ctc_greedy_decode(pred_indices)
        
        # 计算置信度（平均概率）
        confidence = torch.exp(pred_probs.max(1)[0]).mean().item()
        
    return pred_text, confidence


def predict_batch(model, image_paths, label_converter, device, target_size=(256, 64)):
    """批量预测图片"""
    results = []
    
    for image_path in image_paths:
        try:
            pred_text, confidence = predict_single_image(
                model, image_path, label_converter, device, target_size
            )
            results.append({
                'image_path': image_path,
                'prediction': pred_text,
                'confidence': confidence
            })
        except Exception as e:
            print(f"预测 {image_path} 时出错: {e}")
            results.append({
                'image_path': image_path,
                'prediction': '',
                'confidence': 0.0,
                'error': str(e)
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='TransformerCNN模型预测脚本')
    parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--input', type=str, required=True, 
                       help='输入图片路径或目录')
    parser.add_argument('--data-dir', type=str, default='dataset', 
                       help='数据集目录（用于构建字符集）')
    parser.add_argument('--output-file', type=str, default='', 
                       help='输出JSON文件路径')
    parser.add_argument('--visualize', action='store_true', 
                       help='可视化预测结果')
    parser.add_argument('--img-height', type=int, default=64, 
                       help='图像高度')
    parser.add_argument('--img-width', type=int, default=256, 
                       help='图像宽度')
    
    args = parser.parse_args()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载模型...")
    model, model_args = load_model(args.model_path, device)
    
    # 模型信息
    model_info = ModelSize.get_model_size(model)
    print(f"模型参数量: {model_info['param_count']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    
    # 构建字符集和标签转换器
    character_set = build_character_set(args.data_dir)
    label_converter = CTCLabelConverter(character_set)
    print(f"字符集大小: {len(character_set)}")
    
    # 准备输入
    target_size = (args.img_width, args.img_height)
    
    if os.path.isfile(args.input):
        # 单张图片预测
        print(f"预测单张图片: {args.input}")
        
        pred_text, confidence = predict_single_image(
            model, args.input, label_converter, device, target_size
        )
        
        print(f"预测结果: {pred_text}")
        print(f"置信度: {confidence:.4f}")
        
        # 可视化
        if args.visualize:
            visualize_prediction(args.input, pred_text, confidence)
        
        # 保存结果
        if args.output_file:
            result = {
                'image_path': args.input,
                'prediction': pred_text,
                'confidence': confidence
            }
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"结果保存至: {args.output_file}")
            
    elif os.path.isdir(args.input):
        # 批量预测
        print(f"批量预测目录: {args.input}")
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_paths = []
        
        for file in os.listdir(args.input):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(args.input, file))
        
        print(f"找到 {len(image_paths)} 张图片")
        
        # 批量预测
        results = predict_batch(model, image_paths, label_converter, device, target_size)
        
        # 打印结果
        for result in results:
            if 'error' not in result:
                print(f"{os.path.basename(result['image_path'])}: {result['prediction']} "
                      f"(置信度: {result['confidence']:.4f})")
            else:
                print(f"{os.path.basename(result['image_path'])}: 预测失败 - {result['error']}")
        
        # 保存结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"结果保存至: {args.output_file}")
        
        # 统计
        successful_predictions = [r for r in results if 'error' not in r]
        print(f"\n预测统计:")
        print(f"成功: {len(successful_predictions)}/{len(results)}")
        
        if successful_predictions:
            avg_confidence = np.mean([r['confidence'] for r in successful_predictions])
            print(f"平均置信度: {avg_confidence:.4f}")
    
    else:
        print(f"错误: 输入路径 {args.input} 不存在")


if __name__ == "__main__":
    main()