"""
测试脚本：读取训练好的模型，对test文件夹中的图片进行推理
生成带可视化的预测结果，保存到test_predict文件夹
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm
import json
import numpy as np

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformer_cnn_net import create_transformer_cnn
from dataset import create_dataset, build_character_set


def get_chinese_font():
    """获取中文字体，优先使用系统字体"""
    try:
        # Windows系统字体
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",      # 黑体
            "C:/Windows/Fonts/simsun.ttc",      # 宋体
            "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
            "C:/Windows/Fonts/simkai.ttf",      # 楷体
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path
                
        # 如果都找不到，使用matplotlib的默认中文字体
        chinese_fonts = [f for f in fm.findSystemFonts() if 'han' in f.lower() or 'cjk' in f.lower()]
        if chinese_fonts:
            return chinese_fonts[0]
            
    except Exception as e:
        print(f"字体加载警告: {e}")
    
    return None


def test_model_predictions():
    """测试模型预测并生成可视化结果"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集（只使用验证集进行测试）
    data_dir = "dataset"
    train_loader, val_loader, label_converter = create_dataset(data_dir, batch_size=1)
    
    # 设置checkpoint目录
    checkpoint_dir = "checkpoints/test"
    
    # 创建模型 - 使用训练时的参数
    num_classes = len(label_converter.character)
    
    # 从配置文件读取参数
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"从配置文件读取参数: {config}")
        
        model = create_transformer_cnn(
            num_classes=num_classes,
            rnn_type=config.get('rnn_type', 'lstm'),
            hidden_size=config.get('hidden_size', 256),
            nhead=config.get('nhead', 8),
            num_layers=config.get('num_layers', 6),
            dropout=config.get('dropout', 0.1)
        ).to(device)
    else:
        # 如果没有配置文件，使用与训练脚本一致的默认参数
        print("未找到配置文件，使用默认参数（与训练脚本一致）")
        model = create_transformer_cnn(
            num_classes=num_classes,
            rnn_type='gru',  # 改为gru，与训练脚本默认值一致
            hidden_size=256,
            nhead=8,
            num_layers=6,

        ).to(device)
    
    # 优先加载最佳模型 - 尝试多种可能的文件名
    possible_best_models = [
        "best_model.pth",
        "best_acc_model.pth", 
        "transformer_cnn_best_ap.pth",
        "gru_best_ap.pth"
    ]

    model_path = "final_model.pth"
    for model_name in possible_best_models:
        candidate_path = os.path.join(checkpoint_dir, model_name)
        if os.path.exists(candidate_path):
            model_path = candidate_path
            break
    
    if model_path:
        print(f"加载最佳模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            # 打印训练时的性能信息（如果有）
            if 'best_accuracy' in checkpoint:
                print(f"  训练时最佳准确率: {checkpoint['best_accuracy']:.4f}")
            if 'epoch' in checkpoint:
                print(f"  模型来自第 {checkpoint['epoch']} 轮")
        else:
            model.load_state_dict(checkpoint)
        epoch_num = "best"
    else:
        # 如果没有best_model.pth，查找最新的checkpoint文件
        model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]

        if model_files:
            # 按epoch号排序，取最新的
            latest_model = max(model_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            model_path = os.path.join(checkpoint_dir, latest_model)
            print(f"加载模型: {model_path}")

            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            epoch_num = int(latest_model.split('_')[-1].split('.')[0])
        else:
            print("找不到模型文件")
            return
    
    # 设置推理模式 - 关闭语义指导
    model.eval()
    if hasattr(model, 'set_inference_mode'):
        model.set_inference_mode()
    
    print(f"模型设置为推理模式")
    
    # 创建输出目录
    output_dir = "test_predict"
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查test文件夹
    test_dir = "test"
    if not os.path.exists(test_dir):
        print(f"test文件夹不存在，使用验证集进行测试")
        use_val_set = True
    else:
        # 检查test文件夹中是否有图片
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        test_images = [f for f in os.listdir(test_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]
        if len(test_images) == 0:
            print(f"test文件夹中没有图片，使用验证集进行测试")
            use_val_set = True
        else:
            print(f"在test文件夹中找到{len(test_images)}张图片")
            use_val_set = False
    
    results = []
    
    if use_val_set:
        # 使用验证集进行测试
        print("对验证集进行测试...")
        
        with torch.no_grad():
            for batch_idx, (images, targets, target_lengths, raw_texts) in enumerate(tqdm(val_loader, desc="测试验证集")):
                if batch_idx >= 20:  # 只测试前20个样本
                    break
                    
                images = images.to(device)
                
                # 预测
                outputs = model(images)
                if isinstance(outputs, tuple):
                    preds = outputs[0]
                else:
                    preds = outputs
                
                # 调整维度 (与训练时一致)
                preds = preds.permute(1, 0, 2)  # (T, N, C)
                
                # 使用log_softmax和CTC解码 (与训练时一致)
                log_probs = torch.log_softmax(preds, dim=2)
                
                # 获取最可能的序列 (贪心解码)
                _, preds_index = log_probs.max(2)  # (T, N)
                
                # 解码预测结果
                preds_str = []
                for i in range(preds_index.size(1)):  # 按batch维度遍历
                    pred_text = label_converter.ctc_greedy_decode(preds_index[:, i])  # 传入时间序列
                    preds_str.append(pred_text)
                
                for i in range(images.size(0)):
                    gt_text = raw_texts[i]
                    pred_text = preds_str[i]  # 新的解码方法已经过滤了特殊字符
                    
                    # 获取原始图像
                    original_image = val_loader.dataset.get_original_image(batch_idx * val_loader.batch_size + i)
                    
                    # 创建可视化图像
                    create_visualization(
                        original_image, gt_text, pred_text,
                        output_path=os.path.join(output_dir, f"test_{batch_idx:04d}_{i}.png")
                    )
                    
                    # 记录结果
                    is_correct = (gt_text == pred_text)
                    results.append({
                        'image': f"test_{batch_idx:04d}_{i}.png",
                        'ground_truth': gt_text,
                        'prediction': pred_text,
                        'correct': is_correct
                    })
    else:
        # 处理test文件夹中的图片
        print("处理test文件夹中的图片...")
        
        # 预处理函数 - 与训练时完全一致
        from torchvision import transforms
        from dataset import Resize  # 使用训练时相同的Resize类
        
        transform = transforms.Compose([
            Resize((256, 64)),  # 修正尺寸：宽256，高64（与训练时一致）
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        with torch.no_grad():
            for img_file in tqdm(test_images, desc="测试图片"):
                img_path = os.path.join(test_dir, img_file)
                
                # 加载和预处理图像
                original_image = Image.open(img_path).convert('RGB')
                image_tensor = transform(original_image).unsqueeze(0).to(device)
                
                # 预测
                outputs = model(image_tensor)
                if isinstance(outputs, tuple):
                    preds = outputs[0]
                else:
                    preds = outputs
                
                # 调整维度 (与训练时一致)
                preds = preds.permute(1, 0, 2)  # (T, N, C)
                
                # 使用log_softmax和CTC解码 (与训练时一致)
                log_probs = torch.log_softmax(preds, dim=2)
                
                # 获取最可能的序列 (贪心解码)
                _, preds_index = log_probs.max(2)  # (T, N)
                
                # 解码预测结果
                pred_text = label_converter.ctc_greedy_decode(preds_index[:, 0])  # 传入时间序列
                
                # 创建可视化图像（没有ground truth）
                output_path = os.path.join(output_dir, f"pred_{img_file}")
                create_visualization(
                    original_image, None, pred_text,
                    output_path=output_path
                )
                
                # 记录结果
                results.append({
                    'image': img_file,
                    'ground_truth': None,
                    'prediction': pred_text,
                    'correct': None
                })
    
    # 保存结果到JSON文件
    results_file = os.path.join(output_dir, "predictions.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 统计准确率（如果有ground truth）
    if use_val_set:
        correct_count = sum(1 for r in results if r['correct'])
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        print(f"\n测试完成！")
        print(f"总样本数: {total_count}")
        print(f"正确数量: {correct_count}")
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    else:
        print(f"\n测试完成！处理了{len(results)}张图片")
    
    print(f"结果保存在: {output_dir}")
    print(f"可视化图片和JSON结果已生成")


def create_visualization(original_image, gt_text, pred_text, output_path):
    """创建预测结果的可视化图像"""
    
    # 设置中文字体
    font_path = get_chinese_font()
    
    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    
    # 显示原始图像
    ax.imshow(np.array(original_image))
    ax.axis('off')
    
    # 准备文本
    if gt_text is not None:
        title_text = f"真实: {gt_text}\n预测: {pred_text}"
        # 检查预测是否正确
        is_correct = (gt_text == pred_text)
        title_color = 'green' if is_correct else 'red'
    else:
        title_text = f"预测: {pred_text}"
        title_color = 'blue'
    
    # 设置标题
    if font_path:
        try:
            font_prop = fm.FontProperties(fname=font_path, size=12)
            ax.set_title(title_text, fontproperties=font_prop, color=title_color, pad=20)
        except:
            ax.set_title(title_text, color=title_color, pad=20)
    else:
        ax.set_title(title_text, color=title_color, pad=20)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_model_predictions()