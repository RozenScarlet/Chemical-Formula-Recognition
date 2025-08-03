import os
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import platform

from src.crnn import CRNN
from src.lcrnn import LCRNN
from src.msf_lcrnn import MSFLCRNN as MSF_LCRNN
from src.utils import CTCLabelConverter
from src.dataset import ChemicalEquationDataset
from src.equation_formatter import EquationFormatter


def preprocess_image(image_path, target_height=32, max_width=512):
    """处理输入图像"""
    # 读取图像
    image = Image.open(image_path).convert('L')  # 转灰度图
    
    # 调整大小
    w, h = image.size
    target_w = int(w * (target_height / h))
    if target_w > max_width:
        target_w = max_width
    
    image = image.resize((target_w, target_height), Image.LANCZOS)
    
    # 转换为张量
    image = np.array(image) / 255.0  # 归一化
    image = image.reshape(1, 1, target_height, target_w)  # 添加批次和通道维度
    return torch.FloatTensor(image)


def predict(model, image_tensor, converter, device):
    """使用模型预测图像内容"""
    model.eval()
    formatter = EquationFormatter()
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # 前向传播
        outputs = model(image_tensor)
        log_probs = outputs.log_softmax(2)
        
        # 解码预测
        _, preds = log_probs.max(2)
        preds_list = preds.detach().cpu().numpy()
        
        # 使用模型输出的序列长度作为解码长度
        length_for_pred = torch.IntTensor([preds_list.shape[1]] * preds_list.shape[0])
        raw_pred = converter.decode(preds_list, length_for_pred, raw=True)[0]
        formatted_pred = converter.decode(preds_list, length_for_pred)[0]
        
        # 返回原始预测和格式化后的方程式
        unicode_result, html_result = formatter.decode_equation_format(formatted_pred)
        return raw_pred, unicode_result, html_result


def visualize_prediction(image_path, prediction, formatted_prediction, html_prediction=None):
    """可视化预测结果"""
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

    # 读取原始图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 8))
    
    # 显示图像
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title('输入图像')
    plt.axis('off')
    
    # 显示预测结果
    plt.subplot(2, 1, 2)
    plt.text(0.1, 0.7, f"原始编码: {prediction}", fontsize=12)
    plt.text(0.1, 0.5, f"Unicode格式: {formatted_prediction}", fontsize=12)
    if html_prediction:
        plt.text(0.1, 0.3, f"HTML格式: {html_prediction}", fontsize=12)
    plt.title('预测结果')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    output_dir = os.path.normpath('prediction_results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.normpath(os.path.join(output_dir, f"pred_{os.path.basename(image_path)}"))
    plt.savefig(output_path)
    print(f"结果已保存到: {output_path}")
    
    # 显示结果
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='手写化学方程式识别预测脚本')
    
    # 输入参数
    parser.add_argument('--image', type=str, required=True, help='待预测的图像路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--model_type', type=str, default='LCRNN', 
                        choices=['CRNN', 'LCRNN', 'MSF_LCRNN'], help='模型类型')
    parser.add_argument('--classes_file', type=str, default=None, help='字符集文件路径')
    
    # 模型相关参数
    parser.add_argument('--input_channel', type=int, default=1, help='输入通道数')
    parser.add_argument('--output_channel', type=int, default=512, help='特征通道数')
    parser.add_argument('--hidden_size', type=int, default=256, help='RNN隐藏层大小')
    
    args = parser.parse_args()
    
    # 标准化路径
    args.image = os.path.normpath(args.image)
    args.model_path = os.path.normpath(args.model_path)
    
    # 确定classes_file路径
    if args.classes_file is None:
        # 尝试几种可能的路径
        possible_paths = [
            os.path.normpath("../dataset/label/classes.txt"),
            os.path.normpath("dataset/label/classes.txt"),
            os.path.normpath(os.path.join(os.path.dirname(args.model_path), "classes.txt"))
        ]
        
        # 检查当前目录
        if os.path.basename(os.path.abspath('.')) == 'src':
            possible_paths.append(os.path.normpath("../dataset/label/classes.txt"))
        
        # 检测操作系统类型并根据需要添加额外的路径
        is_windows = platform.system() == 'Windows'
        if is_windows:
            possible_paths.append(os.path.normpath("..\\dataset\\label\\classes.txt"))
        
        # 寻找存在的路径
        for path in possible_paths:
            if os.path.exists(path):
                args.classes_file = path
                break
        
        if args.classes_file is None:
            raise FileNotFoundError(f"无法找到字符集文件，请使用--classes_file参数指定路径。尝试过的路径: {possible_paths}")
    else:
        args.classes_file = os.path.normpath(args.classes_file)
    
    print(f"使用字符集文件: {args.classes_file}")
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图像文件不存在: {args.image}")
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    if not os.path.exists(args.classes_file):
        raise FileNotFoundError(f"字符集文件不存在: {args.classes_file}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 读取字符集
    try:
        with open(args.classes_file, 'r', encoding='utf-8') as f:
            character_set = [line.strip() for line in f.readlines() if line.strip()]
        print(f"成功加载了 {len(character_set)} 个字符")
    except Exception as e:
        raise RuntimeError(f"读取字符集文件出错: {str(e)}")
    
    # 创建CTC解码器
    converter = CTCLabelConverter(character_set)
    num_classes = converter.num_classes
    
    # 创建模型
    if args.model_type == 'CRNN':
        model = CRNN(
            input_channel=args.input_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_classes=num_classes
        )
    elif args.model_type == 'LCRNN':
        model = LCRNN(
            input_channel=args.input_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_classes=num_classes
        )
    elif args.model_type == 'MSF_LCRNN':
        model = MSF_LCRNN(
            input_channel=args.input_channel,
            output_channel=args.output_channel,
            hidden_size=args.hidden_size,
            num_classes=num_classes
        )
    else:
        raise ValueError(f'不支持的模型类型: {args.model_type}')
    
    # 加载模型
    try:
        if torch.cuda.is_available():
            checkpoint = torch.load(args.model_path)
        else:
            checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f'已加载模型: {args.model_path}')
    except Exception as e:
        raise RuntimeError(f"加载模型出错: {str(e)}")
    
    # 处理图像
    try:
        image = preprocess_image(args.image)
        print(f"已处理图像: {args.image}, 大小: {image.shape}")
    except Exception as e:
        raise RuntimeError(f"处理图像出错: {str(e)}")
    
    # 预测
    try:
        prediction, unicode_result, html_result = predict(model, image, converter, device)
        
        # 打印结果
        print('\n预测结果:')
        print(f'- 原始编码: {prediction}')
        print(f'- Unicode格式: {unicode_result}')
        print(f'- HTML格式: {html_result}')
        
        # 可视化结果
        visualize_prediction(args.image, prediction, unicode_result, html_result)
    except Exception as e:
        raise RuntimeError(f"预测过程出错: {str(e)}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc() 