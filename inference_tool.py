"""
化学方程式OCR推理工具
双击运行，自动推理当前目录下的所有图片文件
包含GUI界面和进度显示
"""

import os
import sys
import glob
import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from datetime import datetime

# 添加src目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from transformer_cnn_net import create_transformer_cnn
    from dataset import build_character_set, CTCLabelConverter
    from utils import setup_seed
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保src目录下的相关文件存在")


class InferenceTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("化学方程式OCR推理工具")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # 设置图标（如果存在）
        try:
            icon_path = os.path.join(current_dir, "icon.ico")
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
        except:
            pass
            
        self.model = None
        self.label_converter = None
        self.device = None
        self.transform = None
        
        self.setup_ui()
        self.load_model()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="化学方程式OCR推理工具", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 状态区域
        status_frame = ttk.LabelFrame(main_frame, text="状态信息", padding="10")
        status_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        ttk.Label(status_frame, text="模型状态:").grid(row=0, column=0, sticky=tk.W)
        self.model_status = ttk.Label(status_frame, text="加载中...", foreground="orange")
        self.model_status.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(status_frame, text="设备:").grid(row=1, column=0, sticky=tk.W)
        self.device_label = ttk.Label(status_frame, text="检测中...")
        self.device_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # 操作按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.start_button = ttk.Button(button_frame, text="开始推理当前目录图片", 
                                      command=self.start_inference, state="disabled")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(button_frame, text="清空结果", 
                                      command=self.clear_results)
        self.clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_button = ttk.Button(button_frame, text="保存结果", 
                                     command=self.save_results, state="disabled")
        self.save_button.pack(side=tk.LEFT)
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                           maximum=100, length=300)
        self.progress_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_label = ttk.Label(main_frame, text="")\n        self.progress_label.grid(row=4, column=0, columnspan=3)
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="推理结果", padding="10")
        result_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 结果文本框（带滚动条）
        self.result_text = scrolledtext.ScrolledText(result_frame, height=15, width=80,
                                                    font=("Consolas", 10))
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 存储结果数据
        self.results_data = []
        
    def load_model(self):
        """加载模型"""
        try:
            # 检测设备
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            device_info = f"{self.device}"
            if torch.cuda.is_available():
                device_info += f" ({torch.cuda.get_device_name()})"
            self.device_label.config(text=device_info)
            
            # 查找最佳模型文件
            model_paths = [
                os.path.join(current_dir, "checkpoints", "enhanced_training", "best_acc_model.pth"),
                os.path.join(current_dir, "checkpoints", "test", "best_acc_model.pth"),
                os.path.join(current_dir, "checkpoints", "transformer_cnn", "best_model.pth"),
                os.path.join(current_dir, "best_model.pth"),
                os.path.join(current_dir, "model.pth")
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
                    
            if model_path is None:
                raise FileNotFoundError("未找到模型文件")
                
            # 构建字符集
            data_dir = os.path.join(current_dir, "dataset")
            if not os.path.exists(data_dir):
                # 如果没有dataset目录，使用预定义字符集
                character_set = self.get_default_character_set()
            else:
                character_set = build_character_set(data_dir)
                
            num_classes = len(character_set) + 3
            
            # 创建标签转换器
            self.label_converter = CTCLabelConverter(character_set)
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 从检查点获取模型参数
            if 'args' in checkpoint:
                args = checkpoint['args']
                model_args = {
                    'num_classes': num_classes,
                    'rnn_type': args.get('rnn_type', 'lstm'),
                    'hidden_size': args.get('hidden_size', 256),
                    'nhead': args.get('nhead', 8),
                    'num_layers': args.get('num_layers', 6),
                    'dropout': args.get('dropout', 0.3)
                }
            else:
                # 默认参数
                model_args = {
                    'num_classes': num_classes,
                    'rnn_type': 'lstm',
                    'hidden_size': 256,
                    'nhead': 8,
                    'num_layers': 6,
                    'dropout': 0.3
                }
            
            # 创建模型
            self.model = create_transformer_cnn(**model_args)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # 创建数据变换
            self.transform = transforms.Compose([
                transforms.Resize((64, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.model_status.config(text="模型加载成功", foreground="green")
            self.start_button.config(state="normal")
            
            # 显示模型信息
            model_info = f"模型文件: {os.path.basename(model_path)}\\n"
            model_info += f"字符集大小: {len(character_set)}\\n"
            model_info += f"设备: {self.device}\\n"
            self.result_text.insert(tk.END, model_info + "\\n")
            
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            self.model_status.config(text="加载失败", foreground="red")
            messagebox.showerror("错误", error_msg)
            self.result_text.insert(tk.END, error_msg + "\\n")
            
    def get_default_character_set(self):
        """获取默认字符集（如果没有dataset目录）"""
        return [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '+', '-', '=', '(', ')', '[', ']', '_', '|', '!', '^', '\\\\', '$', '~', '*'
        ]
        
    def find_image_files(self):
        """查找当前目录下的所有图片文件"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.gif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(current_dir, ext)))
            image_files.extend(glob.glob(os.path.join(current_dir, ext.upper())))
            
        return sorted(image_files)
        
    def predict_image(self, image_path):
        """预测单张图片"""
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            
            # 应用变换
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(image_tensor)
                if isinstance(outputs, tuple):
                    preds = outputs[0]  # 只取主要输出
                else:
                    preds = outputs
                    
                # 转换维度 (N, T, C) -> (T, N, C)
                preds = preds.permute(1, 0, 2)
                
                # 应用softmax并解码
                probs = F.softmax(preds, dim=2)
                _, indices = torch.max(probs, dim=2)
                
                # CTC解码
                indices = indices.squeeze(1)  # 移除batch维度
                pred_text = self.label_converter.ctc_greedy_decode(indices)
                
                return pred_text, True
                
        except Exception as e:
            return f"推理失败: {str(e)}", False
            
    def start_inference(self):
        """开始推理"""
        if self.model is None:
            messagebox.showerror("错误", "模型未加载")
            return
            
        # 查找图片文件
        image_files = self.find_image_files()
        
        if not image_files:
            messagebox.showinfo("提示", "当前目录下没有找到图片文件")
            return
            
        # 清空之前的结果
        self.results_data = []
        self.result_text.delete(1.0, tk.END)
        
        # 禁用按钮
        self.start_button.config(state="disabled")
        
        # 开始推理
        total_files = len(image_files)
        success_count = 0
        
        self.result_text.insert(tk.END, f"开始推理 {total_files} 个图片文件...\\n\\n")
        self.root.update()
        
        start_time = time.time()
        
        for i, image_path in enumerate(image_files):
            # 更新进度
            progress = (i / total_files) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"正在处理: {os.path.basename(image_path)} ({i+1}/{total_files})")
            self.root.update()
            
            # 推理
            pred_text, success = self.predict_image(image_path)
            
            # 记录结果
            result = {
                'file_name': os.path.basename(image_path),
                'file_path': image_path,
                'prediction': pred_text,
                'success': success,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.results_data.append(result)
            
            # 显示结果
            status_icon = "✓" if success else "✗"
            result_line = f"{status_icon} {os.path.basename(image_path)}: {pred_text}\\n"
            self.result_text.insert(tk.END, result_line)
            self.result_text.see(tk.END)
            
            if success:
                success_count += 1
                
        # 完成
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        self.progress_var.set(100)
        self.progress_label.config(text="推理完成!")
        
        # 显示统计信息
        summary = f"\\n{'='*50}\\n"
        summary += f"推理完成!\\n"
        summary += f"总文件数: {total_files}\\n"
        summary += f"成功推理: {success_count}\\n"
        summary += f"失败数量: {total_files - success_count}\\n"
        summary += f"成功率: {success_count/total_files*100:.1f}%\\n"
        summary += f"用时: {elapsed_time:.2f}秒\\n"
        summary += f"平均速度: {total_files/elapsed_time:.1f}张/秒\\n"
        summary += f"{'='*50}\\n"
        
        self.result_text.insert(tk.END, summary)
        self.result_text.see(tk.END)
        
        # 重新启用按钮
        self.start_button.config(state="normal")
        self.save_button.config(state="normal")
        
        messagebox.showinfo("完成", f"推理完成!\\n成功: {success_count}/{total_files}")
        
    def clear_results(self):
        """清空结果"""
        self.result_text.delete(1.0, tk.END)
        self.results_data = []
        self.progress_var.set(0)
        self.progress_label.config(text="")
        self.save_button.config(state="disabled")
        
    def save_results(self):
        """保存结果到文件"""
        if not self.results_data:
            messagebox.showwarning("警告", "没有结果可保存")
            return
            
        try:
            # 保存JSON格式
            json_file = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.results_data, f, ensure_ascii=False, indent=2)
                
            # 保存TXT格式
            txt_file = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write("化学方程式OCR推理结果\\n")
                f.write("="*50 + "\\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"总文件数: {len(self.results_data)}\\n\\n")
                
                for result in self.results_data:
                    status = "成功" if result['success'] else "失败"
                    f.write(f"文件: {result['file_name']}\\n")
                    f.write(f"状态: {status}\\n")
                    f.write(f"预测: {result['prediction']}\\n")
                    f.write("-" * 30 + "\\n")
                    
            messagebox.showinfo("保存成功", f"结果已保存到:\\n{json_file}\\n{txt_file}")
            
        except Exception as e:
            messagebox.showerror("保存失败", f"保存文件时出错: {str(e)}")
            
    def run(self):
        """运行应用"""
        self.root.mainloop()


def main():
    """主函数"""
    # 设置随机种子
    setup_seed(42)
    
    # 创建并运行应用
    app = InferenceTool()
    app.run()


if __name__ == "__main__":
    main()