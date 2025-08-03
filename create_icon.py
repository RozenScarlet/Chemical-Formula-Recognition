"""
创建一个简单的图标文件
如果没有专业图标，这个脚本可以生成一个基本的ICO文件
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_icon():
    """创建应用图标"""
    # 创建一个64x64的图像
    size = 64
    img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # 绘制背景圆形
    margin = 4
    draw.ellipse([margin, margin, size-margin, size-margin], 
                 fill=(52, 152, 219, 255), outline=(41, 128, 185, 255), width=2)
    
    # 绘制化学符号 "OCR"
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # 绘制文本
    text = "OCR"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    text_x = (size - text_width) // 2
    text_y = (size - text_height) // 2 - 2
    
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    
    # 保存为ICO文件
    img.save('icon.ico', format='ICO', sizes=[(64, 64), (32, 32), (16, 16)])
    print("✓ icon.ico 创建成功")

if __name__ == "__main__":
    create_icon()