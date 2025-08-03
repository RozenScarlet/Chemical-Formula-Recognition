
import os
import cv2
import numpy as np
import shutil
import glob

# 定义路径
input_image_dir = 'dataset/images'
input_label_dir = 'dataset/label'
output_image_dir = input_image_dir  # 增强后的图像和原始图像放在同一目录
output_label_dir = input_label_dir  # 标签也放在同一目录

# 确保输出目录存在
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 获取所有图片文件
image_files = glob.glob(os.path.join(input_image_dir, '*.jpg'))
print(f"找到 {len(image_files)} 个图片文件")

# 对每个图片应用高斯模糊
for idx, image_path in enumerate(image_files):
    # 读取原始图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图片: {image_path}")
        continue
    
    # 应用半径为1的高斯模糊
    blurred_img = cv2.GaussianBlur(img, (3, 3), 1)
    
    # 生成新的文件名 (原文件名 + "_blurred")
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    blur_suffix = "_blurred"
    new_image_name = f"{name_without_ext}{blur_suffix}.jpg"
    new_image_path = os.path.join(output_image_dir, new_image_name)
    
    # 保存模糊后的图片
    cv2.imwrite(new_image_path, blurred_img)
    
    # 复制对应的标签文件
    label_path = os.path.join(input_label_dir, f"{name_without_ext}.txt")
    if os.path.exists(label_path):
        new_label_name = f"{name_without_ext}{blur_suffix}.txt"
        new_label_path = os.path.join(output_label_dir, new_label_name)
        shutil.copy(label_path, new_label_path)
    else:
        print(f"警告: 找不到标签文件 {label_path}")
    
    # 显示处理进度
    if (idx + 1) % 100 == 0 or idx == len(image_files) - 1:
        print(f"已处理 {idx + 1}/{len(image_files)} 图片")

print("处理完成!")
print(f"增强图片已保存到: {output_image_dir}")
print(f"对应标签已保存到: {output_label_dir}") 