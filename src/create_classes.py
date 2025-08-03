#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从标签文件中提取字符集
"""

import os
import sys
import argparse

def extract_unique_chars(label_file):
    """
    从标签文件中提取唯一的字符集
    """
    unique_chars = set()
    
    try:
        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) > 1:
                    # 第二部分是标签文本
                    text = parts[1]
                    for char in text:
                        unique_chars.add(char)
        
        # 按照字符排序
        return sorted(list(unique_chars))
    
    except Exception as e:
        print(f"处理标签文件时出错: {e}")
        return []

def create_classes_file(label_file, output_file, force=False):
    """
    创建字符集文件
    
    参数:
        label_file: 输入的标签文件路径
        output_file: 输出的字符集文件路径
        force: 是否强制覆盖已存在的文件
    """
    # 确保输入文件存在
    if not os.path.exists(label_file):
        print(f"错误: 标签文件不存在 - {label_file}")
        return False
    
    # 检查输出文件是否已存在
    if os.path.exists(output_file) and not force:
        print(f"警告: 字符集文件已存在 - {output_file}")
        print("如果要覆盖现有文件，请使用 --force 参数")
        return False
    
    # 提取唯一字符集
    print(f"从标签文件中提取字符集: {label_file}")
    chars = extract_unique_chars(label_file)
    
    if not chars:
        print("错误: 未能从标签文件中提取任何字符")
        return False
    
    print(f"提取了 {len(chars)} 个唯一字符")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建目录: {output_dir}")
    
    # 写入字符集文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for char in chars:
                f.write(char + '\n')
        
        print(f"字符集文件已保存: {output_file}")
        return True
    
    except Exception as e:
        print(f"保存字符集文件时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="从标签文件中提取字符集")
    parser.add_argument('label_file', help='输入的标签文件路径')
    parser.add_argument('output_file', help='输出的字符集文件路径')
    parser.add_argument('--force', action='store_true', help='强制覆盖已存在的文件')
    
    args = parser.parse_args()
    
    # 检查输出文件是否是classes.txt
    output_basename = os.path.basename(args.output_file)
    if output_basename == "classes.txt" and os.path.exists(args.output_file) and not args.force:
        print(f"错误: 禁止覆盖 {output_basename} 文件，除非明确指定 --force 参数。")
        print("这个保护机制是为了防止意外覆盖现有的字符映射文件。")
        print("如果您确认要覆盖此文件，请使用 --force 参数。")
        return 1
    
    success = create_classes_file(args.label_file, args.output_file, args.force)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 