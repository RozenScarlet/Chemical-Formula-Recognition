#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查dataset/labels.txt文件中的化学符号错误
主要检查常见的OCR识别错误，如 AI → Al, CI → Cl 等
"""

import os
import re
import sys
from collections import defaultdict


def check_chemical_symbol_errors(labels_file):
    """
    检查标签文件中的化学符号错误
    
    Args:
        labels_file (str): 标签文件路径
    
    Returns:
        dict: 检查结果
    """
    if not os.path.exists(labels_file):
        print(f"错误: 文件 {labels_file} 不存在")
        return None
    
    # 常见的化学符号OCR错误模式
    # 格式: (错误模式, 正确符号, 描述)
    error_patterns = [
        (r'\bAI\b', 'Al', '铝符号错误：AI → Al'),  # 铝：AI错误写成Al
        (r'\bCI\b', 'Cl', '氯符号错误：CI → Cl'),  # 氯：CI错误写成Cl  
        (r'\bCa\b', 'Ca', '钙符号检查'),  # 钙符号检查（可能被写成Ca）
        (r'\bMg\b', 'Mg', '镁符号检查'),  # 镁符号检查
        (r'\bNa\b', 'Na', '钠符号检查'),  # 钠符号检查
        (r'\bK\b', 'K', '钾符号检查'),    # 钾符号检查
        (r'\bFe\b', 'Fe', '铁符号检查'), # 铁符号检查
        (r'\bCu\b', 'Cu', '铜符号检查'), # 铜符号检查
        (r'\bZn\b', 'Zn', '锌符号检查'), # 锌符号检查
        (r'\bAg\b', 'Ag', '银符号检查'), # 银符号检查
        (r'\bPb\b', 'Pb', '铅符号检查'), # 铅符号检查
        (r'\bHg\b', 'Hg', '汞符号检查'), # 汞符号检查
    ]
    
    # 检查确定的化学符号错误（OCR常见错误）
    actual_errors = {
        'AI': 'Al',  # 铝：大写I错误写成小写l
        'CI': 'Cl',  # 氯：大写I错误写成小写l
    }
    
    results = {
        'total_lines': 0,
        'problematic_lines': [],
        'error_counts': defaultdict(int),
        'unique_errors': defaultdict(set),
        'summary': {}
    }
    
    print(f"正在检查文件: {labels_file}")
    print("检查化学符号错误...")
    print("=" * 50)
    
    try:
        with open(labels_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                results['total_lines'] += 1
                
                # 分割每行，通常格式是: 图片路径 标签文本
                parts = line.split('\t')
                if len(parts) >= 2:
                    image_path = parts[0]
                    label_text = parts[1]
                else:
                    # 如果没有tab分隔，尝试空格分隔
                    parts = line.split(' ', 1)
                    if len(parts) >= 2:
                        image_path = parts[0]
                        label_text = parts[1]
                    else:
                        image_path = ""
                        label_text = line
                
                # 检查确定的错误模式
                found_errors = []

                for wrong_symbol, correct_symbol in actual_errors.items():
                    # 使用单词边界确保完整匹配
                    pattern = r'\b' + re.escape(wrong_symbol) + r'\b'
                    matches = re.findall(pattern, label_text)
                    
                    if matches:
                        error_type = f"{wrong_symbol} → {correct_symbol}"
                        found_errors.append({
                            'wrong': wrong_symbol,
                            'correct': correct_symbol,
                            'count': len(matches),
                            'description': error_type
                        })
                        
                        results['error_counts'][error_type] += len(matches)
                        results['unique_errors'][error_type].add(label_text)
                
                # 如果发现错误，记录这一行
                if found_errors:
                    results['problematic_lines'].append({
                        'line_number': line_num,
                        'image_path': image_path,
                        'label_text': label_text,
                        'full_line': line,
                        'errors': found_errors
                    })
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    # 生成摘要
    results['summary'] = {
        'total_lines': results['total_lines'],
        'problematic_lines_count': len(results['problematic_lines']),
        'problem_rate': len(results['problematic_lines']) / results['total_lines'] if results['total_lines'] > 0 else 0,
        'total_errors': sum(results['error_counts'].values())
    }
    
    return results


def print_results(results):
    """打印检查结果"""
    if not results:
        return
    
    print("\n化学符号错误检查结果:")
    print("=" * 50)
    print(f"总行数: {results['summary']['total_lines']}")
    print(f"有错误的行数: {results['summary']['problematic_lines_count']}")
    print(f"错误比例: {results['summary']['problem_rate']:.2%}")
    print(f"总错误数量: {results['summary']['total_errors']}")
    
    if results['error_counts']:
        print("\n错误类型统计:")
        print("-" * 30)
        for error_type, count in results['error_counts'].items():
            print(f"{error_type}: {count} 次")
    
    if results['problematic_lines']:
        print(f"\n有错误的行详情:")
        print("-" * 60)
        for i, problem in enumerate(results['problematic_lines'], 1):
            print(f"{i:2d}. 行号 {problem['line_number']:4d}: {problem['label_text']}")
            if problem['image_path']:
                print(f"    图片: {problem['image_path']}")
            
            # 显示具体错误
            for error in problem['errors']:
                print(f"    错误: {error['description']} (出现{error['count']}次)")
        
        print(f"\n总共发现 {len(results['problematic_lines'])} 行包含化学符号错误")


def save_error_lines(results, output_file):
    """将有错误的行保存到txt文件"""
    if not results or not results['problematic_lines']:
        print("没有错误行需要保存")
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 包含化学符号错误的行\n")
            f.write(f"# 总共 {len(results['problematic_lines'])} 行\n")
            f.write("# 格式: 图片路径<TAB>标签文本\n")
            f.write("# 主要错误: AI → Al, CI → Cl\n\n")
            
            for problem in results['problematic_lines']:
                # 写入原始的完整行
                f.write(problem['full_line'] + '\n')
        
        print(f"化学符号错误行已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存错误行时出错: {e}")


def save_detailed_report(results, output_file):
    """保存详细报告到文件"""
    if not results:
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("化学符号错误检查报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 摘要信息
            f.write("检查摘要:\n")
            f.write(f"总行数: {results['summary']['total_lines']}\n")
            f.write(f"有错误的行数: {results['summary']['problematic_lines_count']}\n")
            f.write(f"错误比例: {results['summary']['problem_rate']:.2%}\n")
            f.write(f"总错误数量: {results['summary']['total_errors']}\n\n")
            
            # 错误统计
            if results['error_counts']:
                f.write("错误类型统计:\n")
                f.write("-" * 30 + "\n")
                for error_type, count in results['error_counts'].items():
                    f.write(f"{error_type}: {count} 次\n")
                f.write("\n")
            
            # 详细错误列表
            if results['problematic_lines']:
                f.write("有错误的行详情:\n")
                f.write("-" * 60 + "\n")
                for i, problem in enumerate(results['problematic_lines'], 1):
                    f.write(f"{i:3d}. 行号 {problem['line_number']:4d}: {problem['label_text']}\n")
                    if problem['image_path']:
                        f.write(f"     图片: {problem['image_path']}\n")
                    
                    # 显示具体错误
                    for error in problem['errors']:
                        f.write(f"     错误: {error['description']} (出现{error['count']}次)\n")
                    
                    f.write(f"     完整行: {problem['full_line']}\n\n")
        
        print(f"详细报告已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存报告时出错: {e}")


def main():
    """主函数"""
    # 默认标签文件路径
    labels_file = "dataset/labels.txt"
    
    # 如果命令行提供了参数，使用提供的路径
    if len(sys.argv) > 1:
        labels_file = sys.argv[1]
    
    print("化学符号错误检查工具")
    print("=" * 50)
    print(f"检查文件: {labels_file}")
    print("主要检查: AI → Al, CI → Cl 等OCR错误")
    
    # 执行检查
    results = check_chemical_symbol_errors(labels_file)
    
    if results:
        # 打印结果
        print_results(results)
        
        # 保存详细报告
        report_file = "chemical_symbol_report.txt"
        save_detailed_report(results, report_file)
        
        # 保存错误行到单独的文件
        error_lines_file = "chemical_symbol_errors.txt"
        save_error_lines(results, error_lines_file)
        
        # 根据结果给出建议
        if results['summary']['problematic_lines_count'] > 0:
            print(f"\n⚠️  发现 {results['summary']['problematic_lines_count']} 行包含化学符号错误")
            print("建议修正:")
            for error_type, count in results['error_counts'].items():
                print(f"- {error_type}: {count} 处")
            print("\n这些通常是OCR识别错误，需要手动修正")
        else:
            print("\n✅ 未发现化学符号错误")
    
    print(f"\n检查完成!")


if __name__ == "__main__":
    main()
