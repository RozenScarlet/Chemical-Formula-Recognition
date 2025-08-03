# -*- coding: utf-8 -*-
"""
检查dataset/labels.txt文件中的化学式标注错误
包括：字母后跟数字、离子电荷格式、括号平衡、特殊符号等
"""

import os
import re
import sys
from collections import defaultdict


def check_chemical_formula_errors(labels_file):
    """
    检查化学式标注错误的全面检查函数
    
    标注规则：
    - |3+ 表示上标（离子电荷）
    - _2 表示下标
    - H2 这种字母后直接跟数字是错误的，应该是H_2
    - ^表示气体箭头
    - !表示沉淀箭头
    - ~表示加热条件
    - *表示点燃条件
    - $表示高温条件

    Args:
        labels_file (str): 标签文件路径

    Returns:
        dict: 检查结果
    """
    if not os.path.exists(labels_file):
        print(f"错误: 文件 {labels_file} 不存在")
        return None

    results = {
        'total_lines': 0,
        'problematic_lines': [],
        'error_types': defaultdict(int),
        'unique_patterns': set(),
        'summary': {}
    }
    
    print(f"正在检查文件: {labels_file}")
    print("=" * 50)
    
    # 常见元素符号（部分）
    valid_elements = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'OH', 'NO', 'SO', 'CO', 'PO'  # 常见的原子团
    }
    
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
                    formula = parts[1]
                else:
                    image_path = ""
                    formula = line

                # 检查各种错误类型
                line_errors = []
                
                # 1. 字母后直接跟数字（应该用下标_）
                letter_number_pattern = r'[A-Za-z](\d+)'
                matches = re.finditer(letter_number_pattern, formula)
                for match in matches:
                    # 排除离子电荷标记（如|3+, |2-）
                    start_pos = match.start()
                    if start_pos > 0 and formula[start_pos-1] == '|':
                        continue
                    # 排除已经是下标的情况（_前面的数字）
                    if start_pos > 0 and formula[start_pos-1] == '_':
                        continue
                    line_errors.append(f"字母后直接跟数字: '{match.group()}' 应该是 '{match.group()[0]}_{match.group()[1:]}'")
                    results['error_types']['letter_number'] += 1

                # 2. 检查离子电荷标记错误
                # 正确格式：|3+, |2-, |+, |-
                ion_charge_pattern = r'\|([^+\-\|]*[\+\-])'
                matches = re.finditer(ion_charge_pattern, formula)
                for match in matches:
                    charge_part = match.group(1)
                    # 检查电荷数字和符号的顺序
                    if not re.match(r'^(\d*[\+\-]|[\+\-]\d*)$', charge_part):
                        line_errors.append(f"离子电荷格式错误: '|{charge_part}' - 应该是数字+符号或符号+数字")
                        results['error_types']['ion_charge'] += 1

                # 3. 检查括号平衡
                open_parens = formula.count('(')
                close_parens = formula.count(')')
                if open_parens != close_parens:
                    line_errors.append(f"括号不平衡: 开括号{open_parens}个，闭括号{close_parens}个")
                    results['error_types']['bracket_balance'] += 1

                # 4. 检查特殊符号使用
                valid_symbols = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]_|+=-^!~*$.\\/: ')
                invalid_chars = set(formula) - valid_symbols
                # 过滤掉一些可能的误报字符
                truly_invalid = invalid_chars - {'\\', '/', ':', ' '}
                if truly_invalid:
                    line_errors.append(f"包含无效字符: {truly_invalid}")
                    results['error_types']['invalid_chars'] += 1


                # 6. 检查数字使用错误
                # 下标数字不应该为0
                zero_subscript = re.search(r'_0(?!\d)', formula)
                if zero_subscript:
                    line_errors.append("下标不应该是0")
                    results['error_types']['zero_subscript'] += 1

                # 7. 检查常见的OCR识别错误
                ocr_errors = [
                    ('AI', 'Al', 'Al被误识别为AI'),
                    ('AIO', 'AlO', 'AlO被误识别为AIO'),
                    ('CI', 'Cl', 'Cl被误识别为CI'),
                    ('NI', 'Ni', 'Ni被误识别为NI'),
                    ('SI', 'Si', 'Si被误识别为SI'),
                    ('TI', 'Ti', 'Ti被误识别为TI'),
                    ('LI', 'Li', 'Li被误识别为LI'),
                    ('BI', 'Bi', 'Bi被误识别为BI'),
                    ('ZN', 'Zn', 'Zn被误识别为ZN'),
                    ('MG', 'Mg', 'Mg被误识别为MG'),
                    ('CA', 'Ca', 'Ca被误识别为CA'),
                    ('NA', 'Na', 'Na被误识别为NA'),
                    ('FE', 'Fe', 'Fe被误识别为FE'),
                    ('CU', 'Cu', 'Cu被误识别为CU'),
                    ('AG', 'Ag', 'Ag被误识别为AG'),
                    ('AU', 'Au', 'Au被误识别为AU'),
                    ('PB', 'Pb', 'Pb被误识别为PB'),
                    ('SN', 'Sn', 'Sn被误识别为SN'),
                    ('HG', 'Hg', 'Hg被误识别为HG'),
                    ('MN', 'Mn', 'Mn被误识别为MN'),
                    ('CR', 'Cr', 'Cr被误识别为CR'),
                    ('CO', 'Co', 'Co被误识别为CO'),
                    ('BR', 'Br', 'Br被误识别为BR'),
                    ('SR', 'Sr', 'Sr被误识别为SR'),
                    ('BA', 'Ba', 'Ba被误识别为BA'),
                    ('RU', 'Ru', 'Ru被误识别为RU'),
                    ('RH', 'Rh', 'Rh被误识别为RH'),
                    ('PD', 'Pd', 'Pd被误识别为PD'),
                    ('CD', 'Cd', 'Cd被误识别为CD'),
                    ('IN', 'In', 'In被误识别为IN'),
                    ('SB', 'Sb', 'Sb被误识别为SB'),
                    ('TE', 'Te', 'Te被误识别为TE'),
                    ('CS', 'Cs', 'Cs被误识别为CS'),
                    ('LA', 'La', 'La被误识别为LA'),
                    ('CE', 'Ce', 'Ce被误识别为CE'),
                    ('PR', 'Pr', 'Pr被误识别为PR'),
                    ('ND', 'Nd', 'Nd被误识别为ND'),
                    ('SM', 'Sm', 'Sm被误识别为SM'),
                    ('EU', 'Eu', 'Eu被误识别为EU'),
                    ('GD', 'Gd', 'Gd被误识别为GD'),
                    ('TB', 'Tb', 'Tb被误识别为TB'),
                    ('DY', 'Dy', 'Dy被误识别为DY'),
                    ('HO', 'Ho', 'Ho被误识别为HO'),
                    ('ER', 'Er', 'Er被误识别为ER'),
                    ('TM', 'Tm', 'Tm被误识别为TM'),
                    ('YB', 'Yb', 'Yb被误识别为YB'),
                    ('LU', 'Lu', 'Lu被误识别为LU'),
                    ('HF', 'Hf', 'Hf被误识别为HF'),
                    ('TA', 'Ta', 'Ta被误识别为TA'),
                    ('RE', 'Re', 'Re被误识别为RE'),
                    ('OS', 'Os', 'Os被误识别为OS'),
                    ('IR', 'Ir', 'Ir被误识别为IR'),
                    ('PT', 'Pt', 'Pt被误识别为PT'),
                    ('TL', 'Tl', 'Tl被误识别为TL'),
                    ('RA', 'Ra', 'Ra被误识别为RA'),
                    ('AC', 'Ac', 'Ac被误识别为AC'),
                    ('TH', 'Th', 'Th被误识别为TH'),
                    ('PA', 'Pa', 'Pa被误识别为PA'),
                    ('NP', 'Np', 'Np被误识别为NP'),
                    ('PU', 'Pu', 'Pu被误识别为PU'),
                    ('AM', 'Am', 'Am被误识别为AM'),
                    ('CM', 'Cm', 'Cm被误识别为CM'),
                    ('BK', 'Bk', 'Bk被误识别为BK'),
                    ('CF', 'Cf', 'Cf被误识别为CF'),
                    ('ES', 'Es', 'Es被误识别为ES'),
                    ('FM', 'Fm', 'Fm被误识别为FM'),
                ]
                
                for wrong, correct, msg in ocr_errors:
                    # 特殊处理：CO在化学中是一氧化碳，不是钴的误识别
                    if wrong == 'CO' and ('CO_' in formula or 'CO+' in formula or 'CO=' in formula or 'CO)' in formula or 'CO^' in formula or formula.endswith('CO') or 'CO\\~' in formula or 'CO\\*' in formula or 'CO\\$' in formula):
                        continue  # 这是正确的一氧化碳，不是OCR错误
                    
                    if wrong in formula:
                        line_errors.append(f"OCR错误: {msg}")
                        results['error_types']['ocr_error'] += 1

                # 如果发现错误，记录这一行
                if line_errors:
                    results['problematic_lines'].append({
                        'line_number': line_num,
                        'image_path': image_path,
                        'formula': formula,
                        'full_line': line,
                        'errors': line_errors
                    })
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    # 生成摘要
    results['summary'] = {
        'total_lines': results['total_lines'],
        'problematic_lines_count': len(results['problematic_lines']),
        'problem_rate': len(results['problematic_lines']) / results['total_lines'] if results['total_lines'] > 0 else 0,
        'total_errors': sum(results['error_types'].values())
    }
    
    return results


def print_results(results):
    """打印检查结果"""
    if not results:
        return
    
    print("\n化学式标注错误检查结果:")
    print("=" * 60)
    print(f"总行数: {results['summary']['total_lines']}")
    print(f"有问题的行数: {results['summary']['problematic_lines_count']}")
    print(f"问题比例: {results['summary']['problem_rate']:.2%}")
    print(f"总错误数: {results['summary']['total_errors']}")
    
    if results['error_types']:
        print("\n错误类型统计:")
        print("-" * 40)
        for error_type, count in sorted(results['error_types'].items()):
            error_type_names = {
                'letter_number': '字母后直接跟数字',
                'ion_charge': '离子电荷格式错误',
                'bracket_balance': '括号不平衡',
                'invalid_chars': '包含无效字符',
                'zero_subscript': '下标为0',
                'ocr_error': 'OCR识别错误'
            }
            error_name = error_type_names.get(error_type, error_type)
            print(f"  {error_name}: {count} 次")
    
    if results['problematic_lines']:
        print(f"\n有问题的行详情 (显示前30行):")
        print("-" * 80)
        for i, problem in enumerate(results['problematic_lines'][:30], 1):
            print(f"{i:2d}. 行号 {problem['line_number']:4d}: {problem['formula']}")
            if problem['image_path']:
                print(f"    图片: {problem['image_path']}")
            for error in problem['errors']:
                print(f"    错误: {error}")
            print()

        if len(results['problematic_lines']) > 30:
            print(f"... 还有 {len(results['problematic_lines']) - 30} 行有问题")


def save_detailed_report(results, output_file):
    """保存详细报告到文件"""
    if not results:
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("化学式标注错误检查报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("检查说明:\n")
            f.write("- |3+ 表示上标（离子电荷）\n")
            f.write("- _2 表示下标\n")
            f.write("- H2 这种字母后直接跟数字是错误的，应该是H_2\n")
            f.write("- ^表示气体箭头, !表示沉淀箭头\n")
            f.write("- ~表示加热条件, *表示点燃条件, $表示高温条件\n\n")

            # 摘要信息
            f.write("检查摘要:\n")
            f.write("-" * 30 + "\n")
            f.write(f"总行数: {results['summary']['total_lines']}\n")
            f.write(f"有问题的行数: {results['summary']['problematic_lines_count']}\n")
            f.write(f"问题比例: {results['summary']['problem_rate']:.2%}\n")
            f.write(f"总错误数: {results['summary']['total_errors']}\n\n")

            # 错误类型统计
            if results['error_types']:
                f.write("错误类型统计:\n")
                f.write("-" * 30 + "\n")
                error_type_names = {
                    'letter_number': '字母后直接跟数字',
                    'ion_charge': '离子电荷格式错误',
                    'bracket_balance': '括号不平衡',
                    'invalid_chars': '包含无效字符',
                    'zero_subscript': '下标为0',
                    'ocr_error': 'OCR识别错误'
                }
                for error_type, count in sorted(results['error_types'].items()):
                    error_name = error_type_names.get(error_type, error_type)
                    f.write(f"{error_name}: {count} 次\n")
                f.write("\n")

            # 详细问题列表
            if results['problematic_lines']:
                f.write("有问题的行详情:\n")
                f.write("-" * 80 + "\n")
                for i, problem in enumerate(results['problematic_lines'], 1):
                    f.write(f"{i:3d}. 行号 {problem['line_number']:4d}: {problem['formula']}\n")
                    if problem['image_path']:
                        f.write(f"     图片: {problem['image_path']}\n")
                    for error in problem['errors']:
                        f.write(f"     错误: {error}\n")
                    f.write(f"     完整行: {problem['full_line']}\n\n")

        print(f"\n详细报告已保存到: {output_file}")

    except Exception as e:
        print(f"保存报告时出错: {e}")


def save_error_lines(results, output_file):
    """将有错误的行单独保存到txt文件"""
    if not results or not results['problematic_lines']:
        print("没有错误行需要保存")
        return

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# 包含字母直接跟数字错误格式的行\n")
            f.write(f"# 总共 {len(results['problematic_lines'])} 行\n")
            f.write("# 格式: 图片路径<TAB>标签文本\n\n")

            for problem in results['problematic_lines']:
                # 写入原始的完整行
                f.write(problem['full_line'] + '\n')

        print(f"错误行已保存到: {output_file}")

    except Exception as e:
        print(f"保存错误行时出错: {e}")


def main():
    """主函数"""
    # 默认标签文件路径
    labels_file = "dataset/labels.txt"
    
    # 如果命令行提供了参数，使用提供的路径
    if len(sys.argv) > 1:
        labels_file = sys.argv[1]
    
    print("化学式标注错误检查工具")
    print("=" * 60)
    print(f"检查文件: {labels_file}")
    
    # 执行检查
    results = check_chemical_formula_errors(labels_file)
    
    if results:
        # 打印结果
        print_results(results)
        
        # 保存详细报告
        report_file = "label_check_report.txt"
        save_detailed_report(results, report_file)

        # 保存错误行到单独的文件
        error_lines_file = "error_lines.txt"
        save_error_lines(results, error_lines_file)
        
        # 根据结果给出建议
        if results['summary']['problematic_lines_count'] > 0:
            print(f"\n警告: 发现 {results['summary']['problematic_lines_count']} 行包含标注错误")
            print("\n修正建议:")
            print("1. 字母后数字错误: H2 → H_2, CO2 → CO_2")
            print("2. 离子电荷格式: 应该是 |3+, |2-, |+, |-")
            print("3. 检查括号平衡: () 应该配对")
            print("4. 检查OCR识别错误: O/0, I/1, CI/Cl等")
            print("5. 特殊符号使用: ^气体, !沉淀, ~加热, *点燃, $高温")
        else:
            print("\n✓ 未发现化学式标注错误！数据集质量良好。")
    
    print(f"\n检查完成!")


if __name__ == "__main__":
    main()
