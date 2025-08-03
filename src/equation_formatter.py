#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
化学方程式格式转换工具类
用于将一维编码的化学方程式转换为标准显示格式和HTML格式
"""

class EquationFormatter:
    """
    化学方程式格式转换器
    支持将一维编码的化学方程式转换为Unicode格式和HTML格式
    """
    
    def __init__(self):
        # 特殊反应符号映射
        self.special_symbols = {
            "^": "↑",      # 气体
            "!": "↓",      # 沉淀
            "\<>": "⇌",     # 可逆反应
            "\->": "→",     # 箭头
            "\~=": "△→",   # 加热
            "\*=": "点燃→",  # 点燃
            "\&=": "通电→",  # 通电
            "\$=": "高温→",  # 高温
            "\*>": "点燃→",  # 点燃（另一种编码）
            "\@=": "光→",    # 光照
        }
        
        # 下标数字映射
        self.subscript_digits = {
            '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
            '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
        }
        
        # 上标数字映射（用于离子电荷）
        self.superscript_digits = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'
        }
    
    def to_unicode(self, text):
        """
        将一维编码的化学方程式转换为Unicode格式
        支持下标、上标（离子电荷）和特殊反应符号
        
        Args:
            text (str): 一维编码的化学方程式
            
        Returns:
            str: Unicode格式的化学方程式
        """
        result = ""
        i = 0
        
        while i < len(text):
            # 处理特殊反应符号 (优先级最高)
            # 先检查带反斜杠的三字符特殊符号
            if i + 2 < len(text) and text[i] == '\\' and text[i:i+3] in self.special_symbols:
                result += self.special_symbols[text[i:i+3]]
                i += 3
                continue
            # 再检查带反斜杠的双字符特殊符号
            elif i + 1 < len(text) and text[i] == '\\' and text[i:i+2] in self.special_symbols:
                result += self.special_symbols[text[i:i+2]]
                i += 2
                continue
            # 再检查常规双字符特殊符号
            elif i + 1 < len(text) and text[i:i+2] in self.special_symbols:
                result += self.special_symbols[text[i:i+2]]
                i += 2
                continue
            # 最后检查单字符特殊符号
            elif i < len(text) and text[i] in self.special_symbols:
                result += self.special_symbols[text[i]]
                i += 1
                continue
            
            # 处理下标 (H_2O -> H₂O)
            if i + 1 < len(text) and text[i+1] == '_':
                element = text[i]
                j = i + 2
                subscript = ""
                while j < len(text) and text[j].isdigit():
                    subscript += self.subscript_digits.get(text[j], text[j])
                    j += 1
                
                result += element + subscript
                i = j
                
            # 处理离子符号 (OH|- -> OH⁻, Mg|2+ -> Mg²⁺)
            elif i + 1 < len(text) and text[i] == '|':
                j = i + 1
                charge = ""
                charge_value = ""
                charge_sign = ""
                
                # 先检查是否有数字
                while j < len(text) and text[j].isdigit():
                    charge_value += text[j]
                    j += 1
                
                # 再检查是否有符号
                if j < len(text) and (text[j] == '+' or text[j] == '-'):
                    charge_sign = text[j]
                    j += 1
                else:
                    # 如果没有找到符号，跳过这个字符
                    result += text[i]
                    if charge_value:  # 如果有解析到数字，也添加进去
                        result += charge_value
                    i = j if j > i + 1 else i + 1
                    continue
                
                # 构建上标表示
                if charge_value:
                    # 转换为上标数字
                    for digit in charge_value:
                        charge += self.superscript_digits.get(digit, digit)
                
                # 添加电荷符号
                charge += "⁺" if charge_sign == "+" else "⁻"
                
                result += charge
                i = j
                
            # 常规字符
            else:
                result += text[i]
                i += 1
                
        return result
    
    def to_html(self, text):
        """
        将一维编码的化学方程式转换为HTML格式，用于网页显示
        
        Args:
            text (str): 一维编码的化学方程式
            
        Returns:
            str: HTML格式的化学方程式
        """
        result = ""
        i = 0
        
        while i < len(text):
            # 处理特殊反应符号 (优先级最高)
            # 先检查带反斜杠的三字符特殊符号
            if i + 2 < len(text) and text[i] == '\\' and text[i:i+3] in self.special_symbols:
                result += self.special_symbols[text[i:i+3]]
                i += 3
                continue
            # 再检查带反斜杠的双字符特殊符号
            elif i + 1 < len(text) and text[i] == '\\' and text[i:i+2] in self.special_symbols:
                result += self.special_symbols[text[i:i+2]]
                i += 2
                continue
            # 再检查常规双字符特殊符号
            elif i + 1 < len(text) and text[i:i+2] in self.special_symbols:
                result += self.special_symbols[text[i:i+2]]
                i += 2
                continue
            # 最后检查单字符特殊符号
            elif i < len(text) and text[i] in self.special_symbols:
                result += self.special_symbols[text[i]]
                i += 1
                continue
            
            # 处理下标
            if i + 1 < len(text) and text[i+1] == '_':
                element = text[i]
                j = i + 2
                subscript = ""
                while j < len(text) and text[j].isdigit():
                    subscript += text[j]
                    j += 1
                
                result += f"{element}<sub>{subscript}</sub>"
                i = j
            
            # 处理离子电荷
            elif i + 1 < len(text) and text[i] == '|':
                j = i + 1
                charge_sign = ""
                charge_value = ""
                
                # 先检查是否有数字
                while j < len(text) and text[j].isdigit():
                    charge_value += text[j]
                    j += 1
                
                # 再检查是否有符号
                if j < len(text) and (text[j] == '+' or text[j] == '-'):
                    charge_sign = text[j]
                    j += 1
                else:
                    # 如果没有找到符号，跳过这个字符
                    result += text[i]
                    if charge_value:  # 如果有解析到数字，也添加进去
                        result += charge_value
                    i = j if j > i + 1 else i + 1
                    continue
                
                # 生成HTML上标，包括数字和符号
                result += f"<sup>{charge_value if charge_value else ''}{charge_sign}</sup>"
                i = j
            
            # 常规字符
            else:
                result += text[i]
                i += 1
                
        return result
    
    def decode_equation_format(self, text):
        """
        解码预测的一维化学方程式格式为Unicode和HTML格式
        
        Args:
            text (str): 一维编码的化学方程式
            
        Returns:
            tuple: (Unicode格式的化学方程式, HTML格式的化学方程式)
        """
        return self.to_unicode(text), self.to_html(text)

    def from_html_to_onedim(self, html_text):
        """
        将HTML格式的化学方程式转换为一维编码格式
        
        Args:
            html_text (str): HTML格式的化学方程式
            
        Returns:
            str: 一维编码格式的化学方程式
        """
        # 替换HTML下标标签
        import re
        
        # 处理下标 <sub>n</sub> -> _n
        text = re.sub(r'<sub>(\d+)</sub>', r'_\1', html_text)
        
        # 处理上标离子 <sup>n+</sup> -> |n+, <sup>n-</sup> -> |n-
        text = re.sub(r'<sup>(\d*)([+-])</sup>', r'|\1\2', text)
        
        # 处理特殊符号（反向映射）
        # 手动定义特殊反映射关系
        special_reverse_mapping = {
            "点燃→": "\\*=",
            "通电→": "\\&=",
            "高温→": "\\$=",
            "△→": "\\~=",
            "光→": "\\@=",
            "→": "\\->",
            "⇌": "\\<>",
            "↑": "^",
            "↓": "!"
        }
        
        # 按长度排序，确保先替换较长的符号
        for symbol, code in sorted(special_reverse_mapping.items(), key=lambda x: len(x[0]), reverse=True):
            if symbol in text:
                text = text.replace(symbol, code)
        
        return text


def test_formatter():
    """测试格式转换器"""
    formatter = EquationFormatter()
    
    # 测试用例
    test_cases = [
        "H_2O",  # 下标
        "OH|-",  # 离子（单电荷）
        "Mg|2+", # 离子（多电荷）
        "CaCO_3!",  # 下标和沉淀符号
        "H_2^",  # 气体
        "2Na+2H_2O+CuSO_4=Cu(OH)_2!+Na_2SO_4+H_2^",  # 完整化学方程式
        "Ca|2++2OH|-+2HCO_3|-=CaCO_3!+2H_2O+CO_2^",   # 完整离子方程式
        "N_2+3H_2\<>2NH_3",  # 可逆反应
        "CH_4+2O_2\*=CO_2+2H_2O",  # 点燃反应
        "2H_2O\&=2H_2^+O_2^",  # 通电反应
        "CaCO_3\$=CaO+CO_2^",  # 高温反应
        "Fe_2O_3\~=Fe_3O_4",   # 加热反应
        "H_2+Cl_2\@=2HCl",     # 光照反应
        "Na+H_2O\->NaOH+H_2^"  # 箭头
    ]
    
    print("===== 测试一维编码转Unicode和HTML =====")
    for test_case in test_cases:
        unicode_result = formatter.to_unicode(test_case)
        html_result = formatter.to_html(test_case)
        print(f"原始: {test_case}")
        print(f"Unicode: {unicode_result}")
        print(f"HTML: {html_result}")
        print("-" * 40)
    
    print("\n===== 测试HTML转一维编码 =====")
    # 测试HTML转一维编码
    html_test_cases = [
        "H<sub>2</sub>O",
        "OH<sup>-</sup>",
        "Mg<sup>2+</sup>",
        "CaCO<sub>3</sub>↓",
        "H<sub>2</sub>↑",
        "2Na+2H<sub>2</sub>O+CuSO<sub>4</sub>=Cu(OH)<sub>2</sub>↓+Na<sub>2</sub>SO<sub>4</sub>+H<sub>2</sub>↑",
        "N<sub>2</sub>+3H<sub>2</sub>⇌2NH<sub>3</sub>",
        "CH<sub>4</sub>+2O<sub>2</sub>点燃→CO<sub>2</sub>+2H<sub>2</sub>O",
        "CaCO<sub>3</sub>高温→CaO+CO<sub>2</sub>↑",
        "2H<sub>2</sub>O通电→2H<sub>2</sub>↑+O<sub>2</sub>↑",
        "Fe<sub>2</sub>O<sub>3</sub>△→Fe<sub>3</sub>O<sub>4</sub>",
        "H<sub>2</sub>+Cl<sub>2</sub>光→2HCl",
        "Na+H<sub>2</sub>O→NaOH+H<sub>2</sub>↑"
    ]
    
    for html_case in html_test_cases:
        onedim_result = formatter.from_html_to_onedim(html_case)
        print(f"HTML: {html_case}")
        print(f"一维编码: {onedim_result}")
        print(f"再次转Unicode: {formatter.to_unicode(onedim_result)}")
        print("-" * 40)


if __name__ == "__main__":
    test_formatter() 