# -*- coding: utf-8 -*-
"""
统计论文中的数学公式（纯文本格式）
通过数学符号和公式特征来识别
"""
from docx import Document
import re

doc_path = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v34.docx'
doc = Document(doc_path)

# 数学符号集合
math_symbols = set('βλρΔθαπμσΩ∑∫√∞±≈≤≥×÷∂∇∈∉⊂⊃∪∩∀∃¬∧∨→←↔⇒⇐⇔∝≡≠∼∽')
# 额外的公式特征词
formula_keywords = ['AP50', 'AP@', 'mAP', 'IoU', 'AUC', 'CI=', 'p=', 'ρ=', 
                    'β=', 'λ=', 'α=', 'γ=', 'k_', 's_', 'p_', 'f_', 'g_',
                    'R(c)', 'R_max', 'ΔAUC', 'ΔAP', '+0.', '-0.', '±0.',
                    'N=', 'n=', 'K=', 'k=', '11点', '101点']

formula_paragraphs = []
total_formulas = 0

print("=" * 70)
print("含数学公式的段落统计")
print("=" * 70)

for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if not text:
        continue
    
    # 检查是否包含数学符号
    has_math_symbol = any(c in math_symbols for c in text)
    # 检查是否包含公式关键词
    has_formula_keyword = any(kw in text for kw in formula_keywords)
    # 检查是否有上下标模式（如 x_y 或 x^y）
    has_sub_super = bool(re.search(r'[a-zA-Z]_[a-zA-Z0-9]', text)) or bool(re.search(r'[a-zA-Z]\^[a-zA-Z0-9]', text))
    # 检查是否有等式模式
    has_equation = '=' in text and any(c in text for c in 'βλρΔθαπμσ+-±×÷')
    
    if has_math_symbol or has_formula_keyword or has_sub_super or has_equation:
        # 估算该段落中的公式数量
        formula_count = 0
        if has_math_symbol:
            formula_count += 1
        if has_equation:
            formula_count += 1
        if has_sub_super and not has_equation:
            formula_count += 1
        
        formula_paragraphs.append((i+1, formula_count, text[:80]))
        total_formulas += formula_count

print(f"\n含公式的段落数: {len(formula_paragraphs)}")
print(f"估算公式总数: {total_formulas}")

print("\n" + "=" * 70)
print("含公式的段落列表")
print("=" * 70)
for para_num, count, text in formula_paragraphs:
    print(f"\n段落{para_num} (估算{count}个公式):")
    print(f"  {text}...")

# 统计表格中的公式
print("\n" + "=" * 70)
print("表格中的公式统计")
print("=" * 70)

table_formula_cells = 0
for t_idx, table in enumerate(doc.tables):
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            text = cell.text.strip()
            if not text:
                continue
            has_math = any(c in math_symbols for c in text)
            has_kw = any(kw in text for kw in formula_keywords)
            if has_math or has_kw:
                table_formula_cells += 1

print(f"含公式/数学符号的表格单元格数: {table_formula_cells}")

# 总结
print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print(f"正文含公式的段落: {len(formula_paragraphs)}")
print(f"正文估算公式数: {total_formulas}")
print(f"表格含数学符号的单元格: {table_formula_cells}")
print(f"\n注意：公式为纯文本格式（非Word公式编辑器），统计为估算值")
print(f"如果按'独立公式行'严格统计，约 {len(formula_paragraphs)} 个公式段落")
