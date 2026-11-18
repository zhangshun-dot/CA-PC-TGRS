# -*- coding: utf-8 -*-
"""
提取论文中的核心数学公式段落
"""
from docx import Document

doc_path = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v34.docx'
doc = Document(doc_path)

# 数学符号集合
math_symbols = set('βλρΔθαπμσΩ∑∫√∞±≈≤≥×÷∂∇∈∉⊂⊃∪∩∀∃¬∧∨→←↔⇒⇐⇔∝≡≠∼∽')

print("=" * 80)
print("论文中含数学公式的段落（按顺序）")
print("=" * 80)

formula_paragraphs = []
for i, para in enumerate(doc.paragraphs):
    text = para.text.strip()
    if not text:
        continue
    
    # 检查是否包含数学符号或公式特征
    has_math = any(c in math_symbols for c in text)
    has_equation = '=' in text and any(c in text for c in 'βλρΔθαπμσ+-±×÷')
    has_subscript = '_' in text and any(c.isalpha() for c in text)
    
    if has_math or has_equation or has_subscript:
        # 只保留有实质公式内容的段落（排除纯数值结果段落）
        if len(text) > 10 and ('=' in text or 'max' in text or 'mean' in text or 'R(' in text or 's_' in text or 'p_' in text or 'g₂' in text or 'D_dom' in text or 'β_eff' in text or 'λ_eff' in text):
            formula_paragraphs.append((i+1, text))
            print(f"\n--- 段落 {i+1} ---")
            print(text)

print(f"\n\n总共找到 {len(formula_paragraphs)} 个含核心公式的段落")
