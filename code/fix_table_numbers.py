# -*- coding: utf-8 -*-
"""fix_table_numbers.py — 修复表格编号混乱（表10、表12各出现两次）"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# 我新加的两个表格需要重新编号：
# 1. "反向迁移三方向消融结果"（段落77）→ 改为表7a（§4.5，作为表7的补充）
# 2. "物理先验预测力ρ的四个口径速查"（段落128）→ 改为表15a（§6.3，作为表15的补充）

count = 0
for i, p in enumerate(doc.paragraphs):
    old_text = p.text
    new_text = old_text
    
    # 反向迁移三方向消融结果 → 表7a
    if '表10　反向迁移三方向消融结果' in new_text:
        new_text = new_text.replace('表10　反向迁移三方向消融结果', '表7a　反向迁移三方向消融结果')
        count += 1
        print(f'  [{i}] 表10→表7a: 反向迁移三方向消融结果')
    
    # 物理先验预测力ρ的四个口径速查 → 表15a
    if '表12　物理先验预测力ρ的四个口径速查' in new_text:
        new_text = new_text.replace('表12　物理先验预测力ρ的四个口径速查', '表15a　物理先验预测力ρ的四个口径速查')
        count += 1
        print(f'  [{i}] 表12→表15a: 物理先验预测力ρ的四个口径速查')
    
    # 同时修改正文里对这两个表格的引用
    if '表10' in new_text and '三方向消融' in new_text:
        new_text = new_text.replace('表10', '表7a')
        count += 1
        print(f'  [{i}] 正文引用: 表10→表7a')
    
    if '表12' in new_text and '四个口径' in new_text:
        new_text = new_text.replace('表12', '表15a')
        count += 1
        print(f'  [{i}] 正文引用: 表12→表15a')
    
    if new_text != old_text:
        for run in p.runs:
            run.text = ''
        p.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print(f'\n表格编号修复完成：共修改{count}处')
print('  反向迁移三方向消融结果: 表10 → 表7a')
print('  物理先验预测力ρ口径速查: 表12 → 表15a')
