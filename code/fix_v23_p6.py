# -*- coding: utf-8 -*-
r"""fix_v23_p6.py — 修复表6缺失问题
表7（β扫描）→ 表6
表8（λ扫描）→ 表7
保持4-5-6-7连续编号
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'

doc = Document(PAPER_PATH)

def replace_in_paragraph(p, old_text, new_text):
    if old_text in p.text:
        full_text = p.text.replace(old_text, new_text)
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = full_text
        else:
            p.add_run(full_text)
        return True
    return False

# ══════════════════════════ 修复：表7→表6，表8→表7 ══════════════════════════
print('修复表号连续：表7→表6（β扫描），表8→表7（λ扫描）')

# 修改题注
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    # β扫描题注：表7 → 表6
    if text.startswith('表7') and 'β扫描' in text:
        replace_in_paragraph(p, '表7　', '表6　')
        print(f'  段落{i}: 表7→表6（β扫描题注）')
    # λ扫描题注：表8 → 表7
    elif text.startswith('表8') and 'λ扫描' in text:
        replace_in_paragraph(p, '表8　', '表7　')
        print(f'  段落{i}: 表8→表7（λ扫描题注）')

# 检查正文中的表7/表8引用
print('\n检查正文中的表7/表8引用：')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 匹配"表7"或"表8"（不是题注）
    if not (text.strip().startswith('表6') and 'β扫描' in text) and \
       not (text.strip().startswith('表7') and 'λ扫描' in text):
        if re.search(r'表\s*[78](?!\d)', text):
            # 排除补充材料中的表S2/S3/S4
            if '补充表' not in text:
                print(f'  段落{i}: {text[:100]}')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('表号连续修复完成')
