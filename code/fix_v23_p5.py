# -*- coding: utf-8 -*-
r"""fix_v23_p5.py — 修复剩余3个次要问题
1. 补充材料表7/8/9 → 补充表S2/S3/S4（S1已被LODO表占用）
2. 表6（top-k消融）→ 表5，保持4-5-6-7-8连续编号
3. 检查并更新正文中的表6引用
"""
from docx import Document
from docx.oxml.ns import qn

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

# ══════════════════════════ 修复1：补充材料表号 ══════════════════════════
print('修复1：补充材料表7/8/9 → 补充表S2/S3/S4')

# 段落207: 表 7 完整消融与效率 → 补充表S2
for i, p in enumerate(doc.paragraphs):
    if '表 7' in p.text and '完整消融' in p.text:
        replace_in_paragraph(p, '表 7　', '补充表S2　')
        replace_in_paragraph(p, '表 7 ', '补充表S2 ')
        print(f'  段落{i}: 表7 → 补充表S2（完整消融与效率）')
        break

# 段落219: 表 8 信息使用审计 → 补充表S3
for i, p in enumerate(doc.paragraphs):
    if '表 8' in p.text and '信息使用审计' in p.text:
        replace_in_paragraph(p, '表 8　', '补充表S3　')
        replace_in_paragraph(p, '表 8 ', '补充表S3 ')
        print(f'  段落{i}: 表8 → 补充表S3（信息使用审计）')
        break

# 段落220: 表 9 实验协议 → 补充表S4
for i, p in enumerate(doc.paragraphs):
    if '表 9' in p.text and '实验协议' in p.text:
        replace_in_paragraph(p, '表 9　', '补充表S4　')
        replace_in_paragraph(p, '表 9 ', '补充表S4 ')
        print(f'  段落{i}: 表9 → 补充表S4（实验协议）')
        break

# ══════════════════════════ 修复2：表6 → 表5 ══════════════════════════
print('\n修复2：表6（top-k消融）→ 表5')

# 修改题注
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('表6') and 'top-k' in p.text:
        replace_in_paragraph(p, '表6　', '表5　')
        print(f'  段落{i}: 表6 → 表5（top-k消融题注）')
        break

# 检查正文中是否有引用"表6"的地方
print('\n检查正文中的"表6"引用：')
found_ref = False
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 精确匹配"表6"（不是"表60"等）
    import re
    if re.search(r'表\s*6(?!\d)', text):
        # 排除题注本身
        if not (p.text.strip().startswith('表5') and 'top-k' in p.text):
            print(f'  段落{i}: {text[:100]}')
            found_ref = True
if not found_ref:
    print('  未找到其他"表6"引用')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('剩余问题修复完成')
