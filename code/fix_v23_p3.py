# -*- coding: utf-8 -*-
r"""fix_v23_p3.py — 修复表14撞车问题
表14: 多目标域对比（tbl[11]）✓
表14: 反向迁移（tbl[12]）❌ → 改为表15
表15: 信息使用审计（tbl[13]）→ 改为表16
表16: 物理先验因子（tbl[14]）→ 改为表17
表17: LODO（tbl[15]）→ 改为表18
"""
from docx import Document
from docx.oxml.ns import qn

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'

doc = Document(PAPER_PATH)

# 修复表14撞车：反向迁移→表15
print('修复表14撞车：反向迁移→表15')
for i, p in enumerate(doc.paragraphs):
    if '表14' in p.text and '反向迁移' in p.text:
        old_text = p.text
        new_text = old_text.replace('表14（§4.5', '表15（§4.5')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 表14→表15（反向迁移）')
        break

# 信息使用审计：表15→表16
print('\n信息使用审计：表15→表16')
for i, p in enumerate(doc.paragraphs):
    if '表15' in p.text and '信息使用审计' in p.text:
        old_text = p.text
        new_text = old_text.replace('表15　', '表16　')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 表15→表16（信息使用审计）')
        break

# 物理先验因子：表16→表17
print('\n物理先验因子：表16→表17')
for i, p in enumerate(doc.paragraphs):
    if '表16' in p.text and '物理先验因子' in p.text:
        old_text = p.text
        new_text = old_text.replace('表16　', '表17　')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 表16→表17（物理先验因子）')
        break

# LODO：表17→表18
print('\nLODO：表17→表18')
for i, p in enumerate(doc.paragraphs):
    if '表17' in p.text and 'LODO' in p.text:
        old_text = p.text
        new_text = old_text.replace('表17　', '表18　')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 表17→表18（LODO）')
        break

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('表14撞车问题已修复')

# 验证
doc2 = Document(OUTPUT_PATH)
print('\n=== 验证：所有表题编号 ===')
for i, p in enumerate(doc2.paragraphs):
    text = p.text.strip()
    if text.startswith('表') and len(text) > 2 and ('表' in text[:5]):
        # 提取表号
        import re
        match = re.match(r'表\s*(\w+)', text)
        if match:
            table_num = match.group(1)
            print(f'  段落{i}: 表{table_num} - {text[:50]}')
