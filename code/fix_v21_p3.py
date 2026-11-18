# -*- coding: utf-8 -*-
r"""fix_v21_p3.py — 修正表16/17编号 + 搜索显著性检验 + 修正交叉引用
"""
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'

doc = Document(PAPER_PATH)

# ══════════════════════════ 修正表16/17编号 ══════════════════════════
print('修正表16/17编号')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if '表16' in text and '信息使用审计' in text:
        new_text = text.replace('表16', '表13')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 表16 → 表13')
    if '表17' in text and 'LODO' in text:
        new_text = text.replace('表17', '表15')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 表17 → 表15')

# ══════════════════════════ 搜索显著性检验 ══════════════════════════
print('\n搜索"显著性检验"')
for i, p in enumerate(doc.paragraphs):
    if '显著性' in p.text or '显著检验' in p.text:
        print(f'  段落{i}: {p.text[:200]}')

# ══════════════════════════ 修正交叉引用 ══════════════════════════
print('\n修正交叉引用')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    changed = False
    # §4.6 → §4.2.1（VOC20扩展）
    if '§4.6' in text and 'VOC' in text:
        text = text.replace('§4.6', '§4.2.1')
        changed = True
        print(f'  段落{i}: §4.6 → §4.2.1 (VOC20)')
    # §5.5 → §4.9（配置审计）
    if '§5.5' in text and ('配置' in text or '调参' in text or 'β' in text or 'λ' in text):
        text = text.replace('§5.5', '§4.9')
        changed = True
        print(f'  段落{i}: §5.5 → §4.9 (配置审计)')
    if changed:
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = text
        else:
            p.add_run(text)

# ══════════════════════════ 修正表15引用错误 ══════════════════════════
print('\n修正表15引用错误（段85）')
p85 = doc.paragraphs[85]
old_text = p85.text
print(f'  旧: {old_text[:200]}')
# 段85写的是"表15（§4.5 COCO→DIOR协议...）反向迁移六行细分协议结果"
# 但表15实际是LODO表，反向迁移六行表是表12
if '表15' in old_text and '反向迁移' in old_text:
    new_text = old_text.replace('表15', '表12')
    for run in p85.runs:
        run.text = ''
    if p85.runs:
        p85.runs[0].text = new_text
    else:
        p85.add_run(new_text)
    print(f'  新: {new_text[:200]}')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
