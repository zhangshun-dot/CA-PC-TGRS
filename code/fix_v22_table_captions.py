# -*- coding: utf-8 -*-
r"""fix_v22_table_captions.py — 将3张表格的题注从下方移到上方
表1（SOTA-a）、表2（SOTA-b）、表3（VOC20）
"""
from docx import Document
from docx.oxml.ns import qn

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v22.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v22.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

# 找到需要移动的表格（题注在下方的）
tables_to_fix = []
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        # 检查后一个元素是否是题注
        next_elem = body[i+1] if i < len(body)-1 else None
        if next_elem is not None and next_elem.tag == qn('w:p'):
            next_text = ''.join([t.text or '' for t in next_elem.findall('.//' + qn('w:t'))])
            if next_text.strip().startswith('表'):
                tables_to_fix.append((i, child, next_elem, next_text))
                print(f'找到需移动的表格 at index {i}: {next_text[:50]}...')

# 从后往前移动，避免索引偏移
for idx, table_elem, caption_elem, caption_text in reversed(tables_to_fix):
    # 把题注段落移到表格前面
    # 先从原位置移除
    caption_elem.getparent().remove(caption_elem)
    # 插入到表格前面
    table_elem.addprevious(caption_elem)
    print(f'已移动题注到表格上方: {caption_text[:40]}...')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print(f'共移动 {len(tables_to_fix)} 张表格的题注')

# 验证
doc2 = Document(OUTPUT_PATH)
body2 = doc2.element.body
caption_before = 0
caption_after = 0
for i, child in enumerate(body2):
    if child.tag == qn('w:tbl'):
        prev_elem = body2[i-1] if i > 0 else None
        next_elem = body2[i+1] if i < len(body2)-1 else None
        prev_text = ''
        next_text = ''
        if prev_elem is not None and prev_elem.tag == qn('w:p'):
            prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
        if next_elem is not None and next_elem.tag == qn('w:p'):
            next_text = ''.join([t.text or '' for t in next_elem.findall('.//' + qn('w:t'))])
        if prev_text.strip().startswith('表'):
            caption_before += 1
        elif next_text.strip().startswith('表'):
            caption_after += 1

print(f'\n验证: 题注在上方 {caption_before}, 题注在下方 {caption_after}')
