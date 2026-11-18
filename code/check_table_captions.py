# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v22.docx')

# 遍历文档body的所有子元素，检查表格和题注的位置关系
body = doc.element.body
table_count = 0
caption_before = 0
caption_after = 0

print('=== 表格与题注位置关系 ===')
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        table_count += 1
        # 检查前一个元素是否是题注
        prev_elem = body[i-1] if i > 0 else None
        next_elem = body[i+1] if i < len(body)-1 else None
        
        prev_text = ''
        next_text = ''
        if prev_elem is not None and prev_elem.tag == qn('w:p'):
            prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
        if next_elem is not None and next_elem.tag == qn('w:p'):
            next_text = ''.join([t.text or '' for t in next_elem.findall('.//' + qn('w:t'))])
        
        is_prev_caption = prev_text.strip().startswith('表') or '表' in prev_text[:10]
        is_next_caption = next_text.strip().startswith('表') or '表' in next_text[:10]
        
        if is_prev_caption:
            caption_before += 1
            print(f'表{table_count}: 题注在表格上方 ✓ ({prev_text[:40]}...)')
        elif is_next_caption:
            caption_after += 1
            print(f'表{table_count}: 题注在表格下方 ✗ ({next_text[:40]}...)')
        else:
            print(f'表{table_count}: 无题注（前: {prev_text[:30]}... / 后: {next_text[:30]}...）')

print(f'\n总计: {table_count}张表格')
print(f'题注在上方: {caption_before}')
print(f'题注在下方: {caption_after}')
