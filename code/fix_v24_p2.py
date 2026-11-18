# -*- coding: utf-8 -*-
r"""fix_v24_p2.py — 修复表4题注位置错误
表4题注应该在tbl[6]（分层失效诊断表）之前，而不是在表5题注之前。
把表4题注从当前位置移回tbl[6]之前。
"""
from docx import Document
from docx.oxml.ns import qn

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v24.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v24.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

def get_paragraph_text(elem):
    return ''.join([t.text or '' for t in elem.findall('.//' + qn('w:t'))])

print('=== 当前表4题注位置 ===')
table4_caption = None
for i, child in enumerate(body):
    if child.tag == qn('w:p'):
        text = get_paragraph_text(child)
        if text.strip().startswith('表4') and '分层失效诊断' in text:
            table4_caption = child
            print(f'  表4题注在位置{i}: {text[:50]}')
            break

print('\n=== tbl[6]（分层失效诊断表）位置 ===')
table_idx = 0
target_table = None
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        if table_idx == 6:
            target_table = child
            print(f'  tbl[6]在位置{i}')
            # 检查前面的题注
            j = i - 1
            while j >= 0 and body[j].tag == qn('w:p'):
                text = get_paragraph_text(body[j])
                if text.strip():
                    print(f'  前面段落{j}: {text[:50]}')
                    break
                j -= 1
            break
        table_idx += 1

if table4_caption is not None and target_table is not None:
    print('\n=== 移动表4题注到tbl[6]之前 ===')
    # 移除表4题注
    body.remove(table4_caption)
    # 插入到tbl[6]之前
    target_table.addprevious(table4_caption)
    print('  已移动表4题注到tbl[6]之前')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')

# 验证
print('\n=== 验证：表题-表格配对 ===')
doc2 = Document(OUTPUT_PATH)
body2 = doc2.element.body
table_idx = 0
for i, child in enumerate(body2):
    if child.tag == qn('w:tbl'):
        caption = ''
        j = i - 1
        while j >= 0 and body2[j].tag == qn('w:p'):
            text = ''.join([t.text or '' for t in body2[j].findall('.//' + qn('w:t'))])
            if text.strip().startswith('表'):
                caption = text[:30]
                break
            elif text.strip() == '':
                j -= 1
                continue
            else:
                break
            j -= 1
        print(f'  tbl[{table_idx}]: {caption}')
        table_idx += 1
