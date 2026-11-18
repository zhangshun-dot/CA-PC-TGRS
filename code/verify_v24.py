# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v24.docx')
body = doc.element.body

print('=== 验证1：表4位置（应在表5之前）===')
for i, child in enumerate(body):
    if child.tag == qn('w:p'):
        text = ''.join([t.text or '' for t in child.findall('.//' + qn('w:t'))])
        if text.strip().startswith('表4') and '分层失效诊断' in text:
            print(f'  表4题注在位置{i}')
        elif text.strip().startswith('表5') and 'top-k' in text:
            print(f'  表5题注在位置{i}')
        elif text.strip().startswith('表6') and 'β扫描' in text:
            print(f'  表6题注在位置{i}')
        elif text.strip().startswith('表7') and 'λ扫描' in text:
            print(f'  表7题注在位置{i}')

print('\n=== 验证2：段68交叉引用 ===')
if 68 < len(doc.paragraphs):
    text = doc.paragraphs[68].text
    if '第 4.3 节表 4' in text:
        print('  ✅ 已修正为第4.3节表4')
    else:
        print(f'  ⚠️ 当前内容: {text[:100]}')

print('\n=== 验证3：补充表S3/S4内容 ===')
for i in [219, 220]:
    if i < len(doc.paragraphs):
        print(f'  段落{i}: {doc.paragraphs[i].text[:100]}')

print('\n=== 验证4：负号检查 ===')
import re
count_hyphen = 0
count_minus = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 检查数字前的连字符
    hyphen_matches = re.findall(r'(?<=\s)-(?=\d)', text)
    minus_matches = re.findall(r'(?<=\s)−(?=\d)', text)
    count_hyphen += len(hyphen_matches)
    count_minus += len(minus_matches)
print(f'  连字符(-)在数字前: {count_hyphen}个')
print(f'  负号(−)在数字前: {count_minus}个')

print('\n=== 验证5：表题-表格配对 ===')
table_idx = 0
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        # 找前面的题注
        caption = ''
        j = i - 1
        while j >= 0 and body[j].tag == qn('w:p'):
            text = ''.join([t.text or '' for t in body[j].findall('.//' + qn('w:t'))])
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
