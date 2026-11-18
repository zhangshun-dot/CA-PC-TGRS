# -*- coding: utf-8 -*-
from docx import Document
doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v20.docx')

print('=== 检查1: 参考文献[49]污染 ===')
for i, p in enumerate(doc.paragraphs):
    if '[49]' in p.text and 'Field' in p.text:
        print(f'段落{i}: {p.text[:300]}')
        print(f'... (长度: {len(p.text)})')
        break

print('\n=== 检查2: NWPU 5/6矛盾 ===')
for i, p in enumerate(doc.paragraphs):
    if 'NWPU' in p.text and '5/6' in p.text:
        print(f'段落{i}: {p.text[:300]}')
        break

print('\n=== 检查3: 表3显著性检验引用 ===')
for i, p in enumerate(doc.paragraphs):
    if '表3' in p.text and '显著性' in p.text:
        print(f'段落{i}: {p.text[:300]}')
        break

print('\n=== 检查4: §6.1 airplane/motorcycle冲突 ===')
for i, p in enumerate(doc.paragraphs):
    if 'airplane' in p.text and '0.113' in p.text:
        print(f'段落{i}: {p.text[:300]}')
        break

print('\n=== 检查5: 全文表编号引用 ===')
table_refs = {}
for i, p in enumerate(doc.paragraphs):
    import re
    refs = re.findall(r'表\s*(\d+)', p.text)
    for r in refs:
        if r not in table_refs:
            table_refs[r] = []
        table_refs[r].append(i)
print('表编号引用:')
for k in sorted(table_refs.keys(), key=int):
    print(f'  表{k}: 段落{table_refs[k][:5]}')

print(f'\n文档表格总数: {len(doc.tables)}')
