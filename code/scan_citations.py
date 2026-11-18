# -*- coding: utf-8 -*-
from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v24.docx')

print('=== 扫描全文引用位置 ===')
citations = []  # (段落号, 引用编号, 位置)

for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 匹配 [数字] 或 [数字,数字] 或 [数字-数字]
    matches = re.finditer(r'\[(\d+(?:[,\-]\d+)*)\]', text)
    for m in matches:
        ref_str = m.group(1)
        # 处理多个引用，如 [1,2,3] 或 [1-3]
        if ',' in ref_str:
            refs = [int(x.strip()) for x in ref_str.split(',')]
        elif '-' in ref_str:
            start, end = map(int, ref_str.split('-'))
            refs = list(range(start, end+1))
        else:
            refs = [int(ref_str)]
        
        for ref in refs:
            citations.append((i, ref, m.start()))

# 按出现顺序排序（段落号 + 位置）
citations.sort(key=lambda x: (x[0], x[2]))

print(f'\n共找到 {len(citations)} 个引用')
print('\n=== 引用出现顺序 ===')
seen = set()
order = []
for para_idx, ref, pos in citations:
    if ref not in seen:
        seen.add(ref)
        order.append(ref)
        print(f'  第{len(order)}个引用: [{ref}] (段落{para_idx})')

print(f'\n共引用 {len(order)} 篇文献')
print(f'引用顺序: {order}')

# 检查是否有未引用的文献
print('\n=== 检查参考文献列表 ===')
ref_start = None
for i, p in enumerate(doc.paragraphs):
    if '参考文献' in p.text and p.style.name.startswith('Heading'):
        ref_start = i
        break

if ref_start:
    ref_list = []
    for i in range(ref_start+1, len(doc.paragraphs)):
        text = doc.paragraphs[i].text.strip()
        if text:
            match = re.match(r'\[(\d+)\]', text)
            if match:
                ref_num = int(match.group(1))
                ref_list.append(ref_num)
            else:
                # 可能是参考文献的续行，跳过
                pass
    
    print(f'参考文献列表中有 {len(ref_list)} 篇')
    print(f'列表编号: {ref_list}')
    
    # 找出未引用的文献
    all_refs = set(ref_list)
    cited_refs = set(order)
    uncited = all_refs - cited_refs
    if uncited:
        print(f'\n未被引用的文献: {sorted(uncited)}')
    else:
        print('\n所有文献都被引用了')
