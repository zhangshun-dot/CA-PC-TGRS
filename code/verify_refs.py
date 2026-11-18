# -*- coding: utf-8 -*-
from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v27.docx')

print('=== 验证1：正文引用顺序 ===')
citations = []
for i, p in enumerate(doc.paragraphs):
    text = p.text
    matches = re.finditer(r'\[(\d{1,2}(?:[,\-]\d{1,2})*)\]', text)
    for m in matches:
        ref_str = m.group(1)
        if ',' in ref_str:
            refs = [int(x.strip()) for x in ref_str.split(',') if x.strip().isdigit()]
        elif '-' in ref_str:
            try:
                start, end = map(int, ref_str.split('-'))
                refs = list(range(start, end+1))
            except:
                refs = []
        else:
            try:
                refs = [int(ref_str)]
            except:
                refs = []
        
        for ref in refs:
            if 1 <= ref <= 51:
                citations.append((i, ref, m.start()))

citations.sort(key=lambda x: (x[0], x[2]))

# 检查是否按顺序
seen = set()
order = []
for para_idx, ref, pos in citations:
    if ref not in seen:
        seen.add(ref)
        order.append(ref)

print(f'引用顺序: {order}')
print(f'是否严格递增: {order == sorted(order)}')

# 检查是否有跳号
expected = list(range(1, 52))
missing = set(expected) - set(order)
extra = set(order) - set(expected)
if missing:
    print(f'缺失的引用: {sorted(missing)}')
if extra:
    print(f'多余的引用: {sorted(extra)}')
if not missing and not extra:
    print('所有51篇文献都被引用，且编号完整')

print('\n=== 验证2：参考文献列表顺序 ===')
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
                # 打印前几个和后几个
                if len(ref_list) <= 5 or len(ref_list) >= 47:
                    print(f'  [{ref_num}] {text[:80]}')
    
    print(f'\n参考文献列表共 {len(ref_list)} 篇')
    print(f'列表编号: {ref_list}')
    print(f'是否严格递增: {ref_list == sorted(ref_list)}')
    print(f'是否完整1-51: {ref_list == list(range(1, 52))}')

print('\n=== 验证3：前几个引用的上下文 ===')
for i, p in enumerate(doc.paragraphs[:30]):
    text = p.text
    if '[' in text and re.search(r'\[\d+\]', text):
        # 提取引用
        refs = re.findall(r'\[\d+\]', text)
        print(f'段落{i}: 引用{refs}')
        print(f'  内容: {text[:100]}')
