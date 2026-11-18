# -*- coding: utf-8 -*-
r"""fix_v27_refs.py — 重新排序和编号参考文献（简化版）
直接通过doc.paragraphs处理
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v25.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v27.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

# 旧→新映射
old_to_new = {
    1: 1, 2: 2, 3: 3, 5: 4, 8: 5, 20: 6, 21: 7, 22: 8, 23: 9, 29: 10,
    24: 11, 26: 12, 10: 13, 50: 14, 11: 15, 51: 16, 16: 17, 17: 18, 18: 19, 19: 20,
    30: 21, 32: 22, 6: 23, 7: 24, 15: 25, 9: 26, 4: 27, 12: 28, 13: 29, 14: 30,
    25: 31, 27: 32, 28: 33, 31: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40,
    39: 41, 40: 42, 41: 43, 42: 44, 43: 45, 44: 46, 45: 47, 46: 48, 47: 49, 48: 50, 49: 51
}

print('第一步：找到参考文献标题位置')
ref_title_idx = None
for i, p in enumerate(doc.paragraphs):
    if '参考文献' in p.text and p.style.name.startswith('Heading'):
        ref_title_idx = i
        print(f'  参考文献标题在段落{i}: {p.text}')
        break

print('\n第二步：收集参考文献段落')
ref_paragraphs = {}  # 旧编号 -> (段落索引, 段落对象)
for i in range(ref_title_idx + 1, len(doc.paragraphs)):
    p = doc.paragraphs[i]
    text = p.text.strip()
    match = re.match(r'\[(\d+)\]', text)
    if match:
        old_num = int(match.group(1))
        if 1 <= old_num <= 51:
            ref_paragraphs[old_num] = (i, p)
            if len(ref_paragraphs) <= 5 or len(ref_paragraphs) >= 47:
                print(f'  收集到旧[{old_num}] (段落{i}): {text[:60]}')

print(f'\n共收集到 {len(ref_paragraphs)} 篇参考文献')

# 按新编号排序
print('\n第三步：按新编号排序并更新文本')
sorted_refs = sorted(ref_paragraphs.items(), key=lambda x: old_to_new.get(x[0], 999))

# 收集所有参考文献段落的XML元素
ref_elems = []
for old_num, (idx, p) in sorted_refs:
    new_num = old_to_new[old_num]
    # 更新编号
    old_text = p.text
    new_text = re.sub(r'^\[\d+\]', f'[{new_num}]', old_text)
    
    # 更新段落文本
    for run in p.runs:
        run.text = ''
    if p.runs:
        p.runs[0].text = new_text
    else:
        p.add_run(new_text)
    
    ref_elems.append(p._element)
    
    if new_num <= 5 or new_num >= 47:
        print(f'  旧[{old_num}] → 新[{new_num}]: {new_text[:60]}')

# 物理重排：先移除所有参考文献段落，再按新顺序插入
print('\n第四步：物理重排')
ref_title_elem = doc.paragraphs[ref_title_idx]._element

# 移除所有参考文献段落
for elem in ref_elems:
    body.remove(elem)

# 按新顺序插入到参考文献标题之后
insert_after = ref_title_elem
for elem in ref_elems:
    insert_after.addnext(elem)
    insert_after = elem

print(f'已重排 {len(ref_elems)} 篇参考文献')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('参考文献重新排序和编号完成')
