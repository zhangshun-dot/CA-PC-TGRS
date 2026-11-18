# -*- coding: utf-8 -*-
r"""fix_v26_refs.py — 重新排序和编号参考文献
1. 收集所有51篇参考文献段落
2. 按旧编号建立映射
3. 按新编号排序
4. 物理重排
5. 更新编号
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v25.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v26.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

# 旧→新映射（按引用顺序）
old_to_new = {
    1: 1, 2: 2, 3: 3, 5: 4, 8: 5, 20: 6, 21: 7, 22: 8, 23: 9, 29: 10,
    24: 11, 26: 12, 10: 13, 50: 14, 11: 15, 51: 16, 16: 17, 17: 18, 18: 19, 19: 20,
    30: 21, 32: 22, 6: 23, 7: 24, 15: 25, 9: 26, 4: 27, 12: 28, 13: 29, 14: 30,
    25: 31, 27: 32, 28: 33, 31: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40,
    39: 41, 40: 42, 41: 43, 42: 44, 43: 45, 44: 46, 45: 47, 46: 48, 47: 49, 48: 50, 49: 51
}

print('第一步：收集参考文献段落')
ref_paragraphs = {}  # 旧编号 -> XML元素
ref_title_elem = None

for child in body.iterchildren(qn('w:p')):
    text = ''.join([t.text or '' for t in child.findall('.//' + qn('w:t'))])
    if '参考文献' in text:
        # 检查是否是标题
        pStyle = child.find('.//' + qn('w:pStyle'))
        if pStyle is not None and 'Heading' in pStyle.get(qn('w:val'), ''):
            ref_title_elem = child
            print(f'  找到参考文献标题')
            continue
    
    if ref_title_elem is not None:
        match = re.match(r'\[(\d+)\]', text.strip())
        if match:
            old_num = int(match.group(1))
            if 1 <= old_num <= 51:
                ref_paragraphs[old_num] = child
                if len(ref_paragraphs) <= 5 or len(ref_paragraphs) >= 47:
                    print(f'  收集到旧[{old_num}]: {text[:60]}')

print(f'\n共收集到 {len(ref_paragraphs)} 篇参考文献')

# 按新编号排序
print('\n第二步：按新编号排序')
sorted_refs = sorted(ref_paragraphs.items(), key=lambda x: old_to_new.get(x[0], 999))
print(f'排序后顺序（旧→新）:')
for old_num, elem in sorted_refs[:10]:
    print(f'  旧[{old_num}] → 新[{old_to_new[old_num]}]')
print('  ...')

# 移除所有参考文献段落
print('\n第三步：移除旧参考文献段落')
for old_num, elem in ref_paragraphs.items():
    body.remove(elem)
print(f'已移除 {len(ref_paragraphs)} 个段落')

# 更新编号并按新顺序插入
print('\n第四步：更新编号并按新顺序插入')
insert_after = ref_title_elem
for old_num, elem in sorted_refs:
    new_num = old_to_new[old_num]
    # 更新编号
    text = ''.join([t.text or '' for t in elem.findall('.//' + qn('w:t'))])
    new_text = re.sub(r'^\[\d+\]', f'[{new_num}]', text)
    
    # 更新段落文本
    runs = elem.findall('.//' + qn('w:r'))
    if runs:
        # 清空所有run
        for r in runs:
            for t in r.findall('.//' + qn('w:t')):
                t.text = ''
        # 在第一个run设置新文本
        first_t = runs[0].find('.//' + qn('w:t'))
        if first_t is not None:
            first_t.text = new_text
            first_t.set(qn('xml:space'), 'preserve')
    
    # 插入到参考文献标题之后
    insert_after.addnext(elem)
    insert_after = elem
    
    if new_num <= 5 or new_num >= 47:
        print(f'  插入新[{new_num}]: {new_text[:60]}')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('参考文献重新排序和编号完成')
