# -*- coding: utf-8 -*-
r"""fix_v25_refs.py — 按文中引用顺序重新编码参考文献
1. 扫描全文引用，建立旧编号→新编号映射
2. 更新正文中的所有引用（排除数学表达式）
3. 重新排序参考文献列表
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v24.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v25.docx'

doc = Document(PAPER_PATH)

# ══════════════════════════ 第一步：扫描引用顺序 ══════════════════════════
print('第一步：扫描引用顺序')
citations = []

for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 匹配 [数字] 或 [数字,数字] 或 [数字-数字]
    # 但要排除数学表达式中的区间，如 [0,1]、[0.25,0.5] 等
    # 文献引用的特点：数字是整数，且通常是1-51的范围
    matches = re.finditer(r'\[(\d+(?:[,\-]\d+)*)\]', text)
    for m in matches:
        ref_str = m.group(1)
        # 检查是否是数学表达式（包含小数或在特定上下文中）
        # 简单判断：如果数字都在1-51范围内，且不是连续的小数区间，则认为是引用
        if ',' in ref_str:
            refs = [x.strip() for x in ref_str.split(',')]
        elif '-' in ref_str:
            start, end = ref_str.split('-')
            refs = [str(x) for x in range(int(start), int(end)+1)]
        else:
            refs = [ref_str]
        
        # 过滤：只保留1-51的整数引用
        valid_refs = []
        for r in refs:
            try:
                r_int = int(r)
                if 1 <= r_int <= 51:
                    valid_refs.append(r_int)
            except:
                pass
        
        for ref in valid_refs:
            citations.append((i, ref, m.start()))

# 按出现顺序排序
citations.sort(key=lambda x: (x[0], x[2]))

# 建立旧编号→新编号映射
old_to_new = {}
order = []
for para_idx, ref, pos in citations:
    if ref not in old_to_new:
        old_to_new[ref] = len(order) + 1
        order.append(ref)

print(f'共引用 {len(order)} 篇文献')
print(f'引用顺序: {order}')
print(f'\n旧→新映射:')
for old, new in sorted(old_to_new.items(), key=lambda x: x[1]):
    print(f'  [{old}] → [{new}]')

# ══════════════════════════ 第二步：更新正文中的引用 ══════════════════════════
print('\n第二步：更新正文中的引用')

def replace_citations_in_text(text):
    """替换文本中的文献引用"""
    def replace_match(m):
        ref_str = m.group(1)
        if ',' in ref_str:
            refs = [x.strip() for x in ref_str.split(',')]
            new_refs = []
            for r in refs:
                try:
                    r_int = int(r)
                    if r_int in old_to_new:
                        new_refs.append(str(old_to_new[r_int]))
                    else:
                        new_refs.append(r)
                except:
                    new_refs.append(r)
            return '[' + ','.join(new_refs) + ']'
        elif '-' in ref_str:
            try:
                start, end = map(int, ref_str.split('-'))
                if start in old_to_new and end in old_to_new:
                    new_start = old_to_new[start]
                    new_end = old_to_new[end]
                    if new_start < new_end:
                        return f'[{new_start}-{new_end}]'
                    else:
                        return f'[{new_start},{new_end}]'
            except:
                pass
            return m.group(0)
        else:
            try:
                r_int = int(ref_str)
                if r_int in old_to_new:
                    return f'[{old_to_new[r_int]}]'
            except:
                pass
            return m.group(0)
    
    # 只替换看起来像文献引用的模式（1-51的整数）
    # 使用更精确的正则，避免匹配数学表达式
    result = re.sub(r'\[(\d{1,2}(?:[,\-]\d{1,2})*)\]', replace_match, text)
    return result

# 更新所有段落
count = 0
for i, p in enumerate(doc.paragraphs):
    old_text = p.text
    new_text = replace_citations_in_text(old_text)
    if new_text != old_text:
        # 更新段落文本
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        count += 1

print(f'更新了 {count} 个段落的引用')

# ══════════════════════════ 第三步：重新排序参考文献列表 ══════════════════════════
print('\n第三步：重新排序参考文献列表')

# 找到参考文献部分
ref_start = None
for i, p in enumerate(doc.paragraphs):
    if '参考文献' in p.text and p.style.name.startswith('Heading'):
        ref_start = i
        break

if ref_start is None:
    print('未找到参考文献部分')
else:
    print(f'参考文献开始于段落{ref_start}')
    
    # 收集所有参考文献条目
    ref_entries = {}  # 旧编号 -> (段落元素, 文本)
    ref_paragraphs = []  # 参考文献段落的索引
    
    i = ref_start + 1
    while i < len(doc.paragraphs):
        text = doc.paragraphs[i].text.strip()
        if text:
            match = re.match(r'\[(\d+)\]', text)
            if match:
                ref_num = int(match.group(1))
                ref_entries[ref_num] = doc.paragraphs[i]
                ref_paragraphs.append(i)
            # 否则可能是续行，跳过
        i += 1
    
    print(f'收集到 {len(ref_entries)} 篇参考文献')
    
    # 按新顺序重新排列参考文献
    # 我们需要修改每个参考文献条目的编号
    for old_num, new_num in old_to_new.items():
        if old_num in ref_entries:
            p = ref_entries[old_num]
            old_text = p.text
            # 替换编号
            new_text = re.sub(r'^\[\d+\]', f'[{new_num}]', old_text)
            if new_text != old_text:
                for run in p.runs:
                    run.text = ''
                if p.runs:
                    p.runs[0].text = new_text
                else:
                    p.add_run(new_text)
    
    print('参考文献编号已更新')
    
    # 注意：物理重排段落比较复杂，因为需要操作XML
    # 我们先只更新编号，然后检查是否需要物理重排
    # 实际上，只要编号正确，读者可以按编号顺序查找
    # 但为了规范，我们应该物理重排
    
    # 物理重排：收集所有参考文献段落，按新编号排序，然后重新插入
    print('\n进行物理重排...')
    
    # 收集参考文献段落元素（包括续行）
    ref_blocks = []  # [(旧编号, [段落元素列表])]
    current_block = None
    current_num = None
    
    body = doc.element.body
    all_paragraphs = list(body.iterchildren(qn('w:p')))
    
    # 找到参考文献标题后的所有段落
    in_refs = False
    ref_blocks_dict = {}  # 旧编号 -> [段落元素]
    current_num = None
    
    for p_elem in all_paragraphs:
        text = ''.join([t.text or '' for t in p_elem.findall('.//' + qn('w:t'))])
        if '参考文献' in text and p_elem.find('.//' + qn('w:pStyle')) is not None:
            # 检查是否是标题
            pStyle = p_elem.find('.//' + qn('w:pStyle'))
            if pStyle is not None and 'Heading' in pStyle.get(qn('w:val'), ''):
                in_refs = True
                continue
        
        if in_refs and text.strip():
            match = re.match(r'\[(\d+)\]', text.strip())
            if match:
                current_num = int(match.group(1))
                ref_blocks_dict[current_num] = [p_elem]
            elif current_num is not None:
                # 续行
                ref_blocks_dict[current_num].append(p_elem)
    
    print(f'收集到 {len(ref_blocks_dict)} 个参考文献块')
    
    # 按新编号排序
    sorted_blocks = sorted(ref_blocks_dict.items(), key=lambda x: old_to_new.get(x[0], 999))
    
    # 移除所有参考文献段落
    for old_num, blocks in ref_blocks_dict.items():
        for p_elem in blocks:
            body.remove(p_elem)
    
    # 按新顺序重新插入（在参考文献标题之后）
    # 找到参考文献标题
    ref_title_elem = None
    for p_elem in all_paragraphs:
        text = ''.join([t.text or '' for t in p_elem.findall('.//' + qn('w:t'))])
        if '参考文献' in text:
            pStyle = p_elem.find('.//' + qn('w:pStyle'))
            if pStyle is not None and 'Heading' in pStyle.get(qn('w:val'), ''):
                ref_title_elem = p_elem
                break
    
    if ref_title_elem is not None:
        insert_after = ref_title_elem
        for old_num, blocks in sorted_blocks:
            for p_elem in blocks:
                insert_after.addnext(p_elem)
                insert_after = p_elem
        print('参考文献已按新顺序重排')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('参考文献重新编码完成')
