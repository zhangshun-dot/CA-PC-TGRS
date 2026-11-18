# -*- coding: utf-8 -*-
r"""fix_v23_p8.py — 修复表号连锁替换问题
使用临时占位符避免连锁替换：
1. 先把所有表号改成临时占位符（表T1, 表T2, ...）
2. 再把临时占位符改成正确的表号
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

# 根据表格内容确定正确的表号
# 先收集所有表格和它们的题注
print('=== 当前表格与题注 ===')
tables_info = []
table_idx = 0
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        # 获取表头
        header = []
        first_row = child.find('.//' + qn('w:tr'))
        if first_row is not None:
            for cell in first_row.findall('.//' + qn('w:tc')):
                cell_text = ''.join([t.text or '' for t in cell.findall('.//' + qn('w:t'))])
                header.append(cell_text[:15])
        # 找前面的题注
        caption = ''
        caption_elem = None
        j = i - 1
        while j >= 0 and body[j].tag == qn('w:p'):
            text = ''.join([t.text or '' for t in body[j].findall('.//' + qn('w:t'))])
            if text.strip().startswith('表'):
                caption = text
                caption_elem = body[j]
                break
            elif text.strip() == '':
                j -= 1
                continue
            else:
                break
            j -= 1
        tables_info.append({
            'idx': table_idx,
            'header': header,
            'caption': caption,
            'caption_elem': caption_elem,
            'body_idx': i
        })
        print(f'tbl[{table_idx}]: 题注={caption[:40]}')
        print(f'  表头={header[:4]}')
        table_idx += 1

# 根据表头内容确定正确的表号
correct_numbers = {
    0: 'SOTA-a',   # SOTA主对比
    1: 'SOTA-b',   # SOTA上界参考
    2: 'VOC20',    # VOC→COCO 20类
    3: '5',        # top-k消融
    4: '6',        # β扫描
    5: '7',        # λ扫描
    6: '4',        # 分层失效诊断
    7: '8',        # 类别映射
    8: '9',        # 三源域复现
    9: '10',       # 负原型构建
    10: '11',      # 检测器谱系
    11: '12',      # 多目标域对比
    12: '13',      # 反向迁移
    13: '14',      # 信息使用审计
    14: '15',      # 物理先验因子
    15: '16',      # LODO
}

print('\n=== 正确的表号映射 ===')
for idx, num in correct_numbers.items():
    print(f'  tbl[{idx}]: 表{num}')

# 第一步：把所有题注中的表号改成临时占位符
print('\n第一步：改成临时占位符')
for info in tables_info:
    if info['caption_elem'] is not None:
        caption_text = info['caption']
        # 匹配开头的"表X"
        match = re.match(r'表\s*(\w+)', caption_text)
        if match:
            old_num = match.group(1)
            temp_num = f'TEMP{info["idx"]}'
            # 替换题注中的表号
            new_caption = re.sub(rf'^表\s*{re.escape(old_num)}', f'表{temp_num}', caption_text)
            # 修改XML元素
            for t in info['caption_elem'].findall('.//' + qn('w:t')):
                if t.text:
                    t.text = t.text.replace(f'表{old_num}', f'表{temp_num}', 1)
                    break
            print(f'  tbl[{info["idx"]}]: 表{old_num} → 表{temp_num}')

# 第二步：把临时占位符改成正确的表号
print('\n第二步：改成正确表号')
for info in tables_info:
    if info['caption_elem'] is not None:
        temp_num = f'TEMP{info["idx"]}'
        correct_num = correct_numbers[info['idx']]
        # 修改XML元素
        for t in info['caption_elem'].findall('.//' + qn('w:t')):
            if t.text and f'表{temp_num}' in t.text:
                t.text = t.text.replace(f'表{temp_num}', f'表{correct_num}', 1)
                print(f'  tbl[{info["idx"]}]: 表{temp_num} → 表{correct_num}')
                break

# 第三步：修改正文中的表引用
print('\n第三步：修改正文中的表引用')
# 先收集正文中所有的表引用位置
ref_locations = []
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 匹配"表X"（X为数字）
    matches = list(re.finditer(r'表\s*(\d+)', text))
    if matches:
        ref_locations.append((i, p, matches))

# 对于每个引用，需要根据上下文判断它引用的是哪个表
# 由于正文中的引用通常是"见表X"或"表X显示"，我们可以根据表号来判断
# 但由于表号已经被修改成了临时占位符（只在题注中），正文中的引用还是旧的
# 所以我们需要先把正文中的引用也改成临时占位符，然后再改成正确的

# 实际上，正文中的引用还是旧的表号（表10-表18），因为我只修改了题注
# 让我检查一下正文中的引用
print('正文中的表引用：')
for i, p, matches in ref_locations:
    for m in matches:
        num = m.group(1)
        if int(num) >= 8:  # 只关注表8及以上的引用
            print(f'  段落{i}: 表{num} - {p.text[:80]}')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('表号修复完成（题注部分）')
