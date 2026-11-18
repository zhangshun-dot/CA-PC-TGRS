# -*- coding: utf-8 -*-
r"""final_format_check.py — 最终格式检查
按照Elsevier期刊格式要求检查：
1. 页面设置
2. 字体字号
3. 段落格式
4. 标题层级
5. 摘要关键词
6. 图表题注
7. 参考文献格式
8. 章节编号连续性
9. 公式编号
10. 整体结构
"""
from docx import Document
from docx.shared import Pt, Cm
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v32.docx'
doc = Document(PAPER_PATH)

print('=' * 70)
print('最终格式检查报告')
print('=' * 70)

# 1. 页面设置
print('\n【1. 页面设置】')
section = doc.sections[0]
print(f'  纸张大小: {section.page_width.cm:.2f} x {section.page_height.cm:.2f} cm')
print(f'  页边距: 上{section.top_margin.cm:.2f} 下{section.bottom_margin.cm:.2f} 左{section.left_margin.cm:.2f} 右{section.right_margin.cm:.2f} cm')
print(f'  建议: A4 (21x29.7cm), 页边距2.5cm左右')

# 2. 字体字号
print('\n【2. 字体字号】')
font_sizes = {}
font_names = {}
for p in doc.paragraphs:
    for run in p.runs:
        if run.font.size:
            size_pt = run.font.size.pt
            font_sizes[size_pt] = font_sizes.get(size_pt, 0) + 1
        if run.font.name:
            font_names[run.font.name] = font_names.get(run.font.name, 0) + 1

print(f'  字号分布: {sorted(font_sizes.items())}')
print(f'  字体分布: {font_names}')
print(f'  建议: 正文10-12pt, 标题14-16pt, Times New Roman(英文)/宋体(中文)')

# 3. 段落格式
print('\n【3. 段落格式】')
line_spacings = {}
first_indents = {}
for p in doc.paragraphs:
    if p.paragraph_format.line_spacing:
        ls = p.paragraph_format.line_spacing
        line_spacings[ls] = line_spacings.get(ls, 0) + 1
    if p.paragraph_format.first_line_indent:
        fi = p.paragraph_format.first_line_indent.cm
        first_indents[fi] = first_indents.get(fi, 0) + 1

print(f'  行距分布: {line_spacings}')
print(f'  首行缩进分布: {first_indents}')
print(f'  建议: 行距1.5倍, 首行缩进0.5cm(2字符)')

# 4. 标题层级
print('\n【4. 标题层级】')
headings = []
for i, p in enumerate(doc.paragraphs):
    if p.style.name.startswith('Heading'):
        level = p.style.name.replace('Heading ', '')
        headings.append((i, level, p.text[:60]))

print(f'  标题总数: {len(headings)}')
print(f'  标题列表:')
for idx, (para_idx, level, text) in enumerate(headings):
    print(f'    {idx+1}. [H{level}] 段落{para_idx}: {text}')

# 检查标题编号连续性
print(f'\n  章节编号检查:')
chapter_nums = []
for para_idx, level, text in headings:
    if level == '2':  # 一级章节
        match = re.match(r'^(\d+)', text)
        if match:
            chapter_nums.append(int(match.group(1)))
print(f'    一级章节编号: {chapter_nums}')
expected = list(range(1, len(chapter_nums)+1))
if chapter_nums == expected:
    print(f'    ✓ 章节编号连续')
else:
    print(f'    ✗ 章节编号不连续，期望{expected}')

# 5. 摘要关键词
print('\n【5. 摘要和关键词】')
for i, p in enumerate(doc.paragraphs[:10]):
    if '摘要' in p.text:
        print(f'  摘要标题: 段落{i}')
    if '关键词' in p.text:
        print(f'  关键词: 段落{i} - {p.text[:100]}')

# 6. 图表题注
print('\n【6. 图表题注】')
fig_captions = []
table_captions = []
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if re.match(r'^图\s*\d+', text):
        fig_captions.append((i, text[:80]))
    if re.match(r'^表\s*\d+', text) or re.match(r'^表SOTA', text) or re.match(r'^表VOC', text):
        table_captions.append((i, text[:80]))

print(f'  图题注数: {len(fig_captions)}')
for idx, (para_idx, text) in enumerate(fig_captions):
    print(f'    图{idx+1}: 段落{para_idx} - {text}')

print(f'\n  表题注数: {len(table_captions)}')
for idx, (para_idx, text) in enumerate(table_captions):
    print(f'    表{idx+1}: 段落{para_idx} - {text}')

# 7. 参考文献格式
print('\n【7. 参考文献格式】')
ref_start = None
for i, p in enumerate(doc.paragraphs):
    if '参考文献' in p.text and p.style.name.startswith('Heading'):
        ref_start = i
        break

if ref_start:
    ref_count = 0
    ref_formats = {'ieee': 0, 'elsevier': 0, 'other': 0}
    for i in range(ref_start+1, len(doc.paragraphs)):
        text = doc.paragraphs[i].text.strip()
        if re.match(r'^\[\d+\]', text):
            ref_count += 1
            # 检查格式
            if re.match(r'^\[\d+\]\s+\w+\s+\w+', text):
                ref_formats['ieee'] += 1
            else:
                ref_formats['other'] += 1

    print(f'  参考文献总数: {ref_count}')
    print(f'  格式分布: {ref_formats}')
    print(f'  建议: Elsevier格式(作者, 年份, 标题, 期刊, 卷, 页) 或 IEEE格式([1] 作者. 标题. 期刊, 年)')

# 8. 公式编号
print('\n【8. 公式编号】')
formula_count = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 检查公式编号 (1), (2), 等
    if re.search(r'\(\d+\)\s*$', text) or re.search(r'^\(\d+\)', text):
        formula_count += 1
        if formula_count <= 5:
            print(f'  公式: 段落{i} - {text[:80]}')
print(f'  公式总数(近似): {formula_count}')

# 9. 整体统计
print('\n【9. 整体统计】')
total_chars = sum(len(p.text) for p in doc.paragraphs)
total_paragraphs = len(doc.paragraphs)
total_tables = len(doc.tables)
print(f'  总段落数: {total_paragraphs}')
print(f'  总表格数: {total_tables}')
print(f'  总字符数: {total_chars}')
print(f'  估计字数: {total_chars} 字(中文)')

# 10. 问题汇总
print('\n' + '=' * 70)
print('【问题汇总和建议】')
print('=' * 70)
print('''
需要修改的格式问题:
1. 字体: 当前为宋体，英文投稿需改为Times New Roman
2. 页边距: 左右3.17cm略宽，建议改为2.5cm
3. 参考文献格式: 当前为IEEE格式[1]，Elsevier期刊可接受，但建议统一
4. 摘要: 需要英文摘要和关键词(投稿时)
5. 图表题注: 需要英文翻译(投稿时)
6. 公式: 建议使用公式编辑器或LaTeX格式

中文版本格式基本规范，英文翻译时需要按照目标期刊格式重新排版。
''')

print('格式检查完成')
