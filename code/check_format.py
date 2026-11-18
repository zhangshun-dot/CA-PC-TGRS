# -*- coding: utf-8 -*-
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v24.docx')

print('=== 1. 页面设置 ===')
section = doc.sections[0]
print(f'  纸张宽度: {section.page_width.cm:.2f} cm ({section.page_width.inches:.2f} inches)')
print(f'  纸张高度: {section.page_height.cm:.2f} cm ({section.page_height.inches:.2f} inches)')
print(f'  上边距: {section.top_margin.cm:.2f} cm')
print(f'  下边距: {section.bottom_margin.cm:.2f} cm')
print(f'  左边距: {section.left_margin.cm:.2f} cm')
print(f'  右边距: {section.right_margin.cm:.2f} cm')
print(f'  页眉距离: {section.header_distance.cm if section.header_distance else "N/A"} cm')
print(f'  页脚距离: {section.footer_distance.cm if section.footer_distance else "N/A"} cm')

print('\n=== 2. 段落格式（前10段）===')
for i, p in enumerate(doc.paragraphs[:15]):
    if p.text.strip():
        align = p.alignment
        align_str = '左对齐' if align == WD_ALIGN_PARAGRAPH.LEFT else \
                    '居中' if align == WD_ALIGN_PARAGRAPH.CENTER else \
                    '右对齐' if align == WD_ALIGN_PARAGRAPH.RIGHT else \
                    '两端对齐' if align == WD_ALIGN_PARAGRAPH.JUSTIFY else '默认'
        line_spacing = p.paragraph_format.line_spacing
        first_indent = p.paragraph_format.first_line_indent
        space_before = p.paragraph_format.space_before
        space_after = p.paragraph_format.space_after
        print(f'  段落{i}: 对齐={align_str}, 行距={line_spacing}, 首行缩进={first_indent}, 段前={space_before}, 段后={space_after}')
        print(f'    内容: {p.text[:50]}')

print('\n=== 3. 字体格式（前10段）===')
for i, p in enumerate(doc.paragraphs[:15]):
    if p.text.strip() and p.runs:
        run = p.runs[0]
        font_name = run.font.name
        font_size = run.font.size
        bold = run.font.bold
        italic = run.font.italic
        print(f'  段落{i}: 字体={font_name}, 字号={font_size}, 加粗={bold}, 斜体={italic}')
        print(f'    内容: {p.text[:50]}')

print('\n=== 4. 标题样式 ===')
for i, p in enumerate(doc.paragraphs[:50]):
    if p.style.name.startswith('Heading') or p.text.strip().startswith(('1 ', '2 ', '3 ', '4 ', '5 ', '6 ', '7 ', '8 ')):
        if p.text.strip():
            print(f'  段落{i}: 样式={p.style.name}, 内容={p.text[:60]}')
            if p.runs:
                print(f'    字体={p.runs[0].font.name}, 字号={p.runs[0].font.size}, 加粗={p.runs[0].font.bold}')

print('\n=== 5. 参考文献格式检查 ===')
ref_start = None
for i, p in enumerate(doc.paragraphs):
    if '参考文献' in p.text or 'References' in p.text:
        ref_start = i
        print(f'  参考文献开始于段落{i}: {p.text[:50]}')
        break

if ref_start:
    for i in range(ref_start, min(ref_start+10, len(doc.paragraphs))):
        if doc.paragraphs[i].text.strip():
            print(f'  段落{i}: {doc.paragraphs[i].text[:80]}')

print('\n=== 6. 图表题注格式 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if text.startswith('表') or text.startswith('图'):
        if p.runs:
            print(f'  段落{i}: 字体={p.runs[0].font.name}, 字号={p.runs[0].font.size}, 加粗={p.runs[0].font.bold}')
            print(f'    内容: {text[:60]}')
        if i > 20:  # 只看前几个
            break

print('\n=== 7. 文档统计 ===')
print(f'  总段落数: {len(doc.paragraphs)}')
print(f'  总表格数: {len(doc.tables)}')
# 统计字数
total_chars = sum(len(p.text) for p in doc.paragraphs)
print(f'  总字符数: {total_chars}')
