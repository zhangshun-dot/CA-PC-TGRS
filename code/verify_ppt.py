# -*- coding: utf-8 -*-
from pptx import Presentation

prs = Presentation(r'C:\Users\张三\Downloads\A-CSPC论文汇报_实验补充版.pptx')

print(f'总幻灯片数: {len(prs.slides)}')

print('\n=== 新增4张幻灯片 ===')
for i in range(17, 21):
    slide = prs.slides[i]
    title = ''
    tables = 0
    for shape in slide.shapes:
        if shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if '实验' in text and len(text) > 10:
                title = text[:60]
        if shape.has_table:
            tables += 1
    print(f'幻灯片{i+1}: {title} | 表格数: {tables}')

print('\n=== 全部幻灯片概览 ===')
for i, slide in enumerate(prs.slides):
    title_found = False
    for shape in slide.shapes:
        if shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if text and len(text) > 5:
                keywords = ['实验', '方法', '结论', '背景', '目录', '物理', '研究', '核心', '消融', '跨源', 'SOTA', 'VOC', '统计', '敏感', '面向', '参考', '探索']
                if any(kw in text for kw in keywords):
                    print(f'  {i+1}. {text[:50]}')
                    title_found = True
                    break
    if not title_found:
        print(f'  {i+1}. (图片/其他内容)')
