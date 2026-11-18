# -*- coding: utf-8 -*-
r"""extract_images.py — 提取docx中的所有内联图片
"""
from docx import Document
import os
from docx.oxml.ns import qn

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'
OUTPUT_DIR = r'C:\Users\张三\Desktop\实验数据\paper_images_v21'

os.makedirs(OUTPUT_DIR, exist_ok=True)

doc = Document(PAPER_PATH)

# 提取所有图片
image_count = 0
for rel in doc.part.rels.values():
    if "image" in rel.reltype:
        image_count += 1
        image_data = rel.target_part.blob
        image_ext = rel.target_part.content_type.split('/')[-1]
        if image_ext == 'jpeg':
            image_ext = 'jpg'
        output_path = os.path.join(OUTPUT_DIR, f'image_{image_count:02d}.{image_ext}')
        with open(output_path, 'wb') as f:
            f.write(image_data)
        print(f'提取图片 {image_count}: {output_path} ({len(image_data)} bytes)')

print(f'\n共提取 {image_count} 张图片')
print(f'输出目录: {OUTPUT_DIR}')

# 检查每个段落中的图片位置
print('\n=== 图片在文档中的位置 ===')
for i, p in enumerate(doc.paragraphs):
    has_image = False
    for run in p.runs:
        drawings = run._element.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing')
        if drawings:
            has_image = True
    if has_image:
        # 检查前后段落是否有图题
        prev_text = doc.paragraphs[i-1].text[:80] if i > 0 else ''
        next_text = doc.paragraphs[i+1].text[:80] if i < len(doc.paragraphs)-1 else ''
        print(f'段落{i}: 包含图片')
        print(f'  前段: {prev_text}')
        print(f'  后段: {next_text}')
        print()
