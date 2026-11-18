# -*- coding: utf-8 -*-
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'

doc = Document(PAPER_PATH)

print('修复表SOTA-a-a → 表SOTA-a，表SOTA-b-b → 表SOTA-b')
for i, p in enumerate(doc.paragraphs):
    if '表SOTA-a-a' in p.text:
        old_text = p.text
        new_text = old_text.replace('表SOTA-a-a', '表SOTA-a')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 表SOTA-a-a → 表SOTA-a')
    elif '表SOTA-b-b' in p.text:
        old_text = p.text
        new_text = old_text.replace('表SOTA-b-b', '表SOTA-b')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 表SOTA-b-b → 表SOTA-b')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('修复完成')
